"""Reinforced Behavior Alignment (RBA) trainer for aligning SpeechLM with LLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
import wandb
from accelerate import Accelerator

from ..model.speech_lm import SpeechLMTransformer
from ..model.multimodal_wrapper import MultimodalLLMWrapper
from .losses import KnowledgeDistillationLoss, MultiTaskLoss


class RBATrainer:
    """
    Reinforced Behavior Alignment trainer for aligning SpeechLM with LLM behavior.
    Uses reinforcement learning to close the knowledge gap between speech-centric
    generation and text-centric understanding.
    """
    
    def __init__(
        self,
        speech_lm: SpeechLMTransformer,
        multimodal_llm: MultimodalLLMWrapper,
        kd_loss: Optional[KnowledgeDistillationLoss] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        use_rl: bool = True,
        rl_reward_weight: float = 0.3,
        ppo_clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        use_wandb: bool = True,
        project_name: str = "speech-llm-rba",
    ):
        """
        Initialize RBA trainer.
        
        Args:
            speech_lm: SpeechLM model (student)
            multimodal_llm: Multimodal LLM wrapper (teacher)
            kd_loss: Knowledge distillation loss
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Warmup steps
            max_grad_norm: Max gradient norm for clipping
            use_rl: Use reinforcement learning for alignment
            rl_reward_weight: Weight for RL reward
            ppo_clip_epsilon: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            use_wandb: Use Weights & Biases logging
            project_name: W&B project name
        """
        self.speech_lm = speech_lm
        self.multimodal_llm = multimodal_llm
        self.kd_loss = kd_loss or KnowledgeDistillationLoss()
        
        # Accelerator for distributed training
        self.accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=4,
        )
        
        # Optimizers
        self.optimizer = torch.optim.AdamW(
            speech_lm.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler(warmup_steps)
        
        # RL parameters
        self.use_rl = use_rl
        self.rl_reward_weight = rl_reward_weight
        self.ppo_clip_epsilon = ppo_clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        if use_rl:
            # Value network for estimating returns
            self.value_head = nn.Sequential(
                nn.Linear(speech_lm.hidden_dim, speech_lm.hidden_dim),
                nn.GELU(),
                nn.Linear(speech_lm.hidden_dim, 1),
            ).to(self.accelerator.device)
            
            self.value_optimizer = torch.optim.AdamW(
                self.value_head.parameters(),
                lr=learning_rate,
            )
        
        self.max_grad_norm = max_grad_norm
        
        # Multi-task loss
        self.multi_task_loss = MultiTaskLoss(
            loss_weights={
                "kd": 1.0,
                "rl": rl_reward_weight,
                "value": value_loss_coef,
                "entropy": entropy_coef,
            },
            use_uncertainty_weighting=True,
        )
        
        # Prepare models with accelerator
        self.speech_lm, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.speech_lm, self.optimizer, self.scheduler
        )
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb and self.accelerator.is_main_process:
            wandb.init(project=project_name, config={
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "use_rl": use_rl,
                "rl_reward_weight": rl_reward_weight,
            })
        
        self.global_step = 0
        
    def _create_scheduler(self, warmup_steps: int):
        """Create learning rate scheduler with warmup."""
        from transformers import get_cosine_schedule_with_warmup
        
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=100000,  # Will be updated during training
        )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        num_epochs: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch
            num_epochs: Total number of epochs
            
        Returns:
            Dict of average metrics for the epoch
        """
        self.speech_lm.train()
        
        total_losses = {}
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not self.accelerator.is_main_process,
        )
        
        for batch in progress_bar:
            losses = self.train_step(batch)
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value
            
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": losses.get("loss", 0.0),
                "kd_loss": losses.get("kd_loss", 0.0),
                "lr": self.optimizer.param_groups[0]["lr"],
            })
            
            self.global_step += 1
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dict of losses
        """
        with self.accelerator.accumulate(self.speech_lm):
            # Unpack batch
            audio_tokens = batch["audio_tokens"]
            speaker_embeddings = batch["speaker_embeddings"]
            text_tokens = batch.get("text_tokens")
            labels = batch.get("labels", audio_tokens)
            
            # Forward pass through SpeechLM
            student_outputs = self.speech_lm(
                audio_tokens=audio_tokens,
                speaker_embedding=speaker_embeddings,
                text_tokens=text_tokens,
                return_hidden_states=self.use_rl,
            )
            student_logits = student_outputs["logits"]
            
            # Get teacher predictions (frozen)
            with torch.no_grad():
                # Adapt audio embeddings for LLM
                adapted_audio = self.multimodal_llm.audio_adapter(
                    student_outputs.get("hidden_states", [student_logits])[-1]
                )
                
                # Get teacher logits
                teacher_outputs = self.multimodal_llm.llm(
                    inputs_embeds=adapted_audio["llm_embeddings"],
                    attention_mask=adapted_audio["attention_mask"],
                )
                teacher_logits = teacher_outputs.logits
            
            # Knowledge distillation loss
            kd_losses = self.kd_loss(student_logits, teacher_logits, labels)
            
            losses = {"kd": kd_losses["loss"]}
            
            # Reinforcement learning alignment
            if self.use_rl:
                rl_losses = self._compute_rl_losses(
                    student_outputs,
                    teacher_logits,
                    labels,
                )
                losses.update(rl_losses)
            
            # Combine losses
            combined_losses = self.multi_task_loss(losses)
            total_loss = combined_losses["loss"]
            
            # Backward pass
            self.accelerator.backward(total_loss)
            
            # Gradient clipping
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.speech_lm.parameters(),
                    self.max_grad_norm,
                )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            if self.use_rl:
                self.value_optimizer.step()
                self.value_optimizer.zero_grad()
            
            # Log metrics
            metrics = {
                "loss": total_loss.item(),
                "kd_loss": kd_losses["loss"].item(),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            
            if self.use_rl:
                metrics.update({
                    f"rl_{k}": v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in rl_losses.items()
                })
            
            if self.use_wandb and self.accelerator.is_main_process:
                wandb.log(metrics, step=self.global_step)
            
            return metrics
    
    def _compute_rl_losses(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reinforcement learning losses for behavior alignment.
        
        Args:
            student_outputs: Outputs from student model
            teacher_logits: Logits from teacher model
            labels: Ground truth labels
            
        Returns:
            Dict of RL losses
        """
        student_logits = student_outputs["logits"]
        hidden_states = student_outputs.get("hidden_states", [])[-1]
        
        # Compute log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Sample actions from student policy
        student_probs = F.softmax(student_logits, dim=-1)
        actions = torch.multinomial(
            student_probs.view(-1, student_probs.size(-1)),
            num_samples=1
        ).view(student_logits.shape[:2])
        
        # Get action log probabilities
        action_log_probs = student_log_probs.gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute rewards based on teacher agreement
        with torch.no_grad():
            # Reward = how much teacher agrees with student action
            teacher_action_probs = teacher_probs.gather(
                -1, actions.unsqueeze(-1)
            ).squeeze(-1)
            
            # Also consider correctness
            correct_actions = (actions == labels).float()
            
            # Combined reward
            rewards = teacher_action_probs + 0.5 * correct_actions
        
        # Compute value estimates
        values = self.value_head(hidden_states).squeeze(-1)
        
        # Compute advantages (rewards - baseline)
        with torch.no_grad():
            advantages = rewards - values.detach()
        
        # PPO policy loss
        ratio = torch.exp(action_log_probs - action_log_probs.detach())
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.ppo_clip_epsilon,
            1 + self.ppo_clip_epsilon,
        )
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages,
        ).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, rewards)
        
        # Entropy bonus (encourage exploration)
        entropy = -(student_probs * student_log_probs).sum(dim=-1).mean()
        
        return {
            "policy": policy_loss,
            "value": value_loss,
            "entropy": -entropy,  # Negative because we want to maximize
            "reward": rewards.mean(),
        }
    
    def save_checkpoint(self, save_path: str, epoch: int):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.accelerator.unwrap_model(self.speech_lm).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        
        if self.use_rl:
            checkpoint["value_head_state_dict"] = self.value_head.state_dict()
            checkpoint["value_optimizer_state_dict"] = self.value_optimizer.state_dict()
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, load_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(load_path, map_location=self.accelerator.device)
        
        self.accelerator.unwrap_model(self.speech_lm).load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.use_rl and "value_head_state_dict" in checkpoint:
            self.value_head.load_state_dict(checkpoint["value_head_state_dict"])
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        
        self.global_step = checkpoint["global_step"]
        
        print(f"Checkpoint loaded from {load_path}")
        return checkpoint["epoch"]
