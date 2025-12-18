"""Reinforced Behavior Alignment (RBA) trainer with Staged Training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from tqdm import tqdm
import wandb
from accelerate import Accelerator

from ..model.speech_lm import SpeechLMTransformer
from ..model.multimodal_wrapper import MultimodalLLMWrapper
from .losses import KnowledgeDistillationLoss, MultiTaskLoss

class RBATrainer:
    """
    Trainer supporting 3-stage training:
    Stage 1: SpeechLM Pre-training (Audio AR)
    Stage 2: Adapter Alignment (Audio -> Text)
    Stage 3: RBA (RL + KD)
    """
    
    def __init__(
        self,
        speech_lm: SpeechLMTransformer,
        multimodal_llm: MultimodalLLMWrapper,
        stage: int = 1,  # 1, 2, or 3
        kd_loss: Optional[KnowledgeDistillationLoss] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        rl_reward_weight: float = 0.3,
        ppo_clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        use_wandb: bool = True,
        project_name: str = "speech-llm-rba",
    ):
        self.speech_lm = speech_lm
        self.multimodal_llm = multimodal_llm
        self.stage = stage
        self.kd_loss = kd_loss or KnowledgeDistillationLoss()
        
        # Accelerator
        self.accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=4,
        )
        
        # Configure Optimizers based on Stage
        if self.stage == 1:
            # Optimize only SpeechLM
            self.optimizer = torch.optim.AdamW(
                self.speech_lm.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            # Freeze other components to be safe
            self.multimodal_llm.requires_grad_(False)
            
        elif self.stage == 2:
            # Optimize only Adapter
            self.speech_lm.requires_grad_(False)
            self.multimodal_llm.llm.requires_grad_(False) # Freeze LLM
            self.multimodal_llm.audio_adapter.requires_grad_(True)
            
            self.optimizer = torch.optim.AdamW(
                self.multimodal_llm.audio_adapter.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
            
        elif self.stage == 3:
            # Full RBA: Optimize SpeechLM
            self.speech_lm.requires_grad_(True)
            # Generally keep Adapter/LLM frozen in stage 3
            self.multimodal_llm.requires_grad_(False)
            
            self.optimizer = torch.optim.AdamW(
                self.speech_lm.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        
        # Scheduler
        self.scheduler = self._create_scheduler(warmup_steps)
        
        # RL Parameters
        self.rl_reward_weight = rl_reward_weight
        self.ppo_clip_epsilon = ppo_clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # Value Head (Only needed for Stage 3)
        if self.stage == 3:
            self.value_head = nn.Sequential(
                nn.Linear(speech_lm.hidden_dim, speech_lm.hidden_dim),
                nn.GELU(),
                nn.Linear(speech_lm.hidden_dim, 1),
            ).to(self.accelerator.device)
            
            self.value_optimizer = torch.optim.AdamW(
                self.value_head.parameters(), lr=learning_rate
            )
        else:
            self.value_head = None
            self.value_optimizer = None
        
        self.max_grad_norm = max_grad_norm
        self.multi_task_loss = MultiTaskLoss()

        # Prepare models
        if self.stage == 1:
            self.speech_lm, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.speech_lm, self.optimizer, self.scheduler
            )
        elif self.stage == 2:
            self.multimodal_llm, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.multimodal_llm, self.optimizer, self.scheduler
            )
            self.speech_lm = self.speech_lm.to(self.accelerator.device)
        elif self.stage == 3:
            self.speech_lm, self.optimizer, self.scheduler, self.value_head, self.value_optimizer = self.accelerator.prepare(
                self.speech_lm, self.optimizer, self.scheduler, self.value_head, self.value_optimizer
            )
            self.multimodal_llm = self.multimodal_llm.to(self.accelerator.device)

        # Logging
        self.use_wandb = use_wandb
        if use_wandb and self.accelerator.is_main_process:
            wandb.init(project=project_name, config={
                "stage": stage,
                "learning_rate": learning_rate,
                "rl_reward_weight": rl_reward_weight,
            })
        
        self.global_step = 0
        
    def _create_scheduler(self, warmup_steps: int):
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=100000
        )
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, num_epochs: int) -> Dict[str, float]:
        if self.stage == 1 or self.stage == 3:
            self.speech_lm.train()
        if self.stage == 2:
            self.multimodal_llm.audio_adapter.train()
            
        total_losses = {}
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Stage {self.stage} | Epoch {epoch+1}/{num_epochs}",
            disable=not self.accelerator.is_main_process,
        )
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            if self.stage == 1:
                losses = self.train_step_stage_1(batch)
            elif self.stage == 2:
                losses = self.train_step_stage_2(batch)
            else:
                losses = self.train_step_stage_3(batch)
            
            # Accumulate logs
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value
            
            num_batches += 1
            self.global_step += 1
            
            progress_bar.set_postfix({"loss": losses.get("loss", 0.0)})
        
        return {k: v / num_batches for k, v in total_losses.items()}

    def train_step_stage_1(self, batch):
        """Stage 1: SpeechLM Pre-training (Autoregressive Audio)"""
        with self.accelerator.accumulate(self.speech_lm):
            audio_tokens = batch["audio_tokens"]
            if "speaker_embeddings" in batch:
                speaker_embeddings = batch["speaker_embeddings"]
            else:
                raise ValueError("No speaker info provided! Check Collator.")

            outputs = self.speech_lm(
                audio_tokens=audio_tokens,
                speaker_embedding=speaker_embeddings,
                attention_mask=batch.get("audio_mask")
            )
            logits = outputs["logits"]
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = audio_tokens[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.speech_lm.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            return {"loss": loss.item()}

    def train_step_stage_2(self, batch):
        """Stage 2: Adapter Alignment (Audio -> Text)"""
        with self.accelerator.accumulate(self.multimodal_llm):
            audio_tokens = batch["audio_tokens"]
            text_tokens = batch["text_tokens"]
            speaker_embeddings = batch["speaker_embeddings"]
            
            with torch.no_grad():
                student_out = self.speech_lm(
                    audio_tokens=audio_tokens,
                    speaker_embedding=speaker_embeddings,
                    return_hidden_states=True
                )
                hidden_states = student_out["hidden_states"][-1] 

            adapter_out = self.multimodal_llm.audio_adapter(
                hidden_states, 
                audio_mask=batch.get("audio_mask")
            )
            audio_embeds = adapter_out["llm_embeddings"]
            audio_mask = adapter_out["attention_mask"]

            batch_size = audio_tokens.size(0)
            llm_out = self.multimodal_llm(
                input_ids=text_tokens,
                attention_mask=batch.get("text_mask"),
                audio_embeddings=audio_embeds,
                audio_attention_mask=audio_mask,
                audio_positions=[0] * batch_size, 
                labels=text_tokens 
            )
            
            loss = llm_out.loss
            
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.multimodal_llm.audio_adapter.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            return {"loss": loss.item()}

    def train_step_stage_3(self, batch):
        """Stage 3: Full RBA (Reinforcement Learning)"""
        with self.accelerator.accumulate(self.speech_lm):
            audio_tokens = batch["audio_tokens"]
            speaker_embeddings = batch["speaker_embeddings"]
            labels = batch.get("labels", audio_tokens)
            
            # 1. Student Forward
            student_outputs = self.speech_lm(
                audio_tokens=audio_tokens,
                speaker_embedding=speaker_embeddings,
                return_hidden_states=True,
            )
            student_logits = student_outputs["logits"]
            
            # 2. Teacher Forward (Frozen)
            with torch.no_grad():
                # Get Teacher Logits for Distillation
                hidden = student_outputs["hidden_states"][-1]
                adapted_audio = self.multimodal_llm.audio_adapter(hidden)
                
                teacher_outputs = self.multimodal_llm.llm(
                    inputs_embeds=adapted_audio["llm_embeddings"],
                    attention_mask=adapted_audio["attention_mask"],
                )
                teacher_logits = teacher_outputs.logits

            # 3. Calculate Losses
            # Note: KD is tricky if vocab sizes differ. 
            # If SpeechLM vocab != LLM vocab, standard KD doesn't work directly on logits.
            # Assuming we rely on RL here or have a mapping. 
            # For this corrected code, we will focus on the RL part + Hard Loss.
            
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)), 
                labels.view(-1)
            )
            losses = {"hard_loss": hard_loss}

            # RL Alignment
            rl_losses = self._compute_rl_losses(
                student_outputs,
                teacher_logits, 
                labels
            )
            losses.update(rl_losses)
            
            # Combine
            # loss = hard_loss + weight * rl_policy_loss + value_loss
            total_loss = hard_loss + \
                         self.rl_reward_weight * rl_losses["policy"] + \
                         self.value_loss_coef * rl_losses["value"] - \
                         self.entropy_coef * rl_losses["entropy"]

            self.accelerator.backward(total_loss)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.speech_lm.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            self.value_optimizer.step()
            self.value_optimizer.zero_grad()
            
            return {"loss": total_loss.item(), "reward": rl_losses["reward"].item()}

    def _compute_rl_losses(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO losses."""
        student_logits = student_outputs["logits"]
        hidden_states = student_outputs["hidden_states"][-1]
        
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        
        # Sample actions
        student_probs = F.softmax(student_logits, dim=-1)
        actions = torch.multinomial(
            student_probs.view(-1, student_probs.size(-1)),
            num_samples=1
        ).view(student_logits.shape[:2])
        
        action_log_probs = student_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Calculate Rewards
        # Simple reward: How confident is teacher about these actions?
        # Note: This implies mapping audio tokens -> text tokens, or we assume implicit alignment.
        # Since we can't map 1:1, we use the teacher's *internal* confidence on the *embedding*
        # (This is a simplification; real RLHF for speech is complex).
        # We will use the 'correctness' reward as a stable baseline.
        
        with torch.no_grad():
            correct_actions = (actions == labels).float()
            rewards = correct_actions # [B, L]
        
        # Value Estimates
        values = self.value_head(hidden_states).squeeze(-1)
        
        # Advantages
        with torch.no_grad():
            advantages = rewards - values.detach()
        
        # PPO Policy Loss
        ratio = torch.exp(action_log_probs - action_log_probs.detach())
        clipped_ratio = torch.clamp(ratio, 1 - self.ppo_clip_epsilon, 1 + self.ppo_clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value Loss
        value_loss = F.mse_loss(values, rewards)
        
        # Entropy
        entropy = -(student_probs * student_log_probs).sum(dim=-1).mean()
        
        return {
            "policy": policy_loss,
            "value": value_loss,
            "entropy": entropy,
            "reward": rewards.mean(),
        }

    def save_checkpoint(self, save_path: str, epoch: int):
        checkpoint = {
            "epoch": epoch,
            "stage": self.stage,
            "model_state": self.accelerator.unwrap_model(
                self.speech_lm if self.stage != 2 else self.multimodal_llm.audio_adapter
            ).state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.stage == 3:
            checkpoint["value_head"] = self.value_head.state_dict()
            
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")