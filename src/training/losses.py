"""Loss functions for speech model training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss for transferring knowledge from teacher to student."""
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        reduction: str = "batchmean",
    ):
        """
        Initialize KD loss.
        
        Args:
            temperature: Temperature for softening distributions
            alpha: Weight for distillation loss vs hard target loss
            reduction: Reduction method for KL divergence
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        self.kl_div = nn.KLDivLoss(reduction=reduction)
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_logits: Logits from student model [batch, seq_len, vocab_size]
            teacher_logits: Logits from teacher model [batch, seq_len, vocab_size]
            labels: Ground truth labels [batch, seq_len]
            
        Returns:
            Dict with total loss and component losses
        """
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Soft predictions from student
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence loss (distillation loss)
        distillation_loss = self.kl_div(soft_predictions, soft_targets) * (self.temperature ** 2)
        
        # Hard target loss (if labels provided)
        if labels is not None:
            hard_loss = self.ce_loss(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1)
            )
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        else:
            hard_loss = torch.tensor(0.0)
            total_loss = distillation_loss
        
        return {
            "loss": total_loss,
            "distillation_loss": distillation_loss,
            "hard_loss": hard_loss,
        }


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning aligned representations."""
    
    def __init__(
        self,
        temperature: float = 0.07,
        use_cosine_similarity: bool = True,
    ):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter
            use_cosine_similarity: Use cosine similarity instead of dot product
        """
        super().__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        
    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute symmetric contrastive loss.
        
        Args:
            embeddings_a: First set of embeddings [batch, dim]
            embeddings_b: Second set of embeddings [batch, dim]
            
        Returns:
            Dict with loss and accuracy
        """
        batch_size = embeddings_a.shape[0]
        device = embeddings_a.device
        
        # Normalize if using cosine similarity
        if self.use_cosine_similarity:
            embeddings_a = F.normalize(embeddings_a, dim=-1)
            embeddings_b = F.normalize(embeddings_b, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=device)
        
        # Symmetric loss
        loss_a = F.cross_entropy(similarity, labels)
        loss_b = F.cross_entropy(similarity.T, labels)
        loss = (loss_a + loss_b) / 2
        
        # Compute accuracy
        with torch.no_grad():
            pred_a = similarity.argmax(dim=-1)
            pred_b = similarity.T.argmax(dim=-1)
            acc_a = (pred_a == labels).float().mean()
            acc_b = (pred_b == labels).float().mean()
            acc = (acc_a + acc_b) / 2
        
        return {
            "loss": loss,
            "accuracy": acc,
        }


class SequenceAlignmentLoss(nn.Module):
    """Loss for aligning speech and text token sequences."""
    
    def __init__(
        self,
        similarity_metric: str = "cosine",
        alignment_temperature: float = 0.1,
    ):
        super().__init__()
        self.similarity_metric = similarity_metric
        self.temperature = alignment_temperature
        
    def forward(
        self,
        speech_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        speech_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute alignment loss between speech and text sequences.
        
        Args:
            speech_embeddings: Speech embeddings [batch, speech_len, dim]
            text_embeddings: Text embeddings [batch, text_len, dim]
            speech_mask: Mask for speech [batch, speech_len]
            text_mask: Mask for text [batch, text_len]
            
        Returns:
            Dict with alignment loss
        """
        # Normalize embeddings
        speech_norm = F.normalize(speech_embeddings, dim=-1)
        text_norm = F.normalize(text_embeddings, dim=-1)
        
        # Compute similarity matrix [batch, speech_len, text_len]
        similarity = torch.matmul(speech_norm, text_norm.transpose(-2, -1))
        
        # Apply masks
        if speech_mask is not None:
            similarity = similarity.masked_fill(~speech_mask.unsqueeze(-1).bool(), float("-inf"))
        if text_mask is not None:
            similarity = similarity.masked_fill(~text_mask.unsqueeze(1).bool(), float("-inf"))
        
        # Soft alignment: compute alignment weights
        speech_to_text = F.softmax(similarity / self.temperature, dim=-1)  # Align speech to text
        text_to_speech = F.softmax(similarity.transpose(-2, -1) / self.temperature, dim=-1)  # Align text to speech
        
        # Alignment loss: encourage each position to align with a single position
        # Use entropy as regularization (lower entropy = more focused alignment)
        speech_entropy = -(speech_to_text * torch.log(speech_to_text + 1e-8)).sum(dim=-1).mean()
        text_entropy = -(text_to_speech * torch.log(text_to_speech + 1e-8)).sum(dim=-1).mean()
        
        # Encourage high maximum similarity
        max_similarity = similarity.max(dim=-1)[0].mean()
        
        loss = speech_entropy + text_entropy - max_similarity
        
        return {
            "loss": loss,
            "max_similarity": max_similarity,
            "speech_entropy": speech_entropy,
            "text_entropy": text_entropy,
        }


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining multiple objectives."""
    
    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        use_uncertainty_weighting: bool = False,
    ):
        """
        Initialize multi-task loss.
        
        Args:
            loss_weights: Manual weights for each loss component
            use_uncertainty_weighting: Use learnable uncertainty-based weighting
        """
        super().__init__()
        
        self.loss_weights = loss_weights or {}
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        if use_uncertainty_weighting:
            # Learnable log variances for uncertainty weighting
            self.log_vars = nn.ParameterDict()
        
    def add_task(self, task_name: str, weight: float = 1.0):
        """Add a task to multi-task learning."""
        if task_name not in self.loss_weights:
            self.loss_weights[task_name] = weight
            
        if self.use_uncertainty_weighting and task_name not in self.log_vars:
            self.log_vars[task_name] = nn.Parameter(torch.zeros(1))
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Combine multiple losses.
        
        Args:
            losses: Dict of individual losses
            
        Returns:
            Dict with total loss and weighted components
        """
        total_loss = 0.0
        weighted_losses = {}
        
        for task_name, loss_value in losses.items():
            if self.use_uncertainty_weighting and task_name in self.log_vars:
                # Uncertainty weighting: loss / (2 * var) + log(var)
                precision = torch.exp(-self.log_vars[task_name])
                weighted_loss = precision * loss_value + self.log_vars[task_name]
            else:
                weight = self.loss_weights.get(task_name, 1.0)
                weighted_loss = weight * loss_value
            
            weighted_losses[f"weighted_{task_name}"] = weighted_loss
            total_loss = total_loss + weighted_loss
        
        return {
            "loss": total_loss,
            **weighted_losses,
            **losses,  # Include original losses
        }
