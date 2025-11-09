"""Voice style encoder using CLIP-style contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from einops import rearrange


class VoiceStyleEncoder(nn.Module):
    """
    Base voice style encoder that extracts speaker/voice embeddings from audio.
    Uses contrastive learning inspired by CLIP for voice style conditioning.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize voice style encoder.
        
        Args:
            input_dim: Input feature dimension (e.g., mel-spectrogram features)
            hidden_dim: Hidden dimension for transformer
            embedding_dim: Output embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling and projection to embedding space
        self.pool_type = "attention"  # "attention", "mean", "last"
        if self.pool_type == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode audio features to voice style embedding.
        
        Args:
            features: Audio features [batch, time, feature_dim]
            attention_mask: Attention mask [batch, time]
            
        Returns:
            Voice style embeddings [batch, embedding_dim]
        """
        batch_size, seq_len, _ = features.shape
        
        # Project input
        x = self.input_proj(features)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Create attention mask for transformer (inverted: 1 = mask, 0 = keep)
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()
        
        # Transform
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Pool across time
        if self.pool_type == "attention":
            attn_weights = self.attention_pool(x)  # [batch, time, 1]
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(-1), float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=1)
            pooled = (x * attn_weights).sum(dim=1)  # [batch, hidden_dim]
        elif self.pool_type == "mean":
            if attention_mask is not None:
                mask_expanded = (~attention_mask).unsqueeze(-1).float()
                pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                pooled = x.mean(dim=1)
        else:  # last
            pooled = x[:, -1, :]
        
        # Project to embedding space
        embeddings = self.output_proj(pooled)
        
        # Normalize embeddings (like CLIP)
        embeddings = F.normalize(embeddings, dim=-1)
        
        return embeddings
    
    def contrastive_loss(
        self,
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CLIP-style contrastive loss between audio and text embeddings.
        
        Args:
            audio_embeddings: Voice style embeddings [batch, embedding_dim]
            text_embeddings: Text embeddings [batch, embedding_dim]
            
        Returns:
            Loss value and metrics dict
        """
        # Normalize
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        # Compute similarity matrix
        logits = audio_embeddings @ text_embeddings.T / self.temperature
        
        # Symmetric cross-entropy loss
        batch_size = audio_embeddings.shape[0]
        labels = torch.arange(batch_size, device=audio_embeddings.device)
        
        loss_audio = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.T, labels)
        loss = (loss_audio + loss_text) / 2
        
        # Compute accuracy
        with torch.no_grad():
            pred_audio = logits.argmax(dim=-1)
            pred_text = logits.T.argmax(dim=-1)
            acc_audio = (pred_audio == labels).float().mean()
            acc_text = (pred_text == labels).float().mean()
        
        metrics = {
            "loss": loss.item(),
            "acc_audio": acc_audio.item(),
            "acc_text": acc_text.item(),
            "temperature": self.temperature.item(),
        }
        
        return loss, metrics


class CLIPVoiceEncoder(nn.Module):
    """
    CLIP-style voice encoder that learns aligned representations between
    audio features and text descriptions of voice characteristics.
    """
    
    def __init__(
        self,
        audio_encoder_config: Optional[Dict[str, Any]] = None,
        text_encoder_config: Optional[Dict[str, Any]] = None,
        embedding_dim: int = 256,
    ):
        """
        Initialize CLIP voice encoder.
        
        Args:
            audio_encoder_config: Config for audio encoder
            text_encoder_config: Config for text encoder
            embedding_dim: Shared embedding dimension
        """
        super().__init__()
        
        # Audio encoder
        audio_config = audio_encoder_config or {}
        audio_config["embedding_dim"] = embedding_dim
        self.audio_encoder = VoiceStyleEncoder(**audio_config)
        
        # Text encoder (simple transformer for voice descriptions)
        text_config = text_encoder_config or {}
        self.text_embedding_dim = text_config.get("vocab_size", 32000)
        self.text_hidden_dim = text_config.get("hidden_dim", 512)
        
        self.text_embed = nn.Embedding(self.text_embedding_dim, self.text_hidden_dim)
        self.text_pos_encoding = nn.Parameter(torch.randn(1, 512, self.text_hidden_dim))
        
        text_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.text_hidden_dim,
            nhead=8,
            dim_feedforward=self.text_hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_transformer = nn.TransformerEncoder(text_encoder_layer, num_layers=6)
        
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_hidden_dim, self.text_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.text_hidden_dim),
            nn.Linear(self.text_hidden_dim, embedding_dim),
        )
        
        self.embedding_dim = embedding_dim
        
    def encode_audio(
        self,
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode audio to voice style embedding."""
        return self.audio_encoder(audio_features, attention_mask)
    
    def encode_text(
        self,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text descriptions to embedding space.
        
        Args:
            text_tokens: Text token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Text embeddings [batch, embedding_dim]
        """
        batch_size, seq_len = text_tokens.shape
        
        # Embed tokens
        x = self.text_embed(text_tokens)
        
        # Add positional encoding
        x = x + self.text_pos_encoding[:, :seq_len, :]
        
        # Create attention mask for transformer
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()
        
        # Transform
        x = self.text_transformer(x, src_key_padding_mask=attention_mask)
        
        # Pool (use first token like BERT [CLS])
        pooled = x[:, 0, :]
        
        # Project to shared embedding space
        embeddings = self.text_proj(pooled)
        
        # Normalize
        embeddings = F.normalize(embeddings, dim=-1)
        
        return embeddings
    
    def forward(
        self,
        audio_features: torch.Tensor,
        text_tokens: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for contrastive training.
        
        Args:
            audio_features: Audio features [batch, time, feature_dim]
            text_tokens: Text descriptions [batch, seq_len]
            audio_mask: Audio attention mask
            text_mask: Text attention mask
            
        Returns:
            Dict with audio_embeddings, text_embeddings, loss, and metrics
        """
        audio_embeddings = self.encode_audio(audio_features, audio_mask)
        text_embeddings = self.encode_text(text_tokens, text_mask)
        
        loss, metrics = self.audio_encoder.contrastive_loss(audio_embeddings, text_embeddings)
        
        return {
            "audio_embeddings": audio_embeddings,
            "text_embeddings": text_embeddings,
            "loss": loss,
            "metrics": metrics,
        }
