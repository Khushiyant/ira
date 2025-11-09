"""Audio-to-LLM adapter for cross-modal alignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from einops import rearrange


class AudioToLLMAdapter(nn.Module):
    """
    Adapter module that transforms SpeechLM audio token embeddings
    into LLM-compatible embedding space for multimodal processing.
    """
    
    def __init__(
        self,
        audio_dim: int = 768,
        llm_dim: int = 4096,
        num_adapter_layers: int = 4,
        num_query_tokens: int = 32,
        use_perceiver: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize audio-to-LLM adapter.
        
        Args:
            audio_dim: Dimension of audio embeddings from SpeechLM
            llm_dim: Dimension of LLM embeddings
            num_adapter_layers: Number of adapter transformer layers
            num_query_tokens: Number of learnable query tokens (for perceiver)
            use_perceiver: Use perceiver-style architecture with query tokens
            dropout: Dropout rate
        """
        super().__init__()
        
        self.audio_dim = audio_dim
        self.llm_dim = llm_dim
        self.use_perceiver = use_perceiver
        self.num_query_tokens = num_query_tokens
        
        # Input projection
        self.input_proj = nn.Linear(audio_dim, llm_dim)
        
        if use_perceiver:
            # Learnable query tokens (perceiver-style)
            self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, llm_dim))
            
            # Cross-attention from queries to audio features
            self.cross_attention_layers = nn.ModuleList([
                PerceiverCrossAttentionLayer(llm_dim, num_heads=16, dropout=dropout)
                for _ in range(num_adapter_layers)
            ])
        else:
            # Standard transformer adapter
            adapter_layer = nn.TransformerEncoderLayer(
                d_model=llm_dim,
                nhead=16,
                dim_feedforward=llm_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.adapter_layers = nn.TransformerEncoder(
                adapter_layer,
                num_layers=num_adapter_layers
            )
        
        # Output normalization and projection
        self.output_norm = nn.LayerNorm(llm_dim)
        self.output_proj = nn.Linear(llm_dim, llm_dim)
        
        # Optional compression/pooling layer
        self.use_compression = True
        if self.use_compression:
            self.compression = nn.Sequential(
                nn.Linear(llm_dim, llm_dim),
                nn.GELU(),
                nn.LayerNorm(llm_dim),
            )
        
    def forward(
        self,
        audio_embeddings: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Transform audio embeddings to LLM embedding space.
        
        Args:
            audio_embeddings: Audio embeddings from SpeechLM [batch, audio_seq_len, audio_dim]
            audio_mask: Attention mask for audio [batch, audio_seq_len]
            hidden_states: Intermediate hidden states from SpeechLM layers (optional)
            
        Returns:
            Dict containing:
                - llm_embeddings: Adapted embeddings [batch, num_tokens, llm_dim]
                - attention_mask: Mask for LLM input [batch, num_tokens]
        """
        batch_size = audio_embeddings.shape[0]
        
        # Optionally aggregate hidden states from multiple layers
        if hidden_states is not None:
            # Use weighted sum of hidden states from different layers
            audio_embeddings = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)
        
        # Project to LLM dimension
        x = self.input_proj(audio_embeddings)  # [batch, audio_seq_len, llm_dim]
        
        if self.use_perceiver:
            # Use perceiver-style cross-attention with learnable queries
            queries = self.query_tokens.expand(batch_size, -1, -1)  # [batch, num_query_tokens, llm_dim]
            
            # Create attention mask for cross-attention
            if audio_mask is not None:
                cross_attn_mask = audio_mask.unsqueeze(1)  # [batch, 1, audio_seq_len]
            else:
                cross_attn_mask = None
            
            # Apply cross-attention layers
            for layer in self.cross_attention_layers:
                queries = layer(queries, x, cross_attn_mask)
            
            adapted = queries
            output_mask = torch.ones(batch_size, self.num_query_tokens, device=x.device)
            
        else:
            # Standard transformer adapter
            if audio_mask is not None:
                # Create padding mask for transformer
                padding_mask = ~audio_mask.bool()
            else:
                padding_mask = None
            
            adapted = self.adapter_layers(x, src_key_padding_mask=padding_mask)
            output_mask = audio_mask if audio_mask is not None else torch.ones_like(adapted[..., 0])
        
        # Apply output normalization and projection
        adapted = self.output_norm(adapted)
        
        if self.use_compression:
            adapted = self.compression(adapted)
        
        llm_embeddings = self.output_proj(adapted)
        
        return {
            "llm_embeddings": llm_embeddings,
            "attention_mask": output_mask,
        }
    
    def compress_sequence(
        self,
        embeddings: torch.Tensor,
        compression_factor: int = 2,
    ) -> torch.Tensor:
        """
        Compress sequence length by pooling (useful for long audio).
        
        Args:
            embeddings: Input embeddings [batch, seq_len, dim]
            compression_factor: Factor to compress by
            
        Returns:
            Compressed embeddings [batch, seq_len // compression_factor, dim]
        """
        batch_size, seq_len, dim = embeddings.shape
        
        # Reshape and pool
        new_seq_len = seq_len // compression_factor
        embeddings = embeddings[:, :new_seq_len * compression_factor, :]
        embeddings = embeddings.reshape(batch_size, new_seq_len, compression_factor, dim)
        embeddings = embeddings.mean(dim=2)
        
        return embeddings


class PerceiverCrossAttentionLayer(nn.Module):
    """Perceiver-style cross-attention layer."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Layer norms
        self.norm_queries = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        
        # Cross-attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention from queries to key-value pairs.
        
        Args:
            queries: Query tokens [batch, num_queries, dim]
            kv: Key-value features [batch, kv_len, dim]
            kv_mask: Mask for key-value [batch, 1, kv_len]
            
        Returns:
            Updated queries [batch, num_queries, dim]
        """
        batch_size, num_queries, _ = queries.shape
        kv_len = kv.shape[1]
        
        # Cross-attention with pre-norm
        residual = queries
        queries_norm = self.norm_queries(queries)
        kv_norm = self.norm_kv(kv)
        
        # Project Q, K, V
        q = self.q_proj(queries_norm)
        k = self.k_proj(kv_norm)
        v = self.v_proj(kv_norm)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask
        if kv_mask is not None:
            attn_scores = attn_scores.masked_fill(~kv_mask.unsqueeze(1).bool(), float("-inf"))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.dim)
        
        queries = residual + self.dropout(self.out_proj(attn_output))
        
        # Feed-forward with pre-norm
        residual = queries
        queries = self.norm_ffn(queries)
        queries = residual + self.ffn(queries)
        
        return queries


class MultiScaleAudioAdapter(nn.Module):
    """
    Multi-scale audio adapter that processes audio at different temporal resolutions
    and aggregates features for richer LLM input.
    """
    
    def __init__(
        self,
        audio_dim: int = 768,
        llm_dim: int = 4096,
        scales: List[int] = [1, 2, 4],
        num_adapter_layers: int = 4,
    ):
        """
        Initialize multi-scale adapter.
        
        Args:
            audio_dim: Audio embedding dimension
            llm_dim: LLM embedding dimension
            scales: Temporal compression scales to use
            num_adapter_layers: Number of adapter layers
        """
        super().__init__()
        
        self.scales = scales
        
        # Create adapter for each scale
        self.scale_adapters = nn.ModuleList([
            AudioToLLMAdapter(
                audio_dim=audio_dim,
                llm_dim=llm_dim,
                num_adapter_layers=num_adapter_layers,
                use_perceiver=True,
                num_query_tokens=32 // scale,
            )
            for scale in scales
        ])
        
        # Aggregation layer
        self.aggregation = nn.Sequential(
            nn.Linear(llm_dim * len(scales), llm_dim),
            nn.GELU(),
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, llm_dim),
        )
        
    def forward(
        self,
        audio_embeddings: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Process audio at multiple scales and aggregate."""
        
        scale_outputs = []
        
        for scale, adapter in zip(self.scales, self.scale_adapters):
            # Compress audio sequence if needed
            if scale > 1:
                compressed = adapter.compress_sequence(audio_embeddings, scale)
                if audio_mask is not None:
                    compressed_mask = audio_mask[:, ::scale]
                else:
                    compressed_mask = None
            else:
                compressed = audio_embeddings
                compressed_mask = audio_mask
            
            # Adapt to LLM space
            output = adapter(compressed, compressed_mask)
            scale_outputs.append(output["llm_embeddings"])
        
        # Aggregate across scales
        # Pad to same length
        max_len = max(x.shape[1] for x in scale_outputs)
        padded_outputs = []
        for x in scale_outputs:
            if x.shape[1] < max_len:
                padding = torch.zeros(
                    x.shape[0], max_len - x.shape[1], x.shape[2],
                    device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, padding], dim=1)
            padded_outputs.append(x)
        
        # Concatenate and aggregate
        concatenated = torch.cat(padded_outputs, dim=-1)
        aggregated = self.aggregation(concatenated)
        
        return {
            "llm_embeddings": aggregated,
            "attention_mask": torch.ones(aggregated.shape[0], aggregated.shape[1], device=aggregated.device),
        }
