"""Knowledge-distilled SpeechLM transformer for autoregressive audio token generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math
from einops import rearrange


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better length extrapolation."""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate rotary embeddings for sequence length."""
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb
    

def apply_rotary_pos_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to input tensor."""
    # x: [batch, seq_len, n_heads, head_dim]
    # freqs: [seq_len, head_dim]
    
    cos = freqs.cos()
    sin = freqs.sin()
    
    # Split x into two halves
    x1, x2 = x.chunk(2, dim=-1)
    
    # Apply rotation
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ], dim=-1)
    
    return rotated


class SpeechLMTransformerBlock(nn.Module):
    """Transformer block with causal self-attention for autoregressive generation."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_rope = use_rope
        
        # Pre-normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Multi-head attention
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Cross-attention for conditioning (optional)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.cross_attn_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_attn_kv = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.cross_out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
        conditioning_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            rope_freqs: Rotary position embeddings
            attention_mask: Causal attention mask
            conditioning: Conditioning vector (e.g., speaker embedding) [batch, cond_len, hidden_dim]
            conditioning_mask: Mask for conditioning
            
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        batch_size, seq_len, _ = x.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rope and rope_freqs is not None:
            q = apply_rotary_pos_emb(q.transpose(1, 2), rope_freqs).transpose(1, 2)
            k = apply_rotary_pos_emb(k.transpose(1, 2), rope_freqs).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        x = residual + self.dropout(self.out_proj(attn_output))
        
        # Cross-attention for conditioning
        if conditioning is not None:
            residual = x
            x = self.norm_cross(x)
            
            q = self.cross_attn_q(x)
            kv = self.cross_attn_kv(conditioning)
            k, v = kv.chunk(2, dim=-1)
            
            # Reshape
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            cond_len = conditioning.shape[1]
            k = k.view(batch_size, cond_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, cond_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute cross-attention
            cross_attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if conditioning_mask is not None:
                cross_attn_scores = cross_attn_scores + conditioning_mask
            
            cross_attn_weights = F.softmax(cross_attn_scores, dim=-1)
            cross_attn_weights = self.dropout(cross_attn_weights)
            
            cross_attn_output = torch.matmul(cross_attn_weights, v)
            cross_attn_output = cross_attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
            
            x = residual + self.dropout(self.cross_out_proj(cross_attn_output))
        
        # Feed-forward with pre-norm
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


class SpeechLMTransformer(nn.Module):
    """
    Knowledge-distilled SpeechLM transformer for autoregressive audio token generation.
    Conditioned on text input and speaker/voice style embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        use_rope: bool = True,
        speaker_embed_dim: int = 256,
        text_vocab_size: Optional[int] = None,
        tie_embeddings: bool = False,
    ):
        """
        Initialize SpeechLM transformer.
        
        Args:
            vocab_size: Audio token vocabulary size
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            use_rope: Use rotary positional embeddings
            speaker_embed_dim: Speaker embedding dimension
            text_vocab_size: Text token vocabulary size (for text conditioning)
            tie_embeddings: Tie input and output embeddings
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_rope = use_rope
        
        # Token embeddings for audio tokens
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # Text token embeddings (if text conditioning is used)
        self.text_vocab_size = text_vocab_size
        if text_vocab_size is not None:
            self.text_token_embed = nn.Embedding(text_vocab_size, hidden_dim)
        
        # Speaker/voice style conditioning
        self.speaker_proj = nn.Sequential(
            nn.Linear(speaker_embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Positional encoding
        if use_rope:
            self.rope = RotaryPositionalEmbedding(hidden_dim // num_heads, max_seq_len)
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            SpeechLMTransformerBlock(hidden_dim, num_heads, dropout, use_rope)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie embeddings if specified
        if tie_embeddings:
            self.lm_head.weight = self.token_embed.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        audio_tokens: torch.Tensor,
        speaker_embedding: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for autoregressive audio token generation.
        
        Args:
            audio_tokens: Audio token IDs [batch, seq_len]
            speaker_embedding: Speaker/voice style embeddings [batch, embed_dim]
            text_tokens: Text token IDs [batch, text_len] (optional)
            attention_mask: Attention mask [batch, seq_len]
            return_hidden_states: Return intermediate hidden states
            
        Returns:
            Dict with logits, loss (if labels provided), and optionally hidden states
        """
        batch_size, seq_len = audio_tokens.shape
        device = audio_tokens.device
        
        # Embed audio tokens
        x = self.token_embed(audio_tokens)
        
        # Add positional encoding
        if self.use_rope:
            rope_freqs = self.rope(seq_len, device)
        else:
            x = x + self.pos_embed[:, :seq_len, :]
            rope_freqs = None
        
        x = self.dropout(x)
        
        # Prepare speaker conditioning
        speaker_cond = self.speaker_proj(speaker_embedding).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Prepare text conditioning if provided
        if text_tokens is not None and self.text_vocab_size is not None:
            text_embed = self.text_token_embed(text_tokens)
            # Concatenate speaker and text conditioning
            conditioning = torch.cat([speaker_cond, text_embed], dim=1)
        else:
            conditioning = speaker_cond
        
        # Create causal attention mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1
        )
        
        if attention_mask is not None:
            # Combine with padding mask
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) + (1 - padding_mask) * float("-inf")
        else:
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Pass through transformer layers
        hidden_states = []
        for layer in self.layers:
            x = layer(x, rope_freqs, causal_mask, conditioning)
            if return_hidden_states:
                hidden_states.append(x)
        
        # Output projection
        x = self.norm_out(x)
        logits = self.lm_head(x)
        
        output = {"logits": logits}
        
        if return_hidden_states:
            output["hidden_states"] = hidden_states
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        speaker_embedding: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        start_token_id: int = 0,
    ) -> torch.Tensor:
        """
        Autoregressive generation of audio tokens.
        
        Args:
            speaker_embedding: Speaker embedding [batch, embed_dim]
            text_tokens: Text conditioning tokens [batch, text_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            start_token_id: Start token ID
            
        Returns:
            Generated audio tokens [batch, seq_len]
        """
        batch_size = speaker_embedding.shape[0]
        device = speaker_embedding.device
        
        # Start with start token
        generated = torch.full((batch_size, 1), start_token_id, device=device, dtype=torch.long)
        
        for _ in range(max_length - 1):
            # Forward pass
            outputs = self.forward(generated, speaker_embedding, text_tokens)
            logits = outputs["logits"][:, -1, :] / temperature  # [batch, vocab_size]
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
