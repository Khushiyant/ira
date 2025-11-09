"""Audio tokenizer for converting between audio and discrete tokens."""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from .encodec_wrapper import EnCodecWrapper


class AudioTokenizer(nn.Module):
    """
    Audio tokenizer that handles conversion between audio waveforms and discrete tokens.
    Supports multiple codebook strategies and token flattening.
    """
    
    def __init__(
        self,
        codec: Optional[EnCodecWrapper] = None,
        flatten_strategy: str = "delay",  # "delay", "interleave", "first_only"
        add_special_tokens: bool = True,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = 2,
    ):
        """
        Initialize audio tokenizer.
        
        Args:
            codec: EnCodec wrapper instance
            flatten_strategy: How to flatten multiple codebooks
                - "delay": Use delay pattern (hierarchical)
                - "interleave": Interleave codebook tokens
                - "first_only": Use only first codebook
            add_special_tokens: Whether to add BOS/EOS tokens
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
        """
        super().__init__()
        self.codec = codec or EnCodecWrapper()
        self.flatten_strategy = flatten_strategy
        self.add_special_tokens = add_special_tokens
        
        # Special token IDs (offset from codec vocabulary)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.special_token_offset = 3  # Reserve first 3 IDs for special tokens
        
    def tokenize(self, audio: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        Tokenize audio to discrete tokens.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate of input
            
        Returns:
            Flattened discrete tokens [batch, seq_len]
        """
        # Encode to multi-codebook tokens
        codes = self.codec.encode(audio, sample_rate)  # [batch, n_codebooks, time]
        
        # Offset codes to account for special tokens
        codes = codes + self.special_token_offset
        
        # Flatten codebooks according to strategy
        tokens = self._flatten_codes(codes)
        
        # Add special tokens if needed
        if self.add_special_tokens:
            tokens = self._add_special_tokens(tokens)
            
        return tokens
    
    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete tokens back to audio.
        
        Args:
            tokens: Flattened discrete tokens [batch, seq_len]
            
        Returns:
            Audio waveform [batch, channels, time]
        """
        # Remove special tokens
        if self.add_special_tokens:
            tokens = self._remove_special_tokens(tokens)
            
        # Unflatten to multi-codebook format
        codes = self._unflatten_tokens(tokens)
        
        # Remove special token offset
        codes = codes - self.special_token_offset
        
        # Decode to audio
        audio = self.codec.decode(codes)
        
        return audio
    
    def _flatten_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Flatten multi-codebook codes to single sequence.
        
        Args:
            codes: Multi-codebook codes [batch, n_codebooks, time]
            
        Returns:
            Flattened tokens [batch, seq_len]
        """
        batch_size, n_codebooks, time_steps = codes.shape
        
        if self.flatten_strategy == "first_only":
            # Use only first codebook
            return codes[:, 0, :]
        
        elif self.flatten_strategy == "interleave":
            # Interleave codebooks: [c0_t0, c1_t0, c2_t0, c0_t1, c1_t1, ...]
            codes = codes.permute(0, 2, 1)  # [batch, time, n_codebooks]
            return codes.reshape(batch_size, -1)
        
        elif self.flatten_strategy == "delay":
            # Delay pattern (hierarchical): [c0_t0, c0_t1, ..., c1_t0, c1_t1, ...]
            return codes.reshape(batch_size, -1)
        
        else:
            raise ValueError(f"Unknown flatten strategy: {self.flatten_strategy}")
    
    def _unflatten_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Unflatten tokens back to multi-codebook format.
        
        Args:
            tokens: Flattened tokens [batch, seq_len]
            
        Returns:
            Multi-codebook codes [batch, n_codebooks, time]
        """
        batch_size, seq_len = tokens.shape
        n_codebooks = self.codec.get_num_codebooks()
        
        if self.flatten_strategy == "first_only":
            # Expand to all codebooks (use first codebook for all)
            time_steps = seq_len
            codes = tokens.unsqueeze(1).expand(-1, n_codebooks, -1)
            
        elif self.flatten_strategy == "interleave":
            # Reverse interleaving
            time_steps = seq_len // n_codebooks
            codes = tokens.reshape(batch_size, time_steps, n_codebooks)
            codes = codes.permute(0, 2, 1)
            
        elif self.flatten_strategy == "delay":
            # Reverse delay pattern
            time_steps = seq_len // n_codebooks
            codes = tokens.reshape(batch_size, n_codebooks, time_steps)
            
        else:
            raise ValueError(f"Unknown flatten strategy: {self.flatten_strategy}")
            
        return codes
    
    def _add_special_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Add BOS and EOS tokens."""
        batch_size = tokens.shape[0]
        device = tokens.device
        
        bos = torch.full((batch_size, 1), self.bos_token_id, device=device, dtype=tokens.dtype)
        eos = torch.full((batch_size, 1), self.eos_token_id, device=device, dtype=tokens.dtype)
        
        return torch.cat([bos, tokens, eos], dim=1)
    
    def _remove_special_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Remove BOS and EOS tokens."""
        # Find EOS token positions
        eos_mask = tokens == self.eos_token_id
        
        # Remove first (BOS) and last non-padding token (EOS)
        tokens = tokens[:, 1:]  # Remove BOS
        
        # Remove EOS (find first EOS per sequence)
        batch_size = tokens.shape[0]
        cleaned_tokens = []
        
        for i in range(batch_size):
            seq = tokens[i]
            eos_positions = (seq == self.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                eos_pos = eos_positions[0]
                cleaned_tokens.append(seq[:eos_pos])
            else:
                cleaned_tokens.append(seq)
                
        # Pad to same length
        max_len = max(len(seq) for seq in cleaned_tokens)
        padded = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            device=tokens.device,
            dtype=tokens.dtype
        )
        
        for i, seq in enumerate(cleaned_tokens):
            padded[i, :len(seq)] = seq
            
        return padded
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self.codec.vocab_size + self.special_token_offset
