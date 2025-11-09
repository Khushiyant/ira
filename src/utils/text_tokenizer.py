"""Proper text tokenizer using HuggingFace transformers."""

import torch
from transformers import AutoTokenizer
from typing import Union, List, Optional, Dict


class TextTokenizerWrapper:
    """Wrapper for HuggingFace tokenizer compatible with the codebase."""
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-1.7B"):
        """
        Initialize text tokenizer.
        
        Args:
            model_name: HuggingFace model name or path
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Ensure we have a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        return_attention_mask: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Tokenize text to token IDs.
        
        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: Return format ('pt' for PyTorch)
            return_attention_mask: Return attention mask along with token IDs
            
        Returns:
            Token IDs tensor or dict with token_ids and attention_mask
        """
        outputs = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        
        if return_attention_mask:
            return {
                "input_ids": outputs["input_ids"],
                "attention_mask": outputs["attention_mask"],
            }
        else:
            return outputs["input_ids"]
    
    def tokenize(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ) -> torch.Tensor:
        """Tokenize text (alias for __call__)."""
        return self(text, max_length, padding, truncation)
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs tensor
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text string or list of strings
        """
        if token_ids.dim() == 1:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.tokenizer.pad_token_id
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self.tokenizer.bos_token_id
    
    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID."""
        return self.tokenizer.eos_token_id
