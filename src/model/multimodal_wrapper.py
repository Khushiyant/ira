"""Multimodal LLM wrapper for speech + text integration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from .audio_adapter import AudioToLLMAdapter


class MultimodalLLMWrapper(nn.Module):
    """
    Wrapper that integrates speech tokens (via audio adapter) with a large language model
    for unified multimodal understanding and generation.
    """
    
    def __init__(
        self,
        llm_name_or_path: str,
        audio_adapter: AudioToLLMAdapter,
        freeze_llm: bool = False,
        use_lora: bool = True,
        load_in_4bit: bool = False,  # Add this argument
        lora_rank: int = 16,
        lora_alpha: int = 32,
        audio_start_token: str = "<|audio_start|>",
        audio_end_token: str = "<|audio_end|>",
    ):
        super().__init__()
        
        # Configure Quantization
        quantization_config = None
        if load_in_4bit:
            # Check if bitsandbytes is actually installed
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes as bnb
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            except ImportError:
                print("WARNING: bitsandbytes not installed. Skipping 4-bit quantization.")
                load_in_4bit = False

        # Load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path,
            quantization_config=quantization_config,
            # Use float16 for Mac (MPS) stability, bfloat16 for Nvidia
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, trust_remote_code=True)
        
        # Add special tokens for audio
        special_tokens = {
            "additional_special_tokens": [audio_start_token, audio_end_token]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        self.audio_start_token_id = self.tokenizer.convert_tokens_to_ids(audio_start_token)
        self.audio_end_token_id = self.tokenizer.convert_tokens_to_ids(audio_end_token)
        
        # Audio adapter
        self.audio_adapter = audio_adapter
        
        # Optionally freeze LLM
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # Apply LoRA if specified
        if use_lora and not freeze_llm:
            self._apply_lora(lora_rank, lora_alpha)
        
        self.llm_hidden_size = self.llm.config.hidden_size
        
    def _apply_lora(self, rank: int = 16, alpha: int = 32):
        """Apply LoRA to LLM for efficient fine-tuning."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
            )
            
            self.llm = get_peft_model(self.llm, lora_config)
            print(f"Applied LoRA with rank={rank}, alpha={alpha}")
            print(f"Trainable parameters: {self.llm.print_trainable_parameters()}")
            
        except ImportError:
            print("Warning: peft not installed, skipping LoRA. Install with: pip install peft")
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        audio_positions: Optional[List[int]] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multimodal input.
        
        Args:
            input_ids: Text token IDs [batch, text_seq_len]
            attention_mask: Text attention mask [batch, text_seq_len]
            audio_embeddings: Audio embeddings from adapter [batch, audio_seq_len, hidden_dim]
            audio_attention_mask: Audio attention mask [batch, audio_seq_len]
            audio_positions: Positions in text sequence where audio should be inserted
            labels: Labels for language modeling loss
            return_dict: Return dict or tuple
            
        Returns:
            Dict with loss, logits, and other outputs
        """
        # Get text embeddings from LLM
        if input_ids is not None:
            text_embeds = self.llm.get_input_embeddings()(input_ids)
        else:
            text_embeds = None
        
        # Integrate audio embeddings into text sequence
        if audio_embeddings is not None and text_embeds is not None:
            # Combine text and audio embeddings
            combined_embeds, combined_mask = self._integrate_modalities(
                text_embeds,
                attention_mask,
                audio_embeddings,
                audio_attention_mask,
                audio_positions,
            )
        elif audio_embeddings is not None:
            combined_embeds = audio_embeddings
            combined_mask = audio_attention_mask
        else:
            combined_embeds = text_embeds
            combined_mask = attention_mask
        
        # Forward through LLM with combined embeddings
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=labels,
            return_dict=return_dict,
            use_cache=False,
        )
        
        return outputs
    
    def _integrate_modalities(
        self,
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor,
        audio_embeds: torch.Tensor,
        audio_mask: torch.Tensor,
        audio_positions: Optional[List[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate audio and text embeddings at specified positions.
        
        Args:
            text_embeds: Text embeddings [batch, text_len, hidden_dim]
            text_mask: Text attention mask [batch, text_len]
            audio_embeds: Audio embeddings [batch, audio_len, hidden_dim]
            audio_mask: Audio attention mask [batch, audio_len]
            audio_positions: Where to insert audio in text sequence
            
        Returns:
            Combined embeddings and attention mask
        """
        batch_size = text_embeds.shape[0]
        
        if audio_positions is None:
            # Default: concatenate audio after text
            combined_embeds = torch.cat([text_embeds, audio_embeds], dim=1)
            combined_mask = torch.cat([text_mask, audio_mask], dim=1)
        else:
            # Insert audio at specified positions
            combined_list = []
            mask_list = []
            
            for b in range(batch_size):
                pos = audio_positions[b] if b < len(audio_positions) else len(text_embeds[b])
                
                # Split text at insertion point
                text_before = text_embeds[b, :pos]
                text_after = text_embeds[b, pos:]
                mask_before = text_mask[b, :pos]
                mask_after = text_mask[b, pos:]
                
                # Concatenate: text_before + audio + text_after
                combined = torch.cat([
                    text_before,
                    audio_embeds[b],
                    text_after,
                ], dim=0)
                
                combined_mask_b = torch.cat([
                    mask_before,
                    audio_mask[b],
                    mask_after,
                ], dim=0)
                
                combined_list.append(combined)
                mask_list.append(combined_mask_b)
            
            # Pad to same length
            max_len = max(x.shape[0] for x in combined_list)
            combined_embeds = torch.zeros(
                batch_size, max_len, text_embeds.shape[-1],
                device=text_embeds.device,
                dtype=text_embeds.dtype,
            )
            combined_mask = torch.zeros(
                batch_size, max_len,
                device=text_mask.device,
                dtype=text_mask.dtype,
            )
            
            for b, (emb, mask) in enumerate(zip(combined_list, mask_list)):
                combined_embeds[b, :len(emb)] = emb
                combined_mask[b, :len(mask)] = mask
        
        return combined_embeds, combined_mask
    
    @torch.no_grad()
    def generate(
        self,
        input_text: str,
        audio_embeddings: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text conditioned on text and audio input.
        
        Args:
            input_text: Input text prompt
            audio_embeddings: Audio embeddings [1, audio_len, hidden_dim]
            audio_attention_mask: Audio attention mask
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated text sequences
        """
        # Tokenize input text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.llm.device)
        attention_mask = inputs["attention_mask"].to(self.llm.device)
        
        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Combine with audio if provided
        if audio_embeddings is not None:
            combined_embeds, combined_mask = self._integrate_modalities(
                text_embeds,
                attention_mask,
                audio_embeddings,
                audio_attention_mask,
                audio_positions=None,  # Append audio after text
            )
        else:
            combined_embeds = text_embeds
            combined_mask = attention_mask
        
        # Generate
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode generated tokens
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return generated_texts
    
    def prepare_inputs_for_training(
        self,
        text: Union[str, List[str]],
        audio_embeddings: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for training.
        
        Args:
            text: Input text(s)
            audio_embeddings: Audio embeddings
            audio_attention_mask: Audio mask
            labels: Labels for loss computation
            
        Returns:
            Dict of model inputs
        """
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "audio_embeddings": audio_embeddings,
            "audio_attention_mask": audio_attention_mask,
            "labels": labels if labels is not None else inputs["input_ids"].clone(),
        }
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        # Save LLM
        self.llm.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        # Save audio adapter separately
        torch.save(
            self.audio_adapter.state_dict(),
            f"{save_directory}/audio_adapter.pt"
        )
    
    @classmethod
    def from_pretrained(cls, load_directory: str, audio_adapter: AudioToLLMAdapter):
        """Load model from directory."""
        # Load LLM
        llm = AutoModelForCausalLM.from_pretrained(load_directory)
        tokenizer = AutoTokenizer.from_pretrained(load_directory)
        
        # Create wrapper
        model = cls(load_directory, audio_adapter, freeze_llm=False)
        
        # Load audio adapter weights
        audio_adapter_weights = torch.load(f"{load_directory}/audio_adapter.pt")
        model.audio_adapter.load_state_dict(audio_adapter_weights)
        
        return model
