"""End-to-end inference pipeline for speech-to-LLM system."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
import torchaudio
from pathlib import Path

from ..codec import EnCodecWrapper, AudioTokenizer
from ..model import (
    VoiceStyleEncoder,
    CLIPVoiceEncoder,
    SpeechLMTransformer,
    AudioToLLMAdapter,
    MultimodalLLMWrapper,
)


class SpeechToLLMPipeline:
    """
    End-to-end pipeline for processing speech input through SpeechLM
    and feeding to LLM for multimodal understanding.
    """
    
    def __init__(
        self,
        speech_lm: SpeechLMTransformer,
        multimodal_llm: MultimodalLLMWrapper,
        audio_codec: Optional[EnCodecWrapper] = None,
        voice_encoder: Optional[Union[VoiceStyleEncoder, CLIPVoiceEncoder]] = None,
        text_tokenizer = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize inference pipeline.
        
        Args:
            speech_lm: SpeechLM model
            multimodal_llm: Multimodal LLM wrapper
            audio_codec: Audio codec for tokenization
            voice_encoder: Voice style encoder
            text_tokenizer: Text tokenizer (uses multimodal_llm's tokenizer if None)
            device: Device to run on
        """
        self.device = device
        
        # Models
        self.speech_lm = speech_lm.to(device).eval()
        self.multimodal_llm = multimodal_llm.to(device).eval()
        
        # Codec and tokenizer
        self.audio_codec = audio_codec or EnCodecWrapper(device=device)
        self.audio_tokenizer = AudioTokenizer(codec=self.audio_codec)
        
        # Text tokenizer - use the one from multimodal_llm if not provided
        self.text_tokenizer = text_tokenizer if text_tokenizer is not None else multimodal_llm.tokenizer
        
        # Voice encoder
        self.voice_encoder = voice_encoder
        if voice_encoder is not None:
            self.voice_encoder = voice_encoder.to(device).eval()
    
    @torch.no_grad()
    def process_audio_file(
        self,
        audio_path: str,
        extract_voice_style: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Process audio file to extract features and embeddings.
        
        Args:
            audio_path: Path to audio file
            extract_voice_style: Extract voice style embedding
            
        Returns:
            Dict with audio tokens and voice embedding
        """
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        audio = audio.to(self.device)
        
        # Tokenize audio
        audio_tokens = self.audio_tokenizer.tokenize(audio, sample_rate=sr)
        
        # Extract voice style if encoder is available
        if extract_voice_style and self.voice_encoder is not None:
            # Extract features (e.g., mel-spectrogram)
            mel_spec = self._extract_mel_features(audio, sr)
            voice_embedding = self.voice_encoder.encode_audio(mel_spec)
        else:
            # Use dummy embedding
            voice_embedding = torch.zeros(
                1, 256, device=self.device
            )
        
        return {
            "audio_tokens": audio_tokens,
            "voice_embedding": voice_embedding,
            "audio_waveform": audio,
        }
    
    def _extract_mel_features(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        n_mels: int = 128,
    ) -> torch.Tensor:
        """Extract mel-spectrogram features."""
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
        ).to(self.device)
        
        mel = mel_transform(audio)
        mel = torch.log(mel + 1e-9)
        
        # Transpose to [batch, time, features]
        mel = mel.transpose(1, 2)
        
        return mel
    
    @torch.no_grad()
    def text_to_speech_tokens(
        self,
        text: str,
        voice_embedding: torch.Tensor,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
    ) -> torch.Tensor:
        """
        Generate speech tokens from text using SpeechLM.
        
        Args:
            text: Input text
            voice_embedding: Voice style embedding
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            Generated audio tokens
        """
        # Tokenize text (use simple tokenizer for now)
        # In production, use proper tokenizer (SmolLM, etc.)
        text_tokens = self._tokenize_text(text)
        text_tokens = text_tokens.to(self.device)
        
        # Generate audio tokens
        audio_tokens = self.speech_lm.generate(
            speaker_embedding=voice_embedding,
            text_tokens=text_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        return audio_tokens
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text using proper HuggingFace tokenizer."""
        return self.text_tokenizer(
            text,
            max_length=512,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
    
    @torch.no_grad()
    def speech_tokens_to_audio(
        self,
        audio_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert audio tokens back to waveform.
        
        Args:
            audio_tokens: Audio tokens [batch, seq_len]
            
        Returns:
            Audio waveform [batch, channels, time]
        """
        audio_waveform = self.audio_tokenizer.detokenize(audio_tokens)
        return audio_waveform
    
    @torch.no_grad()
    def speech_to_llm_understanding(
        self,
        audio_path: str,
        prompt_text: str,
        max_llm_length: int = 512,
    ) -> str:
        """
        Process speech through SpeechLM and feed to LLM for understanding.
        
        Args:
            audio_path: Path to audio file
            prompt_text: Text prompt for LLM
            max_llm_length: Maximum LLM generation length
            
        Returns:
            LLM response text
        """
        # Process audio
        audio_data = self.process_audio_file(audio_path)
        audio_tokens = audio_data["audio_tokens"]
        voice_embedding = audio_data["voice_embedding"]
        
        # Forward through SpeechLM to get embeddings
        outputs = self.speech_lm(
            audio_tokens=audio_tokens,
            speaker_embedding=voice_embedding,
            return_hidden_states=True,
        )
        
        # Get hidden states from last layer
        speech_embeddings = outputs["hidden_states"][-1]
        
        # Adapt to LLM embedding space
        adapted = self.multimodal_llm.audio_adapter(speech_embeddings)
        llm_embeddings = adapted["llm_embeddings"]
        llm_mask = adapted["attention_mask"]
        
        # Generate response from LLM
        response = self.multimodal_llm.generate(
            input_text=prompt_text,
            audio_embeddings=llm_embeddings,
            audio_attention_mask=llm_mask,
            max_length=max_llm_length,
        )
        
        return response[0]
    
    @torch.no_grad()
    def end_to_end_pipeline(
        self,
        input_text: str,
        voice_style_reference: Optional[str] = None,
        llm_prompt: Optional[str] = None,
        return_audio: bool = True,
    ) -> Dict[str, Any]:
        """
        End-to-end pipeline: text → speech tokens → LLM understanding.
        
        Args:
            input_text: Input text to synthesize
            voice_style_reference: Optional audio file for voice style
            llm_prompt: Optional prompt for LLM processing
            return_audio: Return synthesized audio waveform
            
        Returns:
            Dict with generated audio, LLM response, and intermediate outputs
        """
        # Extract voice style if reference provided
        if voice_style_reference is not None:
            reference_data = self.process_audio_file(
                voice_style_reference,
                extract_voice_style=True,
            )
            voice_embedding = reference_data["voice_embedding"]
        else:
            # Use default voice embedding
            voice_embedding = torch.zeros(1, 256, device=self.device)
        
        # Step 1: Text → Speech tokens via SpeechLM
        speech_tokens = self.text_to_speech_tokens(
            text=input_text,
            voice_embedding=voice_embedding,
        )
        
        # Step 2: Speech tokens → Audio waveform (optional)
        if return_audio:
            audio_waveform = self.speech_tokens_to_audio(speech_tokens)
        else:
            audio_waveform = None
        
        # Step 3: Speech tokens → LLM embeddings
        outputs = self.speech_lm(
            audio_tokens=speech_tokens,
            speaker_embedding=voice_embedding,
            return_hidden_states=True,
        )
        speech_embeddings = outputs["hidden_states"][-1]
        
        # Step 4: Adapt to LLM space
        adapted = self.multimodal_llm.audio_adapter(speech_embeddings)
        
        # Step 5: LLM processing
        llm_response = None
        if llm_prompt is not None:
            llm_response = self.multimodal_llm.generate(
                input_text=llm_prompt,
                audio_embeddings=adapted["llm_embeddings"],
                audio_attention_mask=adapted["attention_mask"],
            )[0]
        
        return {
            "speech_tokens": speech_tokens,
            "audio_waveform": audio_waveform,
            "speech_embeddings": speech_embeddings,
            "llm_embeddings": adapted["llm_embeddings"],
            "llm_response": llm_response,
        }
    
    def save_audio(self, audio: torch.Tensor, output_path: str):
        """Save audio waveform to file."""
        audio = audio.cpu().squeeze(0)
        torchaudio.save(output_path, audio, self.audio_codec.sample_rate)
        print(f"Audio saved to {output_path}")


def create_pipeline_from_checkpoints(
    speech_lm_checkpoint: str,
    llm_name_or_path: str,
    adapter_checkpoint: str,
    voice_encoder_checkpoint: Optional[str] = None,
    device: str = "cuda",
) -> SpeechToLLMPipeline:
    """
    Create pipeline from saved checkpoints.
    
    Args:
        speech_lm_checkpoint: Path to SpeechLM checkpoint
        llm_name_or_path: HuggingFace LLM name or path
        adapter_checkpoint: Path to audio adapter checkpoint
        voice_encoder_checkpoint: Optional voice encoder checkpoint
        device: Device to load on
        
    Returns:
        Configured pipeline
    """
    # Import here to avoid circular imports
    from ..utils import TextTokenizerWrapper
    
    # Create text tokenizer
    text_tokenizer = TextTokenizerWrapper(model_name=llm_name_or_path)
    
    # Load SpeechLM
    speech_lm = SpeechLMTransformer(
        vocab_size=1024,
        hidden_dim=768,
        num_layers=12,
        text_vocab_size=text_tokenizer.vocab_size,  # Use actual vocab size
    )
    speech_lm.load_state_dict(torch.load(speech_lm_checkpoint, map_location=device))
    
    # Load adapter
    audio_adapter = AudioToLLMAdapter(
        audio_dim=768,
        llm_dim=4096,
    )
    audio_adapter.load_state_dict(torch.load(adapter_checkpoint, map_location=device))
    
    # Create multimodal LLM
    multimodal_llm = MultimodalLLMWrapper(
        llm_name_or_path=llm_name_or_path,
        audio_adapter=audio_adapter,
    )
    
    # Load voice encoder if provided
    voice_encoder = None
    if voice_encoder_checkpoint is not None:
        voice_encoder = VoiceStyleEncoder()
        voice_encoder.load_state_dict(
            torch.load(voice_encoder_checkpoint, map_location=device)
        )
    
    # Create pipeline
    pipeline = SpeechToLLMPipeline(
        speech_lm=speech_lm,
        multimodal_llm=multimodal_llm,
        voice_encoder=voice_encoder,
        text_tokenizer=text_tokenizer,  # Pass the text tokenizer
        device=device,
    )
    
    return pipeline
