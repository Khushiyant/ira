"""EnCodec wrapper for 24kHz audio tokenization."""

import torch
import torch.nn as nn
from encodec import EncodecModel
from encodec.utils import convert_audio
from typing import Optional, Tuple
import torchaudio


class EnCodecWrapper(nn.Module):
    """Wrapper for EnCodec model to generate discrete audio tokens at 24kHz."""
    
    def __init__(
        self,
        model_name: str = "encodec_24khz",
        bandwidth: float = 6.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize EnCodec wrapper.
        
        Args:
            model_name: EnCodec model variant (encodec_24khz or encodec_48khz)
            bandwidth: Target bandwidth in kbps (1.5, 3.0, 6.0, 12.0, 24.0)
            device: Device to load model on
        """
        super().__init__()
        self.device = device
        self.bandwidth = bandwidth
        
        # Load pretrained EnCodec model
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.to(device)
        self.model.eval()
        
        self.sample_rate = self.model.sample_rate
        self.channels = self.model.channels
        
    @torch.no_grad()
    def encode(self, audio: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        Encode audio to discrete tokens.
        
        Args:
            audio: Audio waveform tensor [batch, channels, time] or [batch, time]
            sample_rate: Sample rate of input audio (will resample if needed)
            
        Returns:
            Discrete audio tokens [batch, num_codebooks, time_steps]
        """
        # Handle different input shapes
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dimension
            
        # Resample if needed
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = convert_audio(
                audio, sample_rate, self.sample_rate, self.channels
            )
            
        audio = audio.to(self.device)
        
        # Encode to discrete tokens
        encoded_frames = self.model.encode(audio)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        
        return codes
    
    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete tokens back to audio.
        
        Args:
            codes: Discrete audio tokens [batch, num_codebooks, time_steps]
            
        Returns:
            Reconstructed audio waveform [batch, channels, time]
        """
        codes = codes.to(self.device)
        
        # Decode from discrete tokens
        frames = [(codes, None)]
        audio = self.model.decode(frames)
        
        return audio
    
    @torch.no_grad()
    def encode_file(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Encode audio file to discrete tokens.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (discrete tokens, original sample rate)
        """
        audio, sr = torchaudio.load(audio_path)
        codes = self.encode(audio, sample_rate=sr)
        return codes, sr
    
    @torch.no_grad()
    def decode_to_file(self, codes: torch.Tensor, output_path: str):
        """
        Decode discrete tokens and save to audio file.
        
        Args:
            codes: Discrete audio tokens
            output_path: Path to save audio file
        """
        audio = self.decode(codes)
        torchaudio.save(output_path, audio.cpu().squeeze(0), self.sample_rate)
        
    def get_num_codebooks(self) -> int:
        """Get number of codebooks used."""
        return self.model.quantizer.n_q
    
    def get_codebook_size(self) -> int:
        """Get vocabulary size per codebook."""
        return self.model.quantizer.bins
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size across all codebooks."""
        return self.get_codebook_size()
