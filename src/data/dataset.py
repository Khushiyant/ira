"""Dataset classes for speech and multimodal data."""

import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List
import torchaudio
from pathlib import Path
import json


class SpeechLMDataset(Dataset):
    """Dataset for SpeechLM training with text and audio pairs."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_audio_length: int = 30,  # seconds
        sample_rate: int = 24000,
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing audio files and metadata
            split: Dataset split (train/val/test)
            max_audio_length: Maximum audio length in seconds
            sample_rate: Target sample rate
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        
        # Load metadata
        metadata_file = self.data_dir / f"{split}.json"
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        item = self.metadata[idx]
        
        # Load audio
        audio_path = self.data_dir / item["audio_file"]
        audio, sr = torchaudio.load(str(audio_path))
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Trim or pad to max length
        max_samples = self.max_audio_length * self.sample_rate
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
        elif audio.shape[1] < max_samples:
            padding = max_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        # Get text
        text = item.get("text", "")
        
        # Get speaker ID
        speaker_id = item.get("speaker_id", 0)
        
        return {
            "audio": audio.squeeze(0),  # Remove channel dimension
            "text": text,
            "speaker_id": speaker_id,
            "audio_path": str(audio_path),
        }


class MultimodalDataset(Dataset):
    """Dataset for multimodal LLM training with speech and text."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_audio_length: int = 30,
        max_text_length: int = 512,
    ):
        """
        Initialize multimodal dataset.
        
        Args:
            data_dir: Directory containing data
            split: Dataset split
            max_audio_length: Max audio length in seconds
            max_text_length: Max text length in tokens
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        
        # Load metadata
        metadata_file = self.data_dir / f"{split}_multimodal.json"
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get multimodal item."""
        item = self.metadata[idx]
        
        # Load audio if available
        audio = None
        if "audio_file" in item:
            audio_path = self.data_dir / item["audio_file"]
            audio, sr = torchaudio.load(str(audio_path))
            audio = audio.squeeze(0)
        
        # Get text components
        input_text = item.get("input_text", "")
        target_text = item.get("target_text", "")
        
        # Get speaker ID
        speaker_id = item.get("speaker_id", 0)
        
        return {
            "audio": audio,
            "input_text": input_text,
            "target_text": target_text,
            "speaker_id": speaker_id,
        }


class SpeechLMCollator:
    """Collator for batching SpeechLM data."""
    
    def __init__(
        self,
        audio_tokenizer,
        text_tokenizer,
        padding_value: int = 0,
    ):
        """
        Initialize collator.
        
        Args:
            audio_tokenizer: Audio tokenizer
            text_tokenizer: Text tokenizer
            padding_value: Padding value
        """
        self.audio_tokenizer = audio_tokenizer
        self.text_tokenizer = text_tokenizer
        self.padding_value = padding_value
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch."""
        # Extract components
        audios = [item["audio"] for item in batch]
        texts = [item["text"] for item in batch]
        speaker_ids = torch.tensor([item["speaker_id"] for item in batch])
        
        # Tokenize audio
        audio_tokens_list = []
        for audio in audios:
            tokens = self.audio_tokenizer.tokenize(audio.unsqueeze(0))
            audio_tokens_list.append(tokens.squeeze(0))
        
        # Pad audio tokens
        max_audio_len = max(t.shape[0] for t in audio_tokens_list)
        audio_tokens = torch.full(
            (len(batch), max_audio_len),
            self.padding_value,
            dtype=torch.long,
        )
        audio_mask = torch.zeros(len(batch), max_audio_len)
        
        for i, tokens in enumerate(audio_tokens_list):
            audio_tokens[i, :len(tokens)] = tokens
            audio_mask[i, :len(tokens)] = 1
        
        # Tokenize text (simple tokenization for now)
        text_tokens = []
        for text in texts:
            tokens = [ord(c) % 1000 for c in text[:512]]
            text_tokens.append(torch.tensor(tokens, dtype=torch.long))
        
        # Pad text tokens
        max_text_len = max(t.shape[0] for t in text_tokens)
        text_tokens_padded = torch.full(
            (len(batch), max_text_len),
            self.padding_value,
            dtype=torch.long,
        )
        text_mask = torch.zeros(len(batch), max_text_len)
        
        for i, tokens in enumerate(text_tokens):
            text_tokens_padded[i, :len(tokens)] = tokens
            text_mask[i, :len(tokens)] = 1
        
        return {
            "audio_tokens": audio_tokens,
            "audio_mask": audio_mask,
            "text_tokens": text_tokens_padded,
            "text_mask": text_mask,
            "speaker_ids": speaker_ids,
            "labels": audio_tokens.clone(),  # For autoregressive training
        }
