"""Dataset classes for speech and multimodal data with Auto-Download."""

import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List
import torchaudio
from pathlib import Path
import json
import os
import random
import soundfile as sf
import csv

class SpeechLMDataset(Dataset):
    """
    Dataset for SpeechLM training.
    Supports auto-downloading and preparing LJSpeech.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_audio_length: int = 30,
        sample_rate: int = 24000,
        download: bool = False,
        dataset_name: str = "ljspeech" 
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        
        # Auto-download and prepare if requested
        if download and dataset_name.lower() == "ljspeech":
            self._prepare_ljspeech()

        # Load metadata
        metadata_file = self.data_dir / f"{split}.json"
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata {metadata_file} not found. "
                f"Did you set download=True in config?"
            )
            
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)
    
    def _prepare_ljspeech(self):
        """Download LJSpeech and generate train/val JSONs."""
        # Check if already prepared
        if (self.data_dir / "train.json").exists():
            return

        print(f"Downloading and preparing LJSpeech in {self.data_dir}...")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 1. Download using Torchaudio
        # This downloads to data_dir/LJSpeech-1.1
        dataset = torchaudio.datasets.LJSPEECH(root=str(self.data_dir), download=True)
        
        # 2. Parse Metadata
        # LJSpeech structure: root/LJSpeech-1.1/wavs/ and metadata.csv
        lj_root = self.data_dir / "LJSpeech-1.1"
        wav_dir = lj_root / "wavs"
        
        items = []
        # torchaudio LJSPEECH dataset object is iterable, but we want the file paths
        # specifically to write to JSON. Let's read the csv directly for control.
        with open(lj_root / "metadata.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                # Format: ID|Transcription|NormalizedTranscription
                file_id = row[0]
                text = row[2] if len(row) > 2 else row[1]
                
                # We store absolute path or relative to data_dir
                # Here we verify the file exists
                audio_path = wav_dir / f"{file_id}.wav"
                if audio_path.exists():
                    items.append({
                        "audio_file": str(audio_path), # Store full path
                        "text": text,
                        "speaker_id": 0 # LJSpeech is single speaker
                    })

        # 3. Split and Save JSONs
        random.seed(42)
        random.shuffle(items)
        
        # 90/10 Split
        split_idx = int(len(items) * 0.9)
        train_items = items[:split_idx]
        val_items = items[split_idx:]
        
        print(f"Generated {len(train_items)} train and {len(val_items)} val samples.")
        
        with open(self.data_dir / "train.json", "w") as f:
            json.dump(train_items, f, indent=2)
            
        with open(self.data_dir / "val.json", "w") as f:
            json.dump(val_items, f, indent=2)
        
        print("LJSpeech preparation complete.")

    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.metadata[idx]
        
        # Handle both relative and absolute paths
        audio_path_str = item["audio_file"]
        if os.path.isabs(audio_path_str):
            audio_path = Path(audio_path_str)
        else:
            audio_path = self.data_dir / audio_path_str
            
        # Load and Resample
        # Note: We do online resampling here to save disk space
        audio, sr = sf.read(str(audio_path))
        audio = torch.from_numpy(audio).float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # (1, samples)
        elif audio.ndim == 2:
            audio = audio.T  # (channels, samples)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Trim or pad
        max_samples = self.max_audio_length * self.sample_rate
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
        elif audio.shape[1] < max_samples:
            padding = max_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        return {
            "audio": audio.squeeze(0),
            "text": item.get("text", ""),
            "speaker_id": item.get("speaker_id", 0),
            "audio_path": str(audio_path),
        }

class MultimodalDataset(Dataset):
    """Dataset for multimodal LLM training."""
    def __init__(self, data_dir: str, split: str = "train", max_audio_length: int = 30, max_text_length: int = 512):
        self.data_dir = Path(data_dir)
        self.split = split
        # Simply load existing json, assuming stage 1/2 prep handles downloads
        metadata_file = self.data_dir / f"{split}_multimodal.json"
        if not metadata_file.exists():
             # Fallback to standard json if multimodal specific not found
             metadata_file = self.data_dir / f"{split}.json"
        
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.metadata[idx]
        
        audio = None
        if "audio_file" in item:
            p = item["audio_file"]
            path = Path(p) if os.path.isabs(p) else self.data_dir / p
            if path.exists():
                audio, _ = torchaudio.load(str(path))
                audio = audio.squeeze(0)
        
        return {
            "audio": audio,
            "input_text": item.get("input_text", item.get("text", "")),
            "target_text": item.get("target_text", ""),
            "speaker_id": item.get("speaker_id", 0),
        }

class SpeechLMCollator:
    """Collator for batching SpeechLM data."""
    
    def __init__(
        self,
        audio_tokenizer,
        text_tokenizer,
        speaker_embedding_table=None,
        padding_value: int = 0,
    ):
        if text_tokenizer is None:
            raise ValueError("text_tokenizer is required for SpeechLMCollator.")
            
        self.audio_tokenizer = audio_tokenizer
        self.text_tokenizer = text_tokenizer
        self.speaker_embedding_table = speaker_embedding_table
        self.padding_value = padding_value
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [item["audio"] for item in batch]
        texts = [item["text"] for item in batch]
        speaker_ids = torch.tensor([item["speaker_id"] for item in batch])
        
        # Audio Tokenization
        audio_tokens_list = []
        for audio in audios:
            # Check for None audio (multimodal case)
            if audio is None:
                # Create dummy silent audio if missing
                tokens = torch.zeros(1, 1, dtype=torch.long)
            else:
                tokens = self.audio_tokenizer.tokenize(audio.unsqueeze(0))
            audio_tokens_list.append(tokens.squeeze(0))
        
        max_audio_len = max(t.shape[0] for t in audio_tokens_list)
        audio_tokens = torch.full((len(batch), max_audio_len), self.padding_value, dtype=torch.long)
        audio_mask = torch.zeros(len(batch), max_audio_len)
        
        for i, tokens in enumerate(audio_tokens_list):
            audio_tokens[i, :len(tokens)] = tokens
            audio_mask[i, :len(tokens)] = 1
        
        # Text Tokenization
        tokens_dict = self.text_tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        result = {
            "audio_tokens": audio_tokens,
            "audio_mask": audio_mask,
            "text_tokens": tokens_dict["input_ids"],
            "text_mask": tokens_dict["attention_mask"],
            "labels": audio_tokens.clone(),
        }

        if self.speaker_embedding_table is not None:
            result["speaker_embeddings"] = self.speaker_embedding_table(speaker_ids)
        else:
            result["speaker_ids"] = speaker_ids
            
        return result