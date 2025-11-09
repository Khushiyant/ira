"""Main training script for Speech-to-LLM pipeline."""

import torch
import argparse
from pathlib import Path
import yaml
from omegaconf import OmegaConf
import random
import numpy as np

from src.codec import EnCodecWrapper, AudioTokenizer
from src.model import (
    CLIPVoiceEncoder,
    SpeechLMTransformer,
    AudioToLLMAdapter,
    MultimodalLLMWrapper,
)
from src.training import RBATrainer, KnowledgeDistillationLoss
from src.data import SpeechLMDataset, SpeechLMCollator
from torch.utils.data import DataLoader


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


def create_models(config):
    """Create all models from configuration."""
    
    print("Creating models...")
    
    # Audio codec
    codec = EnCodecWrapper(
        model_name=config.codec.model_name,
        bandwidth=config.codec.bandwidth,
    )
    tokenizer = AudioTokenizer(codec=codec)
    
    # Voice encoder
    voice_encoder = CLIPVoiceEncoder(
        audio_encoder_config=OmegaConf.to_container(config.model.voice_encoder),
        embedding_dim=config.model.voice_encoder.embedding_dim,
    )
    
    # SpeechLM
    speech_lm = SpeechLMTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=config.model.speech_lm.hidden_dim,
        num_layers=config.model.speech_lm.num_layers,
        num_heads=config.model.speech_lm.num_heads,
        max_seq_len=config.model.speech_lm.max_seq_len,
        dropout=config.model.speech_lm.dropout,
        use_rope=config.model.speech_lm.use_rope,
        speaker_embed_dim=config.model.speech_lm.speaker_embed_dim,
        text_vocab_size=config.model.speech_lm.text_vocab_size,
        tie_embeddings=config.model.speech_lm.tie_embeddings,
    )
    
    # Audio adapter
    audio_adapter = AudioToLLMAdapter(
        audio_dim=config.model.audio_adapter.audio_dim,
        llm_dim=config.model.audio_adapter.llm_dim,
        num_adapter_layers=config.model.audio_adapter.num_adapter_layers,
        num_query_tokens=config.model.audio_adapter.num_query_tokens,
        use_perceiver=config.model.audio_adapter.use_perceiver,
        dropout=config.model.audio_adapter.dropout,
    )
    
    # Multimodal LLM
    multimodal_llm = MultimodalLLMWrapper(
        llm_name_or_path=config.model.llm.name_or_path,
        audio_adapter=audio_adapter,
        freeze_llm=config.model.llm.freeze_llm,
        use_lora=config.model.llm.use_lora,
        lora_rank=config.model.llm.lora_rank,
        lora_alpha=config.model.llm.lora_alpha,
    )
    
    print(f"✓ Models created successfully")
    print(f"  - SpeechLM: {sum(p.numel() for p in speech_lm.parameters())/1e6:.1f}M params")
    print(f"  - Audio Adapter: {sum(p.numel() for p in audio_adapter.parameters())/1e6:.1f}M params")
    
    return {
        "codec": codec,
        "tokenizer": tokenizer,
        "voice_encoder": voice_encoder,
        "speech_lm": speech_lm,
        "audio_adapter": audio_adapter,
        "multimodal_llm": multimodal_llm,
    }


def create_dataloaders(config, tokenizer):
    """Create training and validation dataloaders."""
    
    print("Creating dataloaders...")
    
    # Training dataset
    train_dataset = SpeechLMDataset(
        data_dir=config.data.train_dir,
        split="train",
        max_audio_length=config.data.max_audio_length,
        sample_rate=config.codec.sample_rate,
    )
    
    # Validation dataset
    val_dataset = SpeechLMDataset(
        data_dir=config.data.val_dir,
        split="val",
        max_audio_length=config.data.max_audio_length,
        sample_rate=config.codec.sample_rate,
    )
    
    # Collator
    collator = SpeechLMCollator(
        audio_tokenizer=tokenizer,
        text_tokenizer=None,  # Use simple tokenizer for now
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collator,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collator,
    )
    
    print(f"✓ Dataloaders created")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def train(config, models, train_loader, val_loader):
    """Main training loop."""
    
    print("\nStarting training...")
    
    # Create KD loss
    kd_loss = KnowledgeDistillationLoss(
        temperature=config.training.kd_temperature,
        alpha=config.training.kd_alpha,
    )
    
    # Create trainer
    trainer = RBATrainer(
        speech_lm=models["speech_lm"],
        multimodal_llm=models["multimodal_llm"],
        kd_loss=kd_loss,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        max_grad_norm=config.training.max_grad_norm,
        use_rl=config.training.use_rba,
        rl_reward_weight=config.training.rl_reward_weight,
        ppo_clip_epsilon=config.training.ppo_clip_epsilon,
        value_loss_coef=config.training.value_loss_coef,
        entropy_coef=config.training.entropy_coef,
        use_wandb=config.logging.use_wandb,
        project_name=config.logging.wandb_project,
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.training.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.training.num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch, config.training.num_epochs)
        
        print(f"\nTraining metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Validation (simplified for now)
        # In production, implement proper validation loop
        
        # Save checkpoint
        if (epoch + 1) % (config.training.save_steps // len(train_loader)) == 0:
            checkpoint_path = Path(config.logging.output_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(checkpoint_path), epoch)
    
    print("\n✓ Training completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Speech-to-LLM pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config.hardware.seed)
    
    # Set device
    device = config.hardware.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Create models
    models = create_models(config)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, models["tokenizer"])
    
    # Train
    train(config, models, train_loader, val_loader)


if __name__ == "__main__":
    main()
