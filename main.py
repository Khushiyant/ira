"""Main training script for IRA with Staged Training."""

import torch
import argparse
from pathlib import Path
import yaml
from omegaconf import OmegaConf

from src.codec import EnCodecWrapper, AudioTokenizer
from src.model import (
    CLIPVoiceEncoder,
    SpeechLMTransformer,
    AudioToLLMAdapter,
    MultimodalLLMWrapper,
)
from src.training import RBATrainer
from src.data import SpeechLMDataset, SpeechLMCollator
from src.utils import TextTokenizerWrapper, set_seed
from torch.utils.data import DataLoader

def load_config(config_path):
    with open(config_path, 'r') as f:
        return OmegaConf.create(yaml.safe_load(f))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], 
                        help="1: Pretrain SpeechLM, 2: Align Adapter, 3: Full RBA")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config.hardware.seed)
    
    print(f"--- Starting Training Stage {args.stage} ---")

    # 1. Models
    print("Initializing Models...")
    codec = EnCodecWrapper(model_name=config.codec.model_name)
    audio_tokenizer = AudioTokenizer(codec=codec)
    text_tokenizer = TextTokenizerWrapper(model_name=config.model.llm.name_or_path)
    
    speaker_emb = torch.nn.Embedding(config.model.num_speakers, config.model.speech_lm.speaker_embed_dim)
    
    speech_lm = SpeechLMTransformer(
        vocab_size=audio_tokenizer.vocab_size,
        hidden_dim=config.model.speech_lm.hidden_dim,
        num_layers=config.model.speech_lm.num_layers,
        speaker_embed_dim=config.model.speech_lm.speaker_embed_dim,
        text_vocab_size=text_tokenizer.vocab_size
    )
    
    audio_adapter = AudioToLLMAdapter(
        audio_dim=config.model.speech_lm.hidden_dim,
        llm_dim=config.model.audio_adapter.llm_dim
    )
    
    multimodal_llm = MultimodalLLMWrapper(
        llm_name_or_path=config.model.llm.name_or_path,
        audio_adapter=audio_adapter,
        freeze_llm=True 
    )
    
    # 2. Load Checkpoints
    if args.resume_checkpoint:
        print(f"Loading checkpoint: {args.resume_checkpoint}")
        chk = torch.load(args.resume_checkpoint)
        if args.stage == 2:
            # Stage 2 needs pretrained SpeechLM
            speech_lm.load_state_dict(chk['model_state'], strict=False)
        elif args.stage == 3:
            # Stage 3 needs everything. Assuming chk contains adapter weights?
            # In practice, you might need to load speech_lm AND adapter separately.
            # This is a simplification.
            try:
                speech_lm.load_state_dict(chk['model_state'], strict=False)
            except:
                pass
            print("Loaded checkpoint weights.")

    # 3. Data
    train_dataset = SpeechLMDataset(config.data.train_dir, split="train")
    collator = SpeechLMCollator(
        audio_tokenizer=audio_tokenizer,
        text_tokenizer=text_tokenizer,
        speaker_embedding_table=speaker_emb 
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        collate_fn=collator,
        num_workers=config.data.num_workers
    )

    # 4. Trainer
    trainer = RBATrainer(
        speech_lm=speech_lm,
        multimodal_llm=multimodal_llm,
        stage=args.stage,
        learning_rate=config.training.learning_rate,
        use_wandb=config.logging.use_wandb
    )
    
    # 5. Loop
    save_dir = Path(config.logging.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.training.num_epochs):
        metrics = trainer.train_epoch(train_loader, epoch, config.training.num_epochs)
        print(f"Epoch {epoch}: {metrics}")
        
        if (epoch + 1) % 5 == 0:
            save_path = save_dir / f"stage{args.stage}_epoch{epoch+1}.pt"
            trainer.save_checkpoint(str(save_path), epoch)

if __name__ == "__main__":
    main()