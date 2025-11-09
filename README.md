# IRA: Integrated Reasoning with Audio

A speech-to-language model pipeline that unifies audio generation and natural language understanding through knowledge distillation and reinforced behavior alignment.

## Overview

IRA bridges speech synthesis and language model reasoning through a unified multimodal system. It trains a SpeechLM to generate audio tokens directly compatible with LLM representations, enabling seamless integration without intermediate transcription steps.

### Key Implementation

- **Knowledge-distilled SpeechLM** aligned with LLM representations
- **Perceiver-style adapter** for efficient cross-modal projection
- **PPO-based reinforced behavior alignment** for training

This enables natural speech generation conditioned on text and voice characteristics, plus immediate downstream language task processing.

## Key Features

- Autoregressive audio token generation conditioned on text and speaker embeddings
- Voice style encoding through CLIP-inspired contrastive learning
- Cross-modal alignment using Perceiver-style architecture
- Knowledge distillation from teacher LLM to student SpeechLM
- Reinforced behavior alignment using PPO
- Unified multimodal processing through SmolLM2-1.7B

## Architecture

### Core Components

1. **Audio Codec**: EnCodec for 24kHz audio tokenization with configurable bandwidth
2. **Voice Encoder**: Transformer-based model extracting speaker embeddings from audio features
3. **SpeechLM**: 12-layer transformer with rotary positional embeddings for audio token generation
4. **Audio Adapter**: Perceiver-based architecture projecting audio embeddings to LLM space
5. **Multimodal LLM**: SmolLM2-1.7B with LoRA fine-tuning for efficient adaptation

### Training Strategy

The system combines knowledge distillation from a frozen teacher LLM with PPO-based reinforcement learning for behavior alignment. Training uses multi-task learning with uncertainty weighting and mixed precision (BF16) with gradient accumulation for efficiency.

## Installation

Ensure you have Python 3.10 or higher and PyTorch 2.0 or higher installed. A CUDA-compatible GPU is recommended for training.

```bash
# Clone the repository
git clone https://github.com/Khushiyant/ira.git
cd ira

# Install dependencies
pip install -e .

```

## Usage

### Training

Start training with the default configuration:

```bash
python main.py --config configs/train_config.yaml
```

The training configuration in `configs/train_config.yaml` controls model dimensions, training hyperparameters, RBA settings, data paths, and logging options. You can modify these parameters to suit your specific requirements.

### Inference

Use the inference pipeline to generate speech and process it with the LLM:

```python
from src.inference import create_pipeline_from_checkpoints

# Create pipeline
pipeline = create_pipeline_from_checkpoints(
    speech_lm_checkpoint="path/to/speech_lm.pt",
    llm_name_or_path="HuggingFaceTB/SmolLM2-1.7B",
    adapter_checkpoint="path/to/adapter.pt",
)

# Generate speech and process with LLM
result = pipeline.end_to_end_pipeline(
    input_text="Hello, how are you today?",
    voice_style_reference="path/to/reference_audio.wav",
    llm_prompt="Analyze the sentiment of this speech",
    return_audio=True,
)

# Save generated audio
pipeline.save_audio(result["audio_waveform"], "output.wav")
```

## Model Architecture Details

**SpeechLM Transformer**:
- 768 hidden dimensions across 12 layers with 12 attention heads
- Supports sequences up to 8192 tokens
- RoPE (Rotary Position Embeddings) for length extrapolation
- Cross-attention conditioning on text and speaker embeddings

**Audio Adapter**:
- Transforms 768-dim audio embeddings to 4096-dim vectors for SmolLM2
- 32 Perceiver query tokens across 4 adapter layers
- Compresses variable-length audio into fixed-size LLM-compatible representation

**Voice Encoder**:
- Processes 128-dim mel-spectrogram features through 512-dim hidden layer
- Produces 256-dimensional embeddings
- Attention pooling for sequence aggregation
- CLIP-style contrastive learning with text descriptions of voice characteristics

## Data Format

Organize your data in the following structure:

```
data/
├── train/
│   ├── train.json
│   └── audio_files/
├── val/
│   ├── val.json
│   └── audio_files/
└── test/
    ├── test.json
    └── audio_files/
```

Each JSON file should contain metadata in this format:

```json
[
  {
    "audio_file": "audio_files/sample_001.wav",
    "text": "This is a sample transcription",
    "speaker_id": 0
  }
]
```

## Performance Considerations

**Hardware Requirements**:
- Recommended: A100 40GB GPU or equivalent
- Minimum: GPU with at least 16GB memory

**Training Metrics**:
- Approximately 100 epochs for convergence
- Checkpoint size: ~1.5GB (SpeechLM + adapter weights)

## Logging and Monitoring

**Supported Platforms**:
- Weights & Biases (wandb)
- TensorBoard
- Console logging

**Tracked Metrics**:
- Training and validation loss
- Knowledge distillation loss
- RL rewards and policy metrics
- Learning rate schedule
- GPU memory usage

**Checkpoints Include**:
- Model state dictionaries
- Optimizer states
- Training step and epoch information
- Best validation metrics for model selection

## Future Work

Planned improvements include:

- Training a custom transformer architecture from scratch
- Support for streaming audio generation
- Enhanced multi-speaker synthesis capabilities
- Additional language support beyond English
- Model quantization for efficient deployment
- Better handling of extended context lengths

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ira,
  title={IRA: Integrated Reasoning with Audio},
  author={Khushiyant},
  year={2025},
  url={https://github.com/Khushiyant/ira}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

This project builds on EnCodec for audio tokenization, HuggingFace transformer implementations, the SmolLM2 base language model, and CLIP's approach to contrastive learning.