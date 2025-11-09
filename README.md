# IRA: Integrated Reasoning with Audio

A speech-to-language model pipeline that unifies audio generation and natural language understanding through knowledge distillation and reinforced behavior alignment.

## Overview

IRA bridges speech synthesis and language model reasoning through a unified multimodal system. It trains a SpeechLM to generate audio tokens directly compatible with LLM representations, enabling seamless integration without intermediate transcription steps.

The architecture features three key innovations: knowledge-distilled SpeechLM aligned with LLM representations, a Perceiver-style adapter for efficient cross-modal projection, and PPO-based reinforced behavior alignment. This enables both natural speech generation conditioned on text and voice characteristics, and immediate downstream language task processing.

## Key Features

The pipeline provides autoregressive audio token generation conditioned on text and speaker embeddings, voice style encoding through CLIP-inspired contrastive learning, and cross-modal alignment using a Perceiver-style architecture. It implements knowledge distillation to transfer learning from teacher LLM to student SpeechLM, reinforced behavior alignment using PPO for training, and unified multimodal processing through SmolLM2-1.7B.

## Architecture

The system consists of five main components. The audio codec uses EnCodec for 24kHz audio tokenization with configurable bandwidth. The voice encoder is a transformer-based model that extracts speaker embeddings from audio features. The SpeechLM is a 12-layer transformer with rotary positional embeddings that generates audio tokens. The audio adapter uses a Perceiver-based architecture to project audio embeddings to LLM space. Finally, the multimodal LLM integrates SmolLM2-1.7B with LoRA fine-tuning for efficient adaptation.

Training combines knowledge distillation from a frozen teacher LLM with PPO-based reinforcement learning for behavior alignment. The system uses multi-task learning with uncertainty weighting and mixed precision training (BF16) with gradient accumulation for efficiency.

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

The SpeechLM transformer uses a hidden dimension of 768 across 12 layers with 12 attention heads. It supports sequences up to 8192 tokens and employs RoPE (Rotary Position Embeddings) for better length extrapolation. The model uses cross-attention to condition on text and speaker embeddings during generation.

The audio adapter transforms 768-dimensional audio embeddings from the SpeechLM into 4096-dimensional vectors compatible with SmolLM2. It uses 32 Perceiver query tokens across 4 adapter layers to compress variable-length audio into a fixed-size representation that the LLM can process.

The voice encoder processes 128-dimensional mel-spectrogram features through a 512-dimensional hidden layer to produce 256-dimensional embeddings. It uses attention pooling for sequence aggregation and trains with CLIP-style contrastive learning against text descriptions of voice characteristics.

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

Training requires substantial computational resources. We recommend an A100 40GB GPU or equivalent, though the system can run on GPUs with at least 16GB of memory. Training typically requires approximately 100 epochs for convergence. Each checkpoint is around 1.5GB in size, containing the SpeechLM and adapter weights.

## Logging and Monitoring

The pipeline supports Weights & Biases (wandb), TensorBoard, and console logging. It tracks training and validation loss, knowledge distillation loss, RL rewards and policy metrics, learning rate schedule, and GPU memory usage throughout training.

Checkpoints are saved periodically and include model state dictionaries, optimizer states, training step and epoch information, and best validation metrics for model selection.

## Future Work

Planned improvements include training a custom transformer architecture from scratch, support for streaming audio generation, enhanced multi-speaker synthesis capabilities, additional language support beyond English, model quantization for efficient deployment, and better handling of extended context lengths.

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