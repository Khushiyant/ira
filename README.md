
# IRA: Integrated Reasoning with Audio

A speech-to-language model pipeline that unifies audio generation and natural language understanding through knowledge distillation and reinforced behavior alignment.

## Overview

IRA bridges speech synthesis and language model reasoning through a unified multimodal system. It trains a SpeechLM to generate audio tokens directly compatible with LLM representations, enabling seamless integration without intermediate transcription steps.

### Key Features

- **Staged Training Pipeline**: A robust 3-stage process (Pre-train $\to$ Align $\to$ Fine-tune) to ensure stability.
- **Native Multimodal Alignment**: No ASR steps; the model learns to "understand" raw audio tokens.
- **Voice Style Control**: CLIP-style encoder for semantic voice descriptions.
- **Hardware Optimized**: Supports training on consumer hardware (Mac M-series, RTX laptops) via quantization and gradient accumulation.

## Installation

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone [https://github.com/Khushiyant/ira.git](https://github.com/Khushiyant/ira.git)
cd ira

# Install dependencies
pip install -e .
````

### Hardware-Specific Setup

**For Mac (M1/M2/M3):**
Mac users should rely on `accelerate` and MPS (Metal Performance Shaders). **Do not** install `bitsandbytes` as it requires CUDA.

```bash
pip install accelerate

**For Nvidia GPUs (Linux/Windows):**
To enable 4-bit quantization for lower VRAM usage:

```bash
pip install bitsandbytes accelerate
```

## Data Preparation

The system is designed to be plug-and-play. You have two options:

### Option 1: Auto-Download (Recommended)

The model can automatically download and format the **LJSpeech-1.1** dataset. simply set `download: true` and `dataset_name: "ljspeech"` in your config file.

### Option 2: Dummy Data (For Testing)

To verify the pipeline without downloading large datasets, generate synthetic data:

```bash
python scripts/create_dummy_data.py
```

## Usage

### Configuration

Two configuration files are provided:

  * `configs/train_config.yaml`: Standard setup for A100/H100 GPUs.
  * `configs/laptop_config.yaml`: Optimized for consumer hardware (12-16GB VRAM/RAM).

**Mac Users:** Ensure your config has `load_in_4bit: false` and `device: "mps"`.

### Training Pipeline

IRA uses a **Staged Training** approach to ensure convergence. You must run these stages in order.

#### Stage 1: SpeechLM Pre-training

Trains the model to generate valid acoustic tokens (audio reconstruction).

```bash
python main.py --stage 1 --config configs/laptop_config.yaml
```

#### Stage 2: Adapter Alignment

Freezes the SpeechLM and LLM. Trains the Adapter to map audio embeddings to the LLM's text space.
*Requires checkpoint from Stage 1.*

```bash
python main.py \
  --stage 2 \
  --config configs/laptop_config.yaml \
  --resume_checkpoint checkpoints_laptop/stage1_epoch50.pt
```

#### Stage 3: Reinforced Behavior Alignment (RBA)

Fine-tunes the model using Reinforcement Learning (PPO). The LLM acts as a judge, rewarding the SpeechLM for generating audio that "makes sense" semantically.
*Requires checkpoint from Stage 2.*

```bash
python main.py \
  --stage 3 \
  --config configs/laptop_config.yaml \
  --resume_checkpoint checkpoints_laptop/stage2_epoch50.pt
```

## Inference

To generate speech and process it with the LLM:

```python
from src.inference import create_pipeline_from_checkpoints

# Load your Stage 3 checkpoint
pipeline = create_pipeline_from_checkpoints(
    speech_lm_checkpoint="checkpoints_laptop/stage3_epoch50.pt",
    llm_name_or_path="HuggingFaceTB/SmolLM2-1.7B",
    adapter_checkpoint="checkpoints_laptop/stage3_epoch50.pt", # Adapter weights are saved here too
    device="mps" # or "cuda"
)

# Generate and Reason
result = pipeline.end_to_end_pipeline(
    input_text="Hello, analyze the sentiment of this speech.",
    return_audio=True,
)

# Save audio
pipeline.save_audio(result["audio_waveform"], "output.wav")
print("LLM Response:", result["llm_response"])
```

## Architecture Details

1.  **SpeechLM**: Autoregressive transformer (512 dim, 8 layers) generating EnCodec tokens.
2.  **Audio Adapter**: Perceiver-style bridge that compresses audio sequences into fixed-length queries for the LLM.
3.  **Multimodal LLM**: `SmolLM2-1.7B` serving as both the semantic grounding (Teacher) and the reasoning engine.

## License

MIT License. See LICENSE file for details.
