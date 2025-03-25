# Stable Diffusion Implementation from Scratch

This repository contains a PyTorch implementation of Stable Diffusion, a state-of-the-art text-to-image generation model. The implementation is built from scratch, focusing on clarity and educational purposes.

## Architecture Overview

The implementation consists of several key components:

### 1. CLIP Text Encoder (`clip.py`)
- Implements the CLIP model for text encoding
- Uses self-attention mechanisms for processing text tokens
- Converts text prompts into embeddings that guide the image generation
- Fixed sequence length of 77 tokens
- Uses learnable position embeddings
- Implements 12 transformer layers with 12 attention heads
- Uses a quick GELU activation (x * sigmoid(1.702 * x))

### 2. VAE (Variational Autoencoder)
- **Encoder** (`encoder.py`):
  - Converts input images to latent space
  - Uses residual blocks and attention mechanisms
  - Outputs mean and variance for latent space sampling
  - Scales output by factor 0.18215
  - Channel progression: 3 → 128 → 256 → 512
  - Uses SiLU activation and GroupNorm
  - Clamps log variance between -30.0 and 20.0

- **Decoder** (`decoder.py`):
  - Reconstructs images from latent space
  - Uses residual blocks and attention mechanisms
  - Upsamples latent representations back to image space
  - Inverse scaling of 0.18215
  - Channel progression: 4 → 512 → 256 → 128 → 3
  - Uses SiLU activation and GroupNorm

### 3. UNet (`diffusion.py`)
- Core component for the diffusion process
- Features:
  - Time embedding for diffusion steps (320 dimensions)
  - Residual blocks with time conditioning
  - Self-attention and cross-attention blocks
  - Skip connections between encoder and decoder paths
  - Upsampling and downsampling operations
  - Channel progression: 4 → 320 → 640 → 1280
  - Uses GroupNorm and SiLU activations
  - Implements GeGLU in attention blocks

### 4. Pipeline (`pipeline.py`)
- Orchestrates the entire generation process
- Handles:
  - Text prompt processing
  - Latent space sampling
  - Diffusion steps
  - Image reconstruction
- Supports:
  - Classifier-free guidance (CFG)
  - Custom inference steps
  - Strength parameter for image-to-image generation

## Key Features

1. **Attention Mechanisms**
   - Self-attention for processing sequences
   - Cross-attention between text and image features
   - Multi-head attention with configurable dimensions
   - Causal masking support in CLIP
   - GeGLU activation in attention blocks

2. **Residual Connections**
   - Used throughout the architecture
   - Helps with gradient flow and feature preservation
   - Implemented in both VAE and UNet components
   - Identity shortcuts when input/output channels match
   - 1x1 convolutions for channel adjustments

3. **Time Conditioning**
   - Embeds diffusion timesteps
   - Conditions the UNet on the current diffusion step
   - Uses linear projections with SiLU activation
   - 320-dimensional time embedding

4. **Latent Space Processing**
   - 8x downsampling in the encoder
   - 8x upsampling in the decoder
   - Efficient processing in compressed space
   - Proper padding handling for stride operations

## Setup and Installation

1. Clone the repository and install dependencies:
```bash
pip install torch transformers pillow requests
```

2. Download the required model files:
```bash
python download_model_files.py
```
This will download:
- CLIP tokenizer files (`vocab.json` and `merges.txt`)
- Stable Diffusion v1.5 model weights (`v1-5-pruned-emaonly.safetensors`)

## Usage

The implementation provides a simple interface for text-to-image generation. Here's how to use it:

```python
from PIL import Image
from transformers import CLIPTokenizer
from model_loader import preload_models_from_standard_weights
from pipeline import generate

# Load models and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = preload_models_from_standard_weights("data/v1-5-pruned-emaonly.ckpt", device)
tokenizer = CLIPTokenizer(vocab_file="data/vocab.json", merges_file="data/merges.txt")

# Generate an image
output_image = generate(
    prompt="A beautiful sunset over mountains",
    uncond_prompt="",
    input_image=None,  # Optional: for img2img generation
    strength=0.8,      # Only used for img2img
    do_cfg=True,       # Enable classifier-free guidance
    cfg_scale=7.5,     # Guidance scale
    sampler_name='ddpm',
    models=models,
    tokenizer=tokenizer,
    device=device,
    idle_device='cpu', # For memory management
    n_inference_steps=50,
    seed=42           # For reproducibility
)

# Save the generated image
Image.fromarray(output_image).save("output.png")
```

### Key Parameters

- `prompt`: Text description of the image to generate
- `uncond_prompt`: Negative prompt for classifier-free guidance
- `input_image`: Optional PIL Image for image-to-image generation
- `strength`: Controls how much of the input image to preserve (0-1)
- `do_cfg`: Enable/disable classifier-free guidance
- `cfg_scale`: Strength of the guidance (higher = more prompt adherence)
- `n_inference_steps`: Number of denoising steps (higher = better quality but slower)
- `seed`: Random seed for reproducibility

## System Requirements

- Python 3.7+
- CUDA-capable GPU (recommended)
- At least 8GB GPU memory
- 4GB disk space for model files

## Implementation Notes

1. The implementation follows the original Stable Diffusion architecture
2. Uses modern PyTorch practices and features
3. Includes detailed comments explaining the architecture
4. Implements all major components from scratch
5. Uses efficient tensor operations and proper shape handling
6. Implements proper padding for stride operations
7. Uses GroupNorm with 32 groups throughout the architecture

## License

This implementation is for educational purposes. Please refer to the original Stable Diffusion paper and repository for licensing information. 