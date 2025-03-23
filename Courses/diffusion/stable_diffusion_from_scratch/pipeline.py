from tkinter import Image
from clip import CLIP
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_loader import preload_models_from_standard_weights
from transformers import CLIPTokenizer


WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(prompt: str, uncond_prompt: str,
             input_image=None,
             strength: float = 0.8,
             do_cfg: bool = True,
             cfg_scale: float = 7.5,
             sampler_name='ddpm',
             n_inference_steps: int=50,
             models=(),
             seed=None,
             device=None,
             idle_device=None,
             tokenizer=None
        ):
    with torch.no_grad():
        if not(0 < strength <= 1):
            raise ValueError(f"Strength must be between 0 and 1")
        
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x
        
        generator = torch.Generator(device=device)
        if seed:
            generator.manual_seed(seed)
        else:
            generator.seed()
        
        # negative prompt or uncond_prompt is used for CFG.
        # Run the inference twice - with and without the prompt.
        clip = models['clip']
        encoder = models['encoder']
        decoder = models['decoder']
        diffusion = models['diffusion']

        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus([prompt],
                                                      padding="max_length",
                                                      max_length=77,
                                                      return_tensors="pt"
                                                        ).input_ids
            cond_tokens = cond_tokens.clone().to(device=device, dtype=torch.long)
            cond_context = clip(cond_tokens)
            # Unconditional prompt and tokens
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt],
                                                        padding="max_length",
                                                        max_length=77,
                                                        return_tensors="pt"
                                                        ).input_ids
            uncond_tokens = uncond_tokens.clone().to(device=device, dtype=torch.long)
            uncond_context = clip(uncond_tokens)

            # Concatenate conditional and unconditional contexts along batch dimension
            context = torch.cat([uncond_context, cond_context], dim=0)
        else:
            tokens = tokenizer.batch_encode_plus([prompt],
                                                padding="max_length",
                                                max_length=77,
                                                return_tensors="pt"
                                                ).input_ids
            tokens = tokens.clone().to(device=device, dtype=torch.long)
            context = clip(tokens)

        # This looks like an attempt to move the context to the idle device when not in use.
        # Moving to the CPU seems like a cost on latency.
        to_idle(context)


if __name__ == "__main__":
    models = preload_models_from_standard_weights("data/v1-5-pruned-emaonly.ckpt", "cuda")
    # load CLIPTokenizer from data/vocab.json and data/merges.txt
    tokenizer = CLIPTokenizer(vocab_file="data/vocab.json", merges_file="data/merges.txt")
    
    output_image = generate(
        prompt="A beautiful landscape with a river and mountains",
        uncond_prompt="A flat, featureless gray background",
        input_image=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
    sampler_name='ddpm',
        models=models,
        tokenizer=tokenizer,
        device=device,
    )





