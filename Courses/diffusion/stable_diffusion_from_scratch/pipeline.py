from PIL import Image
from clip import CLIP
from ddpm import DDPMSampler
import numpy as np
from tqdm import tqdm
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

def generate(prompt: str,
             uncond_prompt: str,
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
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()
        
        # Process CLIP embeddings
        clip = models['clip']
        clip.to(device)

        if do_cfg:
            # Process conditional and unconditional prompts
            cond_tokens = tokenizer.batch_encode_plus([prompt],
                                                    padding="max_length",
                                                    max_length=77,
                                                    return_tensors="pt"
                                                    ).input_ids.to(device)
            
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt],
                                                      padding="max_length",
                                                      max_length=77,
                                                      return_tensors="pt"
                                                      ).input_ids.to(device)
            
            # Get embeddings
            cond_context = clip(cond_tokens)
            uncond_context = clip(uncond_tokens)
            context = torch.cat([cond_context, uncond_context], dim=0)
        else:
            tokens = tokenizer.batch_encode_plus([prompt],
                                               padding="max_length",
                                               max_length=77,
                                               return_tensors="pt"
                                               ).input_ids.to(device)
            context = clip(tokens)

        to_idle(clip)

        # Initialize sampler
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator=generator, device=device)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Sampler {sampler_name} not found")
        
        # Initialize latents
        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)
        if input_image:
            encoder = models['encoder']
            encoder.to(device)
            
            # Process input image
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = torch.tensor(np.array(input_image_tensor), 
                                           device=device, 
                                           dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            # Encode image
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents = encoder(encoder_noise, input_image_tensor)
            
            # Add noise based on strength
            sampler.set_strength(strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            to_idle(encoder)
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # Run diffusion process
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)
            
            # Prepare model input
            model_input = latents
            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            # Get model prediction
            model_output = diffusion(model_input, context, time_embedding)

            # Apply classifier-free guidance
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Update latents
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        # Decode latents to image
        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(latents)
        to_idle(decoder)

        # Post-process images
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1).cpu()
        images = images.to(torch.uint8).numpy()
        
        return images[0]

def rescale(x, old_range, new_range, clamp=False):
    """Rescale tensor values from old_range to new_range with optional clamping."""
    old_min, old_max = old_range
    new_min, new_max = new_range
    
    # Ensure we're working with float32 for better precision
    x = x.to(torch.float32)
    
    # Normalize to [0, 1]
    x = (x - old_min) / (old_max - old_min)
    
    # Scale to new range
    x = x * (new_max - new_min) + new_min
    
    if clamp:
        x = x.clamp(new_min, new_max)
    
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

if __name__ == "__main__":
    models = preload_models_from_standard_weights("data/v1-5-pruned-emaonly.ckpt", device)
    # load CLIPTokenizer from data/vocab.json and data/merges.txt
    tokenizer = CLIPTokenizer(vocab_file="data/vocab.json", merges_file="data/merges.txt")
    
    output_image = generate(
        prompt="An astronaut in a Ferrari on Mars, under moonlight, 4k, cinematic, hyper-realistic",
        uncond_prompt="",
        input_image=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name='ddpm',
        models=models,
        tokenizer=tokenizer,
        device=device,
        idle_device='cpu',
        n_inference_steps=100,
        seed=42
    )

    Image.fromarray(output_image).save("output.png")
