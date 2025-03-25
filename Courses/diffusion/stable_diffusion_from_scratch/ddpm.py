# DDPM Scheduler/Sampler - remove the predicted noise from the image
# Beta Scheduler - used to control the amount of noise at each timestep

import torch
import numpy as np


class DDPMSampler:
    def __init__(self, generator: torch.Generator,
                 num_training_steps: int=1000,
                 device: torch.device=torch.device("cuda"),
                 beta_start: float=0.00085,
                 beta_end: float=0.0120,
                 beta_schedule: str="linear",
            ):
        # Forward pass adds noise based on the beta schedule
        # Linear schedule as used in the original DDPM paper
        self.betas = torch.linspace(beta_start, beta_end, num_training_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)
        
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.device = device
        self.one = torch.tensor(1.0, device=device)
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return max(prev_t, 0)
    
    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        t = timestep
        prev_timestep = self._get_previous_timestep(t)
        
        # Get alphas for current and previous timestep
        alpha_prod_t = self.alpha_cumprod[t]
        alpha_prod_t_prev = self.alpha_cumprod[prev_timestep] if prev_timestep >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Current timestep values
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        # Predict x_0 from current latent and predicted noise
        pred_original_sample = (latents - self.sqrt_one_minus_alpha_cumprod[t] * model_output) / self.sqrt_alpha_cumprod[t]
        pred_original_sample = pred_original_sample.clamp(-1, 1)
        
        # Compute coefficients for the mean
        pred_original_sample_coeff = alpha_prod_t_prev.sqrt() * current_beta_t / beta_prod_t
        current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t
        
        # Compute mean of the posterior distribution
        mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
        
        # Compute variance
        variance = current_beta_t * beta_prod_t_prev / beta_prod_t
        variance = torch.clamp(variance, min=1e-10)  # Less aggressive clamping
        
        # Add noise only if not the last step
        if t > 0:
            noise = torch.randn(model_output.shape, device=latents.device, dtype=latents.dtype, generator=self.generator)
            std = variance.sqrt() * noise
            return mean + std
        
        return mean

    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        timesteps = timesteps.to(original_samples.device)
        sqrt_alpha_prod = self.sqrt_alpha_cumprod[timesteps].flatten()
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alpha_cumprod[timesteps].flatten()
        
        # Expand dimensions to match original_samples shape
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        noise = torch.randn(original_samples.shape, device=original_samples.device,
                          dtype=original_samples.dtype, generator=self.generator)
        
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    # To remove the noise, we use the prediction from the unet
    def remove_noise():
        pass
        