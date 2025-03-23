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
        # Hugging Face calls this scaled linear schedule
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5,
                                    num_training_steps, device=device) ** 2
        self.alphas = 1.0 - self.betas
        # [alpha_0, alpha_0 * alpha_1, alpha_0 * alpha_1 * alpha_2, ...]
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.device = device
        self.one = torch.tensor(1.0, device=device)
        self.timesteps = torch.arange(0, num_training_steps, device=device)[::-1]

    def set_inference_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        # 999, 998, 997, ... 0
        # If we want to reduce the number of inference steps, to say 50 - 
        # 999, 979, 959, ... 0 = 50 steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        self.timesteps = (np.arange(0, num_inference_steps) * step_ratio).round().long()[::-1]
        self.timesteps = torch.from_numpy(self.timesteps).to(self.device)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        if prev_t < 0:
            prev_t = 0
        return prev_t

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        # latents - the Unet operats on the latents and predictrs the noise which is the
        # model_output - epsilon_theta(x_t, t)
        t = timestep
        # prev_timestep is the next timestep, in reverse diffusion process
        prev_timestep = self._get_previous_timestep(t)
        alpha_prod_t = self.alpha_cumprod[t]
        alpha_prod_t_prev = self.alpha_cumprod[prev_timestep] if prev_timestep >= 0 else self.one

        # I don't think it is correct to use beta_prod as 1 - alpha_prod.
        # beta = 1 - alpha; alpha_prod = prod of alphas till t.
        # thus, beta_prod = prod of betas till t = product [1-alpha_0, 1-alpha_1, ... 1-alpha_t]
        # which is not the same as 1 - alpha_prod_t.
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        # Took a minute but I got what I was missing - the difference in notations for
        # cumulative product of alphas and betas and the values at each timestep.
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = beta_prod_t / beta_prod_t_prev
        
        # There are two ways to go from more noisy image to less noisy image
        # Algorithm 2 in the paper and Equation 6-7.
        # q(xt-1|xt,x0) is conditioned on x0 which is the original image. Since this is not
        # available, Equation 15 provides a way to predict x0 from xt.
        pred_original_sample = (latents - (beta_prod_t ** 0.5) * model_output) / alpha_prod_t ** 0.5

        # Compute the coefficients for the current sample x_t and the predicted x0
        # coefficient of predicted original sample is sqrt(prod of alpha till t-1) * beta for the
        # current timestep / (1 - prod of alpha till t)
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5) * current_beta_t / beta_prod_t
        curr_sample_coeff = (alpha_prod_t ** 0.5) * (1 - beta_prod_t_prev) / beta_prod_t
        mean = pred_original_sample_coeff * pred_original_sample + curr_sample_coeff * latents
        # From the image, the variance is calculated as tilde_beta_t = (1 - alpha_cumprod_{t-1})/(1 - alpha_cumprod_t) * beta_t
        # This is equivalent to: (1 - alpha_prod_t_prev)/(1 - alpha_prod_t) * current_beta_t
        variance = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) * current_beta_t
        # Why clamping the variance?
        variance = torch.clamp(variance, min=1e-20)

        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, device=device, dtype=model_output.dtype, generator=self.generator)
            std = torch.sqrt(variance) * noise
        
        # Now, sample from the distribution, assuming initial distribution is N(0,1)
        # This is equivalent to: mean + std * Z, where Z ~ N(0,1)
        pred_prev_sample = mean + std * noise
        return pred_prev_sample

    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod) < len(original_samples):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        # We want the standard deviation of the noise
        sqrt_one_minus_alpha_prod = ((1 - alpha_cumprod[timesteps]) ** 0.5).flatten()
        while len(sqrt_one_minus_alpha_prod) < len(original_samples):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.randn(original_samples.shape,
                            device=original_samples.device,
                            dtype=original_samples.dtype,
                            generator=self.generator)
        # According to the equation in the paper, for DDPM
        # mean = sqrt_alpha_prod * original_samples
        # and simply follows X = mean + std * Z
        noise_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noise_samples

    # To remove the noise, we use the prediction from the unet
    def remove_noise():
        pass
        