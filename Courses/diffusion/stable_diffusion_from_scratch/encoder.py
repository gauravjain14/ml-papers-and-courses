# Encoder for Stable Diffusion (from scratch)

import torch
from torch import nn
import torch.nn.functional as F
from decoder import VAE_ResidualBlock, VAE_AttentionBlock


class VAE_Encoder(nn.Module):
    
    # VAE_ResidualBlock is defined in decoder.py
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 3, 512, 512] - [batc_size, channel, height, width]
        # noise: [batch_size, output_channels, height/8, width/8]

        for module in self:
            # Why is this padding needed?
            if getattr(module, "stride", None) == 2:
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # At the end of the encoder, we get the image in latent space.
        # So, the output of the VAE is mean and log_variance of the latent space.
        # Divide [batch, 8, height/8, width/8] to two tensors of shape
        # [batch, 4, height/8, width/8] and [batch, 4, height/8, width/8]
        # where first is the mean and second is the log_variance.
        mean, log_variance = x.chunk(2, dim=1)

        # clamping the log_variance to avoid negative values
        log_variance = torch.clamp(log_variance, -30.0, 20.0)

        # Get the variance from the log_variance
        variance = torch.exp(log_variance)
        std_dev = torch.sqrt(variance)

        # How do we sample from the normal distribution?
        x = mean + std_dev * noise

        # Scale the output by a constant (from the repository)
        x = x * 0.18215

        return x
