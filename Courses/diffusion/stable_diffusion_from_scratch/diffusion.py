# diffusion.py

from stable_diffusion_from_scratch.attention import SelfAttention
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, 4 * dim)
        self.linear2 = nn.Linear(4 * dim, 4 * dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        x = self.linear1(time)
        x = F.silu(x)
        x = self.linear2(x)
        return x


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch size, Channels, Height, Width) -> (Batch size, Channels, Height * 2, Width * 2)
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch size, Channels, Height/8, Width/8-> (Batch size, Channels, Height, Width)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        # (Batch size, 4, Height/8, Width/8)
        return x


class UNet_ResidualBlock(nn.Module):
    # n_time is the embedding of the time step
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        residual = x
        
        feature = self.groupnorm1(x)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual(residual)


class UNet_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embed: int):
        super().__init__()
        channels = n_head * n_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Linear(channels, channels)
        
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = SelfAttention(n_head, n_embed)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Linear(channels, channels, kernel_size=1, padding=0)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (Batch size, Channels, Height, Width)
        # context: (Batch size, seq_length, Dim)
        residual = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        n, c, h, w = x.shape
        x = x.view(n, c, h * w)

        # (Batch size, Channels, Height * Width) -> (Batch size, Height * Width, Channels)
        x = x.transpose(1, 2)

        residual_short = x

        # (Batch size, Height * Width, Channels) -> (Batch size, Height * Width, Channels)
        x = self.layernorm_1(x)

        # (Batch size, Height * Width, Channels) -> (Batch size, Height * Width, Channels)
        x = self.attention_1(x)

        x += residual_short

        residual_short = x

        # (Batch size, Height * Width, Channels) -> (Batch size, Height * Width, Channels)
        x = self.layernorm_2(x)

        # Cross Attention between the latent and the prompt
        self.attention_2(x, context)

        x += residual_short

        residual_short = x
        
        # FFN with GeGLU and skip connection
        x = self.layernorm_3(x)

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)

        x += residual_short

        # (Batch size, Height * Width, Channels) -> (Batch size, Channels, Height * Width)
        x = x.transpose(1, 2)

        # (Batch size, Channels, Height * Width) -> (Batch size, Channels, Height, Width)
        x = x.view(n, c, h, w)

        x = self.conv_output(x)

        return x + residual
        


class SwitchSequential(nn.Module):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNet_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNet_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNet_ResidualBlock(320, 320),
                            UNet_AttentionBlock(8, 40)),

            SwitchSequential(UNet_ResidualBlock(320, 320),
                            UNet_AttentionBlock(8, 40)),

            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNet_ResidualBlock(320, 640),
                            UNet_AttentionBlock(8, 80)),

            SwitchSequential(UNet_ResidualBlock(640, 640),
                            UNet_AttentionBlock(8, 80)),

            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNet_ResidualBlock(640, 1280),
                            UNet_AttentionBlock(8, 160)),

            SwitchSequential(UNet_ResidualBlock(1280, 1280),
                            UNet_AttentionBlock(8, 160)),
            
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNet_ResidualBlock(1280, 1280),
                            UNet_AttentionBlock(8, 160)),

            SwitchSequential(UNet_ResidualBlock(1280, 1280),
                            UNet_AttentionBlock(8, 160)),
        ])

        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(1280, 1280),
            UNet_AttentionBlock(8, 160),
            UNet_ResidualBlock(1280, 1280),
        )
        
        # The decoder is going to be 2x in the channels because of the skip connections
        self.decoders = nn.ModuleList([
            # (Batch size, 2560, Height/64, Width/64) -> (Batch size, 1280, Height/64, Width/64)
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UpSample(1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),

            SwitchSequential(UNet_ResidualBlock(1920, 1280), UNet_AttentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(UNet_ResidualBlock(1920, 640), UNet_AttentionBlock(8, 80)),

            SwitchSequential(UNet_ResidualBlock(1280, 640), UNet_AttentionBlock(8, 80)),

            SwitchSequential(UNet_ResidualBlock(960, 640), UNet_AttentionBlock(8, 40), UpSample(640)),

            SwitchSequential(UNet_ResidualBlock(960, 320), UNet_AttentionBlock(8, 40)),

            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),

            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40))
        ])


class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNet_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor,
                context: torch.Tensor,
                time: torch.Tensor) -> torch.Tensor:
        # latent: (Batch size, 4, Height/8, Width/8)
        # context: (Batch size, seq_length, Dim=768)
        # time: (1, 320) (These are derived from the model)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch size, 4, Height/8, Width/8) -> (Batch size, 320, Height/8, Width/8)
        output = self.unet(latent, context, time)
        
        # (Batch size, 320, Height/8, Width/8) -> (Batch size, 4, Height/8, Width/8)
        output = self.final(output)
        return output
    
    