# self attention and cross attention
# On images, the channels are the embedding dimensions.

import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        batch_size, sequence_length, d_embed = x.shape
    
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        # split into query, key, value
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # Why transpose and not use interim shape as (batch, n_heads, sequence_length, d_head)?
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # compute attention weights
        weights = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        if causal_mask:
            # fill the upper triangle with -inf
            mask = torch.ones_like(weights, dtype=torch.bool).triu(1)
            weights.masked_fill_(mask, float("-inf"))
        
        weights = F.softmax(weights, dim=-1)
        # Out -> (batch, n_heads, sequence_length, d_head)
        out = weights @ v
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, sequence_length, d_embed)

        return self.out_proj(out)
        

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int,
                    in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.to_cross = nn.Linear(d_cross, d_embed * 3, bias=False)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: latent: (Batch size, Channels, Height, Width)
        # y: context: (Batch_size, Seq_Len_KV, Dim_KV) = (Batch_size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = y.shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # Multiply q, k, v by the projection matrices
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        
        weights = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        weights = F.softmax(weights, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(input_shape)

        return self.out_proj(out)
        
        