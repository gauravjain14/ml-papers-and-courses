# CLIP

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, sequence_length: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        # position embedding is learnable
        self.position_embedding = nn.Parameter(torch.zeros(sequence_length, embedding_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch size, Sequence length) -> (Batch size, Sequence length, Embedding dim)
        x = self.token_embedding(x)
        x += self.position_embedding
        return x
    

class CLIPLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.attention = SelfAttention(num_heads, embedding_dim)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = nn.Linear(4 * embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layernorm1(x)
        x = self.attention(x, causal_mask=True)
        x += residual

        # Feed Forward Layer
        residual = x
        x = self.layernorm2(x)
        x = self.linear1(x)
        
        # Quick GELU activation
        x = x * torch.sigmoid(1.702 * x)
        x = self.linear2(x)
        x += residual
        return x


class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for _ in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.type(torch.long)

        # Sequence length is fixed to 77 because we are preloading
        # (Batch size, 77) -> (Batch size, 77, 768)
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.layernorm(x)
        
        return x