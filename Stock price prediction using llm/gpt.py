import random
import torch
from torch import nn
import tiktoken
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)

class MaskedMultiheadAttention(nn.Module):
    def __init__(self, context_length, dim_in, dim_out, n_heads, dropout_rate, qkv_bias=False):
        super().__init__()
        
        self.context_length = context_length
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_heads = n_heads

        assert self.dim_out % self.n_heads == 0, "dim_out should be divisible by n_heads"
        
        self.head_dim = self.dim_out // self.n_heads
        self.W_key = nn.Linear(self.dim_in, self.dim_out, bias=qkv_bias)
        self.W_query = nn.Linear(self.dim_in, self.dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(self.dim_in, self.dim_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(self.dim_out, self.dim_out)
        self.register_buffer("mask", torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1).bool())

    def forward(self, x):
        batch_size, num_tokens, dim_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.n_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)
        attention_scores.masked_fill_(self.mask[:num_tokens, :num_tokens], -torch.inf)
        attention_weights = torch.softmax(attention_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)
        context_vectors = (attention_weights @ values).transpose(1, 2)
        context_vectors = context_vectors.contiguous().view(batch_size, num_tokens, self.dim_out)
        context_vectors = self.out_proj(context_vectors)
        return context_vectors

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MaskedMultiheadAttention(
            context_length = cfg["context_length"],
            dim_in = cfg["emb_dim"],
            dim_out = cfg["emb_dim"],
            dropout_rate = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"],
            n_heads = cfg["n_heads"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_embedding(in_idx)
        pos_embeds = self.pos_embedding(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def total_params(model):
    return sum(p.numel() for p in model.parameters())