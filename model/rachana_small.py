# model/mini_mistral.py

import torch
import torch.nn as nn
import math
from model.config import RachanaSmall

# SwiGLU activation block
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff * 2)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.w1(x)
        x1, x2 = x.chunk(2, dim=-1)
        return self.w2(torch.nn.functional.silu(x1) * x2)

# Rotary embedding helper
def apply_rope(x, sin, cos):
    # x: (B, T, n_heads, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(2)  # → (1, T, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot

def build_rope_cache(seq_len, head_dim, device):
    theta = 10000 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    seq = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.einsum('i,j->ij', seq, theta)
    sin, cos = torch.sin(freqs), torch.cos(freqs)
    return sin.to(device), cos.to(device)

# PreNorm wrapper
class PreNorm(nn.Module):
    def __init__(self, d_model, fn, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

# Attention block
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout, max_seq_len, use_rope=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)

        if use_rope:
            self.register_buffer("sin", None, persistent=False)
            self.register_buffer("cos", None, persistent=False)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)

        if self.use_rope:
            if self.sin is None or self.sin.shape[0] < T:
                sin, cos = build_rope_cache(T, self.head_dim, x.device)
                self.sin, self.cos = sin, cos
            q = apply_rope(q, self.sin[:T, :], self.cos[:T, :])
            k = apply_rope(k, self.sin[:T, :], self.cos[:T, :])

        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(
            torch.triu(torch.ones_like(attn_scores), diagonal=1) == 1, float('-inf')
        )

        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        out = (attn_probs @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, config: RachanaSmall):
        super().__init__()
        attn = MultiHeadAttention(
            config.d_model, config.n_heads,
            config.attn_dropout, config.max_seq_len,
            use_rope=config.use_rope
        )
        ffn = SwiGLU(config.d_model, config.d_ff)

        self.attn = PreNorm(config.d_model, attn, eps=config.layer_norm_eps) if config.use_preenorm else attn
        self.ffn = PreNorm(config.d_model, ffn, eps=config.layer_norm_eps) if config.use_preenorm else ffn
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(x))
        x = x + self.dropout(self.ffn(x))
        return x

#RachanaSmallModel
class RachanaSmallModel(nn.Module):
    def __init__(self, config: RachanaSmall):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.output_head.weight = self.token_emb.weight  # weight tying

    def forward(self, input_ids):
        x = self.token_emb(input_ids)  # (B, T, d_model)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.output_head(x)  # (B, T, vocab_size)
        return logits
