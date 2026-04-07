"""Temporal attention classifier over frozen DINO frame features."""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Single learned query attending over N frame tokens with key-padding mask."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor):
        # x: (B, N, D); key_padding_mask: (B, N) True = PAD (to ignore)
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out, weights = self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=True)
        return out.squeeze(1), weights.squeeze(1)  # (B, D), (B, N)


class TemporalAttentionClassifier(nn.Module):
    def __init__(self, feature_dim: int = 384, d_model: int = 192, n_heads: int = 4,
                 n_layers: int = 2, mlp_dim: int = 384, dropout: float = 0.3,
                 num_classes: int = 2, max_len: int = 40):
        super().__init__()
        self.proj = nn.Linear(feature_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=mlp_dim,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = AttentionPool(d_model, n_heads=n_heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        """features: (B, N, feature_dim); mask: (B, N) True = REAL frame."""
        x = self.proj(features) + self.pos[:, : features.size(1)]
        kpm = ~mask  # True where padded
        x = self.encoder(x, src_key_padding_mask=kpm)
        pooled, attn_weights = self.pool(x, key_padding_mask=kpm)
        logits = self.head(pooled)
        return logits, attn_weights
