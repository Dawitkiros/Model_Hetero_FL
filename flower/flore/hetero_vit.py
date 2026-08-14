"""Width-sliceable Vision Transformer for model-heterogeneous federated learning.

Mirrors the HeteroFL width-scaling contract of hetero_gpt2.py / hetero_models.py:
embed_dim and num_heads scale with model_rate at constant head_dim, so a rate-r
sub-model is an exact channel-prefix of the full model and the generic
static-prefix extractor yields a coherent core. ScalerV2 (1/rate fan-in
compensation, with always_scale for core eval) follows every parametric layer.
Normalization is LayerNorm: per-sample, no running stats, so sBN calibration is a
no-op on this model. pos_embed and cls_token are 2-D parameters so the extractor
trims their embedding dimension like any other weight.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hetero_models import ScalerV2


def _scaler(model_rate: float, use_scaler: bool) -> nn.Module:
    return ScalerV2(model_rate) if (use_scaler and model_rate < 1.0) else nn.Identity()


class HeteroPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, model_rate, use_scaler):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.scaler = _scaler(model_rate, use_scaler)

    def forward(self, x):
        x = self.scaler(self.proj(x))
        return x.flatten(2).transpose(1, 2)


class HeteroAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, model_rate, use_scaler):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.q_scaler = _scaler(model_rate, use_scaler)
        self.k_scaler = _scaler(model_rate, use_scaler)
        self.v_scaler = _scaler(model_rate, use_scaler)
        self.out_scaler = _scaler(model_rate, use_scaler)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, E = x.shape
        q = self.q_scaler(self.q_proj(x)).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_scaler(self.k_proj(x)).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_scaler(self.v_proj(x)).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, E)
        return self.out_scaler(self.out_proj(out))


class HeteroMLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout, model_rate, use_scaler):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.s1 = _scaler(model_rate, use_scaler)
        self.s2 = _scaler(model_rate, use_scaler)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(F.gelu(self.s1(self.fc1(x))))
        return self.drop(self.s2(self.fc2(x)))


class HeteroViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout, model_rate, use_scaler):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = HeteroAttention(embed_dim, num_heads, dropout, model_rate, use_scaler)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = HeteroMLP(embed_dim, mlp_dim, dropout, model_rate, use_scaler)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class HeteroViT(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3,
                 num_classes: int = 100, base_embed_dim: int = 256, base_num_heads: int = 8,
                 depth: int = 7, mlp_ratio: int = 4, dropout: float = 0.1,
                 model_rate: float = 1.0, use_scaler: bool = True):
        super().__init__()
        assert base_embed_dim % base_num_heads == 0
        self.model_rate = model_rate
        self.use_scaler = use_scaler
        head_dim = base_embed_dim // base_num_heads

        num_heads = max(1, int(np.ceil(model_rate * base_num_heads)))
        embed_dim = num_heads * head_dim
        mlp_dim = embed_dim * mlp_ratio
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_embed = HeteroPatchEmbed(img_size, patch_size, in_chans, embed_dim,
                                            model_rate, use_scaler)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            HeteroViTBlock(embed_dim, num_heads, mlp_dim, dropout, model_rate, use_scaler)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed.unsqueeze(0)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        return self.linear(self.forward_features(x))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


VIT_BASE = dict(base_embed_dim=256, base_num_heads=8, depth=7, mlp_ratio=4, dropout=0.1)


def build_hetero_vit(model_name: str, in_channels: int = 3, num_classes: int = 100,
                     model_rate: float = 1.0, use_scaler: bool = True,
                     overrides: Optional[dict] = None) -> HeteroViT:
    cfg = dict(VIT_BASE)
    if overrides:
        cfg.update(overrides)
    if model_name in ("HeteroViT", "HeteroViT32"):
        img_size, patch_size = 32, 4
    elif model_name == "HeteroViT64":
        img_size, patch_size = 64, 8
    else:
        raise ValueError(f"Unknown ViT model name {model_name!r}")
    return HeteroViT(img_size=img_size, patch_size=patch_size, in_chans=in_channels,
                     num_classes=num_classes, model_rate=model_rate, use_scaler=use_scaler, **cfg)
