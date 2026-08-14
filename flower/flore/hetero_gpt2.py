"""Heterogeneous GPT-2 Model for Model-Heterogeneous Federated Learning.

This module implements a width-scalable GPT-2 style Transformer that supports
HeteroFL-style model heterogeneity for language modeling tasks.

Based on:
- HeteroFL: Computation and Communication Efficient Federated Learning (ICLR 2021)
- GPT-2: Language Models are Unsupervised Multitask Learners (Radford et al., 2019)

Key features:
- Width-scalable architecture (embed_dim, num_heads, ffn_dim all scale with model_rate)
- Scaler modules after linear layers (HeteroFL Section 3.3)
- Static Layer Normalization (analogous to sBN for Transformers)
"""

import math
from collections.abc import Mapping
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalerLM(nn.Module):
    """Scaling layer for heterogeneous language model training.

    According to HeteroFL paper Section 3.3:
    "The Scaler module scales representations by 1/r during the training phase."

    IMPORTANT: Scaling is only applied during training. During inference,
    the global model is used without any scaling (identity).
    """

    def __init__(self, rate: float):
        super().__init__()
        self.scale = 1.0 / rate if rate > 0 else 1.0

    def forward(self, x):
        if self.training:
            return x * self.scale
        return x


class StaticLayerNorm(nn.LayerNorm):
    """Static Layer Normalization for HeteroFL.

    Analogous to Static Batch Normalization (sBN) from HeteroFL paper.
    For LayerNorm, we don't have running statistics, so this behaves
    like standard LayerNorm but can be configured for calibration if needed.

    Note: Unlike BatchNorm, LayerNorm normalizes over the feature dimension
    per sample, so there are no "running statistics" to calibrate.
    This class exists for API consistency with the vision models.
    """

    def __init__(self, normalized_shape, **kwargs):
        super().__init__(normalized_shape, **kwargs)


class HeteroMultiHeadAttention(nn.Module):
    """Multi-head attention with HeteroFL width scaling.

    Supports width-heterogeneous models where embed_dim and num_heads
    are scaled by model_rate.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        model_rate: float = 1.0,
        use_scaler: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.model_rate = model_rate

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Scalers after projections (HeteroFL Section 3.3)
        if use_scaler and model_rate < 1.0:
            self.q_scaler = ScalerLM(model_rate)
            self.k_scaler = ScalerLM(model_rate)
            self.v_scaler = ScalerLM(model_rate)
            self.out_scaler = ScalerLM(model_rate)
        else:
            self.q_scaler = nn.Identity()
            self.k_scaler = nn.Identity()
            self.v_scaler = nn.Identity()
            self.out_scaler = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            attention_mask: Optional attention mask
            is_causal: Whether to apply causal masking (for autoregressive LM)

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project and scale
        q = self.q_scaler(self.q_proj(x))
        k = self.k_scaler(self.k_proj(x))
        v = self.v_scaler(self.v_proj(x))

        # Reshape for multi-head attention: (batch, seq, embed) -> (batch, heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply causal mask for autoregressive modeling
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, embed)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Output projection with scaler
        output = self.out_scaler(self.out_proj(attn_output))

        return output


class HeteroFeedForward(nn.Module):
    """Feed-forward network with HeteroFL width scaling.

    Standard Transformer FFN: Linear -> GELU -> Linear
    with scalers after each linear layer.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
        model_rate: float = 1.0,
        use_scaler: bool = True,
    ):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

        if use_scaler and model_rate < 1.0:
            self.scaler1 = ScalerLM(model_rate)
            self.scaler2 = ScalerLM(model_rate)
        else:
            self.scaler1 = nn.Identity()
            self.scaler2 = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scaler1(self.fc1(x))
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.scaler2(self.fc2(x))
        x = self.dropout(x)
        return x


class HeteroTransformerBlock(nn.Module):
    """Transformer block with HeteroFL width scaling.

    Pre-norm architecture (like GPT-2):
    x -> LayerNorm -> Attention -> + -> LayerNorm -> FFN -> +
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        model_rate: float = 1.0,
        use_scaler: bool = True,
    ):
        super().__init__()

        self.ln1 = StaticLayerNorm(embed_dim)
        self.attn = HeteroMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            model_rate=model_rate,
            use_scaler=use_scaler,
        )

        self.ln2 = StaticLayerNorm(embed_dim)
        self.ffn = HeteroFeedForward(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
            model_rate=model_rate,
            use_scaler=use_scaler,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        # Pre-norm FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class HeteroGPT2(nn.Module):
    """GPT-2 style language model with HeteroFL width scaling.

    Supports model heterogeneity through width scaling:
    - embed_dim: scales with model_rate
    - num_heads: scales with model_rate (maintains head_dim)
    - ffn_dim: scales with model_rate

    Args:
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        embed_dim: Base embedding dimension (scaled by model_rate)
        num_heads: Base number of attention heads (scaled by model_rate)
        num_layers: Number of transformer layers
        ffn_multiplier: FFN dimension multiplier (ffn_dim = embed_dim * ffn_multiplier)
        dropout: Dropout probability
        model_rate: Width scaling factor (1.0 = full model)
        use_scaler: Whether to use HeteroFL scalers
        tie_weights: Whether to tie input/output embeddings
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        max_seq_len: int = 256,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        model_rate: float = 1.0,
        use_scaler: bool = True,
        tie_weights: bool = True,
    ):
        super().__init__()

        self.model_rate = model_rate
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Scale dimensions by model_rate
        self.embed_dim = int(np.ceil(model_rate * embed_dim))
        self.num_heads = max(1, int(np.ceil(model_rate * num_heads)))
        self.ffn_dim = int(np.ceil(model_rate * embed_dim * ffn_multiplier))
        self.num_layers = num_layers

        # Ensure embed_dim is divisible by num_heads
        while self.embed_dim % self.num_heads != 0:
            self.embed_dim += 1

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, self.embed_dim)

        # Scaler after embeddings
        if use_scaler and model_rate < 1.0:
            self.embed_scaler = ScalerLM(model_rate)
        else:
            self.embed_scaler = nn.Identity()

        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            HeteroTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                dropout=dropout,
                model_rate=model_rate,
                use_scaler=use_scaler,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = StaticLayerNorm(self.embed_dim)

        # Output head (language modeling head)
        self.lm_head = nn.Linear(self.embed_dim, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights like GPT-2."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Optional attention mask
            labels: Optional labels for computing loss (same shape as input_ids)

        Returns:
            Tuple of (logits, loss) where loss is None if labels not provided
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get position indices
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)

        x = self.embed_scaler(token_embeds + position_embeds)
        x = self.dropout(x)

        # Transform through layers
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        # Final layer norm
        x = self.ln_f(x)

        # Language modeling head
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Ignore padding
            )

        return logits, loss

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.position_embedding.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ============================================================================
# Factory functions for different model sizes
# ============================================================================

def HeteroGPT2Small(
    vocab_size: int = 10000,
    max_seq_len: int = 256,
    model_rate: float = 1.0,
    use_scaler: bool = True,
    **kwargs
) -> HeteroGPT2:
    """GPT-2 Small style model (~50M params at rate=1.0).

    Architecture (at rate=1.0):
    - embed_dim: 768
    - num_heads: 12
    - num_layers: 6 (reduced from 12 to target ~50M params)
    - ffn_dim: 3072 (4x embed_dim)

    Note: Original GPT-2 Small has 12 layers (~124M with vocab=50K).
    We use 6 layers to target ~50M params with vocab=10K.
    """
    return HeteroGPT2(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=768,
        num_heads=12,
        num_layers=6,  # 6 layers for ~50M params
        ffn_multiplier=4,
        model_rate=model_rate,
        use_scaler=use_scaler,
        **kwargs
    )


def HeteroGPT2Medium(
    vocab_size: int = 10000,
    max_seq_len: int = 256,
    model_rate: float = 1.0,
    use_scaler: bool = True,
    **kwargs
) -> HeteroGPT2:
    """GPT-2 Medium style model (~100M params at rate=1.0)."""
    return HeteroGPT2(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=1024,
        num_heads=16,
        num_layers=12,
        ffn_multiplier=4,
        model_rate=model_rate,
        use_scaler=use_scaler,
        **kwargs
    )


def HeteroLSTM(
    vocab_size: int = 10000,
    embed_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 2,
    dropout: float = 0.1,
    model_rate: float = 1.0,
    use_scaler: bool = True,
    tie_weights: bool = True,
) -> nn.Module:
    """LSTM language model with HeteroFL width scaling.

    Simpler alternative to GPT-2 for faster experiments.
    """

    class HeteroLSTMModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.model_rate = model_rate
            self.vocab_size = vocab_size

            # Scale dimensions
            self.embed_dim = int(np.ceil(model_rate * embed_dim))
            self.hidden_dim = int(np.ceil(model_rate * hidden_dim))

            self.embedding = nn.Embedding(vocab_size, self.embed_dim)

            if use_scaler and model_rate < 1.0:
                self.embed_scaler = ScalerLM(model_rate)
            else:
                self.embed_scaler = nn.Identity()

            self.lstm = nn.LSTM(
                self.embed_dim,
                self.hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(self.hidden_dim, vocab_size)

            if use_scaler and model_rate < 1.0:
                self.fc_scaler = ScalerLM(model_rate)
            else:
                self.fc_scaler = nn.Identity()

            if tie_weights:
                # Tie weights requires matching dimensions
                if self.embed_dim == self.hidden_dim:
                    self.fc.weight = self.embedding.weight

        def forward(self, input_ids, labels=None):
            embeds = self.embed_scaler(self.embedding(input_ids))
            lstm_out, _ = self.lstm(embeds)
            lstm_out = self.dropout(lstm_out)
            logits = self.fc(self.fc_scaler(lstm_out))

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            return logits, loss

        def get_num_params(self):
            return sum(p.numel() for p in self.parameters())

    return HeteroLSTMModel()


def create_hetero_lm_model(
    model_type: str = "gpt2",
    model_rate: float = 1.0,
    vocab_size: int = 10000,
    max_seq_len: int = 256,
    model_config: Optional[Mapping[str, Any]] = None,
    use_scaler: bool = True,
) -> nn.Module:
    """Create a configurable heterogeneous language model.

    This centralizes LM model creation so the client, strategy, and server all
    honor the same config knobs for tiny smoke tests and larger experiments.
    """
    model_config = model_config or {}

    if model_type == "gpt2":
        return HeteroGPT2(
            vocab_size=int(model_config.get("vocab_size", vocab_size)),
            max_seq_len=int(model_config.get("max_seq_len", max_seq_len)),
            embed_dim=int(model_config.get("embed_dim", 768)),
            num_heads=int(model_config.get("num_heads", 12)),
            num_layers=int(model_config.get("num_layers", 6)),
            ffn_multiplier=int(model_config.get("ffn_multiplier", 4)),
            dropout=float(model_config.get("dropout", 0.1)),
            model_rate=model_rate,
            use_scaler=bool(model_config.get("use_scaler", use_scaler)),
            tie_weights=bool(model_config.get("tie_weights", True)),
        )

    if model_type == "lstm":
        return HeteroLSTM(
            vocab_size=int(model_config.get("vocab_size", vocab_size)),
            embed_dim=int(model_config.get("embed_dim", 256)),
            hidden_dim=int(model_config.get("hidden_dim", 512)),
            num_layers=int(model_config.get("num_layers", 2)),
            dropout=float(model_config.get("dropout", 0.1)),
            model_rate=model_rate,
            use_scaler=bool(model_config.get("use_scaler", use_scaler)),
            tie_weights=bool(model_config.get("tie_weights", True)),
        )

    raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================================
# Complexity levels (matching HeteroFL paper)
# ============================================================================

def get_lm_complexity_levels():
    """Return complexity level configuration for language models."""
    return {
        'a': 1.0,      # Full model
        'b': 0.5,      # r^1 = 0.5
        'c': 0.25,     # r^2 = 0.25
        'd': 0.125,    # r^3 = 0.125
        'e': 0.0625,   # r^4 = 0.0625
    }


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HeteroGPT2 - Width-Scalable GPT-2 for HeteroFL")
    print("=" * 70)

    print("\nModel complexity levels (GPT-2 Small style, vocab=10000):")
    print("-" * 70)
    print(f"{'Level':<6} {'Rate':<8} {'Embed':<8} {'Heads':<8} {'FFN':<10} {'Params':<15} {'Size (MB)':<10}")
    print("-" * 70)

    levels = get_lm_complexity_levels()

    for level, rate in levels.items():
        model = HeteroGPT2Small(vocab_size=10000, model_rate=rate)
        params = model.get_num_params()
        size_mb = params * 4 / (1024**2)
        print(f"{level:<6} {rate:<8.4f} {model.embed_dim:<8} {model.num_heads:<8} {model.ffn_dim:<10} {params:>12,}   {size_mb:>8.2f}")

    # Forward pass test
    print("\n" + "-" * 70)
    print("Forward pass test (50% width model):")
    print("-" * 70)

    model = HeteroGPT2Small(vocab_size=10000, model_rate=0.5)

    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    labels = input_ids.clone()

    logits, loss = model(input_ids, labels=labels)

    print(f"  Input shape:  {tuple(input_ids.shape)}")
    print(f"  Output shape: {tuple(logits.shape)}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Params: {model.get_num_params():,}")

    # LSTM test
    print("\n" + "-" * 70)
    print("HeteroLSTM test (for faster experiments):")
    print("-" * 70)

    for rate in [1.0, 0.5, 0.25, 0.125]:
        lstm_model = HeteroLSTM(vocab_size=10000, model_rate=rate)
        params = lstm_model.get_num_params()
        print(f"  Rate {rate}: {params:,} params")

    print("\n" + "=" * 70)
    print("All tests passed!")
