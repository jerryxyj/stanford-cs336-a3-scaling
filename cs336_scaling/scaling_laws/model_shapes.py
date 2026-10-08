"""Model-shape bookkeeping: build API architecture configs, count parameters, count FLOPs.

The parameter count mirrors :class:`cs336_scaling.training.model.basic_model.BasicCausalLM`
exactly (verified against ``count_params`` in the tests), so that the ``N`` used in the
scaling laws corresponds to the model the API actually trains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from cs336_scaling.training.model.config import BasicTransformerConfig
from cs336_scaling.training.training_config import TrainingConfig

VOCAB_SIZE = 32_000
SEQ_LEN = TrainingConfig.seq_len
DEFAULT_HEAD_DIM = 64
FLOPS_PER_PARAM_TOKEN = (
    6  # forward (2) + backward (4) matmul FLOPs per parameter per token
)

ParamCountMode = Literal["non_embedding", "total"]


def round_up_to_multiple(x: float, multiple: int) -> int:
    return int(math.ceil(x / multiple)) * multiple


def default_intermediate_size(hidden_size: int) -> int:
    """SwiGLU width ``~8/3 d`` rounded up to a multiple of 128 (448 -> 1280 as in the handout)."""
    return round_up_to_multiple(8 / 3 * hidden_size, 128)


def make_architecture_config(
    *,
    num_layers: int,
    hidden_size: int,
    head_dim: int = DEFAULT_HEAD_DIM,
    intermediate_size: int | None = None,
    vocab_size: int = VOCAB_SIZE,
    dtype: Literal["float32", "bfloat16"] = "bfloat16",
    tie_word_embeddings: bool = False,
    rope_theta: int = 1_000_000,
    rms_norm_eps: float = 1e-6,
) -> BasicTransformerConfig:
    if hidden_size % head_dim != 0:
        raise ValueError(f"{hidden_size=} must be a multiple of {head_dim=}")
    num_heads = hidden_size // head_dim
    return BasicTransformerConfig(
        attention_bias=False,
        head_dim=head_dim,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size or default_intermediate_size(hidden_size),
        num_attention_heads=num_heads,
        num_hidden_layers=num_layers,
        num_key_value_heads=num_heads,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        tie_word_embeddings=tie_word_embeddings,
        dtype=dtype,
        vocab_size=vocab_size,
    )


def ladder_shape(
    k: int, *, aspect_ratio: int = DEFAULT_HEAD_DIM
) -> BasicTransformerConfig:
    """One-parameter family of shapes: ``k`` layers, ``d_model = aspect_ratio * k``.

    With ``aspect_ratio = head_dim = 64`` this gives ``k`` heads as well, so the family is
    (4L, 256d), (6L, 384d), (8L, 512d), ... (32L, 2048d). Kaplan et al. (2020, Fig. 5) found
    that the loss depends only weakly on the aspect ratio ``d_model / n_layer`` over a wide
    range, so fixing it lets us treat model size as a single scalar ``N``.
    """
    return make_architecture_config(num_layers=k, hidden_size=aspect_ratio * k)


@dataclass(frozen=True)
class ParameterCounts:
    embedding: int
    lm_head: int
    per_layer: int
    layers: int
    final_norm: int

    @property
    def non_embedding(self) -> int:
        """Everything except the token embedding matrix and the (untied) output projection."""
        return self.layers + self.final_norm

    @property
    def total(self) -> int:
        return self.embedding + self.lm_head + self.non_embedding

    def get(self, mode: ParamCountMode) -> int:
        return self.non_embedding if mode == "non_embedding" else self.total

    def to_dict(self) -> dict[str, int]:
        return {
            "embedding": self.embedding,
            "lm_head": self.lm_head,
            "per_layer": self.per_layer,
            "layers": self.layers,
            "final_norm": self.final_norm,
            "non_embedding": self.non_embedding,
            "total": self.total,
        }


def count_parameters(config: BasicTransformerConfig) -> ParameterCounts:
    """Exact parameter count of :class:`BasicCausalLM` for ``config``."""
    d = config.hidden_size
    q_out = config.num_attention_heads * config.head_dim
    kv_out = config.num_key_value_heads * config.head_dim
    bias = 1 if config.attention_bias else 0
    attention = (
        d * q_out
        + bias * q_out  # q_proj
        + 2 * (d * kv_out + bias * kv_out)  # k_proj, v_proj
        + q_out * d
        + bias * d  # o_proj
        + 2 * config.head_dim  # q_norm, k_norm (RMSNorm weights over head_dim)
    )
    mlp = 3 * d * config.intermediate_size  # gate, up, down (no biases)
    norms = 2 * d  # input_layernorm, post_attention_layernorm
    per_layer = attention + mlp + norms
    return ParameterCounts(
        embedding=config.vocab_size * d,
        lm_head=0 if config.tie_word_embeddings else d * config.vocab_size,
        per_layer=per_layer,
        layers=per_layer * config.num_hidden_layers,
        final_norm=d,
    )


def kaplan_non_embedding_estimate(config: BasicTransformerConfig) -> int:
    """The handout's ``12 * n_layer * d_model**2`` estimate of non-embedding parameters."""
    return 12 * config.num_hidden_layers * config.hidden_size**2


def flops_per_token(
    config: BasicTransformerConfig,
    *,
    seq_len: int = SEQ_LEN,
    include_attention_scores: bool = True,
    include_lm_head: bool = True,
) -> float:
    """Training FLOPs per token.

    ``6 N`` for the dense matmuls (Kaplan et al. ``C ~= 6 N D``), plus, optionally, the
    output projection (``6 d V``, significant for small models with a 32K vocabulary) and the
    attention-score FLOPs (``6 n_layer seq_len d``, halved for the causal mask that a
    block-sparse flash-attention kernel skips).
    """
    counts = count_parameters(config)
    flops = FLOPS_PER_PARAM_TOKEN * counts.non_embedding
    if include_lm_head:
        flops += FLOPS_PER_PARAM_TOKEN * config.hidden_size * config.vocab_size
    if include_attention_scores:
        flops += (
            FLOPS_PER_PARAM_TOKEN
            * config.num_hidden_layers
            * seq_len
            * config.hidden_size
            * 0.5
        )
    return float(flops)


def training_flops(
    config: BasicTransformerConfig,
    total_train_tokens: int,
    *,
    param_count_mode: ParamCountMode = "non_embedding",
) -> float:
    """The textbook ``C = 6 N D`` used on the x-axis of IsoFLOPs plots."""
    return (
        FLOPS_PER_PARAM_TOKEN
        * count_parameters(config).get(param_count_mode)
        * total_train_tokens
    )


def describe_shape(config: BasicTransformerConfig) -> str:
    counts = count_parameters(config)
    return (
        f"{config.num_hidden_layers}L x {config.hidden_size}d "
        f"(ff {config.intermediate_size}, {config.num_attention_heads} heads) "
        f"N_ne={counts.non_embedding / 1e6:.1f}M total={counts.total / 1e6:.1f}M"
    )
