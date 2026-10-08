"""Throughput model: how many tokens per second does the API train for a given model shape?

The assignment's budgets are wall-clock seconds on one B200, not FLOPs. To turn a time budget
into a dataset size for each candidate model we need ``tokens/sec`` as a function of the
model. We use a latency-plus-compute ("roofline-style") model

    seconds_per_token = overhead + flops_per_token / (peak_flops * mfu_max)

whose two parameters are fit on completed runs: ``overhead`` captures the fixed per-step cost
that dominates for small models (kernel launches, optimizer update, memory-bound ops), while
``mfu_max`` is the utilisation that large, compute-bound models approach. Fitting in log
space weights relative errors equally across model sizes. Unlike a single power law this
form extrapolates to large models *conservatively* (towards the compute-bound limit), which
is the safe direction for a 48-hour run that must not time out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from cs336_scaling.scaling_laws.fit import FitMetrics, fit_metrics
from cs336_scaling.scaling_laws.model_shapes import flops_per_token
from cs336_scaling.training.model.config import BasicTransformerConfig

B200_PEAK_BF16_FLOPS = 2.25e15  # dense BF16 tensor-core peak of one B200


@dataclass(frozen=True)
class ThroughputObservation:
    name: str
    flops_per_token: float
    tokens_per_second: float
    batch_tokens: int

    @property
    def achieved_flops_per_second(self) -> float:
        return self.flops_per_token * self.tokens_per_second


@dataclass(frozen=True)
class ThroughputModel:
    overhead_seconds_per_token: float
    seconds_per_flop: float
    peak_flops: float = B200_PEAK_BF16_FLOPS
    metrics: FitMetrics | None = None
    n_observations: int = 0

    @property
    def asymptotic_mfu(self) -> float:
        """Utilisation approached by very large (compute-bound) models."""
        return 1.0 / (self.seconds_per_flop * self.peak_flops)

    def seconds_per_token_for_flops(self, fpt):
        return self.overhead_seconds_per_token + self.seconds_per_flop * np.asarray(fpt)

    def tokens_per_second_for_flops(self, fpt):
        return 1.0 / self.seconds_per_token_for_flops(fpt)

    def tokens_per_second(self, config: BasicTransformerConfig) -> float:
        return float(self.tokens_per_second_for_flops(flops_per_token(config)))

    def seconds_for_tokens(
        self, config: BasicTransformerConfig, tokens: float
    ) -> float:
        return tokens / self.tokens_per_second(config)

    def tokens_for_seconds(
        self, config: BasicTransformerConfig, seconds: float
    ) -> float:
        return seconds * self.tokens_per_second(config)

    def mfu(self, config: BasicTransformerConfig) -> float:
        fpt = flops_per_token(config)
        return float(fpt * self.tokens_per_second_for_flops(fpt) / self.peak_flops)

    def to_dict(self) -> dict[str, object]:
        return {
            "overhead_seconds_per_token": self.overhead_seconds_per_token,
            "seconds_per_flop": self.seconds_per_flop,
            "asymptotic_mfu": self.asymptotic_mfu,
            "peak_flops": self.peak_flops,
            "n_observations": self.n_observations,
            "metrics_log_space": self.metrics.to_dict() if self.metrics else None,
        }

    def describe(self) -> str:
        return (
            f"seconds/token = {self.overhead_seconds_per_token:.3e} + "
            f"{self.seconds_per_flop:.3e} * flops/token "
            f"(asymptotic MFU {self.asymptotic_mfu:.1%})"
        )


def prior_throughput_model(
    *,
    peak_flops: float = B200_PEAK_BF16_FLOPS,
    asymptotic_mfu: float = 0.4,
    calibration_flops_per_token: float = 2.29e8,
    calibration_tokens_per_second: float = 8.2e5,
) -> ThroughputModel:
    """Prior used to size the first probe runs before any measurements exist.

    It assumes large models reach ``asymptotic_mfu`` and sets the per-token overhead so the
    model reproduces the handout's example run (9L x 448d, batch 128: 8.4M tokens in ~10 s,
    i.e. ~0.82M tokens/s at 2.29e8 FLOPs/token).
    """
    seconds_per_flop = 1.0 / (peak_flops * asymptotic_mfu)
    overhead = max(
        1.0 / calibration_tokens_per_second
        - seconds_per_flop * calibration_flops_per_token,
        0.0,
    )
    return ThroughputModel(
        overhead_seconds_per_token=overhead,
        seconds_per_flop=seconds_per_flop,
        peak_flops=peak_flops,
    )


def fit_throughput_model(
    observations: list[ThroughputObservation],
    *,
    peak_flops: float = B200_PEAK_BF16_FLOPS,
    max_asymptotic_mfu: float = 0.55,
) -> ThroughputModel:
    """Fit the latency-plus-compute model in log space.

    ``max_asymptotic_mfu`` bounds the utilisation the model may extrapolate to for very large
    models: single-GPU bf16 training of ~1B-parameter dense Transformers realistically tops
    out around 40-55% of peak, and a fit that only sees small/medium models can otherwise
    drift to unphysical values and under-estimate the runtime of the final large run.
    """
    if len({o.flops_per_token for o in observations}) < 2:
        raise ValueError(
            "need at least two distinct model shapes to fit a throughput model"
        )
    fpt = np.array([o.flops_per_token for o in observations])
    log_tps = np.log([o.tokens_per_second for o in observations])
    prior = prior_throughput_model(peak_flops=peak_flops)
    min_log_seconds_per_flop = np.log(1.0 / (peak_flops * max_asymptotic_mfu))

    def residuals(log_params):
        overhead, seconds_per_flop = np.exp(log_params)
        return -np.log(overhead + seconds_per_flop * fpt) - log_tps

    result = least_squares(
        residuals,
        x0=np.log(
            [
                max(prior.overhead_seconds_per_token, 1e-9),
                max(prior.seconds_per_flop, np.exp(min_log_seconds_per_flop) * 1.001),
            ]
        ),
        bounds=([-np.inf, min_log_seconds_per_flop], [np.inf, np.inf]),
        method="trf",
    )
    overhead, seconds_per_flop = np.exp(result.x)
    model = ThroughputModel(
        overhead_seconds_per_token=float(overhead),
        seconds_per_flop=float(seconds_per_flop),
        peak_flops=peak_flops,
        n_observations=len(observations),
    )
    pred = np.log(model.tokens_per_second_for_flops(fpt))
    return ThroughputModel(
        overhead_seconds_per_token=model.overhead_seconds_per_token,
        seconds_per_flop=model.seconds_per_flop,
        peak_flops=peak_flops,
        metrics=fit_metrics(log_tps, pred),
        n_observations=len(observations),
    )
