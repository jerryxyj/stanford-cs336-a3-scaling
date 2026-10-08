"""A synthetic stand-in for the training API, used to validate the pipeline offline.

The simulator hides a "ground truth" that the scaling-law pipeline must recover:

* loss surface ``L(N, D_eff) = E + A / N**alpha + B / D_eff**beta`` (Hoffmann et al. form),
* a learning-rate sensitivity: the optimal peak LR decays as a power of ``N`` and the loss
  grows quadratically in ``log(lr / lr_opt)``,
* a batch-size inefficiency ``D_eff = D / (1 + batch_tokens / B_crit)`` (McCandlish et al.,
  2018): large batches waste tokens,
* a throughput curve whose model-FLOP utilisation rises with model size and batch size
  (small models are launch/memory bound), calibrated to the ~0.8M tokens/s the handout's
  example run (9L x 448d, batch 128) achieved,
* small multiplicative noise, and timeouts when the run would exceed ``max_runtime_seconds``.

Nothing in here is used by the real pipeline except through the
:class:`cs336_scaling.scaling_laws.backends.SimulatedBackend` wrapper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cs336_scaling.scaling_laws.model_shapes import count_parameters, flops_per_token
from cs336_scaling.scaling_laws.throughput import B200_PEAK_BF16_FLOPS
from cs336_scaling.training.training_config import TrainingConfig


@dataclass(frozen=True)
class SimulatedOutcome:
    completed: bool
    used_runtime_seconds: float
    val_losses: list[float]
    tokens_per_second: float


@dataclass(frozen=True)
class SimulatorTruth:
    E: float = 1.9
    A: float = 250.0
    alpha: float = 0.33
    B: float = 300.0
    beta: float = 0.29
    # optimal peak learning rate: lr_opt(N) = lr_ref * (N / N_ref) ** lr_exponent
    lr_ref: float = 3e-3
    N_ref: float = 1e7
    lr_exponent: float = -0.30
    lr_penalty: float = 0.2  # loss += lr_penalty * log10(lr / lr_opt) ** 2
    lr_divergence_factor: float = 8.0  # lr > factor * lr_opt diverges
    # Critical batch size B_crit(L) = B_star / L**(1/alpha_B) tokens (Kaplan et al. 2020,
    # eq. 2.3 - 2.4): ~270K tokens at L=4, ~2.5M tokens at L=2.5.
    critical_batch_b_star: float = 2.0e8
    critical_batch_alpha: float = 0.21
    # throughput: MFU = mfu_max * fpt / (fpt + fpt_half) * batch / (batch + batch_half)
    mfu_max: float = 0.5
    fpt_half: float = 1.0e9
    batch_half_tokens: float = 2.0**14
    peak_flops: float = B200_PEAK_BF16_FLOPS
    eval_tokens: int = TrainingConfig.n_val_tokens
    noise_std: float = 0.003
    seed: int = 1234

    def lr_opt(self, N: float) -> float:
        return self.lr_ref * (N / self.N_ref) ** self.lr_exponent

    def critical_batch_tokens(self, loss: float) -> float:
        return self.critical_batch_b_star / loss ** (1.0 / self.critical_batch_alpha)

    def effective_tokens(
        self, tokens: float, batch_tokens: float, loss: float
    ) -> float:
        """Tokens "wasted" by a batch above the critical size (McCandlish et al., 2018)."""
        return tokens / (1.0 + batch_tokens / self.critical_batch_tokens(loss))

    def loss(
        self, N: float, tokens: float, *, batch_tokens: float, peak_lr: float
    ) -> float:
        # The critical batch size depends on the loss, so iterate the fixed point a few times.
        base = self.ideal_compute_optimal_loss(N, max(tokens, 1.0))
        for _ in range(4):
            D_eff = max(self.effective_tokens(tokens, batch_tokens, base), 1.0)
            base = self.E + self.A / N**self.alpha + self.B / D_eff**self.beta
        lr_ratio = peak_lr / self.lr_opt(N)
        penalty = self.lr_penalty * math.log10(lr_ratio) ** 2
        if lr_ratio > self.lr_divergence_factor:
            penalty += 0.5 * (lr_ratio / self.lr_divergence_factor - 1.0) + 0.3
        return base + penalty

    def mfu(self, fpt: float, batch_tokens: float) -> float:
        return (
            self.mfu_max
            * fpt
            / (fpt + self.fpt_half)
            * batch_tokens
            / (batch_tokens + self.batch_half_tokens)
        )

    def tokens_per_second(self, fpt: float, batch_tokens: float) -> float:
        return self.peak_flops * self.mfu(fpt, batch_tokens) / fpt

    def ideal_compute_optimal_loss(self, N: float, tokens: float) -> float:
        """Loss with optimal LR and negligible batch, for judging pipeline predictions."""
        return self.E + self.A / N**self.alpha + self.B / tokens**self.beta

    def simulate(self, config: TrainingConfig) -> SimulatedOutcome:
        arch = config.architecture_config
        N = count_parameters(arch).non_embedding
        fpt = flops_per_token(arch)
        batch_tokens = config.tokens_per_optimizer_step
        tps = self.tokens_per_second(fpt, batch_tokens)
        rng = np.random.default_rng(
            [self.seed, int.from_bytes(bytes.fromhex(config.unique_id[:8]), "big")]
        )
        noise = float(np.exp(rng.normal(0.0, self.noise_std)))

        # Each chunk trains eval_every_tokens then runs a forward-only validation pass
        # (~1/3 the cost of a training token per validation token).
        chunk_seconds = (config.eval_every_tokens + self.eval_tokens / 3.0) / tps
        val_losses: list[float] = []
        elapsed = 0.0
        for i in range(config.n_evals):
            elapsed += chunk_seconds
            if elapsed > config.max_runtime_seconds:
                return SimulatedOutcome(
                    completed=False,
                    used_runtime_seconds=elapsed,
                    val_losses=val_losses,
                    tokens_per_second=tps,
                )
            tokens_so_far = (i + 1) * config.eval_every_tokens
            progress = tokens_so_far / config.total_train_tokens
            # Mid-training losses sit above the final loss because the cosine schedule has
            # not annealed yet; the gap closes as progress -> 1.
            anneal_gap = 0.12 * (1.0 - progress)
            loss = (
                self.loss(
                    N,
                    tokens_so_far,
                    batch_tokens=batch_tokens,
                    peak_lr=config.optimizer_config.lr_scheduler.peak_value,
                )
                + anneal_gap
            )
            val_losses.append(loss * noise)
        return SimulatedOutcome(
            completed=True,
            used_runtime_seconds=elapsed,
            val_losses=val_losses,
            tokens_per_second=tps,
        )
