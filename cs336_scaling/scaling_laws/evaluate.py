"""Score a final submission against the simulator's hidden truth (simulated backend only).

Answers the questions a real leaderboard would: does the chosen 48-hour run finish in time,
what loss does it actually reach, how far is that from the predicted loss, and how far is it
from the best configuration the simulator would have allowed?
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cs336_scaling.scaling_laws.analysis import FINAL_CANDIDATE_KS
from cs336_scaling.scaling_laws.model_shapes import (
    count_parameters,
    describe_shape,
    flops_per_token,
    ladder_shape,
)
from cs336_scaling.scaling_laws.plan import batch_size_rule, make_training_config
from cs336_scaling.scaling_laws.simulate import SimulatorTruth
from cs336_scaling.training.training_config import TrainingConfig


@dataclass(frozen=True)
class OracleCandidate:
    k: int
    N: int
    tokens: int
    loss: float
    seconds: float


@dataclass(frozen=True)
class FinalEvaluation:
    predicted_loss: float
    actual_loss: float | None
    completed: bool
    actual_seconds: float
    budget_seconds: float
    oracle_best: OracleCandidate
    chosen_shape: str
    chosen_k: int

    @property
    def regret(self) -> float | None:
        return (
            None
            if self.actual_loss is None
            else self.actual_loss - self.oracle_best.loss
        )

    def describe(self) -> str:
        lines = [
            f"chosen: k={self.chosen_k} {self.chosen_shape}",
            f"  predicted loss {self.predicted_loss:.4f}; "
            + (
                f"actual loss {self.actual_loss:.4f} (error {self.actual_loss - self.predicted_loss:+.4f})"
                if self.actual_loss is not None
                else "TIMED OUT"
            ),
            f"  runtime {self.actual_seconds / 3600:.2f}h of {self.budget_seconds / 3600:.2f}h budget "
            f"({'completed' if self.completed else 'FAILED'})",
            f"oracle (best k with optimal LR, {self.budget_seconds / 3600:.0f}h fully used): "
            f"k={self.oracle_best.k} N={self.oracle_best.N / 1e6:.0f}M D={self.oracle_best.tokens / 1e9:.1f}B "
            f"loss {self.oracle_best.loss:.4f}",
        ]
        if self.regret is not None:
            lines.append(f"  regret vs oracle: {self.regret:+.4f} nats")
        return "\n".join(lines)


def oracle_best_candidate(
    truth: SimulatorTruth, *, budget_seconds: float, utilisation: float = 0.98
) -> OracleCandidate:
    """Brute force the best ladder shape under the truth, using the truth's own throughput."""
    best: OracleCandidate | None = None
    for k in FINAL_CANDIDATE_KS:
        arch = ladder_shape(k)
        N = count_parameters(arch).non_embedding
        batch_tokens = TrainingConfig.seq_len * batch_size_rule(N)
        tps = truth.tokens_per_second(flops_per_token(arch), batch_tokens)
        tokens = int(tps * budget_seconds * utilisation)
        loss = truth.loss(N, tokens, batch_tokens=batch_tokens, peak_lr=truth.lr_opt(N))
        if best is None or loss < best.loss:
            best = OracleCandidate(
                k=k, N=N, tokens=tokens, loss=loss, seconds=tokens / tps
            )
    assert best is not None
    return best


def evaluate_final(
    truth: SimulatorTruth,
    training_config: TrainingConfig,
    predicted_loss: float,
) -> FinalEvaluation:
    outcome = truth.simulate(training_config)
    arch = training_config.architecture_config
    k = arch.num_hidden_layers
    return FinalEvaluation(
        predicted_loss=predicted_loss,
        actual_loss=outcome.val_losses[-1] if outcome.completed else None,
        completed=outcome.completed,
        actual_seconds=outcome.used_runtime_seconds,
        budget_seconds=training_config.max_runtime_seconds,
        oracle_best=oracle_best_candidate(
            truth, budget_seconds=training_config.max_runtime_seconds
        ),
        chosen_shape=describe_shape(arch),
        chosen_k=k,
    )


def loss_if_rebuilt_with(
    truth: SimulatorTruth, k: int, *, budget_seconds: float, peak_lr: float
) -> float:
    """Helper for sensitivity checks: truth loss for ladder shape ``k`` at a given LR."""
    arch = ladder_shape(k)
    N = count_parameters(arch).non_embedding
    batch = batch_size_rule(N)
    tps = truth.tokens_per_second(flops_per_token(arch), TrainingConfig.seq_len * batch)
    config = make_training_config(
        arch,
        batch_size=batch,
        peak_lr=peak_lr,
        target_tokens=tps * budget_seconds * 0.98,
        max_runtime_seconds=budget_seconds,
        n_evals=8,
    )
    return truth.loss(
        N,
        config.total_train_tokens,
        batch_tokens=config.tokens_per_optimizer_step,
        peak_lr=peak_lr,
    )


def summarise_for_json(evaluation: FinalEvaluation) -> dict[str, object]:
    return {
        "chosen_k": evaluation.chosen_k,
        "chosen_shape": evaluation.chosen_shape,
        "predicted_loss": evaluation.predicted_loss,
        "actual_loss": evaluation.actual_loss,
        "completed": evaluation.completed,
        "actual_seconds": evaluation.actual_seconds,
        "budget_seconds": evaluation.budget_seconds,
        "oracle_best": evaluation.oracle_best.__dict__,
        "regret": evaluation.regret,
        "prediction_error": (
            None
            if evaluation.actual_loss is None
            else evaluation.actual_loss - evaluation.predicted_loss
        ),
        "loss_is_finite": evaluation.actual_loss is not None
        and math.isfinite(evaluation.actual_loss),
    }
