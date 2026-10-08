"""Experiment planning for the 12 B200-hour scaling-law budget.

The study is staged, because each stage needs information from the previous one:

1. ``probe``     -- a short run per model shape to measure tokens/sec (the throughput model
                    converts wall-clock budgets into token counts for every later run).
2. ``lr_sweep``  -- peak-LR sweeps at two small scales to fit how the optimal LR scales with
                    ``N`` (used to set the LR of every larger run, including the final one).
3. ``iso_time``  -- the IsoFLOPs analogue for a wall-clock budget: for several time budgets
                    ``T_i`` train 3-5 model sizes, each for as many tokens as fit in ``T_i``.
                    The per-budget minima give ``N_opt(T)``, ``D_opt(T)``, ``L_opt(T)``, and
                    all runs together feed the parametric ``L(N, D)`` fit.

Every planned run reserves ``max_runtime = headroom * target + constant`` seconds, so the
API refunds the unused part when the run finishes early but a modest throughput
mis-estimate does not turn into a timeout.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from cs336_scaling.scaling_laws.model_shapes import (
    count_parameters,
    ladder_shape,
)
from cs336_scaling.scaling_laws.state import ExperimentRecord, PipelineState
from cs336_scaling.scaling_laws.throughput import ThroughputModel
from cs336_scaling.training.model.config import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig, WarmupCosineDecay
from cs336_scaling.training.training_config import TrainingConfig

LRRule = Callable[[float], float]
"""Maps non-embedding parameter count ``N`` to a peak learning rate."""

# Ladder of model sizes (k layers, 64k hidden): ~1.3-2x spacing in N from 3.4M to 5.3B.
DEFAULT_LADDER_KS: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48)


@dataclass(frozen=True)
class PlannerConfig:
    total_budget_seconds: float = 12 * 3600.0
    budget_fraction: float = (
        0.92  # never plan reservations beyond this share of the budget
    )
    ladder_ks: tuple[int, ...] = DEFAULT_LADDER_KS

    probe_ks: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 20)
    probe_target_seconds: float = 45.0
    probe_runtime_multiplier: float = (
        4.0  # probes are sized from a prior, so over-reserve
    )
    probe_n_evals: int = 4

    lr_sweep_ks: tuple[int, ...] = (6, 10)
    lr_sweep_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    lr_sweep_target_seconds: float = 240.0

    iso_time_budgets: tuple[float, ...] = (300.0, 600.0, 1200.0, 2400.0, 4800.0)
    # The most expensive tier starts with two shapes; bracketing adds the third adaptively.
    iso_time_shapes_per_budget: tuple[int, ...] = (4, 4, 3, 3, 2)
    max_bracket_rounds: int = (
        3  # extra shapes added per tier until the minimum is bracketed
    )
    tokens_per_param_prior: float = 20.0  # Chinchilla's D/N ~ 20 to centre the profiles

    runtime_headroom: float = 1.25
    runtime_headroom_constant_seconds: float = 30.0
    expected_runtime_margin: float = (
        1.1  # budget feasibility assumes runs take 10% longer
    )
    n_evals: int = 8
    val_batch_size: int = 32
    model_seed: int = 0

    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    final_lr_frac: float = 0.1

    extra: dict[str, float] = field(default_factory=dict)


def batch_size_rule(N: float) -> int:
    """Sequences per optimizer step: ``64 * sqrt(N / 1e7)`` rounded to a power of two.

    Larger models tolerate (and need, for throughput) larger batches; critical batch size
    grows as the loss falls (McCandlish et al., 2018). Clipped to ``[32, 1024]`` sequences,
    i.e. 16K-512K tokens per step.
    """
    raw = 64 * math.sqrt(N / 1e7)
    return int(min(1024, max(32, 2 ** round(math.log2(raw)))))


def prior_lr_rule(N: float) -> float:
    """Peak LR before any sweep data: ``2e-3 * (N / 1e7) ** -0.2`` (AdamW with QK-norm)."""
    return 2e-3 * (N / 1e7) ** -0.2


def make_training_config(
    arch: BasicTransformerConfig,
    *,
    batch_size: int,
    peak_lr: float,
    target_tokens: float,
    max_runtime_seconds: float,
    n_evals: int,
    planner: PlannerConfig = PlannerConfig(),
) -> TrainingConfig:
    """Build a valid API config, rounding ``target_tokens`` to the API's divisibility rules.

    ``total_train_tokens`` must be a multiple of ``seq_len * batch_size * n_evals`` (whole
    optimizer steps, evenly split across evaluation chunks).
    """
    tokens_per_chunk_step = TrainingConfig.seq_len * batch_size * n_evals
    steps_per_eval = max(1, round(target_tokens / tokens_per_chunk_step))
    return TrainingConfig(
        architecture_config=arch,
        optimizer_config=AdamWConfig(
            lr_scheduler=WarmupCosineDecay(
                peak_value=float(f"{peak_lr:.3g}"),
                final_lr_frac=planner.final_lr_frac,
                warmup_frac=planner.warmup_frac,
                init_value=0.0,
            ),
            weight_decay=planner.weight_decay,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            eps_root=1e-8,
            grad_clip_norm=1.0,
        ),
        train_batch_size=batch_size,
        val_batch_size=planner.val_batch_size,
        n_evals=n_evals,
        total_train_tokens=steps_per_eval * tokens_per_chunk_step,
        max_runtime_seconds=round(max_runtime_seconds, 1),
        model_seed=planner.model_seed,
    )


def _reserve_seconds(target_seconds: float, planner: PlannerConfig) -> float:
    return (
        planner.runtime_headroom * target_seconds
        + planner.runtime_headroom_constant_seconds
    )


def _timed_run(
    *,
    name: str,
    stage: str,
    k: int,
    target_seconds: float,
    throughput: ThroughputModel,
    lr_rule: LRRule,
    planner: PlannerConfig,
    n_evals: int,
    max_runtime_seconds: float,
    peak_lr_multiplier: float = 1.0,
    time_budget_seconds: float | None = None,
    planning_round: int,
) -> ExperimentRecord:
    arch = ladder_shape(k)
    N = count_parameters(arch).non_embedding
    batch_size = batch_size_rule(N)
    # Size the run so that training (plus the per-chunk validation passes) fits the target.
    eval_overhead = n_evals * TrainingConfig.n_val_tokens / 3.0
    target_tokens = max(
        throughput.tokens_for_seconds(arch, target_seconds) - eval_overhead,
        TrainingConfig.seq_len * batch_size * n_evals,
    )
    config = make_training_config(
        arch,
        batch_size=batch_size,
        peak_lr=lr_rule(N) * peak_lr_multiplier,
        target_tokens=target_tokens,
        max_runtime_seconds=max_runtime_seconds,
        n_evals=n_evals,
        planner=planner,
    )
    return ExperimentRecord(
        name=name,
        stage=stage,  # type: ignore[arg-type]
        training_config=config,
        ladder_k=k,
        time_budget_seconds=time_budget_seconds,
        planned_at_round=planning_round,
    )


def plan_probes(
    planner: PlannerConfig,
    throughput: ThroughputModel,
    *,
    lr_rule: LRRule = prior_lr_rule,
    planning_round: int = 0,
) -> list[ExperimentRecord]:
    return [
        _timed_run(
            name=f"probe/k{k:02d}",
            stage="probe",
            k=k,
            target_seconds=planner.probe_target_seconds,
            throughput=throughput,
            lr_rule=lr_rule,
            planner=planner,
            n_evals=planner.probe_n_evals,
            max_runtime_seconds=planner.probe_target_seconds
            * planner.probe_runtime_multiplier,
            planning_round=planning_round,
        )
        for k in planner.probe_ks
    ]


def plan_lr_sweep(
    planner: PlannerConfig,
    throughput: ThroughputModel,
    *,
    lr_rule: LRRule = prior_lr_rule,
    planning_round: int = 0,
) -> list[ExperimentRecord]:
    records = []
    for k in planner.lr_sweep_ks:
        for multiplier in planner.lr_sweep_multipliers:
            records.append(
                _timed_run(
                    name=f"lr/k{k:02d}/x{multiplier:g}",
                    stage="lr_sweep",
                    k=k,
                    target_seconds=planner.lr_sweep_target_seconds,
                    throughput=throughput,
                    lr_rule=lr_rule,
                    planner=planner,
                    n_evals=planner.n_evals,
                    max_runtime_seconds=_reserve_seconds(
                        planner.lr_sweep_target_seconds, planner
                    ),
                    peak_lr_multiplier=multiplier,
                    time_budget_seconds=planner.lr_sweep_target_seconds,
                    planning_round=planning_round,
                )
            )
    return records


def prior_optimal_k(
    planner: PlannerConfig,
    throughput: ThroughputModel,
    time_budget_seconds: float,
    *,
    tokens_per_param: float | None = None,
) -> int:
    """Ladder index whose tokens-in-budget / N ratio is closest to ``tokens_per_param``.

    Defaults to the Chinchilla prior (``D/N ~ 20``); once iso-time data exist the caller
    passes the empirically best ``D/N`` instead, which under a wall-clock budget is smaller
    (bigger models run at higher utilisation, so the optimum shifts towards fewer tokens per
    parameter).
    """
    target = tokens_per_param or planner.tokens_per_param_prior

    def log_ratio_error(k: int) -> float:
        arch = ladder_shape(k)
        N = count_parameters(arch).non_embedding
        tokens = throughput.tokens_for_seconds(arch, time_budget_seconds)
        return abs(math.log(tokens / N) - math.log(target))

    return min(planner.ladder_ks, key=log_ratio_error)


def shapes_around(ladder_ks: Sequence[int], centre_k: int, n: int) -> list[int]:
    """``n`` consecutive ladder entries centred on ``centre_k`` (clipped to the ladder)."""
    if n <= 0:
        return []
    idx = min(range(len(ladder_ks)), key=lambda i: abs(ladder_ks[i] - centre_k))
    start = max(0, min(idx - (n - 1) // 2, len(ladder_ks) - n))
    return list(ladder_ks[start : start + n])


def plan_iso_time(
    planner: PlannerConfig,
    throughput: ThroughputModel,
    *,
    lr_rule: LRRule,
    optimal_k_for_budget: Callable[[float], int] | None = None,
    budgets: Sequence[float] | None = None,
    planning_round: int = 0,
) -> list[ExperimentRecord]:
    """Iso-time profiles: for each ``T_i`` train several shapes for ``~T_i`` seconds each.

    ``budgets`` selects a subset of ``planner.iso_time_budgets`` so that tiers can be planned
    one at a time, each centred (via ``optimal_k_for_budget``) on what the previous tiers
    predict the optimum to be.
    """
    if len(planner.iso_time_budgets) != len(planner.iso_time_shapes_per_budget):
        raise ValueError("iso_time_budgets and iso_time_shapes_per_budget must align")
    records = []
    for budget, n_shapes in zip(
        planner.iso_time_budgets, planner.iso_time_shapes_per_budget, strict=True
    ):
        if budgets is not None and budget not in budgets:
            continue
        centre = (
            optimal_k_for_budget(budget)
            if optimal_k_for_budget is not None
            else prior_optimal_k(planner, throughput, budget)
        )
        for k in shapes_around(planner.ladder_ks, centre, n_shapes):
            records.append(
                _timed_run(
                    name=f"iso/T{int(budget):05d}/k{k:02d}",
                    stage="iso_time",
                    k=k,
                    target_seconds=budget,
                    throughput=throughput,
                    lr_rule=lr_rule,
                    planner=planner,
                    n_evals=planner.n_evals,
                    max_runtime_seconds=_reserve_seconds(budget, planner),
                    time_budget_seconds=budget,
                    planning_round=planning_round,
                )
            )
    return records


def iso_time_profile_ks(
    state: PipelineState,
) -> dict[float, dict[int, ExperimentRecord]]:
    """``{time_budget: {ladder_k: record}}`` for all tracked iso-time runs."""
    profiles: dict[float, dict[int, ExperimentRecord]] = {}
    for r in state.stage("iso_time"):
        if r.time_budget_seconds is not None and r.ladder_k is not None:
            profiles.setdefault(r.time_budget_seconds, {})[r.ladder_k] = r
    return profiles


def plan_bracket_extensions(
    state: PipelineState,
    planner: PlannerConfig,
    throughput: ThroughputModel,
    *,
    lr_rule: LRRule,
    planning_round: int = 0,
) -> list[ExperimentRecord]:
    """Extend profiles whose best run sits on the edge of the sampled shapes.

    A profile only pins down ``N_opt(T)`` if its minimum is bracketed on both sides. For each
    fully finished profile whose lowest loss is at the smallest or largest sampled shape,
    add the next ladder shape beyond that edge (one per profile per round).
    """
    records = []
    for budget, by_k in iso_time_profile_ks(state).items():
        if any(not r.is_finished for r in by_k.values()):
            continue
        completed = {k: r for k, r in by_k.items() if r.status == "completed"}
        if len(completed) < 2:
            continue
        ks = sorted(completed)
        best_k = min(ks, key=lambda k: completed[k].final_loss)
        ladder = list(planner.ladder_ks)
        if best_k == ks[-1] and best_k != ladder[-1]:
            next_k = ladder[ladder.index(best_k) + 1]
        elif best_k == ks[0] and best_k != ladder[0]:
            next_k = ladder[ladder.index(best_k) - 1]
        else:
            continue
        if next_k in by_k:
            continue
        records.append(
            _timed_run(
                name=f"iso/T{int(budget):05d}/k{next_k:02d}",
                stage="iso_time",
                k=next_k,
                target_seconds=budget,
                throughput=throughput,
                lr_rule=lr_rule,
                planner=planner,
                n_evals=planner.n_evals,
                max_runtime_seconds=_reserve_seconds(budget, planner),
                time_budget_seconds=budget,
                planning_round=planning_round,
            )
        )
    return records


def empirical_tokens_per_param(state: PipelineState) -> float | None:
    """``D/N`` of the best run in the largest finished iso-time profile, if any."""
    best: ExperimentRecord | None = None
    for budget, by_k in sorted(iso_time_profile_ks(state).items()):
        completed = [r for r in by_k.values() if r.status == "completed"]
        if completed:
            best = min(completed, key=lambda r: r.final_loss)
    if best is None:
        return None
    return best.tokens / best.non_embedding_params


@dataclass(frozen=True)
class BudgetCheck:
    """Feasibility of adding runs to the study.

    The API charges finished runs their *actual* runtime and only reserves
    ``max_runtime_seconds`` while a run is queued/running, so the binding constraint for a
    sequentially-submitted study is the expected total runtime (with a margin), not the sum
    of reservations. The runner enforces the instantaneous reservation limit at submit time.
    """

    charged_so_far_seconds: float
    pending_reserved_seconds: float
    new_expected_seconds: float
    new_reserved_seconds: float
    margin: float
    limit_seconds: float

    @property
    def projected_seconds(self) -> float:
        return (
            self.charged_so_far_seconds
            + self.pending_reserved_seconds
            + self.margin * self.new_expected_seconds
        )

    @property
    def fits(self) -> bool:
        return self.projected_seconds <= self.limit_seconds

    def describe(self) -> str:
        return (
            f"charged {self.charged_so_far_seconds / 3600:.2f}h + pending reservations "
            f"{self.pending_reserved_seconds / 3600:.2f}h + new runs "
            f"{self.new_expected_seconds / 3600:.2f}h expected x{self.margin} margin "
            f"= {self.projected_seconds / 3600:.2f}h projected vs limit "
            f"{self.limit_seconds / 3600:.2f}h (new reservations if all submitted at once: "
            f"{self.new_reserved_seconds / 3600:.2f}h)"
        )


def check_budget(
    state: PipelineState,
    new_records: Sequence[ExperimentRecord],
    planner: PlannerConfig,
    throughput: ThroughputModel,
) -> BudgetCheck:
    expected = 0.0
    for r in new_records:
        arch = r.training_config.architecture_config
        expected += min(
            throughput.seconds_for_tokens(arch, r.tokens),
            r.training_config.max_runtime_seconds,
        )
    charged = sum(
        min(max(1.0, r.used_runtime_seconds), r.training_config.max_runtime_seconds)
        for r in state.experiments
        if r.is_finished and r.used_runtime_seconds is not None
    )
    pending = sum(r.training_config.max_runtime_seconds for r in state.pending())
    return BudgetCheck(
        charged_so_far_seconds=charged,
        pending_reserved_seconds=pending,
        new_expected_seconds=expected,
        new_reserved_seconds=sum(
            r.training_config.max_runtime_seconds for r in new_records
        ),
        margin=planner.expected_runtime_margin,
        limit_seconds=planner.budget_fraction * planner.total_budget_seconds,
    )


def add_to_state(
    state: PipelineState, records: Sequence[ExperimentRecord]
) -> list[ExperimentRecord]:
    """Append records whose config is not already tracked; returns the ones added."""
    added = []
    for record in records:
        if state.add(record):
            added.append(record)
    return added
