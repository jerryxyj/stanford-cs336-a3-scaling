"""Fit scaling laws to the finished runs and pick the configuration for the 48 B200-hour run.

Two complementary estimates of the compute-optimal model are produced:

* **Iso-time profiles** (Hoffmann et al. approach 2 with wall-clock instead of FLOPs as the
  budget): the best run of each time budget ``T_i`` gives ``N_opt(T_i)``, ``D_opt(T_i)`` and
  ``L_opt(T_i)``; power laws in ``T`` are extrapolated to 48 hours.
* **Parametric loss surface** (approach 3): ``L(N, D) = E + A/N^alpha + B/D^beta`` is fit to
  all iso-time runs. Combined with the throughput model (``D = tokens/sec(N) * T``) the final
  model size is the minimiser of ``L(N, D(N, T_final))`` over the shape ladder.

The learning rate of every run (including the final one) comes from a power law
``lr_opt(N)`` fit to the LR sweeps, and the batch size from :func:`batch_size_rule`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cs336_scaling.scaling_laws.fit import (
    ChinchillaParametricFit,
    PowerLaw,
    SaturatingPowerLaw,
    fit_chinchilla_parametric,
    fit_metrics,
    fit_power_law,
    fit_saturating_power_law,
)
from cs336_scaling.scaling_laws.model_shapes import (
    count_parameters,
    describe_shape,
    flops_per_token,
    kaplan_non_embedding_estimate,
    ladder_shape,
)
from cs336_scaling.scaling_laws.plan import (
    LRRule,
    PlannerConfig,
    batch_size_rule,
    make_training_config,
    prior_lr_rule,
)
from cs336_scaling.scaling_laws.state import ExperimentRecord, PipelineState
from cs336_scaling.scaling_laws.throughput import (
    ThroughputModel,
    fit_throughput_model,
    prior_throughput_model,
)
from cs336_scaling.training.training_config import TrainingConfig

FINAL_BUDGET_SECONDS = 48 * 3600.0
FINAL_CANDIDATE_KS = tuple(range(4, 65))


# --------------------------------------------------------------------------------------
# Learning-rate sweep
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LRSweepScale:
    k: int
    N: float
    lrs: list[float]
    losses: list[float]
    lr_opt: float
    loss_at_opt: float


@dataclass(frozen=True)
class LRSweepFit:
    scales: list[LRSweepScale]
    law: PowerLaw

    def rule(self, N: float) -> float:
        return float(self.law(N))

    def to_dict(self) -> dict[str, object]:
        return {
            "law": self.law.to_dict(),
            "scales": [
                {
                    "k": s.k,
                    "N": s.N,
                    "lrs": s.lrs,
                    "losses": s.losses,
                    "lr_opt": s.lr_opt,
                    "loss_at_opt": s.loss_at_opt,
                }
                for s in self.scales
            ],
        }


def _parabola_argmin(log_lrs: np.ndarray, losses: np.ndarray) -> tuple[float, float]:
    """Vertex of the parabola through ``(log lr, loss)``, clipped to the sampled range."""
    if len(log_lrs) >= 3:
        c2, c1, c0 = np.polyfit(log_lrs, losses, deg=2)
        if c2 > 0:
            x = float(np.clip(-c1 / (2 * c2), log_lrs.min(), log_lrs.max()))
            return x, float(c0 + c1 * x + c2 * x**2)
    i = int(np.argmin(losses))
    return float(log_lrs[i]), float(losses[i])


def fit_lr_sweep(
    records: Sequence[ExperimentRecord],
    *,
    prior_exponent: float = -0.2,
    exponent_bounds: tuple[float, float] = (-0.5, 0.0),
) -> LRSweepFit | None:
    """Fit ``lr_opt(N) = c * N**b`` from the per-scale optima of the LR sweeps.

    The exponent is clamped to ``exponent_bounds``: the optimal Adam LR of Transformers does
    not *grow* with model size, and with only a few sweep scales a noisy, nearly flat sweep
    could otherwise produce a positive exponent that extrapolates to a divergent LR for the
    (10x larger) final model. Under-estimating the LR costs a little loss; over-estimating
    it can cost the whole run, so the clamp errs low.
    """
    by_k: dict[int, list[ExperimentRecord]] = {}
    for r in records:
        if r.stage == "lr_sweep" and r.status == "completed" and r.ladder_k is not None:
            by_k.setdefault(r.ladder_k, []).append(r)
    scales = []
    for k, runs in sorted(by_k.items()):
        if len(runs) < 2:
            continue
        runs = sorted(runs, key=lambda r: r.peak_lr)
        log_lrs = np.log10([r.peak_lr for r in runs])
        losses = np.array([r.final_loss for r in runs])
        x_opt, loss_opt = _parabola_argmin(log_lrs, losses)
        scales.append(
            LRSweepScale(
                k=k,
                N=runs[0].non_embedding_params,
                lrs=[r.peak_lr for r in runs],
                losses=losses.tolist(),
                lr_opt=10**x_opt,
                loss_at_opt=loss_opt,
            )
        )
    if not scales:
        return None
    Ns = np.array([s.N for s in scales], dtype=float)
    lr_opts = np.array([s.lr_opt for s in scales])
    if len(scales) == 1:
        exponent = prior_exponent
    else:
        exponent = float(np.clip(fit_power_law(Ns, lr_opts).exponent, *exponent_bounds))
    # With the exponent fixed, the coefficient is the geometric-mean residual.
    coefficient = float(np.exp(np.mean(np.log(lr_opts) - exponent * np.log(Ns))))
    law = PowerLaw(
        coefficient=coefficient,
        exponent=exponent,
        metrics=fit_metrics(
            np.log(lr_opts), np.log(coefficient) + exponent * np.log(Ns)
        ),
    )
    return LRSweepFit(scales=scales, law=law)


# --------------------------------------------------------------------------------------
# Iso-time profiles
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class IsoTimeProfile:
    time_budget_seconds: float
    runs: list[ExperimentRecord]
    best: ExperimentRecord
    # Parabola-in-log(N) interpolation of the minimum (Hoffmann et al. approach 2); falls
    # back to the best grid point when a profile has fewer than three runs or no curvature.
    n_opt_interp: float
    loss_interp: float

    @property
    def mean_runtime_seconds(self) -> float:
        return float(np.mean([r.used_runtime_seconds for r in self.runs]))

    def to_dict(self) -> dict[str, object]:
        return {
            "time_budget_seconds": self.time_budget_seconds,
            "mean_runtime_seconds": self.mean_runtime_seconds,
            "runs": [
                {
                    "name": r.name,
                    "k": r.ladder_k,
                    "N": r.non_embedding_params,
                    "D": r.tokens,
                    "loss": r.final_loss,
                    "seconds": r.used_runtime_seconds,
                }
                for r in self.runs
            ],
            "best": {
                "name": self.best.name,
                "k": self.best.ladder_k,
                "N_opt": self.best.non_embedding_params,
                "D_opt": self.best.tokens,
                "loss": self.best.final_loss,
            },
            "interpolated": {"N_opt": self.n_opt_interp, "loss": self.loss_interp},
        }


@dataclass(frozen=True)
class IsoTimeFit:
    profiles: list[IsoTimeProfile]
    n_opt_law: PowerLaw | None
    d_opt_law: PowerLaw | None
    l_opt_law: SaturatingPowerLaw | None

    def to_dict(self) -> dict[str, object]:
        return {
            "profiles": [p.to_dict() for p in self.profiles],
            "n_opt_law": self.n_opt_law.to_dict() if self.n_opt_law else None,
            "d_opt_law": self.d_opt_law.to_dict() if self.d_opt_law else None,
            "l_opt_law": self.l_opt_law.to_dict() if self.l_opt_law else None,
        }


def iso_time_profiles(records: Sequence[ExperimentRecord]) -> list[IsoTimeProfile]:
    by_budget: dict[float, list[ExperimentRecord]] = {}
    for r in records:
        if r.stage == "iso_time" and r.status == "completed" and r.time_budget_seconds:
            by_budget.setdefault(r.time_budget_seconds, []).append(r)
    profiles = []
    for budget, runs in sorted(by_budget.items()):
        runs = sorted(runs, key=lambda r: r.non_embedding_params)
        best = min(runs, key=lambda r: r.final_loss)
        log_n = np.log10([r.non_embedding_params for r in runs])
        losses = np.array([r.final_loss for r in runs])
        x_opt, loss_opt = _parabola_argmin(log_n, losses)
        profiles.append(
            IsoTimeProfile(
                time_budget_seconds=budget,
                runs=runs,
                best=best,
                n_opt_interp=10**x_opt,
                loss_interp=loss_opt,
            )
        )
    return profiles


def fit_iso_time(
    records: Sequence[ExperimentRecord], *, throughput: ThroughputModel | None = None
) -> IsoTimeFit | None:
    """Power laws for ``N_opt``, ``D_opt`` and ``L_opt`` versus the wall-clock budget.

    Uses the parabola-interpolated minimum of each profile when a throughput model is
    available to convert the interpolated ``N_opt`` into the tokens that fit the budget;
    otherwise the best grid run.
    """
    profiles = iso_time_profiles(records)
    if not profiles:
        return None
    # Use the time the best run actually took as the budget coordinate.
    T = np.array([p.best.used_runtime_seconds for p in profiles])
    if throughput is not None:
        n_opt = np.array([p.n_opt_interp for p in profiles])
        d_opt = np.array(
            [
                throughput.tokens_per_second_for_flops(
                    _flops_per_token_for_n(p.n_opt_interp)
                )
                * p.best.used_runtime_seconds
                for p in profiles
            ]
        )
        l_opt = np.array([p.loss_interp for p in profiles])
    else:
        n_opt = np.array([p.best.non_embedding_params for p in profiles], dtype=float)
        d_opt = np.array([p.best.tokens for p in profiles], dtype=float)
        l_opt = np.array([p.best.final_loss for p in profiles])
    n_law = fit_power_law(T, n_opt) if len(profiles) >= 2 else None
    d_law = fit_power_law(T, d_opt) if len(profiles) >= 2 else None
    l_law = fit_saturating_power_law(T, l_opt) if len(profiles) >= 3 else None
    return IsoTimeFit(
        profiles=profiles, n_opt_law=n_law, d_opt_law=d_law, l_opt_law=l_law
    )


def _flops_per_token_for_n(N: float) -> float:
    """FLOPs/token of the ladder shape with (interpolated, possibly non-integer) size ``N``."""
    Ns = np.array(
        [count_parameters(ladder_shape(k)).non_embedding for k in FINAL_CANDIDATE_KS]
    )
    fpts = np.array([flops_per_token(ladder_shape(k)) for k in FINAL_CANDIDATE_KS])
    return float(np.exp(np.interp(np.log(N), np.log(Ns), np.log(fpts))))


# --------------------------------------------------------------------------------------
# Parametric fit and final prediction
# --------------------------------------------------------------------------------------
def loss_fit_records(
    records: Sequence[ExperimentRecord], *, min_tokens_per_param: float = 1.0
) -> list[ExperimentRecord]:
    """Runs used for the parametric fit: completed iso-time (and extra) runs.

    Runs with fewer than ``min_tokens_per_param`` tokens per parameter are excluded: they
    are far from the compute-optimal frontier we extrapolate along, and the additive
    ``E + A/N^a + B/D^b`` form fits that under-trained regime poorly (it distorts ``E``).
    """
    return [
        r
        for r in records
        if r.status == "completed"
        and r.stage in ("iso_time", "extra")
        and r.final_loss is not None
        and r.tokens / r.non_embedding_params >= min_tokens_per_param
    ]


def fit_parametric(
    records: Sequence[ExperimentRecord], *, fast: bool = False
) -> ChinchillaParametricFit | None:
    runs = loss_fit_records(records)
    if len(runs) < 5:
        return None
    N = [r.non_embedding_params for r in runs]
    D = [r.tokens for r in runs]
    L = [r.final_loss for r in runs]
    if fast:
        return fit_chinchilla_parametric(
            N,
            D,
            L,
            alpha_grid=(0.2, 0.5, 1.0),
            beta_grid=(0.2, 0.5, 1.0),
            e_grid=(-0.5, 0.0, 0.5),
            a_grid=(5.0, 10.0, 15.0),
            b_grid=(5.0, 10.0, 15.0),
        )
    return fit_chinchilla_parametric(N, D, L)


@dataclass(frozen=True)
class FinalCandidate:
    k: int
    N: int
    N_kaplan: int
    total_params: int
    batch_size: int
    peak_lr: float
    tokens: int
    predicted_seconds: float
    predicted_loss: float | None
    training_config: TrainingConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "shape": describe_shape(self.training_config.architecture_config),
            "N_non_embedding": self.N,
            "N_kaplan_12nd2": self.N_kaplan,
            "total_params": self.total_params,
            "batch_size": self.batch_size,
            "peak_lr": self.peak_lr,
            "tokens": self.tokens,
            "tokens_per_param": self.tokens / self.N,
            "predicted_seconds": self.predicted_seconds,
            "predicted_loss": self.predicted_loss,
        }


def build_final_candidate(
    k: int,
    *,
    throughput: ThroughputModel,
    lr_rule: LRRule,
    parametric: ChinchillaParametricFit | None,
    planner: PlannerConfig,
    final_budget_seconds: float,
    safety: float,
    n_evals: int,
) -> FinalCandidate:
    arch = ladder_shape(k)
    counts = count_parameters(arch)
    N = counts.non_embedding
    batch_size = batch_size_rule(N)
    eval_overhead_tokens = n_evals * TrainingConfig.n_val_tokens / 3.0
    target_tokens = max(
        throughput.tokens_for_seconds(arch, safety * final_budget_seconds)
        - eval_overhead_tokens,
        TrainingConfig.seq_len * batch_size * n_evals,
    )
    config = make_training_config(
        arch,
        batch_size=batch_size,
        peak_lr=lr_rule(N),
        target_tokens=target_tokens,
        max_runtime_seconds=final_budget_seconds,
        n_evals=n_evals,
        planner=planner,
    )
    predicted_seconds = throughput.seconds_for_tokens(
        arch, config.total_train_tokens + eval_overhead_tokens
    )
    return FinalCandidate(
        k=k,
        N=N,
        N_kaplan=kaplan_non_embedding_estimate(arch),
        total_params=counts.total,
        batch_size=batch_size,
        peak_lr=config.optimizer_config.lr_scheduler.peak_value,
        tokens=config.total_train_tokens,
        predicted_seconds=predicted_seconds,
        predicted_loss=(
            float(parametric(N, config.total_train_tokens)) if parametric else None
        ),
        training_config=config,
    )


@dataclass(frozen=True)
class HoldoutValidation:
    """Fit on all but the largest time budget, predict the held-out profile."""

    held_out_budget_seconds: float
    n_train_runs: int
    parametric_rmse: float | None
    parametric_max_abs_error: float | None
    parametric_predicted_best_k: int | None
    isotime_predicted_n_opt: float | None
    isotime_predicted_loss: float | None
    actual_best_k: int
    actual_best_n: float
    actual_best_loss: float

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def holdout_validation(
    records: Sequence[ExperimentRecord],
    *,
    throughput: ThroughputModel | None = None,
    fast: bool = False,
) -> HoldoutValidation | None:
    profiles = iso_time_profiles(records)
    if len(profiles) < 3:
        return None
    held_out = profiles[-1]
    train = [
        r
        for r in loss_fit_records(records)
        if r.time_budget_seconds != held_out.time_budget_seconds
    ]
    parametric = fit_parametric(train, fast=fast) if len(train) >= 5 else None
    rmse = max_err = None
    best_k = None
    if parametric is not None:
        pred = parametric(
            [r.non_embedding_params for r in held_out.runs],
            [r.tokens for r in held_out.runs],
        )
        metrics = fit_metrics([r.final_loss for r in held_out.runs], pred)
        rmse, max_err = metrics.rmse, metrics.max_abs_error
        best_k = held_out.runs[int(np.argmin(pred))].ladder_k
    iso_train = fit_iso_time(train, throughput=throughput)
    n_pred = l_pred = None
    if iso_train is not None and iso_train.n_opt_law is not None:
        n_pred = float(iso_train.n_opt_law(held_out.best.used_runtime_seconds))
    if iso_train is not None and iso_train.l_opt_law is not None:
        l_pred = float(iso_train.l_opt_law(held_out.best.used_runtime_seconds))
    return HoldoutValidation(
        held_out_budget_seconds=held_out.time_budget_seconds,
        n_train_runs=len(train),
        parametric_rmse=rmse,
        parametric_max_abs_error=max_err,
        parametric_predicted_best_k=best_k,
        isotime_predicted_n_opt=n_pred,
        isotime_predicted_loss=l_pred,
        actual_best_k=held_out.best.ladder_k or -1,
        actual_best_n=held_out.n_opt_interp,
        actual_best_loss=held_out.loss_interp,
    )


@dataclass(frozen=True)
class BootstrapSummary:
    """Uncertainty of the final-loss prediction from resampling the fitted runs.

    With ~15-25 runs spanning barely more than a decade in ``N`` and ``D`` the five
    parameters of the loss surface are only weakly identified (``E`` trades off against
    ``B / D**beta``), so a single fit's extrapolation can wander. Resampling runs with
    replacement and refitting gives a distribution of predictions at the chosen final
    configuration; the median is a more robust point estimate than any single fit.
    """

    n_bootstrap: int
    predicted_losses: list[float]
    optimal_ks: list[int]

    @property
    def median(self) -> float:
        return float(np.median(self.predicted_losses))

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.predicted_losses, q))

    def to_dict(self) -> dict[str, object]:
        return {
            "n_bootstrap": self.n_bootstrap,
            "median": self.median,
            "p10": self.percentile(10),
            "p90": self.percentile(90),
            "predicted_losses": self.predicted_losses,
            "optimal_ks": self.optimal_ks,
        }


def bootstrap_final_prediction(
    records: Sequence[ExperimentRecord],
    candidates: Sequence[FinalCandidate],
    chosen: FinalCandidate,
    *,
    n_bootstrap: int = 20,
    seed: int = 0,
) -> BootstrapSummary | None:
    runs = loss_fit_records(records)
    if len(runs) < 5 or n_bootstrap <= 0:
        return None
    rng = np.random.default_rng(seed)
    N_all = np.array([r.non_embedding_params for r in runs], dtype=float)
    D_all = np.array([r.tokens for r in runs], dtype=float)
    L_all = np.array([r.final_loss for r in runs])
    cand_N = np.array([c.N for c in candidates], dtype=float)
    cand_D = np.array([c.tokens for c in candidates], dtype=float)
    losses: list[float] = []
    ks: list[int] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(runs), len(runs))
        if len(set(idx.tolist())) < 5:
            continue
        fit = fit_chinchilla_parametric(
            N_all[idx],
            D_all[idx],
            L_all[idx],
            alpha_grid=(0.2, 0.5, 1.0),
            beta_grid=(0.2, 0.5, 1.0),
            e_grid=(-0.5, 0.0, 0.5),
            a_grid=(5.0, 10.0, 15.0),
            b_grid=(5.0, 10.0, 15.0),
        )
        losses.append(float(fit(chosen.N, chosen.tokens)))
        ks.append(candidates[int(np.argmin(fit(cand_N, cand_D)))].k)
    if not losses:
        return None
    return BootstrapSummary(
        n_bootstrap=len(losses), predicted_losses=losses, optimal_ks=ks
    )


@dataclass
class ScalingLawAnalysis:
    throughput: ThroughputModel
    throughput_is_prior: bool
    lr_fit: LRSweepFit | None
    iso_fit: IsoTimeFit | None
    parametric: ChinchillaParametricFit | None
    holdout: HoldoutValidation | None
    final_budget_seconds: float
    safety: float
    candidates: list[FinalCandidate] = field(default_factory=list)
    parametric_choice: FinalCandidate | None = None
    isotime_choice: FinalCandidate | None = None
    isotime_predicted_loss: float | None = None
    chosen_method: str = "parametric"
    chinchilla_flops_optimum: dict[str, float] | None = None
    bootstrap: BootstrapSummary | None = None

    @property
    def chosen(self) -> FinalCandidate | None:
        return (
            self.parametric_choice
            if self.chosen_method == "parametric"
            else self.isotime_choice
        )

    @property
    def predicted_final_loss(self) -> float | None:
        if self.chosen_method == "parametric":
            if self.bootstrap is not None:
                return self.bootstrap.median
            return (
                self.parametric_choice.predicted_loss
                if self.parametric_choice
                else None
            )
        return self.isotime_predicted_loss

    def lr_rule(self) -> LRRule:
        return self.lr_fit.rule if self.lr_fit else prior_lr_rule

    def optimal_k_for_budget(self, time_budget_seconds: float) -> int | None:
        """Best ladder index for a time budget under the current fits (used by the planner)."""
        if self.parametric is None:
            return None
        best_k, best_loss = None, math.inf
        for k in FINAL_CANDIDATE_KS:
            arch = ladder_shape(k)
            N = count_parameters(arch).non_embedding
            D = self.throughput.tokens_for_seconds(arch, time_budget_seconds)
            loss = float(self.parametric(N, D))
            if loss < best_loss:
                best_k, best_loss = k, loss
        return best_k

    def to_dict(self) -> dict[str, object]:
        return {
            "final_budget_seconds": self.final_budget_seconds,
            "safety": self.safety,
            "throughput": self.throughput.to_dict()
            | {"is_prior": self.throughput_is_prior},
            "lr_fit": self.lr_fit.to_dict() if self.lr_fit else None,
            "iso_time": self.iso_fit.to_dict() if self.iso_fit else None,
            "parametric": self.parametric.to_dict() if self.parametric else None,
            "holdout_validation": self.holdout.to_dict() if self.holdout else None,
            "chinchilla_flops_optimum": self.chinchilla_flops_optimum,
            "candidates": [c.to_dict() for c in self.candidates],
            "parametric_choice": self.parametric_choice.to_dict()
            if self.parametric_choice
            else None,
            "isotime_choice": self.isotime_choice.to_dict()
            if self.isotime_choice
            else None,
            "isotime_predicted_loss": self.isotime_predicted_loss,
            "chosen_method": self.chosen_method,
            "bootstrap": self.bootstrap.to_dict() if self.bootstrap else None,
            "predicted_final_loss": self.predicted_final_loss,
        }


def fit_throughput_from_state(state: PipelineState) -> tuple[ThroughputModel, bool]:
    observations = [o for r in state.experiments if (o := r.throughput_observation())]
    if len({o.flops_per_token for o in observations}) >= 2:
        return fit_throughput_model(observations), False
    return prior_throughput_model(), True


def analyze(
    state: PipelineState,
    planner: PlannerConfig = PlannerConfig(),
    *,
    final_budget_seconds: float = FINAL_BUDGET_SECONDS,
    safety: float = 0.85,
    selection: str = "parametric",
    final_n_evals: int = 32,
    fast: bool = False,
    n_bootstrap: int = 0,
) -> ScalingLawAnalysis:
    throughput, is_prior = fit_throughput_from_state(state)
    lr_fit = fit_lr_sweep(state.experiments)
    iso_fit = fit_iso_time(state.experiments, throughput=throughput)
    parametric = fit_parametric(state.experiments, fast=fast)
    holdout = holdout_validation(state.experiments, throughput=throughput, fast=fast)
    analysis = ScalingLawAnalysis(
        throughput=throughput,
        throughput_is_prior=is_prior,
        lr_fit=lr_fit,
        iso_fit=iso_fit,
        parametric=parametric,
        holdout=holdout,
        final_budget_seconds=final_budget_seconds,
        safety=safety,
        chosen_method=selection,
    )
    lr_rule = analysis.lr_rule()
    analysis.candidates = [
        build_final_candidate(
            k,
            throughput=throughput,
            lr_rule=lr_rule,
            parametric=parametric,
            planner=planner,
            final_budget_seconds=final_budget_seconds,
            safety=safety,
            n_evals=final_n_evals,
        )
        for k in FINAL_CANDIDATE_KS
    ]
    if parametric is not None:
        analysis.parametric_choice = min(
            analysis.candidates, key=lambda c: c.predicted_loss or math.inf
        )
        # FLOPs-based Chinchilla optimum at the FLOPs the chosen run will actually execute.
        chosen = analysis.parametric_choice
        flops = (
            flops_per_token(chosen.training_config.architecture_config) * chosen.tokens
        )
        n_opt, d_opt = parametric.compute_optimal(flops)
        analysis.chinchilla_flops_optimum = {
            "flops": float(flops),
            "n_opt": float(n_opt),
            "d_opt": float(d_opt),
            "loss": float(parametric(n_opt, d_opt)),
        }
    if iso_fit is not None and iso_fit.n_opt_law is not None:
        n_target = float(iso_fit.n_opt_law(safety * final_budget_seconds))
        analysis.isotime_choice = min(
            analysis.candidates, key=lambda c: abs(math.log(c.N / n_target))
        )
        if iso_fit.l_opt_law is not None:
            analysis.isotime_predicted_loss = float(
                iso_fit.l_opt_law(safety * final_budget_seconds)
            )
    if selection == "parametric" and analysis.parametric_choice is None:
        analysis.chosen_method = "isotime"
    if analysis.chosen_method == "isotime" and analysis.isotime_choice is None:
        analysis.chosen_method = "parametric"
    if (
        analysis.chosen_method == "parametric"
        and analysis.parametric_choice is not None
    ):
        analysis.bootstrap = bootstrap_final_prediction(
            state.experiments,
            analysis.candidates,
            analysis.parametric_choice,
            n_bootstrap=n_bootstrap,
        )
    return analysis


# --------------------------------------------------------------------------------------
# Plots and report
# --------------------------------------------------------------------------------------
def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f}h" if seconds >= 3600 else f"{seconds / 60:.1f}min"


def _fmt_n(n: float) -> str:
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def plot_throughput(
    analysis: ScalingLawAnalysis, state: PipelineState, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = [(r, o) for r in state.experiments if (o := r.throughput_observation())]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fpt_grid = np.logspace(7.5, 11, 200)
    tm = analysis.throughput
    for ax, metric in zip(axes, ("tps", "mfu"), strict=True):
        for stage, colour in (
            ("probe", "C0"),
            ("lr_sweep", "C2"),
            ("iso_time", "C1"),
            ("extra", "C4"),
        ):
            pts = [
                (o.flops_per_token, o.tokens_per_second)
                for r, o in obs
                if r.stage == stage
            ]
            if not pts:
                continue
            x = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            if metric == "mfu":
                y = x * y / tm.peak_flops
            ax.scatter(x, y, s=18, color=colour, label=stage, alpha=0.8)
        fit = tm.tokens_per_second_for_flops(fpt_grid)
        if metric == "mfu":
            fit = fpt_grid * fit / tm.peak_flops
        ax.plot(
            fpt_grid,
            fit,
            "k--",
            lw=1,
            label="fit" + (" (prior)" if analysis.throughput_is_prior else ""),
        )
        ax.set_xscale("log")
        ax.set_xlabel("training FLOPs per token")
        ax.grid(True, which="both", alpha=0.3)
        if metric == "tps":
            ax.set_yscale("log")
            ax.set_ylabel("tokens / second")
            ax.set_title(tm.describe(), fontsize=8)
        else:
            ax.set_ylabel("model FLOP utilisation")
            ax.set_title("MFU vs model size")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lr_sweep(analysis: ScalingLawAnalysis, path: Path) -> None:
    if analysis.lr_fit is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, s in enumerate(analysis.lr_fit.scales):
        axes[0].plot(
            s.lrs, s.losses, "o-", color=f"C{i}", label=f"k={s.k} (N={_fmt_n(s.N)})"
        )
        axes[0].axvline(s.lr_opt, color=f"C{i}", ls=":", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("peak learning rate")
    axes[0].set_ylabel("final validation loss")
    axes[0].set_title("LR sweeps (dotted = fitted optimum)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    Ns = np.array([s.N for s in analysis.lr_fit.scales])
    grid = np.logspace(np.log10(Ns.min() / 3), 9.7, 100)
    axes[1].scatter(
        Ns, [s.lr_opt for s in analysis.lr_fit.scales], color="C1", zorder=3
    )
    axes[1].plot(
        grid,
        analysis.lr_fit.law(grid),
        "k--",
        lw=1,
        label=analysis.lr_fit.law.describe("N", "lr_opt"),
    )
    if analysis.chosen:
        axes[1].scatter(
            [analysis.chosen.N],
            [analysis.chosen.peak_lr],
            marker="*",
            s=150,
            color="C3",
            label="final run",
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("non-embedding parameters N")
    axes[1].set_ylabel("optimal peak LR")
    axes[1].set_title("LR scaling rule")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_iso_time_profiles(analysis: ScalingLawAnalysis, path: Path) -> None:
    if analysis.iso_fit is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    profiles = analysis.iso_fit.profiles
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(profiles)))
    for colour, p in zip(colours, profiles, strict=True):
        Ns = [r.non_embedding_params for r in p.runs]
        Ls = [r.final_loss for r in p.runs]
        ax.plot(Ns, Ls, "o", color=colour, label=f"T={_hours(p.time_budget_seconds)}")
        ax.scatter(
            [p.best.non_embedding_params],
            [p.best.final_loss],
            marker="*",
            s=160,
            color=colour,
            zorder=4,
        )
        if analysis.parametric is not None:
            grid_k = np.arange(2, 65)
            grid_N = np.array(
                [count_parameters(ladder_shape(k)).non_embedding for k in grid_k]
            )
            grid_D = np.array(
                [
                    analysis.throughput.tokens_for_seconds(
                        ladder_shape(k), p.mean_runtime_seconds
                    )
                    for k in grid_k
                ]
            )
            ax.plot(
                grid_N, analysis.parametric(grid_N, grid_D), "--", color=colour, lw=1
            )
    ax.set_xscale("log")
    ax.set_xlabel("non-embedding parameters N")
    ax.set_ylabel("final validation loss")
    ax.set_title("Iso-time profiles (stars = best run; dashed = parametric fit)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_iso_time_scaling(analysis: ScalingLawAnalysis, path: Path) -> None:
    iso = analysis.iso_fit
    if iso is None or iso.n_opt_law is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = np.array([p.best.used_runtime_seconds for p in iso.profiles])
    T_final = analysis.safety * analysis.final_budget_seconds
    grid = np.logspace(
        np.log10(T.min() / 2), np.log10(analysis.final_budget_seconds * 1.5), 200
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    d_interp = [
        analysis.throughput.tokens_per_second_for_flops(
            _flops_per_token_for_n(p.n_opt_interp)
        )
        * p.best.used_runtime_seconds
        for p in iso.profiles
    ]
    panels = [
        (
            axes[0],
            [p.best.non_embedding_params for p in iso.profiles],
            [p.n_opt_interp for p in iso.profiles],
            iso.n_opt_law,
            "N_opt (params)",
            "N_opt",
        ),
        (
            axes[1],
            [p.best.tokens for p in iso.profiles],
            d_interp,
            iso.d_opt_law,
            "D_opt (tokens)",
            "D_opt",
        ),
        (
            axes[2],
            [p.best.final_loss for p in iso.profiles],
            [p.loss_interp for p in iso.profiles],
            iso.l_opt_law,
            "L_opt",
            "L_opt",
        ),
    ]
    for ax, raw, interp, law, ylabel, name in panels:
        ax.scatter(T, raw, color="C1", marker="x", zorder=3, label="best grid run")
        ax.scatter(
            T, interp, color="C0", zorder=3, label="parabola-interpolated minimum"
        )
        if law is not None:
            ax.plot(grid, law(grid), "C0", label=law.describe("T", name))
            pred = float(law(T_final))
            ax.scatter(
                [T_final],
                [pred],
                marker="*",
                s=180,
                color="C3",
                zorder=4,
                label=f"48h x {analysis.safety}: {pred:.3g}",
            )
        ax.axvline(analysis.final_budget_seconds, color="gray", ls=":", lw=1)
        ax.set_xscale("log")
        if name != "L_opt":
            ax.set_yscale("log")
        ax.set_xlabel("wall-clock budget T (seconds)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Iso-time scaling laws extrapolated to the 48 B200-hour budget")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_parametric(
    analysis: ScalingLawAnalysis, state: PipelineState, path: Path
) -> None:
    if analysis.parametric is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = loss_fit_records(state.experiments)
    actual = np.array([r.final_loss for r in runs])
    pred = analysis.parametric(
        [r.non_embedding_params for r in runs], [r.tokens for r in runs]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(actual, pred, s=18, color="C1")
    lo, hi = min(actual.min(), pred.min()) - 0.05, max(actual.max(), pred.max()) + 0.05
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xlabel("actual final loss")
    axes[0].set_ylabel("predicted loss")
    m = analysis.parametric.metrics
    axes[0].set_title(
        f"Parametric fit: RMSE={m.rmse:.4f}, max |err|={m.max_abs_error:.4f}"
        if m
        else "Parametric fit"
    )
    axes[0].grid(True, alpha=0.3)

    cands = [c for c in analysis.candidates if c.predicted_loss is not None]
    axes[1].plot(
        [c.N for c in cands],
        [c.predicted_loss for c in cands],
        "C0.-",
        label=f"L(N, D(N, {analysis.safety} x 48h))",
    )
    if analysis.parametric_choice:
        c = analysis.parametric_choice
        axes[1].scatter(
            [c.N],
            [c.predicted_loss],
            marker="*",
            s=200,
            color="C3",
            zorder=4,
            label=f"parametric optimum: k={c.k}, N={_fmt_n(c.N)}, L={c.predicted_loss:.3f}",
        )
    if analysis.isotime_choice:
        c = analysis.isotime_choice
        axes[1].axvline(
            c.N, color="C2", ls=":", label=f"iso-time optimum: k={c.k}, N={_fmt_n(c.N)}"
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("non-embedding parameters N")
    axes[1].set_ylabel("predicted final loss at the 48h budget")
    axes[1].set_title("Predicted loss vs model size for the final run")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(
    analysis: ScalingLawAnalysis, state: PipelineState, outdir: Path
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    plot_throughput(analysis, state, outdir / "throughput.png")
    plot_lr_sweep(analysis, outdir / "lr_sweep.png")
    plot_iso_time_profiles(analysis, outdir / "iso_time_profiles.png")
    plot_iso_time_scaling(analysis, outdir / "iso_time_scaling.png")
    plot_parametric(analysis, state, outdir / "parametric_fit.png")
    (outdir / "fit_summary.json").write_text(
        json.dumps(analysis.to_dict(), indent=2, default=str) + "\n"
    )

    chosen = analysis.chosen
    if chosen is not None and analysis.predicted_final_loss is not None:
        (outdir / "final_submission.json").write_text(
            json.dumps(
                {
                    "training_config": chosen.training_config.model_dump(mode="json"),
                    "predicted_final_loss": analysis.predicted_final_loss,
                },
                indent=2,
            )
            + "\n"
        )

    lines: list[str] = []
    add = lines.append
    add(f"# Scaling-law report ({state.backend} backend)\n")
    finished = [r for r in state.experiments if r.is_finished]
    charged = sum(
        min(
            max(1.0, r.used_runtime_seconds or 0), r.training_config.max_runtime_seconds
        )
        for r in finished
    )
    add(
        f"- experiments: {len(state.experiments)} tracked, {len(state.completed())} completed, "
        f"{sum(r.status == 'failed' for r in state.experiments)} failed"
    )
    add(f"- budget charged: {charged / 3600:.2f} B200-hours of 12\n")

    add("## Throughput model\n")
    add(
        f"- {analysis.throughput.describe()}"
        + (" **(prior, no measurements yet)**" if analysis.throughput_is_prior else "")
    )
    if analysis.throughput.metrics:
        add(
            f"- fit on {analysis.throughput.n_observations} runs, log-space R^2 = {analysis.throughput.metrics.r2:.3f}, "
            f"RMSE of log(tokens/s) = {analysis.throughput.metrics.rmse:.3f}"
        )
    add("")

    add("## Learning-rate rule\n")
    if analysis.lr_fit:
        add(f"- {analysis.lr_fit.law.describe('N', 'lr_opt')}")
        for s in analysis.lr_fit.scales:
            add(
                f"  - k={s.k} (N={_fmt_n(s.N)}): lr_opt = {s.lr_opt:.2e}, losses {[round(x, 4) for x in s.losses]} at lrs {[f'{x:.1e}' for x in s.lrs]}"
            )
    else:
        add("- no LR sweep yet; using prior lr(N) = 2e-3 (N/1e7)^-0.2")
    add("")

    add("## Iso-time profiles\n")
    if analysis.iso_fit:
        add(
            "| budget | best k | N_opt (grid) | D_opt (grid) | D/N | loss | runtime | N_opt (interp.) | loss (interp.) |"
        )
        add("|---|---|---|---|---|---|---|---|---|")
        for p in analysis.iso_fit.profiles:
            b = p.best
            add(
                f"| {_hours(p.time_budget_seconds)} | {b.ladder_k} | {_fmt_n(b.non_embedding_params)} | "
                f"{b.tokens / 1e9:.2f}B | {b.tokens / b.non_embedding_params:.1f} | {b.final_loss:.4f} | {b.used_runtime_seconds:.0f}s | "
                f"{_fmt_n(p.n_opt_interp)} | {p.loss_interp:.4f} |"
            )
        add("")
        add(
            "Power laws below are fit to the parabola-interpolated minima (Hoffmann et al. approach 2)."
        )
        for law, name in (
            (analysis.iso_fit.n_opt_law, "N_opt"),
            (analysis.iso_fit.d_opt_law, "D_opt"),
        ):
            if law:
                add(
                    f"- {law.describe('T', name)} (log-space R^2 = {law.metrics.r2:.3f})"
                )
        if analysis.iso_fit.l_opt_law:
            add(
                f"- {analysis.iso_fit.l_opt_law.describe('T', 'L_opt')} (R^2 = {analysis.iso_fit.l_opt_law.metrics.r2:.3f})"
            )
    else:
        add("- no completed iso-time runs yet")
    add("")

    add("## Parametric loss surface\n")
    if analysis.parametric:
        m = analysis.parametric.metrics
        add(f"- {analysis.parametric.describe()}")
        add(
            f"- fit on {m.n_points} runs: RMSE = {m.rmse:.4f}, max |error| = {m.max_abs_error:.4f}, R^2 = {m.r2:.4f}"
        )
        a = analysis.parametric.beta / (
            analysis.parametric.alpha + analysis.parametric.beta
        )
        add(
            f"- implied FLOPs-optimal exponents: N_opt ~ C^{a:.3f}, D_opt ~ C^{1 - a:.3f}"
        )
    else:
        add("- fewer than five completed iso-time runs; parametric fit not available")
    add("")

    add("## Hold-out validation (largest time budget held out)\n")
    if analysis.holdout:
        h = analysis.holdout
        add(
            f"- held out T = {_hours(h.held_out_budget_seconds)} ({h.n_train_runs} training runs)"
        )
        if h.parametric_rmse is not None:
            add(
                f"- parametric: RMSE = {h.parametric_rmse:.4f}, max |error| = {h.parametric_max_abs_error:.4f}; "
                f"predicted best k = {h.parametric_predicted_best_k} vs actual best k = {h.actual_best_k}"
            )
        if h.isotime_predicted_n_opt is not None:
            add(
                f"- iso-time N_opt law: predicted {_fmt_n(h.isotime_predicted_n_opt)} vs actual {_fmt_n(h.actual_best_n)}"
            )
        if h.isotime_predicted_loss is not None:
            add(
                f"- iso-time L_opt law: predicted {h.isotime_predicted_loss:.4f} vs actual {h.actual_best_loss:.4f}"
            )
    else:
        add("- needs at least three iso-time budgets")
    add("")

    add(
        f"## Final run ({_hours(analysis.final_budget_seconds)} budget, tokens sized for {analysis.safety:.0%} of it)\n"
    )
    for label, cand in (
        ("parametric", analysis.parametric_choice),
        ("iso-time", analysis.isotime_choice),
    ):
        if cand is None:
            continue
        loss = (
            cand.predicted_loss
            if label == "parametric"
            else analysis.isotime_predicted_loss
        )
        add(
            f"- **{label}**: k={cand.k} -> {describe_shape(cand.training_config.architecture_config)}, "
            f"D={cand.tokens / 1e9:.2f}B tokens (D/N={cand.tokens / cand.N:.1f}), batch {cand.batch_size} x 512, "
            f"peak lr {cand.peak_lr:.2e}, predicted runtime {_hours(cand.predicted_seconds)}, "
            f"predicted loss {loss if loss is None else f'{loss:.4f}'}"
        )
    if analysis.chinchilla_flops_optimum:
        c = analysis.chinchilla_flops_optimum
        add(
            f"- FLOPs-based Chinchilla optimum at C={c['flops']:.3e}: N={_fmt_n(c['n_opt'])}, D={c['d_opt'] / 1e9:.2f}B, loss {c['loss']:.4f}"
        )
    if analysis.bootstrap:
        b = analysis.bootstrap
        ks = sorted(set(b.optimal_ks))
        k_counts = ", ".join(f"k={k}: {b.optimal_ks.count(k)}" for k in ks)
        add(
            f"- bootstrap over runs ({b.n_bootstrap} refits): predicted loss at the chosen config "
            f"median {b.median:.4f}, 10-90% [{b.percentile(10):.4f}, {b.percentile(90):.4f}]; "
            f"optimal shape across refits: {k_counts}"
        )
    if chosen is not None:
        add(
            f"\n**Chosen ({analysis.chosen_method})**: predicted final loss **{analysis.predicted_final_loss:.4f}**"
            + (" (bootstrap median)" if analysis.bootstrap else "")
            + "\n"
        )
        add("```json")
        add(json.dumps(chosen.training_config.model_dump(mode="json"), indent=2))
        add("```")
    (outdir / "report.md").write_text("\n".join(lines) + "\n")
    return outdir / "report.md"
