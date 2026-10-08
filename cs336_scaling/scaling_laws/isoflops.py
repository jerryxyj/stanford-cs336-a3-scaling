"""Problem ``chinchilla_isoflops``: IsoFLOPs scaling laws (Hoffmann et al., 2022, approach 2).

Given training runs ``(parameters, compute_budget, final_loss)`` we

1. group the runs by compute budget ``C_i`` (each group is an "IsoFLOPs profile"),
2. take the run with the lowest final loss in each profile as ``N_opt(C_i)`` (the handout
   recommends this over fitting a parabola to each profile) and set
   ``D_opt(C_i) = C_i / (6 N_opt(C_i))`` using ``C = 6 N D``,
3. fit power laws ``N_opt = k_N C**a`` and ``D_opt = k_D C**b`` in log-log space,
4. extrapolate to larger budgets (``1e23`` and ``1e24`` FLOPs in the handout).

Run ``python -m cs336_scaling.scaling_laws isoflops`` (or ``scripts/chinchilla_isoflops.py``)
to produce the plots and predictions.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cs336_scaling.scaling_laws.fit import PowerLaw, fit_power_law

FLOPS_PER_PARAM_TOKEN = 6.0
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "isoflops_curves.json"
)
DEFAULT_TARGET_BUDGETS = (1e23, 1e24)


@dataclass(frozen=True)
class IsoFlopsRun:
    parameters: float
    compute_budget: float
    final_loss: float

    @property
    def tokens(self) -> float:
        return self.compute_budget / (FLOPS_PER_PARAM_TOKEN * self.parameters)


@dataclass(frozen=True)
class IsoFlopsOptimum:
    """The best run of one IsoFLOPs profile."""

    compute_budget: float
    n_opt: float
    d_opt: float
    loss: float

    def to_dict(self) -> dict[str, float]:
        return {
            "compute_budget": self.compute_budget,
            "n_opt": self.n_opt,
            "d_opt": self.d_opt,
            "loss": self.loss,
        }


@dataclass(frozen=True)
class IsoFlopsScalingLaws:
    optima: list[IsoFlopsOptimum]
    n_opt_law: PowerLaw
    d_opt_law: PowerLaw

    def predict_n_opt(self, compute_budget: float) -> float:
        return float(self.n_opt_law(compute_budget))

    def predict_d_opt(self, compute_budget: float) -> float:
        return float(self.d_opt_law(compute_budget))

    def summary(self, target_budgets=DEFAULT_TARGET_BUDGETS) -> dict[str, object]:
        return {
            "optima": [o.to_dict() for o in self.optima],
            "n_opt_law": self.n_opt_law.to_dict(),
            "d_opt_law": self.d_opt_law.to_dict(),
            "predictions": {
                f"{budget:.0e}": {
                    "n_opt": self.predict_n_opt(budget),
                    "d_opt": self.predict_d_opt(budget),
                }
                for budget in target_budgets
            },
        }


def load_runs(path: Path = DEFAULT_DATA_PATH) -> list[IsoFlopsRun]:
    with Path(path).open() as f:
        raw = json.load(f)
    return [
        IsoFlopsRun(
            parameters=float(r["parameters"]),
            compute_budget=float(r["compute_budget"]),
            final_loss=float(r["final_loss"]),
        )
        for r in raw
    ]


def group_profiles(runs: list[IsoFlopsRun]) -> dict[float, list[IsoFlopsRun]]:
    profiles: dict[float, list[IsoFlopsRun]] = defaultdict(list)
    for run in runs:
        profiles[run.compute_budget].append(run)
    return {
        c: sorted(rs, key=lambda r: r.parameters) for c, rs in sorted(profiles.items())
    }


def profile_optima(runs: list[IsoFlopsRun]) -> list[IsoFlopsOptimum]:
    """Lowest-loss run of each IsoFLOPs profile, in increasing order of compute budget."""
    optima = []
    for compute_budget, profile in group_profiles(runs).items():
        best = min(profile, key=lambda r: r.final_loss)
        optima.append(
            IsoFlopsOptimum(
                compute_budget=compute_budget,
                n_opt=best.parameters,
                d_opt=best.tokens,
                loss=best.final_loss,
            )
        )
    return optima


def fit_isoflops_scaling_laws(runs: list[IsoFlopsRun]) -> IsoFlopsScalingLaws:
    optima = profile_optima(runs)
    budgets = np.array([o.compute_budget for o in optima])
    return IsoFlopsScalingLaws(
        optima=optima,
        n_opt_law=fit_power_law(budgets, [o.n_opt for o in optima]),
        d_opt_law=fit_power_law(budgets, [o.d_opt for o in optima]),
    )


def _format_count(x: float, unit: str) -> str:
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= scale:
            return f"{x / scale:.2f}{suffix} {unit}"
    return f"{x:.0f} {unit}"


def plot_scaling_law(
    laws: IsoFlopsScalingLaws,
    *,
    quantity: str,
    output_path: Path,
    target_budgets=DEFAULT_TARGET_BUDGETS,
    runs: list[IsoFlopsRun] | None = None,
) -> Path:
    """Plot ``N_opt`` (``quantity="n"``) or ``D_opt`` (``quantity="d"``) against compute."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if quantity == "n":
        law, label, unit = (
            laws.n_opt_law,
            "Compute-optimal model size $N_{opt}$",
            "params",
        )
        short_name = "N_opt"
        points = [(o.compute_budget, o.n_opt) for o in laws.optima]
        run_points = [(r.compute_budget, r.parameters) for r in runs or []]
    elif quantity == "d":
        law, label, unit = (
            laws.d_opt_law,
            "Compute-optimal dataset size $D_{opt}$",
            "tokens",
        )
        short_name = "D_opt"
        points = [(o.compute_budget, o.d_opt) for o in laws.optima]
        run_points = [(r.compute_budget, r.tokens) for r in runs or []]
    else:
        raise ValueError("quantity must be 'n' or 'd'")

    budgets = np.array([p[0] for p in points])
    lo = budgets.min() / 3
    hi = max(budgets.max(), *target_budgets) * 3
    grid = np.logspace(np.log10(lo), np.log10(hi), 200)

    fig, ax = plt.subplots(figsize=(7, 5))
    if run_points:
        ax.scatter(
            [p[0] for p in run_points],
            [p[1] for p in run_points],
            s=12,
            color="lightgray",
            label="all training runs",
            zorder=1,
        )
    ax.plot(grid, law(grid), color="C0", label=law.describe("C", short_name))
    ax.scatter(
        budgets,
        [p[1] for p in points],
        color="C1",
        zorder=3,
        label="profile minima (fit points)",
    )
    for budget in target_budgets:
        pred = float(law(budget))
        ax.scatter([budget], [pred], marker="*", s=160, color="C3", zorder=4)
        ax.annotate(
            f"C={budget:.0e}\n{_format_count(pred, unit)}",
            (budget, pred),
            textcoords="offset points",
            xytext=(-70, 10),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute budget C (FLOPs)")
    ax.set_ylabel(f"{label} ({unit})")
    ax.set_title(f"IsoFLOPs scaling law: {label}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_profiles(runs: list[IsoFlopsRun], output_path: Path) -> Path:
    """Plot every IsoFLOPs profile (loss vs. N for each budget) with its minimum marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    profiles = group_profiles(runs)
    colors = plt.cm.viridis(np.linspace(0, 1, len(profiles)))
    for color, (budget, profile) in zip(colors, profiles.items(), strict=True):
        ns = [r.parameters for r in profile]
        losses = [r.final_loss for r in profile]
        ax.plot(ns, losses, "o-", color=color, ms=4, label=f"C={budget:.0e}")
        best = min(profile, key=lambda r: r.final_loss)
        ax.scatter([best.parameters], [best.final_loss], color=color, s=90, marker="*")
    ax.set_xscale("log")
    ax.set_xlabel("Model size N (parameters)")
    ax.set_ylabel("Final training loss")
    ax.set_title("IsoFLOPs profiles (stars = lowest loss per budget)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_isoflops_analysis(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = Path("results/isoflops"),
    target_budgets=DEFAULT_TARGET_BUDGETS,
) -> IsoFlopsScalingLaws:
    """Fit the laws, write ``summary.json`` and the plots, and print the deliverables."""
    runs = load_runs(data_path)
    laws = fit_isoflops_scaling_laws(runs)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = laws.summary(target_budgets)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_profiles(runs, output_dir / "isoflops_profiles.png")
    plot_scaling_law(
        laws,
        quantity="n",
        output_path=output_dir / "model_size_scaling_law.png",
        target_budgets=target_budgets,
        runs=runs,
    )
    plot_scaling_law(
        laws,
        quantity="d",
        output_path=output_dir / "dataset_size_scaling_law.png",
        target_budgets=target_budgets,
        runs=runs,
    )

    print(f"Loaded {len(runs)} runs across {len(laws.optima)} compute budgets")
    print(f"{'C (FLOPs)':>12} {'N_opt':>12} {'D_opt':>12} {'loss':>8}")
    for o in laws.optima:
        print(
            f"{o.compute_budget:>12.2e} {o.n_opt:>12.3e} {o.d_opt:>12.3e} {o.loss:>8.4f}"
        )
    print()
    print(
        "Model size law:   ",
        laws.n_opt_law.describe("C", "N_opt"),
        f"(R^2 in log space = {laws.n_opt_law.metrics.r2:.4f})",
    )
    print(
        "Dataset size law: ",
        laws.d_opt_law.describe("C", "D_opt"),
        f"(R^2 in log space = {laws.d_opt_law.metrics.r2:.4f})",
    )
    print()
    for budget in target_budgets:
        n = laws.predict_n_opt(budget)
        d = laws.predict_d_opt(budget)
        print(
            f"(a) For C = {budget:.0e} FLOPs the predicted compute-optimal model size is "
            f"N_opt ~= {n:.3e} ({_format_count(n, 'parameters')})."
        )
        print(
            f"(b) For C = {budget:.0e} FLOPs the predicted compute-optimal dataset size is "
            f"D_opt ~= {d:.3e} ({_format_count(d, 'tokens')})."
        )
    print(f"\nWrote plots and summary.json to {output_dir}/")
    return laws
