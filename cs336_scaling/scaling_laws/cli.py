"""Command-line entry point: ``python -m cs336_scaling.scaling_laws <command>``.

Commands
--------
isoflops      Problem ``chinchilla_isoflops`` on ``data/isoflops_curves.json``.
plan          Add a stage (``probe`` | ``lr_sweep`` | ``iso_time``) of runs to the state file.
run           Submit planned runs to the backend and poll until they finish.
fit           Fit the scaling laws, write plots/report, and the proposed final submission.
submit-final  POST ``final_submission.json`` to the backend.
all           probe -> run -> lr_sweep -> run -> iso_time -> run -> fit (the whole study).

``--backend api`` talks to the hosted training API (needs ``A3_API_KEY``); ``--backend
simulated`` uses the offline simulator so the whole pipeline can be exercised without GPUs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from cs336_scaling.scaling_laws import analysis as analysis_mod
from cs336_scaling.scaling_laws import plan as plan_mod
from cs336_scaling.scaling_laws.backends import ExperimentBackend, make_backend
from cs336_scaling.scaling_laws.runner import run_pending
from cs336_scaling.scaling_laws.state import PipelineState
from cs336_scaling.training.training_config import TrainingConfig

console = Console()
STAGES = ("probe", "lr_sweep", "iso_time", "bracket")


def _workdir(args: argparse.Namespace) -> Path:
    return (
        Path(args.workdir)
        if args.workdir
        else Path("results/scaling_laws") / args.backend
    )


def _state_path(workdir: Path) -> Path:
    return workdir / "state.json"


def _planner(args: argparse.Namespace) -> plan_mod.PlannerConfig:
    overrides = {}
    if getattr(args, "budget_fraction", None) is not None:
        overrides["budget_fraction"] = args.budget_fraction
    return plan_mod.PlannerConfig(**overrides)


def _iso_centre_function(
    state: PipelineState,
    current: analysis_mod.ScalingLawAnalysis,
    planner: plan_mod.PlannerConfig,
):
    """How to centre a new iso-time tier: parametric fit > empirical D/N > Chinchilla prior.

    The parametric prediction is bias-corrected by the ladder offset between the predicted
    and the observed best shape of the largest finished tier (zero if the fit is unbiased),
    so a systematic mis-prediction does not cost a bracket extension at every tier.
    """
    if current.parametric is not None:
        ladder = list(planner.ladder_ks)

        def nearest_index(k: int) -> int:
            return min(range(len(ladder)), key=lambda i: abs(ladder[i] - k))

        offset = 0
        finished = [
            (budget, by_k)
            for budget, by_k in plan_mod.iso_time_profile_ks(state).items()
            if all(r.is_finished for r in by_k.values())
            and any(r.status == "completed" for r in by_k.values())
        ]
        if finished:
            budget, by_k = max(finished)
            completed = {k: r for k, r in by_k.items() if r.status == "completed"}
            observed = min(completed, key=lambda k: completed[k].final_loss)
            predicted = current.optimal_k_for_budget(budget)
            offset = nearest_index(observed) - nearest_index(predicted)

        def centre(budget: float) -> int:
            idx = nearest_index(current.optimal_k_for_budget(budget))
            return ladder[max(0, min(len(ladder) - 1, idx + offset))]

        return centre
    empirical = plan_mod.empirical_tokens_per_param(state)
    if empirical is not None:
        return lambda budget: plan_mod.prior_optimal_k(
            planner, current.throughput, budget, tokens_per_param=empirical
        )
    return None


def plan_stage(
    stage: str,
    state: PipelineState,
    planner: plan_mod.PlannerConfig,
    *,
    budgets: list[float] | None = None,
    fast: bool = False,
    quiet: bool = False,
) -> list:
    """Plan one stage using everything the state already knows.

    ``iso_time`` plans the next not-yet-planned tier (or the given ``budgets``); ``bracket``
    extends finished tiers whose minimum sits on an edge of the sampled shapes.
    """
    # Planning only needs the fits for centring/sizing; the fast grid is plenty here.
    current = analysis_mod.analyze(state, planner, fast=True)
    throughput = current.throughput
    lr_rule = current.lr_rule()
    if stage == "probe":
        records = plan_mod.plan_probes(
            planner, throughput, planning_round=state.planning_rounds
        )
    elif stage == "lr_sweep":
        if current.throughput_is_prior:
            console.print(
                "[yellow]warning:[/yellow] no throughput measurements yet; sizing LR sweep from the prior"
            )
        records = plan_mod.plan_lr_sweep(
            planner, throughput, lr_rule=lr_rule, planning_round=state.planning_rounds
        )
    elif stage == "iso_time":
        if current.throughput_is_prior:
            console.print(
                "[yellow]warning:[/yellow] no throughput measurements yet; sizing iso-time runs from the prior"
            )
        if current.lr_fit is None:
            console.print(
                "[yellow]warning:[/yellow] no LR sweep results; using the prior LR rule"
            )
        if budgets is None:
            planned = set(plan_mod.iso_time_profile_ks(state))
            remaining = [b for b in planner.iso_time_budgets if b not in planned]
            budgets = remaining[:1]
            if not budgets:
                console.print("all iso-time tiers are already planned")
                return []
        records = plan_mod.plan_iso_time(
            planner,
            throughput,
            lr_rule=lr_rule,
            budgets=budgets,
            optimal_k_for_budget=_iso_centre_function(state, current, planner),
            planning_round=state.planning_rounds,
        )
    elif stage == "bracket":
        records = plan_mod.plan_bracket_extensions(
            state,
            planner,
            throughput,
            lr_rule=lr_rule,
            planning_round=state.planning_rounds,
        )
    else:
        raise ValueError(f"unknown stage {stage!r}")

    new_records = [r for r in records if not state.has_config(r.training_config)]
    if stage == "bracket":
        # Extensions are optional: add them cheapest-first while they fit the budget.
        affordable = []
        for r in sorted(
            new_records, key=lambda r: r.training_config.max_runtime_seconds
        ):
            if plan_mod.check_budget(state, affordable + [r], planner, throughput).fits:
                affordable.append(r)
        skipped = len(new_records) - len(affordable)
        if skipped and not quiet:
            console.print(
                f"[yellow]skipping {skipped} bracket extension(s) that do not fit the budget[/yellow]"
            )
        new_records = affordable
    check = plan_mod.check_budget(state, new_records, planner, throughput)
    if not quiet:
        console.print(
            f"[bold]{stage}[/bold]: {len(new_records)} new run(s); budget: {check.describe()}"
        )
    if not new_records:
        return []
    if not check.fits:
        raise SystemExit(
            f"planned {stage} runs do not fit the budget ({check.describe()}); "
            "reduce the stage's budgets/shapes in PlannerConfig or raise --budget-fraction"
        )
    added = plan_mod.add_to_state(state, new_records)
    state.planning_rounds += 1
    if not quiet:
        for r in added:
            cfg = r.training_config
            console.print(
                f"  + {r.name:<22} N={r.non_embedding_params / 1e6:7.1f}M D={r.tokens / 1e6:8.1f}M "
                f"B={cfg.train_batch_size:4d} lr={r.peak_lr:.2e} "
                f"expect {throughput.seconds_for_tokens(cfg.architecture_config, r.tokens):6.0f}s "
                f"reserve {cfg.max_runtime_seconds:6.0f}s"
            )
    return added


def cmd_isoflops(args: argparse.Namespace) -> None:
    from cs336_scaling.scaling_laws.isoflops import run_isoflops_analysis

    run_isoflops_analysis(data_path=Path(args.data), output_dir=Path(args.out))


def cmd_plan(args: argparse.Namespace) -> None:
    workdir = _workdir(args)
    state = PipelineState.load_or_create(_state_path(workdir), args.backend)
    budgets = [float(b) for b in args.budget] if args.budget else None
    plan_stage(args.stage, state, _planner(args), budgets=budgets, fast=args.fast)
    state.save(_state_path(workdir))
    console.print(f"state written to {_state_path(workdir)}")


def cmd_run(args: argparse.Namespace) -> None:
    workdir = _workdir(args)
    backend = make_backend(args.backend, workdir=workdir)
    state = PipelineState.load_or_create(_state_path(workdir), args.backend)
    poll = 0.0 if args.backend == "simulated" else args.poll_interval
    run_pending(state, backend, _state_path(workdir), poll_interval_seconds=poll)


def cmd_fit(args: argparse.Namespace) -> analysis_mod.ScalingLawAnalysis:
    workdir = _workdir(args)
    state = PipelineState.load(_state_path(workdir))
    result = analysis_mod.analyze(
        state,
        _planner(args),
        final_budget_seconds=args.final_hours * 3600.0,
        safety=args.safety,
        selection=args.selection,
        fast=args.fast,
        n_bootstrap=args.bootstrap,
    )
    report = analysis_mod.write_report(result, state, workdir / "fit")
    console.print(report.read_text())
    console.print(
        f"[dim]plots, fit_summary.json and final_submission.json written to {workdir / 'fit'}[/dim]"
    )
    return result


def _submit_final(
    backend: ExperimentBackend, workdir: Path, *, assume_yes: bool
) -> None:
    path = workdir / "fit" / "final_submission.json"
    payload = json.loads(path.read_text())
    config = TrainingConfig.model_validate(payload["training_config"])
    loss = float(payload["predicted_final_loss"])
    console.print(f"final submission from {path}: predicted loss {loss:.4f}")
    console.print_json(data=payload["training_config"])
    if not assume_yes and backend.name == "api":
        answer = (
            input("submit this as your final submission to the API? [y/N] ")
            .strip()
            .lower()
        )
        if answer not in ("y", "yes"):
            console.print("aborted")
            return
    response = backend.save_final_submission(config, loss)
    console.print(f"[green]final submission saved[/green] at {response.submitted_at}")


def cmd_submit_final(args: argparse.Namespace) -> None:
    workdir = _workdir(args)
    _submit_final(
        make_backend(args.backend, workdir=workdir), workdir, assume_yes=args.yes
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Simulated backend only: score fit/final_submission.json against the hidden truth."""
    from cs336_scaling.scaling_laws.evaluate import evaluate_final, summarise_for_json
    from cs336_scaling.scaling_laws.simulate import SimulatorTruth

    if args.backend != "simulated":
        raise SystemExit("evaluate only makes sense with --backend simulated")
    workdir = _workdir(args)
    payload = json.loads((workdir / "fit" / "final_submission.json").read_text())
    evaluation = evaluate_final(
        SimulatorTruth(),
        TrainingConfig.model_validate(payload["training_config"]),
        float(payload["predicted_final_loss"]),
    )
    console.print("[bold]evaluation against the simulator's hidden truth[/bold]")
    console.print(evaluation.describe())
    (workdir / "fit" / "evaluation.json").write_text(
        json.dumps(summarise_for_json(evaluation), indent=2) + "\n"
    )


def cmd_all(args: argparse.Namespace) -> None:
    workdir = _workdir(args)
    backend = make_backend(args.backend, workdir=workdir)
    state_path = _state_path(workdir)
    state = PipelineState.load_or_create(state_path, args.backend)
    planner = _planner(args)
    poll = 0.0 if args.backend == "simulated" else args.poll_interval

    def plan_and_run(stage: str, **kwargs) -> None:
        added = plan_stage(stage, state, planner, fast=args.fast, **kwargs)
        state.save(state_path)
        if added or state.pending():
            run_pending(state, backend, state_path, poll_interval_seconds=poll)

    for stage in ("probe", "lr_sweep"):
        if not state.stage(stage):  # resumable: skip stages that were already planned
            plan_and_run(stage)
        else:
            run_pending(state, backend, state_path, poll_interval_seconds=poll)
    # Iso-time tiers one at a time, each centred on the fit so far, then bracket the minimum.
    for budget in planner.iso_time_budgets:
        if budget not in plan_mod.iso_time_profile_ks(state):
            console.rule(f"iso-time tier T={budget:.0f}s")
            plan_and_run("iso_time", budgets=[budget])
        for _ in range(planner.max_bracket_rounds):
            if not plan_stage("bracket", state, planner, fast=args.fast, quiet=True):
                break
            state.save(state_path)
            run_pending(state, backend, state_path, poll_interval_seconds=poll)
    cmd_fit(args)
    if args.backend == "simulated" or args.yes:
        _submit_final(backend, workdir, assume_yes=True)
    if args.backend == "simulated":
        cmd_evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cs336_scaling.scaling_laws",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("isoflops", help="Problem chinchilla_isoflops")
    p.add_argument("--data", default="data/isoflops_curves.json")
    p.add_argument("--out", default="results/isoflops")
    p.set_defaults(func=cmd_isoflops)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--backend", choices=("api", "simulated"), default="api")
        p.add_argument(
            "--workdir", default=None, help="defaults to results/scaling_laws/<backend>"
        )
        p.add_argument(
            "--fast",
            action="store_true",
            help="smaller initialisation grid for the parametric fit",
        )
        p.add_argument(
            "--budget-fraction",
            type=float,
            default=None,
            help="share of the 12h budget the planner may use (default 0.92)",
        )

    def add_fit_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--final-hours", type=float, default=48.0)
        p.add_argument(
            "--safety",
            type=float,
            default=0.85,
            help="fraction of the final budget the token count is sized for",
        )
        p.add_argument(
            "--selection", choices=("parametric", "isotime"), default="parametric"
        )
        p.add_argument(
            "--bootstrap",
            type=int,
            default=20,
            help="bootstrap refits for the loss-prediction interval (0 disables)",
        )

    p = sub.add_parser("plan", help="plan a stage of runs")
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument(
        "--budget",
        nargs="*",
        default=None,
        help="iso_time only: time budget(s) in seconds to plan (default: next unplanned tier)",
    )
    add_common(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="submit planned runs and poll")
    add_common(p)
    p.add_argument("--poll-interval", type=float, default=30.0)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("fit", help="fit scaling laws and write the report")
    add_common(p)
    add_fit_options(p)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("submit-final", help="submit fit/final_submission.json")
    add_common(p)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_submit_final)

    p = sub.add_parser(
        "evaluate",
        help="(simulated) score the final submission against the hidden truth",
    )
    add_common(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("all", help="run the complete study")
    add_common(p)
    add_fit_options(p)
    p.add_argument("--poll-interval", type=float, default=30.0)
    p.add_argument(
        "--yes",
        action="store_true",
        help="also submit the final submission (api backend)",
    )
    p.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
