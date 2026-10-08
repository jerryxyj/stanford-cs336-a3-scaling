"""Tests for the ``scaling_laws`` pipeline against the offline simulator (no database/network)."""

import json
from pathlib import Path

import numpy as np
import pytest

from cs336_scaling.scaling_laws import analysis as analysis_mod
from cs336_scaling.scaling_laws import plan as plan_mod
from cs336_scaling.scaling_laws.backends import (
    DuplicateExperimentError,
    InsufficientBudgetError,
    SimulatedBackend,
)
from cs336_scaling.scaling_laws.cli import main
from cs336_scaling.scaling_laws.evaluate import evaluate_final
from cs336_scaling.scaling_laws.model_shapes import (
    count_parameters,
    flops_per_token,
    kaplan_non_embedding_estimate,
    ladder_shape,
    make_architecture_config,
)
from cs336_scaling.scaling_laws.runner import run_pending
from cs336_scaling.scaling_laws.simulate import SimulatorTruth
from cs336_scaling.scaling_laws.state import ExperimentRecord, PipelineState
from cs336_scaling.scaling_laws.throughput import (
    ThroughputObservation,
    fit_throughput_model,
    prior_throughput_model,
)
from cs336_scaling.training.training_config import TrainingConfig


# -- model shapes ----------------------------------------------------------------------
def test_parameter_count_matches_real_model():
    jax = pytest.importorskip("jax")
    import equinox.nn as nn

    from cs336_scaling.training.model.basic_model import BasicCausalLM
    from cs336_scaling.training.model.jax_utils import count_params

    for config in (
        make_architecture_config(num_layers=2, hidden_size=128, dtype="float32"),
        make_architecture_config(
            num_layers=3, hidden_size=192, tie_word_embeddings=True
        ),
        ladder_shape(4),
    ):
        model, _ = nn.make_with_state(BasicCausalLM)(config, key=jax.random.PRNGKey(0))
        assert count_parameters(config).total == count_params(model), config


def test_handout_example_shape_counts():
    config = make_architecture_config(
        num_layers=9, hidden_size=448, intermediate_size=1280
    )
    counts = count_parameters(config)
    assert counts.total == 51_389_888
    assert counts.embedding == counts.lm_head == 448 * 32_000
    assert kaplan_non_embedding_estimate(config) == 12 * 9 * 448**2
    # 12 n d^2 is a slight under-estimate of the SwiGLU + QK-norm layer here.
    assert 0.9 < kaplan_non_embedding_estimate(config) / counts.non_embedding < 1.0
    assert flops_per_token(config) > 6 * counts.non_embedding


def test_ladder_is_monotone_and_valid():
    sizes = [
        count_parameters(ladder_shape(k)).non_embedding
        for k in plan_mod.DEFAULT_LADDER_KS
    ]
    assert sizes == sorted(sizes)
    for k in plan_mod.DEFAULT_LADDER_KS:
        cfg = ladder_shape(k)
        assert cfg.hidden_size == cfg.num_attention_heads * cfg.head_dim
        assert cfg.intermediate_size >= cfg.hidden_size


# -- throughput ------------------------------------------------------------------------
def test_throughput_fit_recovers_latency_plus_compute_model():
    overhead, seconds_per_flop = 1.2e-6, 1.0 / (2.25e15 * 0.45)
    obs = [
        ThroughputObservation(
            name=f"o{i}",
            flops_per_token=fpt,
            tokens_per_second=1 / (overhead + seconds_per_flop * fpt),
            batch_tokens=65536,
        )
        for i, fpt in enumerate(np.logspace(7.8, 10, 8))
    ]
    model = fit_throughput_model(obs)
    assert model.overhead_seconds_per_token == pytest.approx(overhead, rel=1e-3)
    assert model.asymptotic_mfu == pytest.approx(0.45, rel=1e-3)
    assert model.metrics is not None and model.metrics.rmse < 1e-6


def test_throughput_fit_caps_asymptotic_mfu():
    seconds_per_flop = 1.0 / (2.25e15 * 0.9)  # unphysically fast
    obs = [
        ThroughputObservation(
            name=f"o{i}",
            flops_per_token=fpt,
            tokens_per_second=1 / (1e-6 + seconds_per_flop * fpt),
            batch_tokens=65536,
        )
        for i, fpt in enumerate(np.logspace(8, 9.5, 5))
    ]
    assert fit_throughput_model(obs).asymptotic_mfu <= 0.55 + 1e-9


def test_prior_throughput_reproduces_handout_example():
    prior = prior_throughput_model()
    example = make_architecture_config(
        num_layers=9, hidden_size=448, intermediate_size=1280
    )
    assert prior.tokens_per_second(example) == pytest.approx(8.2e5, rel=0.02)


# -- planning --------------------------------------------------------------------------
def test_make_training_config_respects_api_divisibility():
    arch = ladder_shape(8)
    cfg = plan_mod.make_training_config(
        arch,
        batch_size=96,
        peak_lr=1.234567e-3,
        target_tokens=123_456_789,
        max_runtime_seconds=300,
        n_evals=8,
    )
    assert cfg.total_train_tokens % (512 * 96 * 8) == 0
    assert cfg.total_optimizer_steps % cfg.n_evals == 0
    assert abs(cfg.total_train_tokens - 123_456_789) < 512 * 96 * 8
    assert cfg.optimizer_config.lr_scheduler.peak_value == pytest.approx(1.23e-3)


def test_batch_size_rule_is_power_of_two_and_clipped():
    for N in (1e6, 1e7, 1e8, 1e9, 1e10):
        b = plan_mod.batch_size_rule(N)
        assert b & (b - 1) == 0 and 32 <= b <= 1024
    assert plan_mod.batch_size_rule(1e7) == 64
    assert plan_mod.batch_size_rule(1e9) <= plan_mod.batch_size_rule(1e10)


def test_staged_plan_fits_budget_and_validates():
    planner = plan_mod.PlannerConfig()
    throughput = prior_throughput_model()
    state = PipelineState(backend="simulated")
    probes = plan_mod.plan_probes(planner, throughput)
    sweep = plan_mod.plan_lr_sweep(planner, throughput)
    iso = plan_mod.plan_iso_time(planner, throughput, lr_rule=plan_mod.prior_lr_rule)
    all_records = probes + sweep + iso
    assert len(probes) == len(planner.probe_ks)
    assert len(sweep) == len(planner.lr_sweep_ks) * len(planner.lr_sweep_multipliers)
    assert len(iso) == sum(planner.iso_time_shapes_per_budget)
    assert len({r.unique_id for r in all_records}) == len(all_records)
    for r in all_records:
        TrainingConfig.model_validate(r.training_config.model_dump(mode="json"))
    check = plan_mod.check_budget(state, all_records, planner, throughput)
    assert check.fits, check.describe()
    assert plan_mod.add_to_state(state, all_records) == all_records
    assert plan_mod.add_to_state(state, all_records) == []  # idempotent


def test_shapes_around_clips_to_ladder():
    ladder = (4, 6, 8, 10, 12)
    assert plan_mod.shapes_around(ladder, 8, 3) == [6, 8, 10]
    assert plan_mod.shapes_around(ladder, 4, 3) == [4, 6, 8]
    assert plan_mod.shapes_around(ladder, 12, 4) == [6, 8, 10, 12]


# -- simulator and backend ---------------------------------------------------------------
def _config(
    k: int = 6, tokens: int = 2**24, max_runtime: float = 300.0
) -> TrainingConfig:
    return plan_mod.make_training_config(
        ladder_shape(k),
        batch_size=64,
        peak_lr=2e-3,
        target_tokens=tokens,
        max_runtime_seconds=max_runtime,
        n_evals=4,
    )


def test_simulated_backend_matches_api_semantics(tmp_path: Path):
    backend = SimulatedBackend(
        persist_path=tmp_path / "server.json", total_budget_seconds=1000.0
    )
    cfg = _config()
    response = backend.submit(cfg)
    assert response.budget_summary.remaining_seconds == 700.0
    with pytest.raises(DuplicateExperimentError):
        backend.submit(cfg)
    with pytest.raises(InsufficientBudgetError):
        backend.submit(_config(k=8, max_runtime=800.0))
    experiment = backend.get_experiment(response.experiment_id)
    assert experiment.status.status_type == "completed"
    assert len(experiment.status.val_losses) == 4
    assert experiment.status.val_losses == sorted(
        experiment.status.val_losses, reverse=True
    )
    budget = backend.get_budget()
    assert 1.0 <= budget.used_seconds < 300.0  # refunded down to the actual runtime
    # persisted across instances
    reloaded = SimulatedBackend(
        persist_path=tmp_path / "server.json", total_budget_seconds=1000.0
    )
    assert reloaded.get_budget() == budget


def test_simulator_times_out_long_runs():
    outcome = SimulatorTruth().simulate(_config(k=16, tokens=2**31, max_runtime=10.0))
    assert not outcome.completed
    assert outcome.used_runtime_seconds > 10.0
    assert outcome.val_losses == []


def test_simulator_loss_is_sensible():
    truth = SimulatorTruth()
    N = 1e8
    best = truth.loss(N, 2e9, batch_tokens=2**16, peak_lr=truth.lr_opt(N))
    assert truth.loss(N, 2e9, batch_tokens=2**16, peak_lr=4 * truth.lr_opt(N)) > best
    assert truth.loss(N, 2e9, batch_tokens=2**16, peak_lr=truth.lr_opt(N) / 4) > best
    assert truth.loss(N, 4e9, batch_tokens=2**16, peak_lr=truth.lr_opt(N)) < best
    assert truth.loss(N, 2e9, batch_tokens=2**19, peak_lr=truth.lr_opt(N)) > best


# -- runner and state ------------------------------------------------------------------
def test_runner_is_resumable(tmp_path: Path):
    backend = SimulatedBackend()
    state_path = tmp_path / "state.json"
    state = PipelineState(backend="simulated")
    state.add(
        ExperimentRecord(
            name="a", stage="probe", training_config=_config(6), ladder_k=6
        )
    )
    state.add(
        ExperimentRecord(
            name="b", stage="probe", training_config=_config(8), ladder_k=8
        )
    )
    run_pending(
        state,
        backend,
        state_path,
        poll_interval_seconds=0,
        sleep=lambda _: None,
        quiet=True,
    )
    assert all(r.status == "completed" for r in state.experiments)
    assert len(backend.list_experiments()) == 2

    # A fresh process that lost its state re-plans the same configs: nothing is resubmitted.
    fresh = PipelineState(backend="simulated")
    fresh.add(
        ExperimentRecord(
            name="a", stage="probe", training_config=_config(6), ladder_k=6
        )
    )
    run_pending(
        fresh,
        backend,
        state_path,
        poll_interval_seconds=0,
        sleep=lambda _: None,
        quiet=True,
    )
    assert fresh.experiments[0].status == "completed"
    assert len(backend.list_experiments()) == 2

    reloaded = PipelineState.load(state_path)
    assert [r.name for r in reloaded.experiments] == ["a"]
    with pytest.raises(ValueError):
        PipelineState.load_or_create(state_path, "api")


def test_throughput_observation_from_timeout():
    cfg = _config(k=16, tokens=2**31, max_runtime=10.0)
    record = ExperimentRecord(
        name="t",
        stage="probe",
        training_config=cfg,
        status="failed",
        failure="timeout",
        used_runtime_seconds=10.0,
        val_losses=[5.0, 4.5],
    )
    obs = record.throughput_observation()
    assert obs is not None
    assert obs.tokens_per_second == pytest.approx(2 * cfg.eval_every_tokens / 10.0)
    assert (
        ExperimentRecord(
            name="u",
            stage="probe",
            training_config=cfg,
            status="failed",
            failure="boom",
            used_runtime_seconds=3.0,
        ).throughput_observation()
        is None
    )


# -- analysis --------------------------------------------------------------------------
def _sweep_records(
    lr_by_k: dict[int, list[tuple[float, float]]],
) -> list[ExperimentRecord]:
    records = []
    for k, points in lr_by_k.items():
        for lr, loss in points:
            cfg = plan_mod.make_training_config(
                ladder_shape(k),
                batch_size=64,
                peak_lr=lr,
                target_tokens=2**26,
                max_runtime_seconds=100,
                n_evals=4,
            )
            records.append(
                ExperimentRecord(
                    name=f"lr/k{k}/{lr}",
                    stage="lr_sweep",
                    training_config=cfg,
                    ladder_k=k,
                    status="completed",
                    used_runtime_seconds=50.0,
                    val_losses=[loss],
                )
            )
    return records


def test_lr_sweep_fit_finds_parabola_minimum_and_clamps_exponent():
    # Minimum at 2e-3 for k=6 and 1e-3 for k=10: a decreasing LR with N.
    records = _sweep_records(
        {
            6: [(5e-4, 3.4), (1e-3, 3.2), (2e-3, 3.1), (4e-3, 3.2), (8e-3, 3.4)],
            10: [(2.5e-4, 3.0), (5e-4, 2.8), (1e-3, 2.7), (2e-3, 2.8), (4e-3, 3.0)],
        }
    )
    fit = analysis_mod.fit_lr_sweep(records)
    assert fit is not None
    assert fit.scales[0].lr_opt == pytest.approx(2e-3, rel=0.05)
    assert fit.scales[1].lr_opt == pytest.approx(1e-3, rel=0.05)
    assert -0.5 <= fit.law.exponent < 0
    # A noisy sweep suggesting a *larger* LR for bigger models is clamped to exponent 0.
    noisy = _sweep_records(
        {
            6: [(5e-4, 3.4), (1e-3, 3.2), (2e-3, 3.1), (4e-3, 3.2), (8e-3, 3.4)],
            10: [(2.5e-4, 3.0), (5e-4, 2.9), (1e-3, 2.8), (2e-3, 2.7), (4e-3, 2.69)],
        }
    )
    clamped = analysis_mod.fit_lr_sweep(noisy)
    assert clamped is not None and clamped.law.exponent == 0.0
    assert clamped.rule(1e10) <= max(s.lr_opt for s in clamped.scales)


# -- end to end ------------------------------------------------------------------------
def test_full_simulated_study(tmp_path: Path):
    workdir = tmp_path / "study"
    main(
        [
            "all",
            "--backend",
            "simulated",
            "--fast",
            "--bootstrap",
            "4",
            "--workdir",
            str(workdir),
        ]
    )

    state = PipelineState.load(workdir / "state.json")
    assert (
        state.completed("probe")
        and state.completed("lr_sweep")
        and state.completed("iso_time")
    )
    charged = sum(
        min(max(1.0, r.used_runtime_seconds), r.training_config.max_runtime_seconds)
        for r in state.experiments
        if r.is_finished
    )
    assert charged <= 12 * 3600, "exceeded the 12 B200-hour scaling-law budget"
    profiles = analysis_mod.iso_time_profiles(state.experiments)
    assert len(profiles) == len(plan_mod.PlannerConfig().iso_time_budgets)
    for p in profiles:  # every profile brackets its minimum
        ks = [r.ladder_k for r in p.runs]
        assert min(ks) < p.best.ladder_k < max(ks), p.to_dict()

    for name in (
        "report.md",
        "fit_summary.json",
        "final_submission.json",
        "evaluation.json",
        "throughput.png",
        "lr_sweep.png",
        "iso_time_profiles.png",
        "iso_time_scaling.png",
        "parametric_fit.png",
    ):
        assert (workdir / "fit" / name).exists(), name

    submission = json.loads((workdir / "fit" / "final_submission.json").read_text())
    config = TrainingConfig.model_validate(submission["training_config"])
    assert config.max_runtime_seconds == 48 * 3600
    assert 2.0 < submission["predicted_final_loss"] < 3.5
    assert 2e8 < count_parameters(config.architecture_config).non_embedding < 5e9

    evaluation = evaluate_final(
        SimulatorTruth(), config, submission["predicted_final_loss"]
    )
    assert evaluation.completed, "the final run must not time out"
    assert evaluation.actual_seconds < config.max_runtime_seconds
    assert evaluation.actual_loss is not None
    assert abs(evaluation.actual_loss - submission["predicted_final_loss"]) < 0.15
    assert evaluation.regret is not None and evaluation.regret < 0.1

    # Re-running is a no-op that still produces the same submission.
    main(
        [
            "all",
            "--backend",
            "simulated",
            "--fast",
            "--bootstrap",
            "4",
            "--workdir",
            str(workdir),
        ]
    )
    assert (
        json.loads((workdir / "fit" / "final_submission.json").read_text())
        == submission
    )
