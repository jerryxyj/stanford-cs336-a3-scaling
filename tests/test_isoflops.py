"""Tests for problem ``chinchilla_isoflops`` (no database or network needed)."""

import json
from pathlib import Path

import numpy as np
import pytest

from cs336_scaling.scaling_laws.fit import (
    fit_chinchilla_parametric,
    fit_power_law,
    fit_saturating_power_law,
)
from cs336_scaling.scaling_laws.isoflops import (
    DEFAULT_DATA_PATH,
    IsoFlopsRun,
    fit_isoflops_scaling_laws,
    load_runs,
    profile_optima,
    run_isoflops_analysis,
)


def test_fit_power_law_recovers_exponent():
    x = np.logspace(18, 22, 12)
    law = fit_power_law(x, 3.0 * x**0.5)
    assert law.coefficient == pytest.approx(3.0, rel=1e-6)
    assert law.exponent == pytest.approx(0.5, abs=1e-9)
    assert law.metrics is not None and law.metrics.r2 == pytest.approx(1.0)


def test_fit_power_law_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_power_law([1.0], [2.0])
    with pytest.raises(ValueError):
        fit_power_law([1.0, -2.0], [2.0, 3.0])


def test_fit_saturating_power_law_recovers_floor():
    x = np.logspace(2, 6, 9)
    law = fit_saturating_power_law(x, 2.0 + 10.0 * x**-0.3)
    assert law.floor == pytest.approx(2.0, abs=0.02)
    assert law.exponent == pytest.approx(-0.3, abs=0.02)


def test_fit_chinchilla_parametric_recovers_surface():
    rng = np.random.default_rng(0)
    N = 10 ** rng.uniform(7, 10, 60)
    D = 10 ** rng.uniform(8.5, 11, 60)
    L = 1.69 + 406.4 / N**0.34 + 410.7 / D**0.28
    fit = fit_chinchilla_parametric(
        N,
        D,
        L,
        alpha_grid=(0.2, 0.5),
        beta_grid=(0.2, 0.5),
        e_grid=(0.0, 0.5),
        a_grid=(5.0, 10.0),
        b_grid=(5.0, 10.0),
    )
    assert fit.E == pytest.approx(1.69, abs=0.05)
    assert fit.alpha == pytest.approx(0.34, abs=0.03)
    assert fit.beta == pytest.approx(0.28, abs=0.03)
    assert fit.metrics is not None and fit.metrics.rmse < 1e-3
    n_opt, d_opt = fit.compute_optimal(1e23)
    # Chinchilla's own numbers for 1e23 FLOPs are roughly N ~ 1.4e10, D ~ 1.2e12.
    assert 5e9 < float(n_opt) < 3e10
    assert 5e11 < float(d_opt) < 3e12


def _synthetic_runs(a: float, b: float) -> list[IsoFlopsRun]:
    """IsoFLOP profiles whose minima follow N_opt = a C^b exactly."""
    runs = []
    for C in np.logspace(18, 21, 7):
        n_opt = a * C**b
        for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
            N = n_opt * factor
            runs.append(
                IsoFlopsRun(
                    parameters=N, compute_budget=C, final_loss=3 + np.log(factor) ** 2
                )
            )
    return runs


def test_isoflops_recovers_known_law():
    laws = fit_isoflops_scaling_laws(_synthetic_runs(0.5, 0.5))
    assert len(laws.optima) == 7
    assert laws.n_opt_law.exponent == pytest.approx(0.5, abs=1e-6)
    assert laws.n_opt_law.coefficient == pytest.approx(0.5, rel=1e-6)
    # D_opt = C / (6 N_opt) => exponent 1 - 0.5 and coefficient 1 / (6 * 0.5).
    assert laws.d_opt_law.exponent == pytest.approx(0.5, abs=1e-6)
    assert laws.d_opt_law.coefficient == pytest.approx(1 / 3, rel=1e-6)


def test_profile_optima_pick_lowest_loss_per_budget():
    runs = [
        IsoFlopsRun(1e8, 1e19, 4.0),
        IsoFlopsRun(2e8, 1e19, 3.5),
        IsoFlopsRun(4e8, 1e19, 3.7),
        IsoFlopsRun(1e9, 1e20, 3.0),
    ]
    optima = profile_optima(runs)
    assert [o.compute_budget for o in optima] == [1e19, 1e20]
    assert optima[0].n_opt == 2e8
    assert optima[0].d_opt == pytest.approx(1e19 / (6 * 2e8))


def test_handout_data_gives_plausible_laws():
    runs = load_runs(DEFAULT_DATA_PATH)
    assert len(runs) == 72
    laws = fit_isoflops_scaling_laws(runs)
    assert len(laws.optima) == 9
    # Chinchilla found N_opt ~ C^0.49, D_opt ~ C^0.51; the synthetic data should be close.
    assert 0.4 < laws.n_opt_law.exponent < 0.6
    assert 0.4 < laws.d_opt_law.exponent < 0.6
    assert laws.n_opt_law.exponent + laws.d_opt_law.exponent == pytest.approx(
        1.0, abs=1e-9
    )
    assert 1e10 < laws.predict_n_opt(1e23) < 1e12
    assert 1e11 < laws.predict_d_opt(1e23) < 1e13
    assert laws.predict_n_opt(1e24) > laws.predict_n_opt(1e23)


def test_run_isoflops_analysis_writes_outputs(tmp_path: Path, capsys):
    laws = run_isoflops_analysis(output_dir=tmp_path)
    assert (tmp_path / "model_size_scaling_law.png").exists()
    assert (tmp_path / "dataset_size_scaling_law.png").exists()
    assert (tmp_path / "isoflops_profiles.png").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["predictions"]["1e+23"]["n_opt"] == pytest.approx(
        laws.predict_n_opt(1e23)
    )
    out = capsys.readouterr().out
    assert "predicted compute-optimal model size" in out
    assert "predicted compute-optimal dataset size" in out
