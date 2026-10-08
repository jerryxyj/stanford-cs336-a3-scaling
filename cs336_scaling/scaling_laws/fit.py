"""Curve-fitting primitives shared by the scaling-law code.

* :func:`fit_power_law` fits ``y = a * x ** b`` by linear least squares in log-log space.
* :class:`ChinchillaParametricFit` fits the Hoffmann et al. (2022) "approach 3" loss surface
  ``L(N, D) = E + A / N**alpha + B / D**beta`` with a Huber loss on log-residuals and a grid
  of initialisations, exactly as described in the Chinchilla paper (Appendix D.2).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import huber, logsumexp


@dataclass(frozen=True)
class FitMetrics:
    """Goodness-of-fit summary. ``r2``/``rmse`` are computed in the space they were fit in."""

    r2: float
    rmse: float
    max_abs_error: float
    n_points: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "r2": self.r2,
            "rmse": self.rmse,
            "max_abs_error": self.max_abs_error,
            "n_points": self.n_points,
        }


def fit_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> FitMetrics:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_true_arr - y_pred_arr
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true_arr - y_true_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return FitMetrics(
        r2=r2,
        rmse=math.sqrt(ss_res / len(y_true_arr)),
        max_abs_error=float(np.max(np.abs(residuals))) if len(residuals) else 0.0,
        n_points=int(len(y_true_arr)),
    )


@dataclass(frozen=True)
class PowerLaw:
    """``y = coefficient * x ** exponent``."""

    coefficient: float
    exponent: float
    metrics: FitMetrics | None = None

    def __call__(self, x: ArrayLike) -> NDArray[np.float64]:
        return self.coefficient * np.asarray(x, dtype=float) ** self.exponent

    def inverse(self, y: ArrayLike) -> NDArray[np.float64]:
        return (np.asarray(y, dtype=float) / self.coefficient) ** (1.0 / self.exponent)

    def to_dict(self) -> dict[str, object]:
        return {
            "coefficient": self.coefficient,
            "exponent": self.exponent,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }

    def describe(self, x_name: str = "x", y_name: str = "y") -> str:
        return f"{y_name} = {self.coefficient:.4g} * {x_name}^{self.exponent:.4f}"


def fit_power_law(x: ArrayLike, y: ArrayLike) -> PowerLaw:
    """Fit ``y = a * x**b`` via ordinary least squares on ``log y = log a + b log x``.

    Fitting in log space weights relative (not absolute) errors equally across the many
    orders of magnitude spanned by compute budgets, which is the standard choice for scaling
    laws (Kaplan et al., 2020; Hoffmann et al., 2022).
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape or x_arr.ndim != 1:
        raise ValueError("x and y must be 1-d arrays of the same length")
    if len(x_arr) < 2:
        raise ValueError("need at least two points to fit a power law")
    if np.any(x_arr <= 0) or np.any(y_arr <= 0):
        raise ValueError("power-law fits require strictly positive x and y")

    log_x = np.log(x_arr)
    log_y = np.log(y_arr)
    design = np.stack([np.ones_like(log_x), log_x], axis=1)
    (intercept, slope), *_ = np.linalg.lstsq(design, log_y, rcond=None)
    metrics = fit_metrics(log_y, design @ np.array([intercept, slope]))
    return PowerLaw(
        coefficient=float(np.exp(intercept)), exponent=float(slope), metrics=metrics
    )


@dataclass(frozen=True)
class SaturatingPowerLaw:
    """``y = floor + coefficient * x ** exponent`` -- a power law with an irreducible floor.

    Used for loss-versus-compute curves, which cannot be a pure power law because the loss
    is bounded below by the entropy of the data.
    """

    floor: float
    coefficient: float
    exponent: float
    metrics: FitMetrics | None = None

    def __call__(self, x: ArrayLike) -> NDArray[np.float64]:
        return (
            self.floor + self.coefficient * np.asarray(x, dtype=float) ** self.exponent
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "floor": self.floor,
            "coefficient": self.coefficient,
            "exponent": self.exponent,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }

    def describe(self, x_name: str = "x", y_name: str = "y") -> str:
        return (
            f"{y_name} = {self.floor:.4f} + {self.coefficient:.4g} * "
            f"{x_name}^{self.exponent:.4f}"
        )


def fit_saturating_power_law(
    x: ArrayLike, y: ArrayLike, *, floor_grid: ArrayLike | None = None
) -> SaturatingPowerLaw:
    """Fit ``y = floor + a * x**b`` by profiling over the floor.

    For a fixed floor the remaining problem is an ordinary power-law fit on ``y - floor``,
    so we scan a grid of candidate floors and keep the one with the smallest squared error
    (in the original space). Requires at least three points.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 3:
        raise ValueError("need at least three points to fit a saturating power law")
    if floor_grid is None:
        floor_grid = np.linspace(0.0, float(y_arr.min()) * 0.999, 400)
    best: SaturatingPowerLaw | None = None
    best_sse = math.inf
    for floor in np.asarray(floor_grid, dtype=float):
        shifted = y_arr - floor
        if np.any(shifted <= 0):
            continue
        power_law = fit_power_law(x_arr, shifted)
        pred = floor + power_law(x_arr)
        sse = float(np.sum((pred - y_arr) ** 2))
        if sse < best_sse:
            best_sse = sse
            best = SaturatingPowerLaw(
                floor=float(floor),
                coefficient=power_law.coefficient,
                exponent=power_law.exponent,
                metrics=fit_metrics(y_arr, pred),
            )
    if best is None:
        raise ValueError("could not fit a saturating power law to the data")
    return best


@dataclass(frozen=True)
class ChinchillaParametricFit:
    """``L(N, D) = E + A / N**alpha + B / D**beta`` (Hoffmann et al., 2022, eq. 2)."""

    E: float
    A: float
    B: float
    alpha: float
    beta: float
    metrics: FitMetrics | None = None
    objective: float = field(default=float("nan"))

    def __call__(self, N: ArrayLike, D: ArrayLike) -> NDArray[np.float64]:
        N_arr = np.asarray(N, dtype=float)
        D_arr = np.asarray(D, dtype=float)
        return self.E + self.A / N_arr**self.alpha + self.B / D_arr**self.beta

    def compute_optimal(self, C: ArrayLike, flops_per_param_token: float = 6.0):
        """Closed-form compute-optimal ``(N_opt, D_opt)`` for FLOPs budget ``C``.

        Minimising ``L(N, C/(kN))`` gives ``N_opt = G (C/k)^a`` and ``D_opt = G^-1 (C/k)^b``
        with ``a = beta/(alpha+beta)``, ``b = alpha/(alpha+beta)`` and
        ``G = (alpha A / (beta B))^(1/(alpha+beta))`` (Hoffmann et al., Appendix D.3).
        """
        C_arr = np.asarray(C, dtype=float)
        a = self.beta / (self.alpha + self.beta)
        b = self.alpha / (self.alpha + self.beta)
        G = (self.alpha * self.A / (self.beta * self.B)) ** (
            1.0 / (self.alpha + self.beta)
        )
        scaled = C_arr / flops_per_param_token
        return G * scaled**a, scaled**b / G

    def to_dict(self) -> dict[str, object]:
        return {
            "E": self.E,
            "A": self.A,
            "B": self.B,
            "alpha": self.alpha,
            "beta": self.beta,
            "objective": self.objective,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }

    def describe(self) -> str:
        return (
            f"L(N, D) = {self.E:.4f} + {self.A:.4g} / N^{self.alpha:.4f} "
            f"+ {self.B:.4g} / D^{self.beta:.4f}"
        )


def _parametric_objective(
    params: NDArray[np.float64],
    log_N: NDArray[np.float64],
    log_D: NDArray[np.float64],
    log_L: NDArray[np.float64],
    delta: float,
) -> float:
    a, b, e, alpha, beta = params
    # log L_hat = LSE(a - alpha log N, b - beta log D, e)
    stacked = np.stack([a - alpha * log_N, b - beta * log_D, np.full_like(log_N, e)])
    log_pred = logsumexp(stacked, axis=0)
    return float(np.sum(huber(delta, log_pred - log_L)))


def fit_chinchilla_parametric(
    N: ArrayLike,
    D: ArrayLike,
    L: ArrayLike,
    *,
    huber_delta: float = 1e-3,
    alpha_grid: ArrayLike = (0.0, 0.5, 1.0, 1.5, 2.0),
    beta_grid: ArrayLike = (0.0, 0.5, 1.0, 1.5, 2.0),
    e_grid: ArrayLike = (-1.0, -0.5, 0.0, 0.5, 1.0),
    a_grid: ArrayLike = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
    b_grid: ArrayLike = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
) -> ChinchillaParametricFit:
    """Fit the parametric loss surface by L-BFGS from a grid of initialisations.

    Following Hoffmann et al. we optimise ``(a, b, e, alpha, beta)`` with ``A = exp(a)``,
    ``B = exp(b)``, ``E = exp(e)`` and minimise the Huber loss (``delta = 1e-3``) of the
    log-space residual ``log L_hat - log L``. The Huber loss makes the fit robust to a few
    outlying runs, and working in log space keeps very different loss magnitudes comparable.
    The default grid matches the paper; pass smaller grids for quick fits.
    """
    log_N = np.log(np.asarray(N, dtype=float))
    log_D = np.log(np.asarray(D, dtype=float))
    L_arr = np.asarray(L, dtype=float)
    log_L = np.log(L_arr)
    if not (log_N.shape == log_D.shape == log_L.shape) or log_N.ndim != 1:
        raise ValueError("N, D and L must be 1-d arrays of equal length")
    if len(log_N) < 5:
        raise ValueError(
            "need at least five runs to fit the five-parameter loss surface"
        )

    best_params: NDArray[np.float64] | None = None
    best_value = math.inf
    for alpha0, beta0, e0, a0, b0 in itertools.product(
        np.asarray(alpha_grid, dtype=float),
        np.asarray(beta_grid, dtype=float),
        np.asarray(e_grid, dtype=float),
        np.asarray(a_grid, dtype=float),
        np.asarray(b_grid, dtype=float),
    ):
        result = minimize(
            _parametric_objective,
            x0=np.array([a0, b0, e0, alpha0, beta0]),
            args=(log_N, log_D, log_L, huber_delta),
            method="L-BFGS-B",
            # Keep the exponents non-negative so the surface decreases in N and D.
            bounds=[(None, None), (None, None), (None, None), (0.0, None), (0.0, None)],
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best_params = result.x
    assert best_params is not None
    a, b, e, alpha, beta = best_params
    fit = ChinchillaParametricFit(
        E=float(np.exp(e)),
        A=float(np.exp(a)),
        B=float(np.exp(b)),
        alpha=float(alpha),
        beta=float(beta),
        objective=best_value,
    )
    pred = fit(np.exp(log_N), np.exp(log_D))
    return replace(fit, metrics=fit_metrics(L_arr, pred))
