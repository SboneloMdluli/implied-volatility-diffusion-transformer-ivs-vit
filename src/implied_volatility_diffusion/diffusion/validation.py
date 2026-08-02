"""Validation helpers for forecast regeneration and reporting visuals."""

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SurfaceGenerationDiagnostics:
    """Metadata for a regeneration run."""

    base_seed: int
    accepted_seed: int
    attempts: int
    arbitrage_free: bool
    worst_butterfly: float
    worst_calendar: float
    worst_bound: float


def arbitrage_violation_score(report: object) -> float:
    """Return a scalar violation score; 0 means arbitrage-free."""
    return float(
        max(
            -min(0.0, float(getattr(report, "worst_butterfly"))),
            -min(0.0, float(getattr(report, "worst_calendar"))),
            -min(0.0, float(getattr(report, "worst_bound"))),
        )
    )


def regenerate_until_arb_free(
    *,
    generate_surface: Callable[[int], np.ndarray],
    check_arbitrage: Callable[[np.ndarray], object],
    base_seed: int,
    max_tries: int,
) -> tuple[np.ndarray, SurfaceGenerationDiagnostics]:
    """Retry generation with incremented seeds until arbitrage-free.

    If no sample is arbitrage-free within ``max_tries``, the least-violating
    candidate is returned so callers can still inspect diagnostics.
    """
    if max_tries <= 0:
        raise ValueError("max_tries must be positive")

    best_surface: np.ndarray | None = None
    best_report: object | None = None
    best_seed = int(base_seed)
    best_attempt = 1
    best_score = float("inf")

    for attempt in range(1, max_tries + 1):
        seed = int(base_seed + attempt - 1)
        candidate = np.asarray(generate_surface(seed), dtype=np.float64)
        report = check_arbitrage(candidate)
        score = arbitrage_violation_score(report)

        if score < best_score:
            best_score = score
            best_surface = candidate
            best_report = report
            best_seed = seed
            best_attempt = attempt

        if bool(getattr(report, "arbitrage_free")):
            break

    assert best_surface is not None and best_report is not None
    diag = SurfaceGenerationDiagnostics(
        base_seed=int(base_seed),
        accepted_seed=int(best_seed),
        attempts=int(best_attempt),
        arbitrage_free=bool(getattr(best_report, "arbitrage_free")),
        worst_butterfly=float(getattr(best_report, "worst_butterfly")),
        worst_calendar=float(getattr(best_report, "worst_calendar")),
        worst_bound=float(getattr(best_report, "worst_bound")),
    )
    return best_surface, diag


def plot_surface_comparison(
    *,
    input_surface: np.ndarray,
    ground_truth_surface: np.ndarray,
    forecast_surface: np.ndarray,
    log_moneyness: np.ndarray,
    tau: np.ndarray,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Show input/ground-truth/forecast/difference as separate heatmaps."""
    inp = np.asarray(input_surface, dtype=float)
    gt = np.asarray(ground_truth_surface, dtype=float)
    pred = np.asarray(forecast_surface, dtype=float)
    diff = pred - gt

    vmin = float(np.nanmin([inp.min(), gt.min(), pred.min()]))
    vmax = float(np.nanmax([inp.max(), gt.max(), pred.max()]))
    dmax = float(np.nanmax(np.abs(diff)))

    fig, axes = plt.subplots(1, 4, figsize=(20.0, 5.0))
    mats = (inp, gt, pred, diff)
    names = ("Input", "Ground truth", "Forecast", "Forecast - truth")
    cmaps = ("viridis", "viridis", "viridis", "coolwarm")
    limits = ((vmin, vmax), (vmin, vmax), (vmin, vmax), (-dmax, dmax))

    extent = (float(log_moneyness[0]), float(log_moneyness[-1]), float(tau[0]), float(tau[-1]))
    for ax, mat, name, cmap, (lo, hi) in zip(axes, mats, names, cmaps, limits, strict=False):
        image = ax.imshow(
            mat.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=lo,
            vmax=hi,
        )
        ax.set_title(name)
        ax.set_xlabel(r"Log-moneyness $k=\log(K/S)$")
        ax.set_ylabel("Tenor (years)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=12, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.14, top=0.88, wspace=0.42)
    return fig, axes


def plot_performance_metrics(
    metrics: pd.DataFrame,
    *,
    group_col: str = "path_type",
    rmse_col: str = "rmse",
    mae_col: str = "mae",
    arb_col: str = "strict_arb_free",
    attempts_col: str = "generation_attempts",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot model quality and feasibility metrics in a dedicated figure."""
    required = {group_col, rmse_col, mae_col, arb_col, attempts_col}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"metrics dataframe missing columns: {missing}")

    grouped = metrics.groupby(group_col, dropna=False)
    rmse_mean = grouped[rmse_col].mean()
    mae_mean = grouped[mae_col].mean()
    arb_pass_rate = grouped[arb_col].mean()
    attempt_mean = grouped[attempts_col].mean()

    labels = [str(x) for x in rmse_mean.index]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    width = 0.38
    axes[0].bar(x - width / 2, rmse_mean.values, width=width, label="RMSE", color="#1f77b4")
    axes[0].bar(x + width / 2, mae_mean.values, width=width, label="MAE", color="#2ca02c")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Error metrics by path type")
    axes[0].set_ylabel("Volatility error")
    axes[0].legend(frameon=False)

    axes[1].bar(x, arb_pass_rate.values * 100.0, color="#9467bd")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 100.0)
    axes[1].set_title("Arbitrage pass-rate")
    axes[1].set_ylabel("Pass-rate (%)")

    axes[2].bar(x, attempt_mean.values, color="#ff7f0e")
    axes[2].set_xticks(x, labels)
    axes[2].set_title("Mean generation attempts")
    axes[2].set_ylabel("Attempts")

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.90, wspace=0.32)
    return fig, axes


__all__ = [
    "SurfaceGenerationDiagnostics",
    "arbitrage_violation_score",
    "plot_performance_metrics",
    "plot_surface_comparison",
    "regenerate_until_arb_free",
]
