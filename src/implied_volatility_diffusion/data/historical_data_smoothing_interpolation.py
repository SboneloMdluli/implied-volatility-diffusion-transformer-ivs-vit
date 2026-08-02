"""Historical IV smoothing, interpolation, and diagnostic plotting utilities."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy.interpolate import griddata, interp1d

from implied_volatility_diffusion.core.normalization import SurfaceNormalizer
from implied_volatility_diffusion.core.unified_grid import (
    UnifiedGrid,
    resample_to_unified_grid,
)


def gaussian_kernel_2d(
    x: np.ndarray,
    y: np.ndarray,
    h1: float,
    h2: float,
) -> np.ndarray:
    """2D Gaussian kernel weights for points (x, y) with bandwidths h1, h2."""
    return (np.exp(-x * x / (2 * h1)) * np.exp(-y * y / (2 * h2))) / (2 * np.pi)


def interpolate_and_extrapolate(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """Interpolate/extrapolate y values at x_new based on (x_data, y_data)."""
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length.")
    if len(x_data) < 2:
        raise ValueError("Need at least two points for interpolation.")

    idx = np.argsort(x_data)
    x_sorted = x_data[idx]
    y_sorted = y_data[idx]

    f = interp1d(
        x_sorted,
        y_sorted,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    return f(x_new)


def interpolate_surface(
    surface_in: np.ndarray,
    x_in: np.ndarray,
    tau_in: np.ndarray,
    x_out: np.ndarray,
    tau_out: np.ndarray,
) -> np.ndarray:
    """Interpolate/extrapolate surface values from (x_in, tau_in) grid to (x_out, tau_out) grid."""
    tmp = np.zeros((len(x_out), len(tau_in)), dtype=float)
    for j in range(len(tau_in)):
        tmp[:, j] = interpolate_and_extrapolate(x_in, surface_in[:, j], x_out)

    out = np.zeros((len(x_out), len(tau_out)), dtype=float)
    for i in range(len(x_out)):
        out[i, :] = interpolate_and_extrapolate(tau_in, tmp[i, :], tau_out)

    return out


def load_cleaned_data(path: str | Path) -> pd.DataFrame:
    """Load cleaned option data from parquet or CSV."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if "quote_date" not in df.columns:
        raise ValueError("Expected 'quote_date' column in the data file.")

    df = df.copy()
    df["quote_date"] = pd.to_datetime(df["quote_date"], errors="coerce")

    if "expire_date" in df.columns:
        df["expire_date"] = pd.to_datetime(df["expire_date"], errors="coerce")

    return df


def filter_day_for_surface(
    day_df: pd.DataFrame,
    k_range: tuple[float, float] = (-0.5, 0.5),
    tau_range: tuple[float, float] = (0.01, 2.0),
    iv_range: tuple[float, float] = (0.01, 2.0),
) -> pd.DataFrame:
    """Filter day_df for valid points in specified (k, tau, iv) ranges."""
    required_cols = {"k", "tau", "iv", "vega"}
    missing_cols = [c for c in required_cols if c not in day_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in day_df: {missing_cols}")

    out = day_df.copy()
    out = out.dropna(subset=["quote_date", "k", "tau", "iv", "vega"]).copy()
    out = out[
        out["k"].between(*k_range) & out["tau"].between(*tau_range) & out["iv"].between(*iv_range) & (out["vega"] > 0)
    ].copy()
    return out


def build_weight_for_smoothing(
    day_df: pd.DataFrame,
    weight_col: str = "vega",
    clip_upper: float | None = None,
    sqrt_weight: bool = False,
) -> np.ndarray:
    """Build weight array for smoothing based on specified column, with optional clipping and sqrt."""
    if weight_col not in day_df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in day_df.")
    w = day_df[weight_col].to_numpy(dtype=float)

    if sqrt_weight:
        w = np.sqrt(np.maximum(w, 0))

    if clip_upper is not None:
        w = np.clip(w, 0, clip_upper)

    return w


def smooth_surface(
    iv: np.ndarray,
    x_in: np.ndarray,
    tau_in: np.ndarray,
    weights: np.ndarray,
    x_grid: np.ndarray,
    tau_grid: np.ndarray,
    h1: float = 0.01,
    h2: float = 0.10,
    min_mass: float = 1e-8,
) -> np.ndarray:
    """Smooth IV surface using weighted Gaussian kernel."""
    iv = np.asarray(iv, dtype=float)
    x_in = np.asarray(x_in, dtype=float)
    tau_in = np.asarray(tau_in, dtype=float)
    weights = np.asarray(weights, dtype=float)

    Z = np.full((len(x_grid), len(tau_grid)), np.nan, dtype=float)
    values = iv

    for i, x0 in enumerate(x_grid):
        dx = x0 - x_in
        for j, tau0 in enumerate(tau_grid):
            dt = tau0 - tau_in
            kernel_weights = gaussian_kernel_2d(dx, dt, h1, h2) * weights
            mass = np.sum(kernel_weights)
            if mass < min_mass:
                continue

            z = np.sum(kernel_weights * values) / mass
            Z[i, j] = z
    return Z


def build_kernel_surface(
    day_df: pd.DataFrame,
    x_col: str = "k",
    weight_col: str = "vega",
    x_grid: np.ndarray | None = None,
    tau_grid: np.ndarray | None = None,
    h1: float = 0.01,
    h2: float = 0.10,
    clip_upper: float | None = None,
    sqrt_weight: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build smoothed IV surface for a day using Gaussian kernel."""
    if x_grid is None or tau_grid is None:
        raise ValueError("x_grid and tau_grid must be provided.")
    if x_col not in day_df.columns:
        raise ValueError(f"x-axis column not found: {x_col}")

    day_sub = filter_day_for_surface(day_df)

    w = build_weight_for_smoothing(
        day_sub,
        weight_col=weight_col,
        clip_upper=clip_upper,
        sqrt_weight=sqrt_weight,
    )

    Z = smooth_surface(
        iv=day_sub["iv"].values,
        x_in=day_sub[x_col].values,
        tau_in=day_sub["tau"].values,
        weights=w,
        x_grid=x_grid,
        tau_grid=tau_grid,
        h1=h1,
        h2=h2,
    )

    return Z, day_sub


## Plot functions:


def plot_smile(
    x_grid: np.ndarray,
    tau_grid: np.ndarray,
    Z: np.ndarray,
    sample_date: pd.Timestamp,
    title: str,
    num_sections: int = 5,
) -> None:
    """Plot volatility smiles at selected maturities for one date."""
    j_idx = [round(i * (len(tau_grid) - 1) / (num_sections - 1)) for i in range(num_sections)]

    fig, ax = plt.subplots(figsize=(8, 5))
    for j in j_idx:
        ax.plot(
            x_grid,
            Z[:, j],
            marker="o",
            ms=3,
            lw=1.2,
            label=rf"$\tau={tau_grid[j]:.4f}$",
        )

    ax.set_xlabel(r"Log-moneyness $k=\log(K/S)$")
    ax.set_ylabel(r"Implied volatility $\sigma_{\mathrm{imp}}$")
    ax.set_title(f"{title} ({sample_date.date()})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()


def plot_surface_3d(
    KGRID: np.ndarray,
    TAUGRID: np.ndarray,
    Z: np.ndarray,
    sample_date: pd.Timestamp,
    title: str = "Implied Volatility Surface",
    cmap: str = "turbo",
    elev: float = 25,
    azim: float = 235,
) -> None:
    """3D surface plot using the existing project style.

    Parameters
    ----------
    KGRID : np.ndarray
        Meshgrid for x-axis (log-moneyness k)
    TAUGRID : np.ndarray
        Meshgrid for tau
    Z : np.ndarray
        IV surface values
    sample_date : pd.Timestamp
        Date for title
    title : str
        Plot title
    cmap : str
        Colormap
    elev : float
        Elevation angle
    azim : float
        Azimuth angle
    """
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        KGRID,
        TAUGRID,
        Z,
        cmap=cmap,
        edgecolor="none",
        alpha=0.92,
        antialiased=True,
    )

    ax.set_xlabel("Log-moneyness k = log(K/S)")
    ax.set_ylabel("Time to maturity (years)")
    ax.set_zlabel("Implied vol")

    ax.set_title(f"{title} ({sample_date})")

    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    cax = fig.add_axes([0.91, 0.2, 0.022, 0.58])
    fig.colorbar(surf, cax=cax, label=r"$\sigma_{\mathrm{imp}}$")

    plt.show()


def fill_surface_interp_extrap(
    Z: np.ndarray,
    x_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> np.ndarray:
    """Fill NaNs in a 2D IV surface via interior interpolation then wing extrapolation.

    Strategy:
      1. Collect every finite pixel as scattered ``(x, tau)`` samples.
      2. Fill NaN cells inside the convex hull with 2D linear interpolation.
      3. Fill any remaining NaN cells (outside the convex hull) with nearest-
         neighbour extrapolation so the entire rectangular grid is populated.

    Returns a copy; never mutates the input.
    """
    Z = np.asarray(Z, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    if Z.shape != (x_grid.size, tau_grid.size):
        raise ValueError(
            f"Z shape {Z.shape} != (len(x_grid), len(tau_grid)) = ({x_grid.size}, {tau_grid.size})",
        )

    XI, TI = np.meshgrid(x_grid, tau_grid, indexing="ij")
    finite = np.isfinite(Z)
    out = Z.copy()

    if not finite.any():
        return out
    if finite.all():
        return out

    src_pts = np.column_stack([XI[finite], TI[finite]])
    src_vals = Z[finite]
    nan_mask = ~finite
    tgt_pts = np.column_stack([XI[nan_mask], TI[nan_mask]])

    lin = griddata(src_pts, src_vals, tgt_pts, method="linear")
    still_nan = ~np.isfinite(lin)
    if still_nan.any():
        near = griddata(src_pts, src_vals, tgt_pts[still_nan], method="nearest")
        lin[still_nan] = near
    out[nan_mask] = lin
    return out


@dataclass(frozen=True)
class HistoricSurfaceStages:
    """Intermediate surfaces produced by :func:`build_historical_surface_pipeline`.

    Attributes:
        sample_date: The quote date being processed.
        day_sub: The filtered quote rows actually used for smoothing.
        x_grid_smooth: Log-moneyness axis of the kernel smoothing grid.
        tau_grid_smooth: Maturity axis of the kernel smoothing grid.
        smoothed: Kernel-smoothed IV surface on the smoothing grid (may contain NaN).
        filled: Same surface after NaN interpolation / nearest-neighbour extrapolation.
        unified_grid: The canonical :class:`UnifiedGrid` both outputs below share.
        iv_unified: Filled surface resampled onto the unified ``(k, tau)`` grid.
        iv_normalized: Per-pixel log-IV z-score of ``iv_unified`` (if a normalizer
            was provided); otherwise ``None``.
    """

    sample_date: pd.Timestamp
    day_sub: pd.DataFrame
    x_grid_smooth: np.ndarray
    tau_grid_smooth: np.ndarray
    smoothed: np.ndarray
    filled: np.ndarray
    unified_grid: UnifiedGrid
    iv_unified: np.ndarray
    iv_normalized: np.ndarray | None


def build_historical_surface_pipeline(
    day_df: pd.DataFrame,
    *,
    sample_date: pd.Timestamp | None = None,
    unified_grid: UnifiedGrid | None = None,
    x_grid_smooth: np.ndarray | None = None,
    tau_grid_smooth: np.ndarray | None = None,
    weight_col: str = "vega",
    clip_upper: float | None = 50.0,
    sqrt_weight: bool = False,
    h1: float = 0.01,
    h2: float = 0.10,
    normalizer: SurfaceNormalizer | None = None,
) -> HistoricSurfaceStages:
    """Run the full historic IV surface pipeline for a single quote day.

    Stages:
      1. Kernel-smooth raw quotes onto ``(x_grid_smooth, tau_grid_smooth)``.
      2. Interpolate interior NaNs and extrapolate the wings via nearest-neighbour.
      3. Resample the dense gap-free surface onto the canonical unified grid so
         the output shares pixels with every synthetic SABR / Heston surface.
      4. Optionally normalize to per-pixel ``log σ`` z-scores.

    Every produced surface is therefore co-registered on the unified grid and,
    when a normalizer is supplied, also on the common normalization scale.
    """
    grid = unified_grid if unified_grid is not None else UnifiedGrid.default()

    if x_grid_smooth is None:
        x_grid_smooth = np.linspace(
            float(grid.log_moneyness[0]),
            float(grid.log_moneyness[-1]),
            max(grid.log_moneyness.size, 41),
        )
    if tau_grid_smooth is None:
        tau_grid_smooth = np.linspace(
            float(grid.tau[0]),
            float(grid.tau[-1]),
            max(grid.tau.size, 40),
        )
    x_grid_smooth = np.asarray(x_grid_smooth, dtype=float)
    tau_grid_smooth = np.asarray(tau_grid_smooth, dtype=float)

    if sample_date is None:
        if "quote_date" not in day_df.columns:
            raise ValueError("day_df must have a 'quote_date' column or an explicit sample_date")
        uniq = pd.to_datetime(day_df["quote_date"]).dropna().unique()
        if len(uniq) != 1:
            raise ValueError(
                "day_df spans multiple quote_dates; pass sample_date or pre-filter to one day",
            )
        sample_date = pd.Timestamp(uniq[0])
    else:
        sample_date = pd.Timestamp(sample_date)

    smoothed, day_sub = build_kernel_surface(
        day_df,
        x_col="k",
        weight_col=weight_col,
        x_grid=x_grid_smooth,
        tau_grid=tau_grid_smooth,
        h1=h1,
        h2=h2,
        clip_upper=clip_upper,
        sqrt_weight=sqrt_weight,
    )

    filled = fill_surface_interp_extrap(smoothed, x_grid_smooth, tau_grid_smooth)

    iv_unified = resample_to_unified_grid(
        filled,
        k_src=x_grid_smooth,
        tau_src=tau_grid_smooth,
        grid=grid,
    )
    if np.any(~np.isfinite(iv_unified)):
        iv_unified = fill_surface_interp_extrap(iv_unified, grid.log_moneyness, grid.tau)

    iv_normalized: np.ndarray | None = None
    if normalizer is not None:
        if tuple(normalizer.grid_shape) != tuple(grid.shape):
            raise ValueError(
                f"normalizer grid_shape {normalizer.grid_shape} != unified grid shape {grid.shape}",
            )
        iv_normalized = normalizer.transform(iv_unified)

    return HistoricSurfaceStages(
        sample_date=sample_date,
        day_sub=day_sub,
        x_grid_smooth=x_grid_smooth,
        tau_grid_smooth=tau_grid_smooth,
        smoothed=smoothed,
        filled=filled,
        unified_grid=grid,
        iv_unified=iv_unified,
        iv_normalized=iv_normalized,
    )


def _add_surface_subplot(
    fig: Figure,
    position: tuple[int, int, int],
    x_axis: np.ndarray,
    tau_axis: np.ndarray,
    Z: np.ndarray,
    *,
    title: str,
    zlabel: str,
    cmap: str = "turbo",
    elev: float = 25.0,
    azim: float = -55.0,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Draw one 3D surface panel on ``fig`` at ``position`` = ``(rows, cols, idx)``."""
    ax = fig.add_subplot(*position, projection="3d")
    XI, TI = np.meshgrid(x_axis, tau_axis, indexing="ij")
    finite_mask = np.isfinite(Z)
    if vmin is None and finite_mask.any():
        vmin = float(np.nanmin(Z))
    if vmax is None and finite_mask.any():
        vmax = float(np.nanmax(Z))
    surf = ax.plot_surface(
        XI,
        TI,
        Z,
        cmap=cmap,
        edgecolor="none",
        alpha=0.92,
        antialiased=True,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("k = log(K/S)")
    ax.set_ylabel(r"$\tau$ (yrs)")
    ax.set_zlabel(zlabel)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.05)


def _add_scatter_subplot(
    fig: Figure,
    position: tuple[int, int, int],
    day_sub: pd.DataFrame,
    *,
    title: str,
    cmap: str = "turbo",
    elev: float = 25.0,
    azim: float = -55.0,
) -> None:
    """3D scatter of raw quotes so the reader sees the data before smoothing."""
    ax = fig.add_subplot(*position, projection="3d")
    if len(day_sub) == 0:
        ax.text2D(0.5, 0.5, "no quotes", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        return
    k = day_sub["k"].to_numpy(dtype=float)
    tau = day_sub["tau"].to_numpy(dtype=float)
    iv = day_sub["iv"].to_numpy(dtype=float)
    sc = ax.scatter(k, tau, iv, c=iv, cmap=cmap, s=8, alpha=0.8)
    ax.set_xlabel("k = log(K/S)")
    ax.set_ylabel(r"$\tau$ (yrs)")
    ax.set_zlabel(r"$\sigma_{\mathrm{imp}}$")
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.05)


def make_pipeline_figure(stages: HistoricSurfaceStages) -> Figure:
    """Four-panel 3D figure: raw scatter, smoothed, filled, unified-grid surface."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Historic IV surface pipeline — {stages.sample_date.date()} (n quotes = {len(stages.day_sub)})",
        fontsize=13,
    )

    iv_vmin = float(np.nanmin(stages.iv_unified))
    iv_vmax = float(np.nanmax(stages.iv_unified))

    _add_scatter_subplot(fig, (2, 2, 1), stages.day_sub, title="1. Raw quotes (before smoothing)")
    _add_surface_subplot(
        fig,
        (2, 2, 2),
        stages.x_grid_smooth,
        stages.tau_grid_smooth,
        stages.smoothed,
        title="2. Kernel smoothed (NaN in sparse cells)",
        zlabel=r"$\sigma_{\mathrm{imp}}$",
        vmin=iv_vmin,
        vmax=iv_vmax,
    )
    _add_surface_subplot(
        fig,
        (2, 2, 3),
        stages.x_grid_smooth,
        stages.tau_grid_smooth,
        stages.filled,
        title="3. Smoothed + interp/extrap NaN fill",
        zlabel=r"$\sigma_{\mathrm{imp}}$",
        vmin=iv_vmin,
        vmax=iv_vmax,
    )
    _add_surface_subplot(
        fig,
        (2, 2, 4),
        stages.unified_grid.log_moneyness,
        stages.unified_grid.tau,
        stages.iv_unified,
        title=f"4. Resampled onto unified grid {stages.unified_grid.shape}",
        zlabel=r"$\sigma_{\mathrm{imp}}$",
        vmin=iv_vmin,
        vmax=iv_vmax,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def make_before_after_figure(stages: HistoricSurfaceStages) -> Figure:
    """Two-panel 3D figure: kernel-smoothed only vs. fully processed on unified grid."""
    fig = plt.figure(figsize=(13, 6))
    fig.suptitle(
        f"Historic IV surface: before vs. after interp/extrap + smoothing + unified-grid "
        f"resample — {stages.sample_date.date()}",
        fontsize=12,
    )
    iv_vmin = float(np.nanmin(stages.iv_unified))
    iv_vmax = float(np.nanmax(stages.iv_unified))
    _add_surface_subplot(
        fig,
        (1, 2, 1),
        stages.x_grid_smooth,
        stages.tau_grid_smooth,
        stages.smoothed,
        title="Before (kernel smoothing only, NaNs in sparse cells)",
        zlabel=r"$\sigma_{\mathrm{imp}}$",
        vmin=iv_vmin,
        vmax=iv_vmax,
    )
    _add_surface_subplot(
        fig,
        (1, 2, 2),
        stages.unified_grid.log_moneyness,
        stages.unified_grid.tau,
        stages.iv_unified,
        title="After (smoothing + interp/extrap + unified grid)",
        zlabel=r"$\sigma_{\mathrm{imp}}$",
        vmin=iv_vmin,
        vmax=iv_vmax,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def make_normalized_figure(stages: HistoricSurfaceStages) -> Figure | None:
    """Side-by-side IV (after pipeline) vs. per-pixel z-score; returns None if no normalizer."""
    if stages.iv_normalized is None:
        return None
    fig = plt.figure(figsize=(13, 6))
    fig.suptitle(
        f"Normalized historic surface — {stages.sample_date.date()} (z = 0 ⇒ per-pixel historical mean log σ)",
        fontsize=12,
    )
    _add_surface_subplot(
        fig,
        (1, 2, 1),
        stages.unified_grid.log_moneyness,
        stages.unified_grid.tau,
        stages.iv_unified,
        title="Pipeline output (raw IV on unified grid)",
        zlabel=r"$\sigma_{\mathrm{imp}}$",
    )
    _add_surface_subplot(
        fig,
        (1, 2, 2),
        stages.unified_grid.log_moneyness,
        stages.unified_grid.tau,
        stages.iv_normalized,
        title="Normalized z-score (per-pixel log σ)",
        zlabel="z",
        cmap="coolwarm",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def make_smile_slices_figure(stages: HistoricSurfaceStages, num_slices: int = 5) -> Figure:
    """Smile slices before (smoothed-only) and after (unified-grid)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        f"Smile slices — before vs. after interp/extrap + unified-grid resample ({stages.sample_date.date()})",
        fontsize=12,
    )

    def _plot_slices(ax: plt.Axes, x_axis: np.ndarray, tau_axis: np.ndarray, Z: np.ndarray, title: str) -> None:
        n = tau_axis.size
        idx = [int(round(i * (n - 1) / (num_slices - 1))) for i in range(num_slices)]
        for j in idx:
            ax.plot(x_axis, Z[:, j], marker="o", ms=3, lw=1.0, label=rf"$\tau={tau_axis[j]:.2f}$")
        ax.set_xlabel("k = log(K/S)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8, loc="best")

    _plot_slices(
        axes[0],
        stages.x_grid_smooth,
        stages.tau_grid_smooth,
        stages.smoothed,
        "Before (kernel smoothed only)",
    )
    axes[0].set_ylabel(r"$\sigma_{\mathrm{imp}}$")
    _plot_slices(
        axes[1],
        stages.unified_grid.log_moneyness,
        stages.unified_grid.tau,
        stages.iv_unified,
        "After (smoothing + interp/extrap + unified grid)",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def _surface_comparison_scales(
    stages: HistoricSurfaceStages,
) -> dict[str, float]:
    """Shared IV / z colour scales and summary stats for the 3-panel figures.

    The scatter panel is visually anchored to the **smoothed surface's
    IV range** (with a small headroom) so day-to-day shifts in the liquid
    core are obvious and deep-wing IV outliers don't dominate the z-axis.
    """
    day_sub = stages.day_sub
    iv_raw = day_sub["iv"].to_numpy(dtype=float)
    finite_iv = iv_raw[np.isfinite(iv_raw)]
    finite_smoothed = stages.iv_unified[np.isfinite(stages.iv_unified)]

    if finite_smoothed.size:
        s_min = float(np.min(finite_smoothed))
        s_max = float(np.max(finite_smoothed))
    elif finite_iv.size:
        s_min = float(np.nanpercentile(finite_iv, 1.0))
        s_max = float(np.nanpercentile(finite_iv, 99.0))
    else:
        s_min, s_max = 0.0, 1.0

    span = max(s_max - s_min, 1e-6)
    iv_vmin = max(0.0, s_min - 0.10 * span)
    iv_vmax = s_max + 0.25 * span

    finite_z = (
        stages.iv_normalized[np.isfinite(stages.iv_normalized)] if stages.iv_normalized is not None else np.empty(0)
    )
    z_abs = float(np.max(np.abs(finite_z))) if finite_z.size else 1.0

    iv_raw_mean = float(np.nanmean(finite_iv)) if finite_iv.size else float("nan")
    iv_raw_p99 = float(np.nanpercentile(finite_iv, 99.0)) if finite_iv.size else float("nan")
    smoothed_mean = float(np.nanmean(finite_smoothed)) if finite_smoothed.size else float("nan")

    return {
        "iv_vmin": iv_vmin,
        "iv_vmax": iv_vmax,
        "z_vmin": -z_abs,
        "z_vmax": z_abs,
        "iv_raw_mean": iv_raw_mean,
        "iv_raw_p99": iv_raw_p99,
        "smoothed_mean": smoothed_mean,
    }


def _scatter_point_sizes(day_sub: pd.DataFrame) -> np.ndarray:
    """Return scatter sizes proportional to sqrt(vega)."""
    if "vega" in day_sub.columns:
        vega = day_sub["vega"].to_numpy(dtype=float)
        vega = np.where(np.isfinite(vega), vega, 0.0)
        vega = np.clip(vega, 0.0, None)
        if vega.max() > 0.0:
            norm = np.sqrt(vega / vega.max())
            return 2.0 + 14.0 * norm
    return np.full(len(day_sub), 6.0)


def make_surface_comparison_3d_figure(
    stages: HistoricSurfaceStages,
    *,
    cmap_iv: str = "turbo",
    cmap_norm: str = "coolwarm",
    elev: float = 25.0,
    azim: float = -55.0,
) -> Figure:
    """3D side-by-side: raw quote scatter, kernel-smoothed surface, normalized.

    All three panels share the day-specific summary in their subplot titles,
    so date changes are immediately visible. The raw-scatter z-axis and colour
    scale are anchored to the smoothed surface range so day-to-day shifts in
    the liquid core dominate over wing outliers.
    """
    if stages.iv_normalized is None:
        raise ValueError(
            "stages.iv_normalized is None; pass a normalizer to "
            "build_historical_surface_pipeline to populate the z-score surface.",
        )

    grid = stages.unified_grid
    K, T = np.meshgrid(grid.log_moneyness, grid.tau, indexing="ij")

    day_sub = stages.day_sub
    k_raw = day_sub["k"].to_numpy(dtype=float)
    tau_raw = day_sub["tau"].to_numpy(dtype=float)
    iv_raw = day_sub["iv"].to_numpy(dtype=float)
    sizes = _scatter_point_sizes(day_sub)

    s = _surface_comparison_scales(stages)
    iv_vmin, iv_vmax = s["iv_vmin"], s["iv_vmax"]
    z_vmin, z_vmax = s["z_vmin"], s["z_vmax"]

    k_lim = (float(grid.log_moneyness[0]), float(grid.log_moneyness[-1]))
    tau_lim = (float(grid.tau[0]), float(grid.tau[-1]))
    title_date = stages.sample_date.date()
    n_quotes = len(day_sub)

    fig = plt.figure(figsize=(16, 5.6))
    fig.suptitle(
        f"IV surface - 3D ({title_date}, n quotes = {n_quotes}, "
        f"raw mean σ = {s['iv_raw_mean']:.2f}, smoothed mean σ = {s['smoothed_mean']:.2f})",
        fontsize=12,
    )

    def _label_3d(ax: plt.Axes, zlabel: str, zlim: tuple[float, float] | None = None) -> None:
        ax.set_xlabel("k = log(K/S)")
        ax.set_ylabel(r"$\tau$ (yrs)")
        ax.set_zlabel(zlabel)
        ax.set_xlim(*k_lim)
        ax.set_ylim(*tau_lim)
        if zlim is not None:
            ax.set_zlim(*zlim)
        ax.view_init(elev=elev, azim=azim)

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    sc = ax1.scatter(
        k_raw,
        tau_raw,
        iv_raw,
        c=iv_raw,
        cmap=cmap_iv,
        s=sizes,
        alpha=0.75,
        vmin=iv_vmin,
        vmax=iv_vmax,
    )
    ax1.set_title(
        f"Raw quotes (p99 σ = {s['iv_raw_p99']:.2f})",
        fontsize=10,
    )
    _label_3d(ax1, r"$\sigma_{\mathrm{imp}}$", zlim=(iv_vmin, iv_vmax))
    fig.colorbar(sc, ax=ax1, shrink=0.55, pad=0.08, label=r"$\sigma_{\mathrm{imp}}$")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    surf2 = ax2.plot_surface(
        K,
        T,
        stages.iv_unified,
        cmap=cmap_iv,
        edgecolor="none",
        antialiased=True,
        vmin=iv_vmin,
        vmax=iv_vmax,
    )
    ax2.set_title("Kernel-smoothed surface", fontsize=10)
    _label_3d(ax2, r"$\sigma_{\mathrm{imp}}$", zlim=(iv_vmin, iv_vmax))
    fig.colorbar(surf2, ax=ax2, shrink=0.55, pad=0.08, label=r"$\sigma_{\mathrm{imp}}$")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    surf3 = ax3.plot_surface(
        K,
        T,
        stages.iv_normalized,
        cmap=cmap_norm,
        edgecolor="none",
        antialiased=True,
        vmin=z_vmin,
        vmax=z_vmax,
    )
    ax3.set_title("Normalized surface (z-score)", fontsize=10)
    _label_3d(ax3, "z", zlim=(z_vmin, z_vmax))
    fig.colorbar(surf3, ax=ax3, shrink=0.55, pad=0.08, label="z")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def make_surface_comparison_heatmap_figure(
    stages: HistoricSurfaceStages,
    *,
    cmap_iv: str = "turbo",
    cmap_norm: str = "coolwarm",
) -> Figure:
    """2D heatmap side-by-side: raw quote scatter, smoothed surface, normalized.

    Raw quotes are drawn as a ``(τ, k)`` scatter with the same IV colour
    scale as the smoothed surface so the reader can overlay the two in
    their head.
    """
    if stages.iv_normalized is None:
        raise ValueError(
            "stages.iv_normalized is None; pass a normalizer to "
            "build_historical_surface_pipeline to populate the z-score surface.",
        )

    grid = stages.unified_grid
    day_sub = stages.day_sub
    k_raw = day_sub["k"].to_numpy(dtype=float)
    tau_raw = day_sub["tau"].to_numpy(dtype=float)
    iv_raw = day_sub["iv"].to_numpy(dtype=float)
    sizes = _scatter_point_sizes(day_sub)

    s = _surface_comparison_scales(stages)
    iv_vmin, iv_vmax = s["iv_vmin"], s["iv_vmax"]
    z_vmin, z_vmax = s["z_vmin"], s["z_vmax"]

    k_lim = (float(grid.log_moneyness[0]), float(grid.log_moneyness[-1]))
    tau_lim = (float(grid.tau[0]), float(grid.tau[-1]))
    extent = (tau_lim[0], tau_lim[1], k_lim[0], k_lim[1])
    title_date = stages.sample_date.date()
    n_quotes = len(day_sub)

    fig = plt.figure(figsize=(16, 5.0))
    fig.suptitle(
        f"IV surface - heatmap ({title_date}, n quotes = {n_quotes}, "
        f"raw mean σ = {s['iv_raw_mean']:.2f}, smoothed mean σ = {s['smoothed_mean']:.2f})",
        fontsize=12,
    )

    def _label_heat(ax: plt.Axes) -> None:
        ax.set_xlim(*tau_lim)
        ax.set_ylim(*k_lim)
        ax.set_xlabel(r"$\tau$ (yrs)")
        ax.set_ylabel("k = log(K/S)")

    axh1 = fig.add_subplot(1, 3, 1)
    sc_h = axh1.scatter(
        tau_raw,
        k_raw,
        c=iv_raw,
        cmap=cmap_iv,
        s=sizes,
        alpha=0.85,
        vmin=iv_vmin,
        vmax=iv_vmax,
    )
    axh1.set_title(
        f"Raw quotes (p99 σ = {s['iv_raw_p99']:.2f})",
        fontsize=10,
    )
    _label_heat(axh1)
    fig.colorbar(sc_h, ax=axh1, shrink=0.9, label=r"$\sigma_{\mathrm{imp}}$")

    axh2 = fig.add_subplot(1, 3, 2)
    im2 = axh2.imshow(
        stages.iv_unified,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap_iv,
        vmin=iv_vmin,
        vmax=iv_vmax,
    )
    axh2.set_title("Kernel-smoothed surface", fontsize=10)
    _label_heat(axh2)
    fig.colorbar(im2, ax=axh2, shrink=0.9, label=r"$\sigma_{\mathrm{imp}}$")

    axh3 = fig.add_subplot(1, 3, 3)
    im3 = axh3.imshow(
        stages.iv_normalized,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap_norm,
        vmin=z_vmin,
        vmax=z_vmax,
    )
    axh3.set_title("Normalized surface (z-score)", fontsize=10)
    _label_heat(axh3)
    fig.colorbar(im3, ax=axh3, shrink=0.9, label="z")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def save_surface_comparison_pdf(
    stages: HistoricSurfaceStages,
    path: str | Path,
    *,
    cmap_iv: str = "turbo",
    cmap_norm: str = "coolwarm",
    elev: float = 25.0,
    azim: float = -55.0,
) -> Path:
    """Write a focused 2-page PDF of the 3D and heatmap comparison figures.

    * Page 1 - 3D side-by-side: raw scatter, kernel-smoothed IV surface on
      the unified grid, and the per-pixel z-score surface.
    * Page 2 - 2D heatmap side-by-side of the same three artefacts.

    Equivalent to calling :func:`make_surface_comparison_3d_figure` and
    :func:`make_surface_comparison_heatmap_figure` and piping both into a
    two-page PDF. Use those directly if you want the figures rendered
    inline in a notebook before persisting.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out) as pdf:
        fig3d = make_surface_comparison_3d_figure(
            stages,
            cmap_iv=cmap_iv,
            cmap_norm=cmap_norm,
            elev=elev,
            azim=azim,
        )
        pdf.savefig(fig3d)
        plt.close(fig3d)

        fig_heat = make_surface_comparison_heatmap_figure(
            stages,
            cmap_iv=cmap_iv,
            cmap_norm=cmap_norm,
        )
        pdf.savefig(fig_heat)
        plt.close(fig_heat)

    return out


def save_historic_pipeline_report_pdf(
    stages: HistoricSurfaceStages,
    path: str | Path,
) -> Path:
    """Write a multi-page PDF report of the historic-surface pipeline stages.

    The PDF contains, in order:

    1. A cover page summarizing the inputs and pipeline.
    2. The 4-panel figure: raw scatter → smoothed → filled → unified grid.
    3. The before/after 3D surface comparison.
    4. Smile-slice diagnostics before and after.
    5. The normalized-surface page (only when a normalizer was used).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    grid = stages.unified_grid
    filled_nan_count = int(np.count_nonzero(~np.isfinite(stages.smoothed)))
    total_cells = stages.smoothed.size
    coverage_after = float(np.isfinite(stages.iv_unified).mean())

    cover_lines = [
        "Historic IV surface pipeline report",
        "",
        f"Quote date       : {stages.sample_date.date()}",
        f"Quotes used      : {len(stages.day_sub)}",
        "",
        "Smoothing grid   : "
        f"k in [{stages.x_grid_smooth[0]:.2f}, {stages.x_grid_smooth[-1]:.2f}] "
        f"({stages.x_grid_smooth.size} pts), "
        f"tau in [{stages.tau_grid_smooth[0]:.2f}, {stages.tau_grid_smooth[-1]:.2f}] "
        f"({stages.tau_grid_smooth.size} pts)",
        "Unified grid     : "
        f"k in [{grid.log_moneyness[0]:.2f}, {grid.log_moneyness[-1]:.2f}] "
        f"({grid.log_moneyness.size} pts), "
        f"tau in [{grid.tau[0]:.2f}, {grid.tau[-1]:.2f}] ({grid.tau.size} pts)",
        "",
        f"NaN cells after smoothing only           : {filled_nan_count} / {total_cells}",
        f"Finite cells on unified grid (after fill): {coverage_after:.1%}",
        "",
        "Pipeline stages:",
        "  1. Kernel smoothing of raw quotes (Gaussian weights * vega).",
        "  2. Interior NaN interpolation (2D linear).",
        "  3. Wing extrapolation (nearest neighbour).",
        "  4. Resample onto the canonical unified (k, tau) grid.",
        "  5. Per-pixel log-IV normalization (z = 0 -> historical mean).",
    ]
    if stages.iv_normalized is None:
        cover_lines.append("     (skipped: no normalizer supplied)")

    with PdfPages(out) as pdf:
        fig_cover = plt.figure(figsize=(11, 8.5))
        fig_cover.text(
            0.08,
            0.95,
            cover_lines[0],
            fontsize=18,
            weight="bold",
            va="top",
        )
        fig_cover.text(
            0.08,
            0.86,
            "\n".join(cover_lines[2:]),
            fontsize=11,
            family="monospace",
            va="top",
        )
        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        fig_stages = make_pipeline_figure(stages)
        pdf.savefig(fig_stages)
        plt.close(fig_stages)

        fig_ba = make_before_after_figure(stages)
        pdf.savefig(fig_ba)
        plt.close(fig_ba)

        fig_smile = make_smile_slices_figure(stages)
        pdf.savefig(fig_smile)
        plt.close(fig_smile)

        fig_norm = make_normalized_figure(stages)
        if fig_norm is not None:
            pdf.savefig(fig_norm)
            plt.close(fig_norm)

    return out
