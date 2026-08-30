"""Run U-Net diffusion validation for a checkpoint (mirrors validation notebook)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

from implied_volatility_diffusion import (
    DiffusionModel,
    MAGViTv2VQVAE,
    ReverseDiffusion,
    SurfaceNormalizer,
    UnifiedGrid,
    UNet,
    VPNoiseScheduler,
    check_iv_surface_arbitrage,
    repair_iv_surface,
    volgan_generative_repair_settings,
)
from implied_volatility_diffusion.diffusion.autoencoders.vqvae_trainer import load_vqvae_checkpoint


def _select_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _find_repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    for candidate in (root, *root.parents):
        if (candidate / "config").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("could not find repository root")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="diffusion_unet_vq.pt",
        help="Checkpoint filename under data/processed/checkpoints/",
    )
    parser.add_argument(
        "--vqvae-checkpoint",
        default="vqvae.pt",
        help="Stage-1 VQ-VAE checkpoint filename under data/processed/checkpoints/",
    )
    parser.add_argument("--ddim-steps", type=int, default=200)
    parser.add_argument(
        "--path-types",
        nargs="+",
        default=["historical", "sabr", "heston"],
        help="Validation path types to evaluate",
    )
    parser.add_argument(
        "--historical-limit",
        type=int,
        default=None,
        help="Optional cap on historical pairs (default: all)",
    )
    args = parser.parse_args()

    repo_root = _find_repo_root()
    cfg_dir = repo_root / "config"
    dataset_root = repo_root / "data" / "processed" / "forecasting_dataset"
    checkpoint_dir = repo_root / "data" / "processed" / "checkpoints"
    checkpoint_path = checkpoint_dir / args.checkpoint
    vqvae_checkpoint_path = checkpoint_dir / args.vqvae_checkpoint
    config_path = checkpoint_dir / "training_config.json"

    device = _select_device()
    grid = UnifiedGrid.load(cfg_dir / "unified_iv_grid.yaml")

    ddim_steps = args.ddim_steps
    arb_accept_tol = 1e-4
    iv_floor_sample = 1e-4
    apply_surface_repair = True
    forecast_repair_settings = volgan_generative_repair_settings(
        tol=arb_accept_tol,
        iv_floor=iv_floor_sample,
    )
    sampler_clip_z = (-4.0, 4.0)
    path_type_to_src_id = {"historical": 0, "heston": 1, "sabr": 2}

    run_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    validation_seed = int(run_config.get("seed", 42))

    manifest = json.loads((dataset_root / "manifest.json").read_text())
    normalizer = SurfaceNormalizer.load(dataset_root / "normalizer.npz")
    assert tuple(normalizer.grid_shape) == tuple(grid.shape)

    val_dir = dataset_root / "validation"
    val_pair_curr = np.load(val_dir / "pair_curr.npy").astype(np.float32)
    val_pair_next = np.load(val_dir / "pair_next.npy").astype(np.float32)
    val_sources = np.load(val_dir / "pair_sources.npy")

    pair_curr_by_src: dict[int, np.ndarray] = {}
    pair_next_by_src: dict[int, np.ndarray] = {}
    for src_id in (0, 1, 2):
        mask = val_sources == src_id
        pair_curr_by_src[src_id] = val_pair_curr[mask]
        pair_next_by_src[src_id] = val_pair_next[mask]

    val_hist_curr_dates = pd.DatetimeIndex(np.load(val_dir / "historical_curr_dates.npy"))
    val_hist_next_dates = pd.DatetimeIndex(np.load(val_dir / "historical_next_dates.npy"))
    val_heston_paths = np.load(val_dir / "heston_path.npy")
    val_heston_steps = np.load(val_dir / "heston_step.npy")
    val_sabr_paths = np.load(val_dir / "sabr_path.npy")
    val_sabr_steps = np.load(val_dir / "sabr_step.npy")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scheduler_timesteps = int(checkpoint.get("scheduler_timesteps", run_config.get("scheduler_timesteps", 400)))
    unet_kw = checkpoint.get("unet_kwargs") or run_config.get("unet_kwargs")
    spot_ref = float(checkpoint.get("spot_ref", run_config.get("spot_ref", 100.0)))
    rate_ref = float(checkpoint.get("rate_ref", run_config.get("rate_ref", 0.03)))
    if not unet_kw:
        raise RuntimeError("Checkpoint missing unet_kwargs")

    backbone = UNet(**dict(unet_kw))
    if int(unet_kw.get("cond_channels", 0)) < 1:
        raise RuntimeError("Checkpoint is unconditional")

    scheduler = VPNoiseScheduler(timesteps=scheduler_timesteps, beta_schedule="cosine")

    state = checkpoint.get("model_state_dict") or {}
    has_vq_weights = any(str(k).startswith("vqvae.") for k in state)
    vqvae_kw = checkpoint.get("vqvae_kwargs") or run_config.get("vqvae_kwargs")
    use_vqvae = bool(
        checkpoint.get(
            "use_magvit_vqvae",
            run_config.get("use_magvit_vqvae", vqvae_kw is not None or has_vq_weights),
        )
    )
    if use_vqvae and vqvae_kw:
        vqvae = MAGViTv2VQVAE(**{k: tuple(v) if isinstance(v, list) else v for k, v in vqvae_kw.items()})
    elif use_vqvae:
        if not vqvae_checkpoint_path.exists():
            raise RuntimeError(
                "VQ-VAE U-Net checkpoint has no vqvae_kwargs and "
                f"{vqvae_checkpoint_path} is missing"
            )
        vqvae = load_vqvae_checkpoint(vqvae_checkpoint_path, map_location="cpu")
        print(f"VQ-VAE architecture loaded from {vqvae_checkpoint_path.name}")
    else:
        vqvae = None

    if vqvae is not None:
        print(
            f"VQ-VAE enabled: latent_channels={vqvae.latent_channels}, "
            f"codebook_size={vqvae.codebook_size}, num_downsample={vqvae.num_downsample}"
        )

    model = DiffusionModel.from_surface_normalizer(
        backbone,
        scheduler,
        normalizer,
        prediction_type=None,
        vqvae=vqvae,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if use_vqvae:
        model.freeze_vqvae()
    model.eval()
    sampler = ReverseDiffusion(model)

    moneyness = np.exp(grid.log_moneyness)
    sampler_use_generator = device.type == "cpu"

    def surface_label(src_id: int, pair_idx: int) -> tuple[str, str]:
        if src_id == 0:
            if pair_idx < len(val_hist_curr_dates):
                return (
                    f"date={val_hist_curr_dates[pair_idx].date()}",
                    f"date={val_hist_next_dates[pair_idx].date()}",
                )
            return f"idx={pair_idx}", f"idx={pair_idx}"
        if src_id == 1:
            path = int(val_heston_paths[pair_idx])
            step = int(val_heston_steps[pair_idx])
            return f"path={path}, step={step}", f"path={path}, step={step + 1}"
        path = int(val_sabr_paths[pair_idx])
        step = int(val_sabr_steps[pair_idx])
        return f"path={path}, step={step}", f"path={path}, step={step + 1}"

    def maybe_guard(iv: np.ndarray) -> np.ndarray:
        if not apply_surface_repair:
            return np.asarray(iv, dtype=np.float64)
        return repair_iv_surface(
            np.asarray(iv, dtype=np.float64),
            moneyness,
            grid.tau,
            spot=spot_ref,
            rate=rate_ref,
            settings=forecast_repair_settings,
        )

    def forecast_vs_gt_metrics(forecast: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
        diff = np.asarray(forecast, dtype=np.float64) - np.asarray(ground_truth, dtype=np.float64)
        gt_abs = np.abs(np.asarray(ground_truth, dtype=np.float64))
        mape_mask = np.isfinite(diff) & np.isfinite(gt_abs) & (gt_abs > float(iv_floor_sample))
        return {
            "rmse": float(np.sqrt(np.nanmean(diff * diff))),
            "mae": float(np.nanmean(np.abs(diff))),
            "mape_pct": (
                float(100.0 * np.mean(np.abs(diff[mape_mask]) / gt_abs[mape_mask])) if np.any(mape_mask) else float("nan")
            ),
        }

    @torch.no_grad()
    def conditional_forecast(s_curr: np.ndarray, base_seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
        cond_iv = torch.as_tensor(s_curr, dtype=torch.float32, device=device)[None, None]
        cond_z = model.normalize(cond_iv)
        seed = int(base_seed)
        if sampler_use_generator:
            sample_generator = torch.Generator().manual_seed(seed)
        else:
            sample_generator = None
            torch.manual_seed(seed)

        forecast_iv = (
            sampler.ddim_sample(
                batch_size=1,
                num_steps=ddim_steps,
                eta=0.0,
                cond=cond_z,
                generator=sample_generator,
                clip_z=sampler_clip_z,
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
            .astype(np.float64)
        )
        forecast_raw = np.asarray(forecast_iv, dtype=np.float64)
        guarded = maybe_guard(forecast_raw)
        report = check_iv_surface_arbitrage(
            guarded,
            moneyness,
            grid.tau,
            spot=spot_ref,
            rate=rate_ref,
            tol=arb_accept_tol,
        )
        raw_report = check_iv_surface_arbitrage(
            forecast_raw,
            moneyness,
            grid.tau,
            spot=spot_ref,
            rate=rate_ref,
            tol=arb_accept_tol,
        )
        meta = {
            "seed": seed,
            "forecast_arb_free": bool(report.arbitrage_free),
            "forecast_raw_arb_free": bool(raw_report.arbitrage_free),
        }
        return guarded, forecast_raw, meta

    print(f"checkpoint: {checkpoint_path.name}")
    print(f"device: {device}")
    print(f"grid: {grid.shape}  timesteps: {scheduler_timesteps}")
    print(f"manifest sample date: {manifest.get('sample_date')}")

    instances: list[dict] = []
    for path_type in args.path_types:
        src_id = path_type_to_src_id[path_type]
        n_pairs = pair_curr_by_src[src_id].shape[0]
        display_rng = np.random.default_rng(validation_seed + 5000 + src_id)
        display_pick = int(display_rng.choice(n_pairs))

        if path_type == "historical":
            if args.historical_limit is not None:
                pair_indices = range(min(args.historical_limit, n_pairs))
            else:
                pair_indices = range(n_pairs)
            print(f"Forecasting {path_type} — {len(pair_indices)}/{n_pairs} pairs...")
        else:
            pair_indices = [display_pick]
            print(f"Forecasting {path_type} — sample 1/{n_pairs} (pair {display_pick})...")

        for j in pair_indices:
            s_curr = pair_curr_by_src[src_id][j].astype(np.float64)
            s_next = pair_next_by_src[src_id][j].astype(np.float64)
            input_label, gt_label = surface_label(src_id, j)
            seed = validation_seed + 9000 + src_id * 10000 + j
            forecast, forecast_raw, meta = conditional_forecast(s_curr, seed)
            instances.append(
                {
                    "path_type": path_type,
                    "pair_idx": j,
                    "input_label": input_label,
                    "gt_label": gt_label,
                    "forecast": forecast,
                    "forecast_raw": forecast_raw,
                    "ground_truth": s_next,
                    "meta": meta,
                }
            )
            if path_type == "historical" and ((j + 1) % 25 == 0 or j == pair_indices[-1]):
                print(f"  historical: {j + 1}/{len(pair_indices)}")

    metrics_rows = []
    for i, inst in enumerate(instances, start=1):
        gt = inst["ground_truth"]
        raw_m = forecast_vs_gt_metrics(inst["forecast_raw"], gt)
        rep_m = forecast_vs_gt_metrics(inst["forecast"], gt)
        metrics_rows.append(
            {
                "instance": i,
                "path_type": inst["path_type"],
                "pair_idx": int(inst["pair_idx"]),
                "input_label": inst["input_label"],
                "gt_label": inst["gt_label"],
                "raw_rmse_vs_gt": raw_m["rmse"],
                "repaired_rmse_vs_gt": rep_m["rmse"],
                "raw_mape_pct_vs_gt": raw_m["mape_pct"],
                "repaired_mape_pct_vs_gt": rep_m["mape_pct"],
                "raw_arb_free": bool(inst["meta"]["forecast_raw_arb_free"]),
                "repaired_arb_free": bool(inst["meta"]["forecast_arb_free"]),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    print("\n=== Per-instance metrics (sample) ===")
    print(metrics_df.to_string(index=False))

    hist_instances = [inst for inst in instances if inst["path_type"] == "historical"]
    if hist_instances:
        hist_metrics = metrics_df[metrics_df["path_type"] == "historical"]
        rmse_inst = hist_metrics["repaired_rmse_vs_gt"].to_numpy(dtype=float)
        mape_inst = hist_metrics["repaired_mape_pct_vs_gt"].to_numpy(dtype=float)
        mean_rmse = float(np.mean(rmse_inst))
        mean_mape = float(np.mean(mape_inst))

        boot_n = 10_000
        boot_rng = np.random.default_rng(validation_seed + 31)
        n_inst = len(hist_instances)
        boot_idx = boot_rng.integers(0, n_inst, size=(boot_n, n_inst))
        rmse_ci95 = tuple(np.percentile(np.mean(rmse_inst[boot_idx], axis=1), [2.5, 97.5]).astype(float))
        mape_ci95 = tuple(np.percentile(np.mean(mape_inst[boot_idx], axis=1), [2.5, 97.5]).astype(float))

        arb_violation_rate = float(np.mean([not bool(inst["meta"]["forecast_arb_free"]) for inst in hist_instances]))
        err_pooled = np.concatenate([(inst["forecast"] - inst["ground_truth"]).ravel() for inst in hist_instances])
        err_pooled = err_pooled[np.isfinite(err_pooled)]

        print("\n=== Historical summary (repaired forecasts) ===")
        print(f"Mean RMSE={mean_rmse:.6g}  (95% CI [{rmse_ci95[0]:.6g}, {rmse_ci95[1]:.6g}])")
        print(f"Mean MAPE={mean_mape:.4f}%  (95% CI [{mape_ci95[0]:.4f}%, {mape_ci95[1]:.4f}%])")
        print(f"Arbitrage violation rate={100.0 * arb_violation_rate:.2f}%")
        print(
            f"Pooled error: mean={np.mean(err_pooled):.6g}, std={np.std(err_pooled, ddof=1):.6g}, "
            f"skew={scipy_stats.skew(err_pooled, bias=False):.6g}, "
            f"excess kurtosis={scipy_stats.kurtosis(err_pooled, fisher=True, bias=False):.6g} "
            f"(n={err_pooled.size})"
        )

    for path_type in ("sabr", "heston"):
        subset = metrics_df[metrics_df["path_type"] == path_type]
        if not subset.empty:
            row = subset.iloc[0]
            print(f"\n=== {path_type} sample ===")
            print(f"{row['input_label']} -> {row['gt_label']}")
            print(
                f"Repaired RMSE={row['repaired_rmse_vs_gt']:.6f}  MAPE={row['repaired_mape_pct_vs_gt']:.2f}%  "
                f"arb_free={row['repaired_arb_free']}"
            )
            print(
                f"Raw RMSE={row['raw_rmse_vs_gt']:.6f}  MAPE={row['raw_mape_pct_vs_gt']:.2f}%  "
                f"arb_free={row['raw_arb_free']}"
            )


if __name__ == "__main__":
    main()
