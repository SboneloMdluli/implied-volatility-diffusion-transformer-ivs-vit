"""Diffusion training loss with optional smoothness and time-annealed arbitrage penalties."""

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn

from implied_volatility_diffusion.diffusion.arbitrage_torch import ArbitragePenalty
from implied_volatility_diffusion.diffusion.model import DiffusionModel
from implied_volatility_diffusion.diffusion.noise_scheduler import VPNoiseScheduler

ArbitrageSchedule = Literal["alpha_bar", "alpha_bar_sq", "sqrt_alpha_bar", "linear", "snr", "constant"]
TimestepSampling = Literal["uniform", "lognormal"]


def _arbitrage_weights(scheduler: VPNoiseScheduler, t: torch.Tensor, schedule: ArbitrageSchedule) -> torch.Tensor:
    """Per-sample weights ~1 near t≈0, ~0 near T (monotone in t).

    Methods: alpha_bar; alpha_bar_sq; sqrt(alpha_bar); SNR clipped to [0,1]; linear 1−t/(T−1); constant 1.
    """
    alpha_bar = scheduler.alpha_bar_at(t)
    if schedule == "alpha_bar":
        return alpha_bar
    if schedule == "alpha_bar_sq":
        return alpha_bar * alpha_bar
    if schedule == "sqrt_alpha_bar":
        return torch.sqrt(alpha_bar)
    if schedule == "snr":
        snr = alpha_bar / torch.clamp(1.0 - alpha_bar, min=1e-8)
        return torch.clamp(snr, max=1.0)
    if schedule == "linear":
        T_max = max(scheduler.timesteps - 1, 1)
        return torch.clamp(1.0 - t.float() / T_max, min=0.0)
    if schedule == "constant":
        return torch.ones_like(alpha_bar)


def _dirichlet_energy_index_mean_per_sample(iv: torch.Tensor) -> torch.Tensor:
    """Per batch row on an IV grid: mean squared index differences along m and τ, summed."""
    device, dtype = iv.device, iv.dtype
    out = torch.zeros(iv.shape[0], device=device, dtype=dtype)
    h, w = iv.shape[-2], iv.shape[-1]
    if h >= 2:
        dg = iv[..., 1:, :] - iv[..., :-1, :]
        out = out + (dg * dg).flatten(1).mean(dim=1)
    if w >= 2:
        dg = iv[..., :, 1:] - iv[..., :, :-1]
        out = out + (dg * dg).flatten(1).mean(dim=1)
    return out


@dataclass
class DiffusionLossConfig:
    """Hyperparameters for :class:`DiffusionLoss`."""

    arbitrage_lambda: float = 0.5
    # schedules as in Zhou et al. (arXiv:2511.07571)
    arbitrage_schedule: ArbitrageSchedule = "alpha_bar"

    predicted_z0_clip: tuple[float, float] | None = (-4.0, 4.0)
    # x0-only SNR weights
    snr_weighting: bool = False

    smoothness_lambda: float = 1e-4
    smoothness_schedule: ArbitrageSchedule = "alpha_bar"

    timestep_sampling: TimestepSampling = "uniform"
    lognormal_mu: float = -3.5
    lognormal_sigma: float = 0.65
    lognormal_max_noise_frac: float | None = 0.15

    eps_loss_schedule: ArbitrageSchedule = "alpha_bar"


class DiffusionLoss(nn.Module):
    """Epsilon MSE plus optional Dirichlet smoothness and arbitrage penalties."""

    def __init__(
        self,
        arbitrage_penalty: ArbitragePenalty | Callable[[torch.Tensor], dict[str, torch.Tensor]] | None = None,
        *,
        config: DiffusionLossConfig | None = None,
    ) -> None:
        """Initialize diffusion loss with optional arbitrage penalty module."""
        super().__init__()
        self.arbitrage_penalty = arbitrage_penalty
        self.config = config or DiffusionLossConfig()

    def sample_timesteps(
        self,
        batch_size: int,
        scheduler: VPNoiseScheduler,
        *,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample integer timesteps per batch row.

        ``uniform`` (default): ``t ~ U{0, …, T-1}``.
        ``lognormal``: lognormal on ``(1 - alpha_bar_t)``; negative ``lognormal_mu``
        and ``lognormal_max_noise_frac`` bias draws toward low noise / ``x_0``.
        """
        if device is None:
            device = scheduler.alpha_bar.device
        if self.config.timestep_sampling == "uniform":
            t = torch.randint(
                0,
                scheduler.timesteps,
                (batch_size,),
                dtype=torch.long,
                generator=generator,
            )
            return t.to(device)
        if self.config.timestep_sampling == "lognormal":
            return _sample_timesteps_lognormal(
                batch_size,
                scheduler,
                device=device,
                mu=self.config.lognormal_mu,
                sigma=self.config.lognormal_sigma,
                max_noise_frac=self.config.lognormal_max_noise_frac,
                generator=generator,
            )
        raise ValueError(f"unknown timestep_sampling: {self.config.timestep_sampling}")

    def forward(
        self,
        model: DiffusionModel,
        iv0: torch.Tensor,
        *,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Training loss on **unnormalized** IV batch.

        Dict includes ``loss``, ``loss_eps``, and when enabled ``loss_smooth`` / ``loss_arb``
        plus per-family ``arb_*`` keys.
        """
        scheduler = model.scheduler
        device = iv0.device
        if t is None:
            t = self.sample_timesteps(iv0.shape[0], scheduler, device=device)
        if t.dim() == 0:
            t = t.expand(iv0.shape[0])

        z_t, z0, eps = model.add_noise(iv0, t, noise=noise)

        # Collect VQ-VAE training losses (commitment + entropy) when a MAGViT-v2

        vq_losses: dict[str, torch.Tensor] = {}
        if getattr(model, "_vq_losses", None):
            vq_losses = dict(model._vq_losses)
            model._vq_losses = {}

        pred = model.predict_noise(z_t, t, cond)
        target = eps if model.prediction_type == "epsilon" else z0
        per_sample_mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
        w_loss = self._eps_loss_weights(scheduler, t, model.prediction_type)
        loss_eps = (w_loss * per_sample_mse).mean()

        out: dict[str, torch.Tensor] = {"loss_eps": loss_eps}
        loss_total = loss_eps

        # Add VQ-VAE commitment and entropy losses
        for k, v in vq_losses.items():
            out[k] = v
            loss_total = loss_total + v

        arb_on = self.arbitrage_penalty is not None and self.config.arbitrage_lambda > 0.0
        smooth_on = self.config.smoothness_lambda > 0.0

        if arb_on or smooth_on:
            x0_z = self._recover_x0_z(model, z_t, pred, t)
            clip = self.config.predicted_z0_clip
            if clip is not None:
                x0_z = torch.clamp(x0_z, clip[0], clip[1])
            iv_pred = torch.nan_to_num(model.denormalize(x0_z), nan=0.0, posinf=0.0, neginf=0.0)

            if smooth_on:
                w_s = _arbitrage_weights(scheduler, t, self.config.smoothness_schedule)
                smooth_per = _dirichlet_energy_index_mean_per_sample(iv_pred)
                loss_smooth = (w_s * smooth_per).mean()
                out["loss_smooth"] = loss_smooth
                out["smooth_weight_mean"] = w_s.mean().detach()
                loss_total = loss_total + self.config.smoothness_lambda * loss_smooth

            if arb_on:
                parts = self.arbitrage_penalty(iv_pred)
                w = _arbitrage_weights(scheduler, t, self.config.arbitrage_schedule)
                arb_per_sample = torch.zeros(iv0.shape[0], device=device)
                for name, value in parts.items():
                    weighted = w * value
                    out[f"arb_{name}"] = weighted.mean()
                    arb_per_sample = arb_per_sample + weighted
                loss_arb = arb_per_sample.mean()
                out["loss_arb"] = loss_arb
                out["arb_weight_mean"] = w.mean().detach()
                loss_total = loss_total + self.config.arbitrage_lambda * loss_arb

        out["loss"] = loss_total
        return out

    def _eps_loss_weights(
        self,
        scheduler: VPNoiseScheduler,
        t: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        """Per-sample MSE weights; ``alpha_bar`` emphasizes low-noise (near ``x_0``) steps."""
        if self.config.eps_loss_schedule != "uniform":
            return _arbitrage_weights(scheduler, t, self.config.eps_loss_schedule)
        if not self.config.snr_weighting:
            return torch.ones(t.shape[0], device=t.device, dtype=scheduler.alpha_bar.dtype)
        alpha_bar = scheduler.alpha_bar_at(t)
        snr = alpha_bar / torch.clamp(1.0 - alpha_bar, min=1e-8)
        if prediction_type == "epsilon":
            return torch.ones_like(snr)
        return snr

    @staticmethod
    def _recover_x0_z(
        model: DiffusionModel,
        z_t: torch.Tensor,
        pred: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Recover x0 in z-space from ``pred`` (avoids an extra backbone pass)."""
        if model.prediction_type == "x0":
            return pred
        alpha_bar = model.scheduler.alpha_bar_at(t).view(t.shape[0], *([1] * (z_t.dim() - 1)))
        sqrt_ab = torch.sqrt(torch.clamp(alpha_bar, min=1e-8))
        sqrt_one_minus_ab = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=0.0))
        return (z_t - sqrt_one_minus_ab * pred) / sqrt_ab


def _sample_timesteps_lognormal(
    batch_size: int,
    scheduler: VPNoiseScheduler,
    *,
    device: torch.device,
    mu: float,
    sigma: float,
    max_noise_frac: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Map lognormal draws on ``(1 - alpha_bar)`` to discrete VP timesteps."""
    z = torch.randn(batch_size, generator=generator).to(device)
    noise_var = torch.exp(z * sigma + mu)
    one_minus_ab = 1.0 - scheduler.alpha_bar.to(device=device)
    lo = one_minus_ab[0]
    hi = one_minus_ab[-1]
    if max_noise_frac is not None:
        if not 0.0 < max_noise_frac <= 1.0:
            raise ValueError("lognormal_max_noise_frac must be in (0, 1]")
        hi = lo + float(max_noise_frac) * (hi - lo)
    noise_var = noise_var.clamp(min=lo, max=hi)
    t = torch.searchsorted(one_minus_ab, noise_var, right=False)
    return t.clamp(max=scheduler.timesteps - 1).to(dtype=torch.long)


__all__ = [
    "ArbitrageSchedule",
    "DiffusionLoss",
    "DiffusionLossConfig",
    "TimestepSampling",
]
