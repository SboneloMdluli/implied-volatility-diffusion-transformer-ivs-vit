"""API works in **unnormalized IV surfaces**.

and ``ReverseDiffusion`` returns sampled IV surfaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from implied_volatility_diffusion.core.normalization import SurfaceNormalizer
from implied_volatility_diffusion.diffusion.autoencoders.latent_blocks import pad_tensor
from implied_volatility_diffusion.diffusion.autoencoders.latent_grid import halving_spatial_factor
from implied_volatility_diffusion.diffusion.backbones.base import (
    DenoisingBackbone,
    build_backbone,
)
from implied_volatility_diffusion.diffusion.autoencoders.magvit_vqvae import MAGViTv2VQVAE
from implied_volatility_diffusion.diffusion.autoencoders.vqvae_trainer import load_vqvae_checkpoint
from implied_volatility_diffusion.diffusion.noise_scheduler import VPNoiseScheduler

_DEFAULT_IV_FLOOR = 1e-8


def _broadcast(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape ``(B,)`` -> ``(B, 1, 1, ...)`` to broadcast against ``ref``."""
    return value.view(value.shape[0], *([1] * (ref.dim() - 1)))


class DiffusionModel(nn.Module):
    """The model takes and returns **unnormalized IV surfaces**.

    Denoising happens in z-space (``z = (log(iv) - mean) / std``), or in
    the VQ-VAE latent space when ``vqvae`` is provided.

    Args:
        backbone: Any :class:`DenoisingBackbone` (e.g. :class:`UNet` /
            :class:`GridTransformer`).
        scheduler: VP noise schedule.
        mean: Per-cell mean of ``log(IV)``.
        std: Per-cell std of ``log(IV)``.
        iv_floor: Clamp applied before ``log`` for numerical safety.
        prediction_type: ``"epsilon"`` (default) or ``"x0"``. Inherits from
            ``backbone.prediction_type`` if not given.
        vqvae: Optional :class:`~implied_volatility_diffusion.diffusion.\
autoencoders.magvit_vqvae.MAGViTv2VQVAE`.  When provided the diffusion
            backbone operates in the VQ-VAE latent space instead of the
            log-normalized IV z-space.  The backbone's ``in_channels``,
            ``out_channels``, and ``cond_channels`` must equal
            ``vqvae.latent_channels``.
    """

    def __init__(
        self,
        backbone: DenoisingBackbone | nn.Module,
        scheduler: VPNoiseScheduler,
        *,
        mean: np.ndarray | torch.Tensor,
        std: np.ndarray | torch.Tensor,
        iv_floor: float = _DEFAULT_IV_FLOOR,
        prediction_type: str | None = None,
        vqvae: nn.Module | None = None,
    ) -> None:
        super().__init__()
        mean_t = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
        std_t = torch.as_tensor(np.asarray(std), dtype=torch.float32)

        self.backbone = backbone
        self.scheduler = scheduler
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)
        self.iv_floor = float(iv_floor)
        self.prediction_type = prediction_type
        self.vqvae = vqvae

        if vqvae is not None and getattr(vqvae, "num_downsample", 0) > 0:
            h, w = int(mean_t.shape[0]), int(mean_t.shape[1])
            f = halving_spatial_factor(vqvae.num_downsample)
            dummy = torch.zeros(1, 1, h, w)
            _, pads = pad_tensor(dummy, multiple_h=f, multiple_w=f)
            self._vq_encode_pads: tuple[int, int, int, int] = pads
        else:
            self._vq_encode_pads = (0, 0, 0, 0)

        self._vq_losses: dict[str, torch.Tensor] = {}
        self._vqvae_frozen: bool = False

    def freeze_vqvae(self) -> None:
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad_(False)
        self._vqvae_frozen = True

    def unfreeze_vqvae(self) -> None:
        self.vqvae.train()
        for p in self.vqvae.parameters():
            p.requires_grad_(True)
        self._vqvae_frozen = False

    def load_vqvae_weights(self, path: str | Path, *, freeze: bool = True) -> None:
        loaded = load_vqvae_checkpoint(path, map_location=next(self.parameters()).device)
        self.vqvae.load_state_dict(loaded.state_dict())
        if freeze:
            self.freeze_vqvae()

    @classmethod
    def from_surface_normalizer(
        cls,
        backbone: DenoisingBackbone | nn.Module,
        scheduler: VPNoiseScheduler,
        normalizer: "SurfaceNormalizer",
        **kwargs: Any,
    ) -> "DiffusionModel":
        """Build from a fitted :class:`SurfaceNormalizer`."""
        return cls(
            backbone,
            scheduler,
            mean=normalizer.mean,
            std=normalizer.std,
            iv_floor=getattr(normalizer, "iv_floor", _DEFAULT_IV_FLOOR),
            **kwargs,
        )

    @classmethod
    def with_unit_stats(
        cls,
        backbone: DenoisingBackbone | nn.Module,
        scheduler: VPNoiseScheduler,
        grid_shape: tuple[int, int],
        **kwargs: Any,
    ) -> "DiffusionModel":
        """Build a passthrough model (mean=0, std=1) for testing or pre-normalized data."""
        return cls(
            backbone,
            scheduler,
            mean=np.zeros(grid_shape, dtype=np.float32),
            std=np.ones(grid_shape, dtype=np.float32),
            **kwargs,
        )

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        scheduler: VPNoiseScheduler,
        *,
        mean: np.ndarray | torch.Tensor,
        std: np.ndarray | torch.Tensor,
    ) -> "DiffusionModel":
        """Build with the backbone selected by ``cfg['backbone']`` (registry name).

        Supports ``vqvae_checkpoint`` to load a pre-trained first-stage encoder;
        it is frozen by default (``freeze_vqvae=True``).
        """
        name = str(cfg.get("backbone", "unet"))
        backbone_cfg = cfg.get("backbone_kwargs") or {}
        backbone = build_backbone(name, backbone_cfg)

        vqvae = None
        should_freeze = False

        if cfg.get("use_magvit_vqvae", False):

            vqvae_checkpoint = cfg.get("vqvae_checkpoint")
            if vqvae_checkpoint is not None:
                vqvae = load_vqvae_checkpoint(vqvae_checkpoint)
                should_freeze = bool(cfg.get("freeze_vqvae", True))
            else:
                vqvae = MAGViTv2VQVAE(**(cfg.get("vqvae_kwargs") or {}))
                should_freeze = bool(cfg.get("freeze_vqvae", False))

        model = cls(
            backbone,
            scheduler,
            mean=mean,
            std=std,
            iv_floor=float(cfg.get("iv_floor", _DEFAULT_IV_FLOOR)),
            prediction_type=cfg.get("prediction_type"),
            vqvae=vqvae,
        )
        if should_freeze and vqvae is not None:
            model.freeze_vqvae()
        return model

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return the 2D grid shape used by normalisation buffers."""
        return int(self.mean.shape[0]), int(self.mean.shape[1])

    @property
    def in_channels(self) -> int:
        """Return expected input channels from the configured backbone."""
        return int(getattr(self.backbone, "in_channels", 1))

    @property
    def out_channels(self) -> int:
        """Return output channels produced by the configured backbone."""
        return int(getattr(self.backbone, "out_channels", 1))

    def _check_grid(self, x: torch.Tensor) -> None:
        if x.shape[-2:] != self.grid_shape:
            raise ValueError(f"trailing shape {tuple(x.shape[-2:])} must match grid {self.grid_shape}")

    def normalize(self, iv: torch.Tensor) -> torch.Tensor:
        """Map unnormalized IV → diffusion latent (z-space or VQ-VAE latent)."""
        self._check_grid(iv)
        log_iv = torch.log(torch.clamp(iv, min=self.iv_floor))
        z = (log_iv - self.mean) / self.std
        if self.vqvae is not None:
            if self._vqvae_frozen:
                with torch.no_grad():
                    z, _ = self.vqvae.encode_latent(z, pads=self._vq_encode_pads)
                self._vq_losses = {}
            else:
                z, self._vq_losses = self.vqvae.encode_latent(z, pads=self._vq_encode_pads)
        return z

    def denormalize(self, z: torch.Tensor, *, return_log_iv: bool = False) -> torch.Tensor:
        """Map diffusion latent back to unnormalized IV values."""
        if self.vqvae is not None:
            z = self.vqvae.decode_latent(
                z,
                pads=self._vq_encode_pads,
                orig_hw=self.grid_shape,
            )
        self._check_grid(z)
        log_iv = z * self.std + self.mean
        return log_iv if return_log_iv else torch.exp(log_iv)

    def add_noise(
        self,
        iv0: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward diffusion; returns ``(z_t, z0, noise)``."""
        z0 = self.normalize(iv0)
        if noise is None:
            noise = torch.randn_like(z0)
        z_t = self.scheduler.q_sample(z0, t, noise=noise)
        return z_t, z0, noise

    def predict_noise(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backbone forward pass in latent space."""
        return self.backbone(z_t, t, cond) if cond is not None else self.backbone(z_t, t)

    def predict_x0_z(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        *,
        clip: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Predicted clean **latent-space** surface from a noisy ``z_t``."""
        pred = self.predict_noise(z_t, t, cond)
        alpha_bar = _broadcast(self.scheduler.alpha_bar_at(t), z_t)
        if self.prediction_type == "epsilon":
            sqrt_one_minus_ab = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=0.0))
            sqrt_ab = torch.sqrt(torch.clamp(alpha_bar, min=1e-8))
            x0 = (z_t - sqrt_one_minus_ab * pred) / sqrt_ab
        else:
            x0 = pred
        if clip is not None:
            x0 = torch.clamp(x0, clip[0], clip[1])
        return x0

    def predict_iv(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        *,
        clip_z: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Predicted clean **IV** surface from a noisy ``z_t`` (denormalized)."""
        return self.denormalize(self.predict_x0_z(z_t, t, cond, clip=clip_z))

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return self.predict_noise(z_t, t, cond)


__all__ = ["DiffusionModel"]
