"""First-stage VQ-VAE training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from implied_volatility_diffusion.diffusion.autoencoders.magvit_vqvae import MAGViTv2VQVAE


class VQVAELoss(nn.Module):
    def __init__(
        self,
        *,
        reconstruction_loss: str = "mse",
        commitment_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.commitment_weight = commitment_weight

    def forward(self, vqvae: MAGViTv2VQVAE, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = vqvae(x, return_output=True)
        loss_recon = F.mse_loss(out.reconstruction, x) if self.reconstruction_loss == "mse" else F.l1_loss(out.reconstruction, x)
        vq_loss = out.vq_commitment_loss if self.commitment_weight is None else self.commitment_weight * out.vq_commitment_loss
        return {"loss": loss_recon + vq_loss, "loss_recon": loss_recon, "vq_commitment_loss": vq_loss}


def save_vqvae_checkpoint(
    vqvae: MAGViTv2VQVAE,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "state_dict": vqvae.state_dict(),
        "vqvae_config": dict(vqvae.vq_model.config),
        "num_downsample": vqvae.num_downsample,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_vqvae_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> MAGViTv2VQVAE:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg: dict[str, Any] = payload["vqvae_config"]
    vqvae = MAGViTv2VQVAE(
        in_channels=cfg.get("in_channels", 1),
        latent_channels=cfg.get("latent_channels", 8),
        block_out_channels=tuple(cfg.get("block_out_channels", (64,))),
        layers_per_block=cfg.get("layers_per_block", 1),
        num_downsample=payload.get("num_downsample", 0),
        num_vq_embeddings=cfg.get("num_vq_embeddings", 256),
        norm_num_groups=cfg.get("norm_num_groups", 32),
        vq_embed_dim=cfg.get("vq_embed_dim", None),
    )
    vqvae.load_state_dict(payload["state_dict"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    return vqvae


__all__ = ["VQVAELoss", "load_vqvae_checkpoint", "save_vqvae_checkpoint"]
