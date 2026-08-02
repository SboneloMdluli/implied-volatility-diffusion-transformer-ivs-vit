"""VQ-VAE for IV surface grids (van den Oord et al., NeurIPS 2017, arXiv:1711.00937)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import VQModel

from implied_volatility_diffusion.diffusion.autoencoders.latent_blocks import crop_tensor, pad_tensor
from implied_volatility_diffusion.diffusion.autoencoders.latent_grid import halving_spatial_factor


@dataclass
class MAGViTv2VQVAEOutput:
    """Return bundle from :meth:`MAGViTv2VQVAE.forward` when ``return_output=True``."""

    reconstruction: torch.Tensor
    z_q: torch.Tensor
    vq_commitment_loss: torch.Tensor


class MAGViTv2VQVAE(nn.Module):
    """VQ-VAE for IV surface grids."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        latent_channels: int = 8,
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 1,
        num_downsample: int = 0,
        num_vq_embeddings: int = 256,
        norm_num_groups: int = 32,
        vq_embed_dim: int | None = None,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()

        _down = tuple("DownEncoderBlock2D" for _ in range(num_downsample))
        _up = tuple("UpDecoderBlock2D" for _ in range(num_downsample))

        self.vq_model = VQModel(
            in_channels=in_channels,
            out_channels=in_channels,
            down_block_types=_down,
            up_block_types=_up,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            latent_channels=latent_channels,
            num_vq_embeddings=num_vq_embeddings,
            norm_num_groups=norm_num_groups,
            vq_embed_dim=vq_embed_dim,
            scaling_factor=1.0,
        )

        self.vq_model.quantize.beta = float(commitment_weight)
        self._num_downsample = int(num_downsample)

    @property
    def latent_channels(self) -> int:
        return self.vq_model.config.latent_channels

    @property
    def num_downsample(self) -> int:
        return self._num_downsample

    @property
    def codebook_size(self) -> int:
        return self.vq_model.config.num_vq_embeddings

    def encode_latent(
        self,
        x: torch.Tensor,
        *,
        pads: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Encode *x* to a VQ-quantized latent (STE) and return the commitment loss."""
        if any(p > 0 for p in pads):
            pad_top, pad_bottom, pad_left, pad_right = pads
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        h = self.vq_model.encode(x).latents
        z_q, commit_loss, _ = self.vq_model.quantize(h)
        return z_q, {"vq_commitment_loss": commit_loss}

    def decode_latent(
        self,
        z: torch.Tensor,
        *,
        pads: tuple[int, int, int, int] = (0, 0, 0, 0),
        orig_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Decode latent *z* back to surface space, cropping padding if applied."""
        recon = self.vq_model.decode(z, force_not_quantize=True).sample
        if any(p > 0 for p in pads):
            if orig_hw is None:
                orig_hw = (z.shape[-2], z.shape[-1])
            recon = crop_tensor(recon, pads, target_h=orig_hw[0], target_w=orig_hw[1])
        return recon

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_output: bool = False,
    ) -> torch.Tensor | MAGViTv2VQVAEOutput:
        """Encode → VQ-quantize → decode.  Use ``return_output=True`` for losses."""
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        if self._num_downsample > 0:
            _, pads = pad_tensor(x, multiple_h=halving_spatial_factor(self._num_downsample),
                                 multiple_w=halving_spatial_factor(self._num_downsample))
        else:
            pads = (0, 0, 0, 0)

        z_q, losses = self.encode_latent(x, pads=pads)
        recon = self.decode_latent(z_q, pads=pads, orig_hw=(orig_h, orig_w))

        if return_output:
            return MAGViTv2VQVAEOutput(
                reconstruction=recon,
                z_q=z_q,
                vq_commitment_loss=losses["vq_commitment_loss"],
            )
        return recon


__all__ = ["MAGViTv2VQVAE", "MAGViTv2VQVAEOutput"]
