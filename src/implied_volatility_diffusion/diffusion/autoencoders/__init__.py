"""Autoencoder-related diffusion modules."""

from implied_volatility_diffusion.diffusion.autoencoders.latent_blocks import (
    DownBlock,
    UpBlock,
    crop_tensor,
    groupnorm,
    pad_tensor,
)
from implied_volatility_diffusion.diffusion.autoencoders.latent_grid import (
    crop_surface,
    halving_spatial_factor,
    latent_padded_hw,
    latent_spatial_hw,
    pad_surface,
    symmetric_pad_widths,
)
from implied_volatility_diffusion.diffusion.autoencoders.magvit_vqvae import (
    MAGViTv2VQVAE,
    MAGViTv2VQVAEOutput,
)

__all__ = [
    "DownBlock",
    "MAGViTv2VQVAE",
    "MAGViTv2VQVAEOutput",
    "UpBlock",
    "crop_tensor",
    "crop_surface",
    "groupnorm",
    "halving_spatial_factor",
    "latent_padded_hw",
    "latent_spatial_hw",
    "pad_tensor",
    "pad_surface",
    "symmetric_pad_widths",
]
