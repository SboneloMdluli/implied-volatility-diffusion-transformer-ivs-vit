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
from implied_volatility_diffusion.diffusion.autoencoders.vqvae_trainer import (
    VQVAELoss,
    load_vqvae_checkpoint,
    save_vqvae_checkpoint,
)

__all__ = [
    "DownBlock",
    "MAGViTv2VQVAE",
    "MAGViTv2VQVAEOutput",
    "UpBlock",
    "VQVAELoss",
    "crop_tensor",
    "crop_surface",
    "groupnorm",
    "halving_spatial_factor",
    "latent_padded_hw",
    "latent_spatial_hw",
    "load_vqvae_checkpoint",
    "pad_tensor",
    "pad_surface",
    "save_vqvae_checkpoint",
    "symmetric_pad_widths",
]
