"""Reverse-diffusion training and sampling for IV surfaces."""

from implied_volatility_diffusion.diffusion.arbitrage_torch import (
    ArbitragePenalty,
    ArbitrageWeights,
    torch_bs_call,
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
from implied_volatility_diffusion.diffusion.autoencoders.latent_grid import (
    crop_surface,
    halving_spatial_factor,
    latent_padded_hw,
    latent_spatial_hw,
    pad_surface,
    symmetric_pad_widths,
)
from implied_volatility_diffusion.diffusion.backbones import (
    DenoisingBackbone,
    GridTransformer,
    SinusoidalTimeEmbedding,
    TimeEmbeddingMLP,
    UNet,
    build_backbone,
    iter_backbone_names,
    register_backbone,
)
from implied_volatility_diffusion.diffusion.losses import (
    ArbitrageSchedule,
    DiffusionLoss,
    DiffusionLossConfig,
    TimestepSampling,
)
from implied_volatility_diffusion.diffusion.model import DiffusionModel
from implied_volatility_diffusion.diffusion.noise_scheduler import VPNoiseScheduler
from implied_volatility_diffusion.diffusion.reverse import ReverseDiffusion
from implied_volatility_diffusion.diffusion.validation import (
    SurfaceGenerationDiagnostics,
    arbitrage_violation_score,
    plot_performance_metrics,
    plot_surface_comparison,
    regenerate_until_arb_free,
)

__all__ = [
    "ArbitragePenalty",
    "ArbitrageSchedule",
    "ArbitrageWeights",
    "DenoisingBackbone",
    "DiffusionLoss",
    "DiffusionLossConfig",
    "DiffusionModel",
    "TimestepSampling",
    "MAGViTv2VQVAE",
    "MAGViTv2VQVAEOutput",
    "ReverseDiffusion",
    "VQVAELoss",
    "load_vqvae_checkpoint",
    "save_vqvae_checkpoint",
    "SinusoidalTimeEmbedding",
    "SurfaceGenerationDiagnostics",
    "TimeEmbeddingMLP",
    "UNet",
    "VPNoiseScheduler",
    "arbitrage_violation_score",
    "build_backbone",
    "crop_surface",
    "halving_spatial_factor",
    "iter_backbone_names",
    "latent_padded_hw",
    "latent_spatial_hw",
    "pad_surface",
    "plot_performance_metrics",
    "plot_surface_comparison",
    "regenerate_until_arb_free",
    "register_backbone",
    "symmetric_pad_widths",
    "torch_bs_call",
    "GridTransformer",
]
