"""Convenience wrappers for HuggingFace Diffusers pipeline tracing."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..api import DiffusionPipelineResult, trace_diffusion_pipeline
from ..config import DiffusionConfig, SimulatorConfig

try:
    from diffusers import DiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    DiffusionPipeline = Any  # type stub


def trace_diffusers_pipeline(
    pipeline: DiffusionPipeline,
    config: SimulatorConfig,
    diffusion_config: DiffusionConfig | None = None,
    prompt_length: int = 77,
    height: int = 512,
    width: int = 512,
    num_frames: int = 1,
    latent_channels: int = 4,
    vae_scale_factor: int = 8,
) -> DiffusionPipelineResult:
    """Trace a HuggingFace Diffusers pipeline for inference simulation.

    Automatically extracts the text encoder, denoiser (UNet/Transformer), and
    VAE decoder from the pipeline, constructs appropriate example inputs, and
    traces each component.

    Args:
        pipeline: A diffusers DiffusionPipeline instance.
        config: SimulatorConfig with HardwareInfo.
        diffusion_config: DiffusionConfig (steps, guidance, etc.).
        prompt_length: Token length for the text encoder input.
        height: Output image/video height in pixels.
        width: Output image/video width in pixels.
        num_frames: Number of video frames (1 for image models).
        latent_channels: Number of latent channels (typically 4 or 16).
        vae_scale_factor: Spatial downsampling factor of the VAE (typically 8).

    Returns:
        DiffusionPipelineResult with per-stage graphs and timing.

    Example:
        >>> from diffusers import StableDiffusionPipeline
        >>> from syssim import HardwareInfo, SimulatorConfig, DiffusionConfig
        >>> pipe = StableDiffusionPipeline.from_pretrained("...", torch_dtype=torch.float16)
        >>> hw = HardwareInfo(989.0, 989.0, 3350.0)
        >>> result = trace_diffusers_pipeline(
        ...     pipe, SimulatorConfig(hw_info=hw),
        ...     DiffusionConfig(num_inference_steps=50),
        ...     height=512, width=512,
        ... )
        >>> print(result.summary())
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError(
            "diffusers package is required for Diffusers integration. "
            "Install with: pip install diffusers"
        )

    if diffusion_config is None:
        diffusion_config = DiffusionConfig(num_frames=num_frames)

    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor
    batch_size = 1

    # --- Extract and prepare text encoder ---
    text_encoder = None
    text_encoder_inputs = None
    if hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None:
        text_encoder = pipeline.text_encoder
        text_encoder.eval()
        text_encoder_inputs = {
            "input_ids": torch.randint(0, 30000, (batch_size, prompt_length)),
        }

    # --- Extract and prepare denoiser ---
    denoise_model = _get_denoiser(pipeline)
    denoise_model.eval()
    denoise_inputs = _build_denoise_inputs(
        pipeline, denoise_model,
        batch_size=batch_size,
        latent_channels=latent_channels,
        latent_h=latent_h, latent_w=latent_w,
        num_frames=num_frames,
        prompt_length=prompt_length,
    )

    # --- Extract and prepare VAE decoder ---
    vae_decoder = None
    vae_decoder_inputs = None
    if hasattr(pipeline, "vae") and pipeline.vae is not None:
        vae_decoder = pipeline.vae.decoder if hasattr(pipeline.vae, "decoder") else None
        if vae_decoder is not None:
            if num_frames > 1:
                # Video: (B, C, T, H, W) latent
                vae_decoder_inputs = torch.randn(
                    batch_size, latent_channels, num_frames, latent_h, latent_w,
                )
            else:
                vae_decoder_inputs = torch.randn(
                    batch_size, latent_channels, latent_h, latent_w,
                )

    return trace_diffusion_pipeline(
        denoise_model=denoise_model,
        denoise_inputs=denoise_inputs,
        config=config,
        diffusion_config=diffusion_config,
        text_encoder=text_encoder,
        text_encoder_inputs=text_encoder_inputs,
        vae_decoder=vae_decoder,
        vae_decoder_inputs=vae_decoder_inputs,
    )


def _get_denoiser(pipeline: DiffusionPipeline) -> nn.Module:
    """Extract the denoising backbone from a diffusers pipeline."""
    # DiT / Transformer-based pipelines (Wan, PixArt, SD3, Flux, etc.)
    if hasattr(pipeline, "transformer") and pipeline.transformer is not None:
        return pipeline.transformer
    # UNet-based pipelines (SD 1.x/2.x, SDXL, etc.)
    if hasattr(pipeline, "unet") and pipeline.unet is not None:
        return pipeline.unet
    raise ValueError(
        "Cannot find denoiser in pipeline. Expected 'transformer' or 'unet' attribute."
    )


def _build_denoise_inputs(
    pipeline: DiffusionPipeline,
    denoise_model: nn.Module,
    *,
    batch_size: int,
    latent_channels: int,
    latent_h: int,
    latent_w: int,
    num_frames: int,
    prompt_length: int,
) -> dict[str, torch.Tensor]:
    """Build synthetic example inputs for one denoising step."""
    # Determine hidden size from the model config
    hidden_size = getattr(
        getattr(denoise_model, "config", None), "cross_attention_dim", 1024
    )

    # Noisy latent sample
    if num_frames > 1:
        # Video model: (B, C, T, H, W)
        hidden_states = torch.randn(
            batch_size, latent_channels, num_frames, latent_h, latent_w,
        )
    else:
        # Image model: (B, C, H, W)
        hidden_states = torch.randn(
            batch_size, latent_channels, latent_h, latent_w,
        )

    inputs: dict[str, Any] = {
        "hidden_states": hidden_states,
        "timestep": torch.tensor([500]),  # mid-schedule timestep
        "encoder_hidden_states": torch.randn(
            batch_size, prompt_length, hidden_size,
        ),
    }

    return inputs


def build_wan2_2_inputs(
    *,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    prompt_length: int = 512,
    latent_channels: int = 16,
    vae_scale_factor: int = 8,
    cross_attention_dim: int = 4096,
    batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    """Build example denoising inputs matching Wan2.2 architecture.

    Wan2.2 uses a 3D DiT that operates on video latents of shape
    (B, C, T, H, W) where T = ceil(num_frames / temporal_compression).
    The temporal compression factor is 4 and spatial is 8.

    Args:
        height: Output video height (pixels). Default 480.
        width: Output video width (pixels). Default 832.
        num_frames: Output video frames. Default 81 (5s @ 16fps + 1).
        prompt_length: Text token sequence length. Default 512.
        latent_channels: VAE latent channels. Default 16 for Wan2.2.
        vae_scale_factor: VAE spatial downscale. Default 8.
        cross_attention_dim: Text encoder hidden dim. Default 4096 (UMT5-XXL).
        batch_size: Batch size. Default 1.

    Returns:
        Dict of tensors suitable as denoiser inputs.
    """
    temporal_compression = 4
    latent_t = (num_frames - 1) // temporal_compression + 1  # ceil division
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor

    return {
        "hidden_states": torch.randn(
            batch_size, latent_channels, latent_t, latent_h, latent_w,
        ),
        "timestep": torch.tensor([500]),
        "encoder_hidden_states": torch.randn(
            batch_size, prompt_length, cross_attention_dim,
        ),
    }
