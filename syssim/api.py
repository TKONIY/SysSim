"""Public API for rlsysim tracing and simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from .config import DiffusionConfig, ExecutionMode, SimulatorConfig
from .tracer import OperatorGraphTracer
from .operator_graph import OperatorGraph


def trace_model_for_training(
    model: nn.Module,
    example_inputs: Any,
    config: SimulatorConfig,
    loss_fn: Any = None,
) -> OperatorGraph:
    """Trace a PyTorch model for training (forward + backward).

    Args:
        model: The PyTorch model to trace.
        example_inputs: Example inputs for shape inference (tensor, tuple, list, or dict).
        config: SimulatorConfig with HardwareInfo for roofline estimation.
        loss_fn: Callable that reduces the model output to a scalar for backward.
                 Defaults to ``lambda out: out.sum()``.

    Returns:
        An OperatorGraph containing the traced operations and their dependencies.
    """
    tracer = OperatorGraphTracer(
        hw_info=config.hw_info,
        execution_mode=ExecutionMode.TRAINING,
        cache_seq_len=0,
    )
    return tracer.trace(model, example_inputs, forward_backward=True, loss_fn=loss_fn)


def trace_model_for_inference(
    model: nn.Module,
    example_inputs: Any,
    config: SimulatorConfig,
    mode: str = "prefill",
) -> OperatorGraph:
    """Trace a PyTorch model for inference (forward only).

    Args:
        model: The PyTorch model to trace.
        example_inputs: Example inputs for shape inference (tensor, tuple, list, or dict).
        config: SimulatorConfig with HardwareInfo for roofline estimation.
        mode: Inference mode, either "prefill" or "decode".

    Returns:
        An OperatorGraph containing the traced operations and their dependencies.
    """
    mode_map = {
        "prefill": ExecutionMode.PREFILL,
        "decode": ExecutionMode.DECODE,
    }
    if mode not in mode_map:
        raise ValueError(f"Invalid inference mode '{mode}', expected 'prefill' or 'decode'")
    execution_mode = mode_map[mode]
    cache_seq_len = config.cache_seq_len if execution_mode == ExecutionMode.DECODE else 0

    tracer = OperatorGraphTracer(
        hw_info=config.hw_info,
        execution_mode=execution_mode,
        cache_seq_len=cache_seq_len,
    )
    return tracer.trace(model, example_inputs, forward_backward=False, loss_fn=None)


@dataclass
class DiffusionPipelineResult:
    """Result of tracing a diffusion pipeline.

    Contains operator graphs for each stage and aggregate timing.

    Attributes:
        denoise_step_graph: OperatorGraph for a single denoising step.
        text_encoder_graph: OperatorGraph for the text encoder (None if not traced).
        vae_decoder_graph: OperatorGraph for the VAE decoder (None if not traced).
        diffusion_config: The DiffusionConfig used for simulation.
        denoise_step_ms: Critical path time for one denoising step (ms).
        text_encoder_ms: Critical path time for text encoding (ms).
        vae_decoder_ms: Critical path time for VAE decoding (ms).
        total_pipeline_ms: Estimated total pipeline time (ms), accounting for
            num_inference_steps and classifier-free guidance.
    """
    denoise_step_graph: OperatorGraph
    text_encoder_graph: OperatorGraph | None
    vae_decoder_graph: OperatorGraph | None
    diffusion_config: DiffusionConfig
    denoise_step_ms: float
    text_encoder_ms: float
    vae_decoder_ms: float
    total_pipeline_ms: float

    def summary(self) -> str:
        cfg = self.diffusion_config
        lines = [
            "=== Diffusion Pipeline Simulation ===",
            f"Denoising steps     : {cfg.num_inference_steps}",
            f"Guidance scale      : {cfg.guidance_scale} (CFG passes/step: {cfg.cfg_multiplier})",
        ]
        if cfg.num_frames > 1:
            lines.append(f"Video frames        : {cfg.num_frames}")
        lines += [
            "",
            f"Text encoder        : {self.text_encoder_ms:.2f} ms",
            f"Single denoise step : {self.denoise_step_ms:.2f} ms",
            f"  x {cfg.num_inference_steps} steps x {cfg.cfg_multiplier} CFG"
            f" = {self.denoise_step_ms * cfg.num_inference_steps * cfg.cfg_multiplier:.2f} ms",
            f"VAE decoder         : {self.vae_decoder_ms:.2f} ms",
            "",
            f"Total pipeline      : {self.total_pipeline_ms:.2f} ms",
            "",
            "--- Denoise step breakdown ---",
            self.denoise_step_graph.summary(),
        ]
        return "\n".join(lines)


def trace_diffusion_pipeline(
    denoise_model: nn.Module,
    denoise_inputs: Any,
    config: SimulatorConfig,
    diffusion_config: DiffusionConfig | None = None,
    text_encoder: nn.Module | None = None,
    text_encoder_inputs: Any = None,
    vae_decoder: nn.Module | None = None,
    vae_decoder_inputs: Any = None,
) -> DiffusionPipelineResult:
    """Trace a diffusion pipeline and estimate end-to-end inference time.

    Traces each component (text encoder, denoiser, VAE decoder) independently,
    then computes total pipeline time accounting for num_inference_steps and
    classifier-free guidance.

    Args:
        denoise_model: The denoising backbone (e.g. DiT, U-Net).
        denoise_inputs: Example inputs for one denoising step
            (noisy latent, timestep, encoder_hidden_states, etc.).
        config: SimulatorConfig with HardwareInfo.
        diffusion_config: DiffusionConfig (defaults to 50 steps, guidance=7.5).
        text_encoder: Optional text encoder model.
        text_encoder_inputs: Example inputs for the text encoder.
        vae_decoder: Optional VAE decoder model.
        vae_decoder_inputs: Example inputs for the VAE decoder.

    Returns:
        DiffusionPipelineResult with per-stage graphs and aggregate timing.

    Example:
        >>> from syssim import HardwareInfo, SimulatorConfig, DiffusionConfig
        >>> hw = HardwareInfo(989.0, 989.0, 3350.0)
        >>> sim_cfg = SimulatorConfig(hw_info=hw)
        >>> diff_cfg = DiffusionConfig(num_inference_steps=50, guidance_scale=7.5)
        >>> result = trace_diffusion_pipeline(
        ...     denoise_model=dit, denoise_inputs=dit_inputs,
        ...     config=sim_cfg, diffusion_config=diff_cfg,
        ... )
        >>> print(result.summary())
    """
    if diffusion_config is None:
        diffusion_config = DiffusionConfig()

    # --- Trace denoiser (single step) ---
    tracer = OperatorGraphTracer(
        hw_info=config.hw_info,
        execution_mode=ExecutionMode.DIFFUSION_DENOISE,
        cache_seq_len=0,
    )
    denoise_step_graph = tracer.trace(
        denoise_model, denoise_inputs, forward_backward=False, loss_fn=None,
    )
    denoise_step_ms = denoise_step_graph.compute_critical_path()

    # --- Trace text encoder (optional) ---
    text_encoder_graph = None
    text_encoder_ms = 0.0
    if text_encoder is not None and text_encoder_inputs is not None:
        te_tracer = OperatorGraphTracer(
            hw_info=config.hw_info,
            execution_mode=ExecutionMode.PREFILL,
            cache_seq_len=0,
        )
        text_encoder_graph = te_tracer.trace(
            text_encoder, text_encoder_inputs, forward_backward=False, loss_fn=None,
        )
        text_encoder_ms = text_encoder_graph.compute_critical_path()

    # --- Trace VAE decoder (optional) ---
    vae_decoder_graph = None
    vae_decoder_ms = 0.0
    if vae_decoder is not None and vae_decoder_inputs is not None:
        vae_tracer = OperatorGraphTracer(
            hw_info=config.hw_info,
            execution_mode=ExecutionMode.PREFILL,
            cache_seq_len=0,
        )
        vae_decoder_graph = vae_tracer.trace(
            vae_decoder, vae_decoder_inputs, forward_backward=False, loss_fn=None,
        )
        vae_decoder_ms = vae_decoder_graph.compute_critical_path()

    # --- Compute total pipeline time ---
    total_denoise_ms = (
        denoise_step_ms
        * diffusion_config.num_inference_steps
        * diffusion_config.cfg_multiplier
    )
    total_pipeline_ms = text_encoder_ms + total_denoise_ms + vae_decoder_ms

    return DiffusionPipelineResult(
        denoise_step_graph=denoise_step_graph,
        text_encoder_graph=text_encoder_graph,
        vae_decoder_graph=vae_decoder_graph,
        diffusion_config=diffusion_config,
        denoise_step_ms=denoise_step_ms,
        text_encoder_ms=text_encoder_ms,
        vae_decoder_ms=vae_decoder_ms,
        total_pipeline_ms=total_pipeline_ms,
    )


def trace_model_for_training(
    model: nn.Module,
    example_inputs: Any,
    config: SimulatorConfig,
    loss_fn: Any = None,
    diffusion_config: DiffusionConfig | None = None,
) -> OperatorGraph:
    """Trace a PyTorch model for training (forward + backward).

    Args:
        model: The PyTorch model to trace.
        example_inputs: Example inputs for shape inference (tensor, tuple, list, or dict).
        config: SimulatorConfig with HardwareInfo for roofline estimation.
        loss_fn: Callable that reduces the model output to a scalar for backward.
                 Defaults to ``lambda out: out.sum()``.
        diffusion_config: Optional DiffusionConfig. When provided, the training
            step is traced with TRAINING mode but the result summary includes
            diffusion-specific context (e.g. steps per epoch).

    Returns:
        An OperatorGraph containing the traced operations and their dependencies.
    """
    tracer = OperatorGraphTracer(
        hw_info=config.hw_info,
        execution_mode=ExecutionMode.TRAINING,
        cache_seq_len=0,
    )
    return tracer.trace(model, example_inputs, forward_backward=True, loss_fn=loss_fn)


def set_efficiency_model_dir(model_dir: str) -> None:
    """Configure directory containing trained efficiency models.

    Args:
        model_dir: Path to directory with model files (*.pth).

    Example:
        >>> from syssim import set_efficiency_model_dir, trace_model_for_inference
        >>> set_efficiency_model_dir("./trained_models")
        >>> graph = trace_model_for_inference(model, inputs, config)
    """
    from .compute.efficiency_models import set_backend_dir
    set_backend_dir(model_dir)
