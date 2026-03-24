# Diffusion Model Support

This document describes SysSim's support for diffusion model performance simulation, using Wan2.2 as the reference implementation.

For the core SysSim architecture (tracing, roofline model, network simulator), see [DESIGN.md](DESIGN.md).

---

## 1. Motivation

Diffusion models (Stable Diffusion, SDXL, Wan2.2, etc.) have fundamentally different execution patterns from autoregressive LLMs:

| Characteristic | LLM Inference | Diffusion Inference |
|---------------|---------------|-------------------|
| Core loop | Autoregressive token generation | Iterative denoising (fixed steps) |
| Per-step shape | Varies (prefill=full seq, decode=1 token) | Constant (same latent shape every step) |
| Components | Single model | Pipeline: text encoder → denoiser → VAE decoder |
| KV cache | Yes (grows with sequence) | No |
| Classifier-free guidance | N/A | 2× forward passes per step when guidance > 1.0 |
| Dominant cost | Attention (memory-bound at decode) | MatMul/Attention in denoiser × num_steps |

These differences mean that a single `trace_model_for_inference()` call is insufficient — the simulator must understand the pipeline decomposition and iterative nature of diffusion.

---

## 2. Pipeline Decomposition

A diffusion inference pipeline consists of three sequential stages:

```
┌──────────────┐     ┌──────────────────────────┐     ┌──────────────┐
│ Text Encoder │     │     Denoiser (DiT/UNet)   │     │ VAE Decoder  │
│   (1× run)   │ ──▶ │  (N steps × CFG passes)  │ ──▶ │   (1× run)   │
│  e.g. T5/CLIP│     │  e.g. DiT, UNet           │     │  latent→pixel│
└──────────────┘     └──────────────────────────┘     └──────────────┘
```

**Total pipeline time**:
```
T_pipeline = T_text_encoder + T_denoise_step × num_steps × cfg_multiplier + T_vae_decoder
```

Where:
- **T_text_encoder**: One-time text encoding cost (typically small)
- **T_denoise_step**: Single denoising step cost (dominant term)
- **num_steps**: Number of denoising iterations (e.g. 20, 50)
- **cfg_multiplier**: 2 if classifier-free guidance is active (guidance_scale > 1.0), else 1
- **T_vae_decoder**: One-time latent-to-pixel decoding cost

**Key insight — trace once, multiply by N**: Every denoising step operates on identical tensor shapes (the latent remains the same size throughout the diffusion process). Therefore the operator graph for step 1 is structurally identical to step 50. We trace a single step and multiply by `num_steps × cfg_multiplier` to get the total denoising cost.

---

## 3. Architecture

### 3.1 Configuration

**DiffusionConfig** (`config.py`):
```python
@dataclass
class DiffusionConfig:
    num_inference_steps: int = 50      # Denoising iterations
    guidance_scale: float = 7.5        # CFG scale (>1.0 doubles passes)
    num_frames: int = 1                # Video frames (1 for image models)

    @property
    def cfg_multiplier(self) -> int:
        return 2 if self.guidance_scale > 1.0 else 1
```

**ExecutionMode.DIFFUSION_DENOISE** (`config.py`):
A new execution mode for single denoising step tracing. This mode:
- Runs forward-only (no backward pass)
- Does NOT activate KV-cache decode path (no cache_seq_len adjustment)
- Uses the standard roofline path, same as PREFILL

The mode exists as a semantic marker — it allows future specialization (e.g. diffusion-specific efficiency models, timestep-aware estimation) without breaking the existing code path.

### 3.2 API

**trace_diffusion_pipeline()** (`api.py`):

The primary API for diffusion model simulation. Traces each pipeline component independently:

```python
def trace_diffusion_pipeline(
    denoise_model: nn.Module,          # DiT, UNet, or any denoiser
    denoise_inputs: Any,               # (latent, timestep, text_emb)
    config: SimulatorConfig,
    diffusion_config: DiffusionConfig | None = None,
    text_encoder: nn.Module | None = None,
    text_encoder_inputs: Any = None,
    vae_decoder: nn.Module | None = None,
    vae_decoder_inputs: Any = None,
) -> DiffusionPipelineResult:
```

**Execution flow**:

1. **Trace denoiser** (required): Creates `OperatorGraphTracer` with `DIFFUSION_DENOISE` mode, traces one forward pass of the denoising model.
2. **Trace text encoder** (optional): Creates separate tracer with `PREFILL` mode, traces text encoding.
3. **Trace VAE decoder** (optional): Creates separate tracer with `PREFILL` mode, traces VAE decoding.
4. **Compute aggregate time**: Combines per-stage critical paths with the multiplication formula.

Each component is traced with an independent `OperatorGraphTracer` instance — this ensures clean dependency graphs without cross-component aliasing.

**DiffusionPipelineResult** (`api.py`):

```python
@dataclass
class DiffusionPipelineResult:
    denoise_step_graph: OperatorGraph   # Graph for ONE denoising step
    text_encoder_graph: OperatorGraph | None
    vae_decoder_graph: OperatorGraph | None
    diffusion_config: DiffusionConfig
    denoise_step_ms: float              # Critical path of one step
    text_encoder_ms: float
    vae_decoder_ms: float
    total_pipeline_ms: float            # Aggregate pipeline time
```

The result preserves per-stage operator graphs for detailed analysis (operator breakdown, bottleneck identification) while also providing the aggregate pipeline timing.

### 3.3 Diffusers Integration

**trace_diffusers_pipeline()** (`integrations/diffusers.py`):

Convenience wrapper that extracts components from a HuggingFace Diffusers `DiffusionPipeline`:

1. **Text encoder**: `pipeline.text_encoder` (if present)
2. **Denoiser**: `pipeline.transformer` (DiT-based) or `pipeline.unet` (UNet-based)
3. **VAE decoder**: `pipeline.vae.decoder` (if present)

The integration handles:
- Automatic latent shape calculation from pixel dimensions and VAE scale factor
- Synthetic input construction for each component
- Cross-attention dimension extraction from model config

**build_wan2_2_inputs()** (`integrations/diffusers.py`):

Helper that constructs denoiser inputs matching Wan2.2's architecture:
- 5D video latent: `(B, 16, T//4, H//8, W//8)` — 16 latent channels, 4× temporal and 8× spatial compression
- Timestep: scalar tensor
- Text embeddings: `(B, seq_len, 4096)` — UMT5-XXL hidden dimension

---

## 4. Wan2.2 Reference Implementation

Wan2.2 is an Alibaba video generation model that serves as the reference for video diffusion support.

### 4.1 Architecture Overview

```
Prompt ──▶ UMT5-XXL ──▶ [B, 512, 4096] text embeddings
                              │
Noise  ──▶ 3D DiT ◀──────────┘ (cross-attention)
           │  30 layers, hidden=1536, 24 heads
           │  3D patch embedding (1×2×2)
           │  Self-attn + Cross-attn + FFN per block
           ▼
Denoised latent ──▶ 3D VAE Decoder ──▶ Video frames
```

**Key architectural parameters** (Wan2.2-1.3B):

| Parameter | Value |
|-----------|-------|
| Latent channels | 16 |
| VAE spatial compression | 8× |
| VAE temporal compression | 4× |
| DiT hidden size | 1536 |
| DiT layers | 30 |
| DiT attention heads | 24 |
| Patch size | (1, 2, 2) — temporal × height × width |
| Text encoder | UMT5-XXL (cross_attention_dim=4096) |
| Default resolution | 480×832, 81 frames (5s @ 16fps + 1) |

### 4.2 Latent Shape Calculation

For a video of `(H, W, F)` pixels/frames:
```
latent_T = ceil(F / temporal_compression)  = ceil(81 / 4) = 21
latent_H = H / spatial_compression         = 480 / 8      = 60
latent_W = W / spatial_compression         = 832 / 8      = 104
```

After patchification `(pt=1, ph=2, pw=2)`:
```
num_patches = (latent_T / pt) × (latent_H / ph) × (latent_W / pw)
            = 21 × 30 × 52 = 32,760 tokens
```

This 32K-token sequence is what the DiT transformer operates on — making self-attention the dominant compute cost.

### 4.3 Compute Profile

For a single denoising step with the sequence above:
- **Self-attention**: O(seq² × hidden) per layer → 32,760² × 1,536 × 30 layers
- **Cross-attention**: O(seq × text_seq × hidden) per layer → 32,760 × 512 × 1,536 × 30 layers
- **FFN**: O(seq × hidden × 4 × hidden) per layer → 32,760 × 1,536 × 6,144 × 30 layers

With 50 steps and CFG (2× per step), the denoiser runs **100 times** — this is why per-step optimization matters.

### 4.4 Example

The example at `examples/diffusion/simulate_wan2_2.py` demonstrates end-to-end simulation:

```python
from syssim import HardwareInfo, SimulatorConfig
from syssim.config import DiffusionConfig
from syssim.api import trace_diffusion_pipeline

# Hardware
hw = HardwareInfo(peak_tflops_mm=989.0, peak_tflops_math=989.0,
                  peak_memory_bandwidth_gbps=3350.0)

# Build simplified DiT (no real weights needed)
with torch.device("meta"):
    dit = SimpleWanDiT(latent_channels=16, hidden_size=1536,
                       num_layers=30, num_heads=24)

# Trace
result = trace_diffusion_pipeline(
    denoise_model=dit,
    denoise_inputs={"hidden_states": latent, "timestep": t, "encoder_hidden_states": text_emb},
    config=SimulatorConfig(hw_info=hw),
    diffusion_config=DiffusionConfig(num_inference_steps=50, guidance_scale=7.5, num_frames=81),
)
print(result.summary())
```

This runs without diffusers or model weights — the simplified DiT captures the computational profile (same operator types and tensor shapes) of the real Wan2.2 model.

---

## 5. Design Decisions

### 5.1 Why Arithmetic Composition Instead of Graph Merging?

The pipeline stages are sequential and independent — there is no data dependency between step 1's graph and step 2's graph (they process different noise levels on the same shape). Merging N identical graphs would:
1. Create a massive graph (100× nodes) with no analytical benefit
2. Make critical path computation slower without improving accuracy
3. Lose the clear per-stage breakdown

Arithmetic composition (`step_time × N`) is exact for identical-shape iterations.

### 5.2 Why a Separate ExecutionMode?

`DIFFUSION_DENOISE` currently follows the same code path as `PREFILL`. The separate mode exists for:
1. **Semantic clarity**: A denoising step is not a "prefill" — the intent is different
2. **Future extensibility**: Timestep-aware efficiency models, diffusion-specific operator patterns (e.g. AdaLN modulation), video-specific optimizations
3. **Profiling**: Allows collecting diffusion-specific profiling data tagged by mode

### 5.3 Why Component-Level Tracing?

Each pipeline component is traced with a separate `OperatorGraphTracer`:
1. **Isolation**: No false dependencies between text encoder and denoiser tensors
2. **Modularity**: Users can trace only the denoiser (dominant cost) without needing the full pipeline
3. **Flexibility**: Components can be swapped (e.g. different text encoders) without re-tracing everything

---

## 6. Comparison with LLM Tracing

```
                    LLM                              Diffusion
                   ┌──────┐                         ┌──────────────────┐
  API layer        │trace_│ → OperatorGraph         │trace_diffusion_  │ → DiffusionPipelineResult
                   │model │                         │pipeline          │   (3 graphs + arithmetic)
                   └──┬───┘                         └──┬───┬───┬──────┘
                      │                                │   │   │
  Tracer layer     trace()                         trace() trace() trace()  ← same tracer
                      │                                │   │   │
  Roofline layer   estimate_runtime()              estimate_runtime()       ← same estimator
                   (DECODE has special path)        (no special path)
```

**What is shared** (zero code duplication):
- `OperatorGraphTracer` — same dispatch interception and dependency tracking
- `estimate_runtime()` — same roofline + efficiency pipeline
- `OperatorGraph` — same DAG IR and critical path analysis
- FLOP counting — same registry (convolution formulas already supported)

**What is new**:
- `DiffusionConfig` — steps, guidance scale, video frames
- `trace_diffusion_pipeline()` — multi-component orchestration + arithmetic composition
- `DiffusionPipelineResult` — multi-graph result with aggregate timing
- `integrations/diffusers.py` — HuggingFace Diffusers pipeline decomposition

---

## 7. Supported Diffusion Architectures

The implementation is architecture-agnostic — any PyTorch `nn.Module` can be traced as the denoiser. Tested patterns:

| Architecture | Pipeline Attribute | Example Models |
|-------------|-------------------|----------------|
| DiT (Diffusion Transformer) | `pipeline.transformer` | Wan2.2, PixArt, SD3, Flux |
| UNet | `pipeline.unet` | Stable Diffusion 1.x/2.x, SDXL |
| Any nn.Module | Direct `trace_diffusion_pipeline()` | Custom architectures |

The `_get_denoiser()` helper in the Diffusers integration auto-detects whether the pipeline uses a transformer or UNet backbone.

---

## 8. Limitations and Future Work

1. **Scheduler overhead not modeled**: The noise scheduler (DDPM, DDIM, Euler, etc.) performs lightweight tensor operations between denoising steps. These are negligible compared to the model forward pass but are not currently traced.

2. **VAE encoder not traced**: For image/video-to-image/video pipelines (e.g. img2img, inpainting), the VAE encoder is also needed. Currently only the decoder is supported.

3. **LoRA/IP-Adapter not modeled**: Adapter-based customizations modify the effective compute but are not explicitly supported.

4. **Multi-GPU diffusion**: Sequence parallelism and tensor parallelism for large DiT models (e.g. Wan2.2-14B) are not yet integrated with the network simulator.

5. **Timestep-dependent efficiency**: Real diffusion models may exhibit timestep-dependent performance patterns (e.g. early steps may trigger different attention patterns). The current model assumes uniform per-step cost.
