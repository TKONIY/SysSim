"""
Simulate Wan2.2 video diffusion inference on GH200 using syssim.

Wan2.2 is a video generation diffusion model from Wan-AI. It uses:
  - UMT5-XXL text encoder (cross_attention_dim=4096)
  - 3D DiT (Diffusion Transformer) denoiser operating on video latents
  - 3D VAE for encoding/decoding video frames

This example traces the DiT denoiser component (the dominant cost) with
synthetic inputs matching Wan2.2-1.3B architecture, then estimates the
full pipeline time for 50 denoising steps with classifier-free guidance.

Wan2.2-1.3B published specs:
  - Latent channels: 16
  - VAE spatial compression: 8x, temporal compression: 4x
  - Default video: 480p (480x832), 81 frames (5s @ 16fps + 1)
  - Text encoder: UMT5-XXL (cross_attention_dim=4096)

Run:
    srun -N 1 --gpus 1 python examples/diffusion/simulate_wan2_2.py

Note:
    This example can run WITHOUT installing diffusers/Wan2.2 weights.
    It constructs synthetic inputs that match the model's expected shapes
    and traces an nn.Module that represents the DiT architecture. To trace
    the real Wan2.2 model, see the diffusers integration example below.
"""

import os
import sys

# Ensure repo root is on path when invoked via srun without pip install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from syssim import HardwareInfo, SimulatorConfig
from syssim.config import DiffusionConfig
from syssim.api import trace_diffusion_pipeline, DiffusionPipelineResult
from syssim.operator_graph import OperatorType

# ---- Wan2.2 architecture parameters ----
WAN2_2_1_3B = dict(
    latent_channels=16,
    vae_spatial_compression=8,
    vae_temporal_compression=4,
    cross_attention_dim=4096,
)

# ---- Video generation parameters ----
HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 81       # 5 seconds @ 16fps + 1
NUM_STEPS = 50
GUIDANCE_SCALE = 7.5
PROMPT_LENGTH = 512   # UMT5-XXL token length


class SimpleDiTBlock(nn.Module):
    """Simplified DiT block for tracing. Captures the operator pattern
    (self-attention + cross-attention + FFN) without needing real weights."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_mult: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_size)
        ffn_hidden = int(hidden_size * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_size),
        )

    def forward(self, x, encoder_hidden_states):
        # Self-attention
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h
        # Cross-attention
        h = self.norm2(x)
        h, _ = self.cross_attn(h, encoder_hidden_states, encoder_hidden_states)
        x = x + h
        # FFN
        h = self.norm3(x)
        h = self.ffn(h)
        x = x + h
        return x


class SimpleWanDiT(nn.Module):
    """Simplified Wan2.2-1.3B DiT for tracing.

    Flattens 3D video latent (B, C, T, H, W) into a sequence of patches,
    runs through transformer blocks with cross-attention to text embeddings,
    then reshapes back. This captures the computational profile without
    needing the real model weights.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_size: int = 1536,
        num_layers: int = 30,
        num_heads: int = 24,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        cross_attention_dim: int = 4096,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        # Patch embedding (project flattened patches into hidden_size)
        patch_dim = latent_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.patch_embed = nn.Linear(patch_dim, hidden_size)

        # Project text encoder output to match hidden_size
        self.text_proj = nn.Linear(cross_attention_dim, hidden_size)

        # Timestep embedding (MLP)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SimpleDiTBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])

        # Output projection back to patch space
        self.out_proj = nn.Linear(hidden_size, patch_dim)

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        B, C, T, H, W = hidden_states.shape
        pt, ph, pw = self.patch_size

        # Patchify: (B, C, T, H, W) -> (B, num_patches, patch_dim)
        num_patches_t = T // pt
        num_patches_h = H // ph
        num_patches_w = W // pw
        num_patches = num_patches_t * num_patches_h * num_patches_w

        x = hidden_states.reshape(
            B, C, num_patches_t, pt, num_patches_h, ph, num_patches_w, pw,
        )
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)  # (B, nT, nH, nW, C, pt, ph, pw)
        x = x.reshape(B, num_patches, -1)         # (B, num_patches, patch_dim)

        # Embed patches
        x = self.patch_embed(x)

        # Timestep conditioning (add to sequence)
        t_emb = self.time_embed(timestep.float().unsqueeze(-1))
        if t_emb.dim() == 2:
            t_emb = t_emb.unsqueeze(1)  # (B, 1, hidden)
        x = x + t_emb

        # Project text embeddings
        context = self.text_proj(encoder_hidden_states)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, context)

        # Un-patchify
        x = self.out_proj(x)  # (B, num_patches, patch_dim)
        x = x.reshape(B, num_patches_t, num_patches_h, num_patches_w, C, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        x = x.reshape(B, C, T, H, W)

        return x


def main():
    # --- Hardware ---
    # Use GH200 specs directly (works without CUDA device present)
    hw = HardwareInfo(
        peak_tflops_mm=989.0,
        peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0,
    )
    print("Hardware: GH200 (Grace Hopper)")
    print(f"  Peak MM TFLOP/s : {hw.peak_tflops_mm:.1f}")
    print(f"  Peak BW GB/s    : {hw.peak_memory_bandwidth_gbps:.1f}")
    print()

    # --- Model ---
    arch = WAN2_2_1_3B
    temporal_compression = arch["vae_temporal_compression"]
    spatial_compression = arch["vae_spatial_compression"]

    latent_t = (NUM_FRAMES - 1) // temporal_compression + 1
    latent_h = HEIGHT // spatial_compression
    latent_w = WIDTH // spatial_compression

    print(f"Wan2.2-1.3B DiT (simplified)")
    print(f"  Video: {HEIGHT}x{WIDTH}, {NUM_FRAMES} frames")
    print(f"  Latent: {arch['latent_channels']}x{latent_t}x{latent_h}x{latent_w}")
    print(f"  Prompt length: {PROMPT_LENGTH}")
    print()

    print("Building simplified DiT model (meta device)...")
    with torch.device("meta"):
        dit = SimpleWanDiT(
            latent_channels=arch["latent_channels"],
            hidden_size=1536,
            num_layers=30,
            num_heads=24,
            patch_size=(1, 2, 2),
            cross_attention_dim=arch["cross_attention_dim"],
        ).to(dtype=torch.bfloat16)
    dit.eval()

    n_params = sum(p.numel() for p in dit.parameters())
    print(f"  Parameters: {n_params / 1e9:.2f}B")
    print()

    # --- Inputs ---
    denoise_inputs = {
        "hidden_states": torch.randn(
            1, arch["latent_channels"], latent_t, latent_h, latent_w,
        ),
        "timestep": torch.tensor([500]),
        "encoder_hidden_states": torch.randn(
            1, PROMPT_LENGTH, arch["cross_attention_dim"],
        ),
    }

    # --- Trace ---
    sim_cfg = SimulatorConfig(hw_info=hw)
    diff_cfg = DiffusionConfig(
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        num_frames=NUM_FRAMES,
    )

    print(f"Tracing single denoising step...")
    result = trace_diffusion_pipeline(
        denoise_model=dit,
        denoise_inputs=denoise_inputs,
        config=sim_cfg,
        diffusion_config=diff_cfg,
    )
    print()

    # --- Report ---
    print(result.summary())
    print()

    # Operator breakdown
    graph = result.denoise_step_graph
    type_counts: dict[OperatorType, int] = {}
    type_times: dict[OperatorType, float] = {}
    for op in graph.operators.values():
        type_counts[op.op_type] = type_counts.get(op.op_type, 0) + 1
        type_times[op.op_type] = type_times.get(op.op_type, 0.0) + op.estimated_time_ms

    print("Operator breakdown (single denoise step):")
    print(f"  {'Type':<12} {'Count':>6} {'Time (ms)':>12} {'% of total':>10}")
    total_time = sum(type_times.values())
    for op_type in OperatorType:
        count = type_counts.get(op_type, 0)
        time = type_times.get(op_type, 0.0)
        if count:
            pct = 100.0 * time / total_time if total_time > 0 else 0.0
            print(f"  {op_type.name:<12} {count:>6} {time:>12.4f} {pct:>9.1f}%")


if __name__ == "__main__":
    main()
