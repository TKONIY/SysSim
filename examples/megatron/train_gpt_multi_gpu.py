"""
Megatron-Core GPT-3 1.3B — Multi-GPU Training Trace (configurable TP)
======================================================================

Traces one forward + backward step of a GPT-3-style 1.3B model sharded
with Tensor Parallelism using syssim. TP size is configurable via --tp-size.

Architecture (per GPU rank, TP=4 example):
  Layers              : 24
  Hidden size         : 2048
  Attention heads     : 16  (16/TP per TP rank)
  FFN hidden size     : 8192 (8192/TP per TP rank)
  Vocab size          : 50257 (rounded up to multiple of TP size)
  Sequence length     : 2048
  Batch size          : 1

Run command (from repo root):
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate cpt
  srun -N 1 --gpus 1 python examples/megatron/train_gpt_multi_gpu.py [--tp-size {1,2,4,8,16}]
"""

import argparse
import math
import os
import sys
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

# Ensure repo root is on path (works when launched via srun from repo root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

from syssim import (
    HardwareInfo,
    SimulatorConfig,
    OperatorType,
    get_hardware_info,
    trace_model_for_training,
)

# ---------------------------------------------------------------------------
# Model config — GPT-3 1.3B
# ---------------------------------------------------------------------------
_VOCAB_SIZE_BASE = 50257  # Original; rounded up to multiple of tp_size at runtime
NUM_LAYERS = 24
HIDDEN_SIZE = 2048
NUM_ATTENTION_HEADS = 16  # Must be divisible by tp_size
FFN_HIDDEN_SIZE = 8192
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1
SEQ_LEN = 2048


def _vocab_size_for_tp(tp_size: int) -> int:
    """Round vocab size up to the nearest multiple of tp_size."""
    return math.ceil(_VOCAB_SIZE_BASE / tp_size) * tp_size


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def build_model(tp_size: int) -> GPTModel:
    """Build a Megatron GPT-3 1.3B model for the given TP size."""
    vocab_size = _vocab_size_for_tp(tp_size)
    transformer_config = TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        bf16=True,
        attention_softmax_in_fp32=True,
    )

    layer_spec = get_gpt_layer_local_spec()

    with torch.device("meta"):
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=MAX_SEQ_LEN,
            pre_process=True,   # PP=1: every rank has embedding
            post_process=True,  # PP=1: every rank has output head
            parallel_output=True,
        )
    return model.train()  # meta params; tracer converts to fake CUDA internally


def make_inputs(vocab_size: int) -> tuple:
    """Create synthetic training inputs (same on each rank)."""
    # CPU tensors — tracer converts to fake CUDA via _make_fake_inputs()
    input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN))
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH_SIZE, -1)
    attention_mask = None  # Megatron handles causal masking internally
    return (input_ids, position_ids, attention_mask)


def make_loss_fn(vocab_size: int):
    """Return a cross-entropy loss over the (sharded) vocab dimension."""
    def loss_fn(logits: torch.Tensor) -> torch.Tensor:
        # Called inside FakeTensorMode — device="cuda" creates fake CUDA labels
        labels = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device="cuda")
        # logits: [B, S, V/TP]  →  compute loss on local vocab shard
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
    return loss_fn


def print_results(graph, hw_name: str, rank: int, tp_size: int) -> None:
    """Print trace summary (rank 0 only)."""
    if rank != 0:
        return

    print("\n" + "=" * 60)
    print(f"  syssim — Megatron GPT-3 1.3B, TP={tp_size}")
    print("=" * 60)
    print(f"  Hardware : {hw_name}")
    print(f"  Rank     : 0 / {tp_size} (TP shard)")
    print(f"  Sequence : {SEQ_LEN}  Batch : {BATCH_SIZE}")
    print()

    # Operator counts by type
    counts: dict[str, int] = {}
    for node in graph.operators.values():
        t = node.op_type.name
        counts[t] = counts.get(t, 0) + 1

    print("  Operator counts by type:")
    for op_type in OperatorType:
        n = counts.get(op_type.name, 0)
        if n:
            print(f"    {op_type.name:<14} {n:4d}")

    total_ops = sum(counts.values())
    print(f"    {'TOTAL':<14} {total_ops:4d}")
    print()

    critical_path_ms = graph.compute_critical_path()
    print(f"  Critical path : {critical_path_ms:.3f} ms")
    print()
    print(graph.summary())
    print("=" * 60)


def _trace_rank(rank: int, world_size: int, port: int, tp_size: int) -> None:
    # -----------------------------------------------------------------------
    # 1. Distributed init
    # -----------------------------------------------------------------------
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(0)  # all ranks share cuda:0 (syssim uses FakeTensors)

    if rank == 0:
        print(f"Distributed init: {world_size} ranks, TP={tp_size}")

    # -----------------------------------------------------------------------
    # 2. Megatron parallel state
    # -----------------------------------------------------------------------
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        create_gloo_process_groups=False,  # skip gloo — CUDA-only environment
    )
    # Initialize model-parallel RNG state (required by TP attention dropout)
    model_parallel_cuda_manual_seed(42)

    # -----------------------------------------------------------------------
    # 3. Hardware info (auto-detect GH200)
    # -----------------------------------------------------------------------
    hw_info, hw_name = get_hardware_info()
    if rank == 0:
        print(f"Hardware detected: {hw_name}")
        print(f"  Peak MM TFLOP/s : {hw_info.peak_tflops_mm:.1f}")
        print(f"  Peak BW  GB/s   : {hw_info.peak_memory_bandwidth_gbps:.1f}")

    sim_cfg = SimulatorConfig(hw_info=hw_info)

    # -----------------------------------------------------------------------
    # 4. Build model
    # -----------------------------------------------------------------------
    vocab_size = _vocab_size_for_tp(tp_size)
    if rank == 0:
        print(f"\nBuilding GPT-3 1.3B (TP={tp_size}, vocab={vocab_size})...")
    model = build_model(tp_size)

    # -----------------------------------------------------------------------
    # 5. Trace forward + backward
    # -----------------------------------------------------------------------
    inputs = make_inputs(vocab_size)

    if rank == 0:
        print("Tracing forward + backward pass with syssim...")

    graph = trace_model_for_training(
        model,
        inputs,
        sim_cfg,
        loss_fn=make_loss_fn(vocab_size),
    )

    # -----------------------------------------------------------------------
    # 6. Report (rank 0 only)
    # -----------------------------------------------------------------------
    print_results(graph, hw_name, rank, tp_size)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Trace Megatron GPT-3 1.3B with syssim")
    parser.add_argument(
        "--tp-size",
        type=int,
        default=4,
        choices=[1, 2, 4, 8, 16],
        help="Tensor parallel size (default: 4). Must divide NUM_ATTENTION_HEADS=16.",
    )
    args = parser.parse_args()
    tp_size = args.tp_size

    port = _find_free_port()
    mp.spawn(_trace_rank, args=(tp_size, port, tp_size), nprocs=tp_size, join=True)


if __name__ == "__main__":
    main()
