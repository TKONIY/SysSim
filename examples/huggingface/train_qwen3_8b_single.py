"""
Simulate Qwen3-8B single-GPU training on GH200 using syssim.

Constructs the Qwen3-8B architecture from its published specs using the Qwen2
model class (same underlying architecture: GQA, SwiGLU, RoPE). Creates
synthetic token inputs, traces a full training step (forward + backward),
and reports operator breakdown and critical path time.

Qwen3-8B published specs:
  - 36 hidden layers
  - Hidden size: 4096
  - Intermediate (FFN) size: 22016
  - Attention heads: 32
  - KV heads: 8 (4x GQA compression)
  - Vocab size: 152064

Run:
    srun -N 1 --gpus 1 python examples/huggingface/train_qwen3_8b_single.py
"""

import os
import sys

# Ensure repo root is on path when invoked via srun without pip install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from syssim import SimulatorConfig, get_hardware_info, trace_hf_model_for_training
from syssim.operator_graph import OperatorType

# Qwen3-8B published architecture dimensions
QWEN3_8B_CONFIG = dict(
    num_hidden_layers=36,
    hidden_size=4096,
    intermediate_size=22016,
    num_attention_heads=32,
    num_key_value_heads=8,
    vocab_size=152064,
    max_position_embeddings=32768,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    hidden_act="silu",
)

BATCH_SIZE = 1
SEQ_LEN = 2048


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def main():
    # --- Hardware ---
    hw, hw_name = get_hardware_info()
    print(f"Detected hardware: {hw_name}")
    print(f"  Peak MM TFLOP/s : {hw.peak_tflops_mm:.1f}")
    print(f"  Peak BW GB/s    : {hw.peak_memory_bandwidth_gbps:.1f}")
    print()

    # --- Model (meta device — no real tensor allocation) ---
    print("Building Qwen3-8B architecture (meta device, no memory allocation)...")
    model_cfg = Qwen2Config(**QWEN3_8B_CONFIG)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_cfg, torch_dtype=torch.bfloat16)
    model.train()

    n_params = param_count(model)
    print("Model: Qwen3-8B (Qwen2 arch with Qwen3-8B dims)")
    print(f"  Layers      : {model_cfg.num_hidden_layers}")
    print(f"  Hidden size : {model_cfg.hidden_size}")
    print(f"  FFN size    : {model_cfg.intermediate_size}")
    print(f"  Attn heads  : {model_cfg.num_attention_heads}")
    print(f"  KV heads    : {model_cfg.num_key_value_heads}")
    print(f"  Parameters  : {n_params / 1e9:.2f}B")
    print()

    # --- Synthetic inputs ---
    print(f"Input: batch={BATCH_SIZE}, seq_len={SEQ_LEN}")
    input_ids = torch.randint(0, model_cfg.vocab_size, (BATCH_SIZE, SEQ_LEN))
    inputs = {"input_ids": input_ids, "labels": input_ids.clone()}
    print()

    # --- Trace ---
    sim_cfg = SimulatorConfig(hw_info=hw)
    print("Tracing training step (forward + backward)...")
    graph = trace_hf_model_for_training(model, inputs, sim_cfg)
    print()

    # --- Report ---
    type_counts: dict[OperatorType, int] = {}
    for op in graph.operators.values():
        type_counts[op.op_type] = type_counts.get(op.op_type, 0) + 1

    print("Operator counts by type:")
    for op_type in OperatorType:
        count = type_counts.get(op_type, 0)
        if count:
            print(f"  {op_type.name:<12}: {count}")
    print()

    critical_path_ms = graph.compute_critical_path()
    print(f"Critical path time : {critical_path_ms:.2f} ms")
    print()

    print(graph.summary())


if __name__ == "__main__":
    main()
