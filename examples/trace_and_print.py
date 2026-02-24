"""End-to-end example: trace a model with diverse ops and print per-node timing."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from syssim import (
    HardwareInfo,
    SimulatorConfig,
    trace_model_for_inference,
    trace_model_for_training,
)


class DiverseModel(nn.Module):
    """A model that exercises many operator types:

    - GEMM:      Linear layers, BMM
    - ATTENTION:  scaled_dot_product_attention
    - COMPUTE:    Conv1d, BatchNorm1d, LayerNorm, GELU, Softmax, Dropout
    """

    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        # Conv + BatchNorm block (COMPUTE)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(embed_dim)

        # Attention projections (GEMM)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Feed-forward (GEMM + COMPUTE)
        self.ln = nn.LayerNorm(embed_dim)
        self.ff1 = nn.Linear(embed_dim, embed_dim * 4)
        self.ff2 = nn.Linear(embed_dim * 4, embed_dim)

        # Final (COMPUTE)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        b, s, d = x.shape

        # ── Conv + BatchNorm (COMPUTE ops) ──
        # Conv1d expects (batch, channels, seq_len)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(h)
        x = x + h

        # ── Multi-head attention (GEMM + ATTENTION) ──
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, d)
        x = x + self.out_proj(attn_out)

        # ── Feed-forward (GEMM + COMPUTE) ──
        h = self.ln(x)
        h = self.ff1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.ff2(h)
        x = x + h

        # ── Final softmax (COMPUTE) ──
        return self.softmax(x)


def print_table(graph, title):
    """Print operator nodes in a tabular form."""
    graph.compute_critical_path()

    w_name = max(len(op.name) for op in graph.operators.values())
    w_name = max(w_name, 4)
    header = (
        f"{'Name':<{w_name}}  {'Type':<10}  {'Time (ms)':>12}  "
        f"{'Start (ms)':>12}  {'Finish (ms)':>12}  {'Stream':>6}"
    )
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f" {title}")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    for name in graph.topological_sort():
        op = graph.operators[name]
        print(
            f"{op.name:<{w_name}}  {op.op_type.value:<10}  "
            f"{op.estimated_time_ms:>12.4e}  {op.earliest_start:>12.4e}  "
            f"{op.earliest_finish:>12.4e}  {op.stream_id:>6}"
        )

    print(sep)
    total = sum(op.estimated_time_ms for op in graph.operators.values())
    cp = max(op.earliest_finish for op in graph.operators.values())
    print(f"Total time: {total:.4e} ms | Critical path: {cp:.4e} ms | Ops: {len(graph)}")
    print()


def main():
    if not torch.cuda.is_available():
        print("ERROR: rlsysim requires a CUDA-capable device for tracing.")
        return

    # ── Hardware config (MI300X-like) ──
    hw = HardwareInfo(
        peak_tflops_mm=989.0,            # TFLOP/s for matrix multiply
        peak_tflops_math=989.0,          # TFLOP/s for vector unit
        peak_memory_bandwidth_gbps=3350.0,  # GB/s
    )

    model = DiverseModel(embed_dim=128, num_heads=4)

    # ── Training (forward + backward) ──
    config = SimulatorConfig(hw_info=hw)
    x = torch.randn(4, 64, 128).cuda()  # (batch=4, seq=64, embed=128)
    model.train()
    graph_train = trace_model_for_training(model, x, config)
    print_table(graph_train, "Training (forward + backward, batch=4, seq=64)")
    print(graph_train.summary())

    # ── Prefill (inference, full sequence) ──
    model.eval()
    graph = trace_model_for_inference(model, x, config, mode="prefill")
    print_table(graph, "Prefill (inference, batch=4, seq=64)")
    print(graph.summary())

    # ── Decode with KV cache ──
    config_decode = SimulatorConfig(hw_info=hw, cache_seq_len=2048)
    x_dec = torch.randn(1, 1, 128).cuda()  # (batch=1, seq=1, embed=128)
    graph_decode = trace_model_for_inference(model, x_dec, config_decode, mode="decode")
    print_table(graph_decode, "Decode (inference, batch=1, seq=1, cache_seq_len=2048)")
    print(graph_decode.summary())


if __name__ == "__main__":
    main()
