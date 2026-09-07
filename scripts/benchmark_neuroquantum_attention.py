#!/usr/bin/env python3
"""Measure dense vs. bounded-window SDPA on the current CPU or Runpod GPU.

No datasets, checkpoints, or network access are needed. This benchmarks the
attention operation only, not full-model training or text-generation quality.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuroquantum_layered import LocalAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if min(args.length, args.window, args.heads, args.head_dim, args.batch_size, args.repeats) < 1:
        parser.error("all sizes and --repeats must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for a CPU-only measurement")

    device = torch.device(args.device)
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float32)
    if device.type == "cpu":
        torch.set_num_threads(1)
    torch.manual_seed(31)
    q, k, v = [torch.randn(args.batch_size, args.heads, args.length, args.head_dim,
                          device=device, dtype=dtype) for _ in range(3)]
    attention = LocalAttention(args.heads * args.head_dim, args.heads,
                               attention_window=args.window, dropout=0).eval()

    def dense():
        positions = torch.arange(args.length, device=device)
        allowed = ((positions[None, :] <= positions[:, None])
                   & (positions[None, :] > positions[:, None] - args.window))
        bias = torch.zeros(args.length, args.length, device=device, dtype=dtype)
        bias.masked_fill_(~allowed, float("-inf"))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0)

    def local():
        return attention._local_attention(q, k, v, None)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure(function):
        for _ in range(2):
            function()
        sync()
        baseline_bytes = 0
        if device.type == "cuda":
            baseline_bytes = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.repeats):
            function()
        sync()
        result = {"milliseconds": (time.perf_counter() - start) * 1000 / args.repeats}
        if device.type == "cuda":
            result["peak_extra_allocated_mib"] = (
                torch.cuda.max_memory_allocated(device) - baseline_bytes
            ) / 1024**2
        return result

    with torch.inference_mode():
        expected, actual = dense(), local()
        tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        max_error = (actual - expected).abs().max().item()
        del actual, expected
        dense_result = measure(dense)
        local_result = measure(local)
    print(json.dumps({
        "torch": torch.__version__, "device": str(device), "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config": vars(args), "max_absolute_output_error": max_error,
        "dense_reference": dense_result, "windowed_attention": local_result,
        "note": "Attention-only inference; dense reference builds its mask each call. GPU allocated memory excludes allocator reserve.",
    }, indent=2))


if __name__ == "__main__":
    main()
