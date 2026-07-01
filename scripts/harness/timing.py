"""Dual-mode timing engine for GPU kernel measurement.

Mode A — single-launch + L2 flush (for latency-bound kernels, t_sol < 5μs):
    Each iteration: flush L2 cache, launch once, record cuda.Event span.
    Returns median of N iterations.

Mode B — back-to-back amortized (for compute/memory-bound, t_sol ≥ 5μs):
    Warmup, then N groups of M back-to-back launches. Amortized per-call
    latency = median(group_elapsed / M).

Auto-switching: if t_sol_us is provided, Mode A is used when t_sol_us < 5μs.
"""
from __future__ import annotations

import statistics
import math

LATENCY_THRESHOLD_US = 5.0


def _need_torch():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for timing.")
    return torch


def _l2_flush(device_id: int = 0):
    """Flush L2 cache by zero-filling a buffer twice the L2 size.

    The L2 cache size is resolved from _hardware.py (default 50 MB for H800).
    """
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "scripts"))
    from _hardware import L2_SIZE

    torch = _need_torch()
    with torch.cuda.device(device_id):
        flush_size = int(2 * L2_SIZE // 4)  # float32 elements (cast to int for torch>=2.1)
        buf = torch.zeros(flush_size, dtype=torch.float32, device="cuda")
        buf.zero_()
        del buf


def time_single_launch(
    call_fn,
    iterations: int = 50,
    l2_flush: bool = True,
) -> float:
    """Mode A: single-launch per iteration, with optional L2 cache flush.

    Returns median latency in microseconds.
    """
    torch = _need_torch()
    timings_us = []

    for _ in range(iterations):
        if l2_flush:
            _l2_flush()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        call_fn()
        e.record()
        torch.cuda.synchronize()
        timings_us.append(s.elapsed_time(e) * 1e3)

    return statistics.median(timings_us)


def time_back_to_back(
    call_fn,
    n_batch: int = 30,
    warmup: int = 25,
    groups: int = 5,
) -> float:
    """Mode B: back-to-back launches, amortized per-call.

    Returns median per-call latency in microseconds.
    """
    torch = _need_torch()
    for _ in range(warmup):
        call_fn()
    torch.cuda.synchronize()

    per_call_us = []
    for _ in range(groups):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_batch):
            call_fn()
        e.record()
        torch.cuda.synchronize()
        per_call_us.append(s.elapsed_time(e) * 1e3 / n_batch)

    return statistics.median(per_call_us)


def time_kernel(
    call_fn,
    t_sol_us: float | None = None,
    n_batch: int = 30,
    warmup: int = 25,
    groups: int = 5,
    single_iterations: int = 50,
    l2_flush: bool = True,
    force_mode: str | None = None,
) -> dict:
    """Auto-select timing mode and return (latency_us, mode, metadata).

    Args:
        call_fn: zero-argument callable that invokes the kernel
        t_sol_us: speed-of-light lower bound — if < 5μs, use single-launch
        n_batch: back-to-back launches per group (Mode B)
        warmup: untimed warmup calls (Mode B)
        groups: timed groups (Mode B)
        single_iterations: timed iterations (Mode A)
        l2_flush: whether to flush L2 in Mode A
        force_mode: "single" | "batch" | None (auto)

    Returns:
        {"latency_us": float, "mode": "single" | "batch", "n_iterations": int}
    """
    if force_mode == "single":
        lat = time_single_launch(call_fn, single_iterations, l2_flush)
        return {"latency_us": lat, "mode": "single", "n_iterations": single_iterations}
    elif force_mode == "batch":
        lat = time_back_to_back(call_fn, n_batch, warmup, groups)
        return {"latency_us": lat, "mode": "batch", "n_iterations": groups * n_batch}
    else:
        # Auto-select
        if t_sol_us is not None and t_sol_us < LATENCY_THRESHOLD_US:
            lat = time_single_launch(call_fn, single_iterations, l2_flush)
            return {"latency_us": lat, "mode": "single", "n_iterations": single_iterations}
        else:
            lat = time_back_to_back(call_fn, n_batch, warmup, groups)
            return {"latency_us": lat, "mode": "batch", "n_iterations": groups * n_batch}