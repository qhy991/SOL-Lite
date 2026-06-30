"""Per-problem roofline analysis on H800 SXM5.

Each problem gets its own `(flops, bytes)` function derived from definition.json
semantics. Same-problem workload rows are classified into regimes; the right
metric (MFU / BW% / latency-floor) is reported per regime.

This is offline analysis — no GPU required. It tells us the *ceiling* and which
metric to report once a real solution is timed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

CONTEST_ROOT = Path(__file__).resolve().parent.parent / "data" / "benchmark" / "Contest"

# ---------------------------------------------------------------------------
# H800 SXM5 peaks
# ---------------------------------------------------------------------------
PEAK_BF16_FP16_FLOPS = 989e12       # Tensor Core dense, BF16/FP16, no sparsity
PEAK_FP8_FLOPS       = 1979e12      # Tensor Core dense, FP8
PEAK_HBM_BW          = 3.35e12      # HBM3, 80 GB SXM5
RIDGE_FP16           = PEAK_BF16_FP16_FLOPS / PEAK_HBM_BW   # 295
RIDGE_FP8            = PEAK_FP8_FLOPS       / PEAK_HBM_BW   # 591

# A kernel that runs in less than this is launch/occupancy-bound, not roofline-bound.
LATENCY_FLOOR_US = 5.0

DTYPE_BYTES = {
    "float32": 4, "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
    "int8": 1, "int32": 4, "int64": 8, "bool": 1,
}

# ---------------------------------------------------------------------------
# Per-problem semantics
# ---------------------------------------------------------------------------
@dataclass
class RooflineSpec:
    """Per-problem analytical model."""
    name: str
    peak_flops: float      # which dtype's tensor-core peak applies
    flops_bytes_fn: Callable[[dict], tuple[float, float]]  # row.axes -> (flops, bytes)
    notes: str = ""

    def analyze_row(self, axes: dict) -> dict:
        flops, bytes_ = self.flops_bytes_fn(axes)
        ai = flops / bytes_ if bytes_ else 0.0
        t_compute = flops / self.peak_flops          # seconds
        t_memory  = bytes_ / PEAK_HBM_BW
        t_sol     = max(t_compute, t_memory)
        ridge     = self.peak_flops / PEAK_HBM_BW

        if t_sol * 1e6 < LATENCY_FLOOR_US:
            regime = "latency"
        elif ai > 2 * ridge:
            regime = "compute"
        elif ai < 0.5 * ridge:
            regime = "memory"
        else:
            regime = "balanced"

        mfu_ceiling = min(1.0, PEAK_HBM_BW * ai / self.peak_flops)
        bw_ceiling  = min(1.0, self.peak_flops / (PEAK_HBM_BW * ai)) if ai > 0 else 1.0

        return {
            "axes": axes,
            "flops": flops,
            "bytes": bytes_,
            "ai": ai,
            "t_compute_us": t_compute * 1e6,
            "t_memory_us":  t_memory  * 1e6,
            "t_sol_us":     t_sol     * 1e6,
            "regime": regime,
            "mfu_ceiling": mfu_ceiling,
            "bw_ceiling":  bw_ceiling,
        }


# ---------------------------------------------------------------------------
# Problem 1: 005_gemm_n256_k7168  (FlashInfer-Bench)
#   C = A @ B.T,  A:[M,K] fp16, B:[N,K] fp16, C:[M,N] fp16
# ---------------------------------------------------------------------------
def gemm_005_flops_bytes(axes: dict) -> tuple[float, float]:
    M, N, K = axes["M"], 256, 7168
    flops = 2 * M * N * K
    bytes_ = (M * K + N * K + M * N) * 2   # all fp16
    return flops, bytes_


# ---------------------------------------------------------------------------
# Problem 2: 023_rmsnorm_h1536  (FlashInfer-Bench)
#   y = rmsnorm(x) * w,  x:[B,H] bf16, w:[H] bf16, y:[B,H] bf16
#   Algorithmic bytes: read x, read w (cached/amortized), write y
# ---------------------------------------------------------------------------
def rmsnorm_023_flops_bytes(axes: dict) -> tuple[float, float]:
    B, H = axes["batch_size"], 1536
    # FLOPs: per element, ~5 ops (square, accumulate, rsqrt amortized, scale, multiply by w).
    # rsqrt is per-row, costs ~H ops dominated by reduction. Total ≈ 5*B*H.
    flops = 5 * B * H
    # Bytes: x read once + y write once + w read once (amortized: only counted once total)
    bytes_ = (2 * B * H + H) * 2   # all bf16
    return flops, bytes_


# ---------------------------------------------------------------------------
# Problem 3: 003_lm_head_projection_with_logit_slicing  (L1)
#   logits[:,-K:,:] = hidden_states[:,-K:,:] @ weight.T
#   hidden_states:[B,S,H] bf16, weight:[V,H] bf16, logits:[B,K,V] bf16
#   K = logits_to_keep, H=2048, V=102400
#   Algorithmic semantics: only the LAST K positions need to be projected.
#   FLOPs = 2 * B * K * H * V    (NOT 2*B*S*H*V — the slice is the whole point)
#   Bytes = sliced_hidden + weight + logits
# ---------------------------------------------------------------------------
def lm_head_003_flops_bytes(axes: dict) -> tuple[float, float]:
    B = axes["batch_size"]
    K = axes["logits_to_keep"]
    H = 2048
    V = 102400
    flops = 2 * B * K * H * V
    # Bytes — semantically minimal: only the sliced hidden is read.
    # weight is read once (large: 200 MB, exceeds L2; treat as full read).
    bytes_ = (B * K * H + V * H + B * K * V) * 2
    return flops, bytes_


PROBLEMS = {
    "005_gemm_n256_k7168": RooflineSpec(
        name="005_gemm_n256_k7168",
        peak_flops=PEAK_BF16_FP16_FLOPS,
        flops_bytes_fn=gemm_005_flops_bytes,
        notes="DS-V3 MoE gate projection. N=256 caps asymptotic AI at 247 < ridge 295 — fundamentally memory-bound.",
    ),
    "023_rmsnorm_h1536": RooflineSpec(
        name="023_rmsnorm_h1536",
        peak_flops=PEAK_BF16_FP16_FLOPS,
        flops_bytes_fn=rmsnorm_023_flops_bytes,
        notes="DS-V3 RMSNorm. AI ~2.5 FLOPs/byte — far below ridge, pure memory-bound for all batch sizes.",
    ),
    "003_lm_head": RooflineSpec(
        name="003_lm_head_projection_with_logit_slicing",
        peak_flops=PEAK_BF16_FP16_FLOPS,
        flops_bytes_fn=lm_head_003_flops_bytes,
        notes="EXAONE LM head. V=102400, H=2048 — huge GEMM, asymptotic AI very high. Compute-bound when B*K large.",
    ),
}


# ---------------------------------------------------------------------------
# Workload loader & per-problem report
# ---------------------------------------------------------------------------
def load_workload(problem_subdir: str) -> list[dict]:
    path = CONTEST_ROOT / problem_subdir / "workload.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report(spec: RooflineSpec, workload_subdir: str) -> None:
    rows = [spec.analyze_row(r["axes"]) for r in load_workload(workload_subdir)]
    rows.sort(key=lambda r: r["t_sol_us"])

    print("=" * 100)
    print(f"PROBLEM: {spec.name}")
    print(f"  peak_flops = {spec.peak_flops/1e12:.0f} TFLOPS   peak_BW = {PEAK_HBM_BW/1e12:.2f} TB/s")
    print(f"  ridge = {spec.peak_flops/PEAK_HBM_BW:.1f} FLOPs/byte   notes: {spec.notes}")
    print("=" * 100)
    print(f"{'axes':<32} {'AI':>7} {'flops':>9} {'bytes':>9} {'t_sol':>9} {'regime':>10} {'mfu_max':>8} {'bw_max':>7}")
    print(f"{'':<32} {'fl/B':>7} {'(G)':>9} {'(MB)':>9} {'(us)':>9} {'':>10} {'':>8} {'':>7}")
    print("-" * 100)

    regime_counts = {}
    for r in rows:
        axes_str = ", ".join(f"{k}={v}" for k, v in r["axes"].items())
        if len(axes_str) > 30:
            axes_str = axes_str[:27] + "..."
        print(
            f"{axes_str:<32} {r['ai']:>7.1f} {r['flops']/1e9:>8.3f} {r['bytes']/1e6:>9.2f} "
            f"{r['t_sol_us']:>9.2f} {r['regime']:>10} {r['mfu_ceiling']:>8.3f} {r['bw_ceiling']:>7.3f}"
        )
        regime_counts[r["regime"]] = regime_counts.get(r["regime"], 0) + 1

    print("-" * 100)
    print(f"regime distribution: {regime_counts}")
    print(_recommend(spec, rows, regime_counts))
    print()


def _recommend(spec: RooflineSpec, rows: list[dict], regimes: dict[str, int]) -> str:
    n = len(rows)
    lines = ["RECOMMENDED REPORTING:"]
    if regimes.get("compute", 0) == n:
        lines.append("  → All rows compute-bound.  Report: MFU (achieved_flops / peak_flops).")
        lines.append("  → Single number per row; comparable across the whole workload.")
    elif regimes.get("memory", 0) == n:
        lines.append("  → All rows memory-bound.   Report: BW%  (achieved_bytes_per_s / peak_BW).")
        lines.append("  → Do NOT report MFU — it will look small but isn't meaningful here.")
    elif regimes.get("latency", 0) == n:
        lines.append("  → All rows latency-bound.  Report: t_measured (us) + speedup_vs_reference.")
        lines.append("  → Roofline metrics are unsafe; problem too small to reach steady state.")
    else:
        lines.append("  → MIXED regime.  Report per-row metric chosen by `regime` field:")
        if regimes.get("compute"):  lines.append(f"      compute  ({regimes['compute']} rows) → MFU")
        if regimes.get("memory"):   lines.append(f"      memory   ({regimes['memory']} rows) → BW%")
        if regimes.get("balanced"): lines.append(f"      balanced ({regimes['balanced']} rows) → both MFU and BW% (also report mfu_ceiling)")
        if regimes.get("latency"):  lines.append(f"      latency  ({regimes['latency']} rows) → time_us + speedup_vs_reference")
        lines.append("  → Aggregate scores across regimes are NOT meaningful — keep them separated.")
    return "\n".join(lines)


if __name__ == "__main__":
    report(PROBLEMS["003_lm_head"],         "L1/003_lm_head_projection_with_logit_slicing")
    report(PROBLEMS["023_rmsnorm_h1536"],   "FlashInfer-Bench/023_rmsnorm_h1536")
    report(PROBLEMS["005_gemm_n256_k7168"], "FlashInfer-Bench/005_gemm_n256_k7168")
