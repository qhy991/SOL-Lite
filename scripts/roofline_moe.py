"""Tier 2 (MoE) batch analyzer.

Three sub-templates:
  G-sim:    full sparse MoE (routing + dispatch + expert MLP + aggregation).
            Token-to-expert routing is data-dependent; we simulate to get
            per-expert token counts.
  G-route:  routing/scoring ONLY (no MLP). Deterministic given axes.
  G-dense:  every expert processes every token (no sparse dispatch).
            Deterministic given axes.

Output per workload row: realized FLOPs/bytes, AI, regime, plus MoE-specific
metrics (imbalance, serial_overhead) for G-sim.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

CONTEST_ROOT = Path(__file__).resolve().parent.parent / "data" / "benchmark" / "Contest"

from _hardware import (PEAK_BF16, PEAK_FP8, PEAK_BW, LATENCY_FLOOR_US)


# ---------------------------------------------------------------------------
# Per-row classification (same as tier-1)
# ---------------------------------------------------------------------------
def classify_regime(flops: float, bytes_: float, peak: float) -> str:
    ridge = peak / PEAK_BW
    ai = flops / bytes_ if bytes_ else 0.0
    t_sol = max(flops/peak, bytes_/PEAK_BW) * 1e6
    if t_sol < LATENCY_FLOOR_US:
        return "latency"
    if ai > 2*ridge:  return "compute"
    if ai < 0.5*ridge: return "memory"
    return "balanced"


# ---------------------------------------------------------------------------
# G-sim: full sparse MoE with routing simulation
# ---------------------------------------------------------------------------
@dataclass
class MoESpec:
    name: str
    subdir: str
    H: int                 # hidden size
    I: int                 # moe_intermediate size
    E: int                 # number of (routed) experts
    topk: int              # experts per token
    # Optional features
    shared_I: int = 0      # shared-expert intermediate size; 0 = no shared
    fused_gate_up: bool = False   # one weight of shape [2I, H] instead of two [I, H]
    has_residual: bool = False    # adds final residual+output bytes
    dtype_bytes: int = 2          # bf16=2, fp8=1 (for activations + weights)
    peak: float = PEAK_BF16
    # FP8 expert parallel: only num_local_experts get tokens
    local_experts: int | None = None
    note: str = ""

    def simulate_routing(self, T: int, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal((T, self.E)).astype(np.float32)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
        topk_idx = np.argpartition(-probs, self.topk-1, axis=1)[:, :self.topk]
        counts = np.bincount(topk_idx.ravel(), minlength=self.E)
        # If expert parallel, only count tokens on local experts
        if self.local_experts is not None:
            counts = counts[: self.local_experts]
        return dict(counts=counts, peak=int(counts.max()),
                    mean=float(counts.mean()), active=int((counts>0).sum()))

    def analyze_row(self, axes: dict, seed: int = 0) -> dict:
        T = _get_tokens(axes)
        routing = self.simulate_routing(T, seed)
        counts = routing["counts"]

        # ---- Stage A: gate matmul + softmax + topk (deterministic) ----
        gate_flops = 2 * T * self.H * self.E
        gate_bytes = (T*self.H + self.E*self.H + T*self.E) * self.dtype_bytes

        # ---- Stage B: expert MLPs (data-dependent) ----
        # Per expert with n tokens: gate+up GEMMs (2*n*H*I each) + silu*up
        # (4*n*I) + down GEMM (2*n*I*H). Bytes: input (n*H), weights
        # (3*H*I or [2I,H]+[H,I] = 3HI), output (n*H), all dtype_bytes.
        flops_per_expert = (2*counts.astype(np.int64)*self.H*self.I*3
                            + 4*counts.astype(np.int64)*self.I)
        bytes_per_expert = np.where(
            counts > 0,
            (2*self.H + 3*self.I) * counts.astype(np.int64) * self.dtype_bytes
            + 3 * self.H * self.I * self.dtype_bytes,
            0,
        )
        expert_flops = int(flops_per_expert.sum())
        expert_bytes = int(bytes_per_expert.sum())

        # ---- Stage B': shared expert (if any) — sees ALL tokens ----
        shared_flops = 0; shared_bytes = 0
        if self.shared_I:
            shared_flops = 2*T*self.H*self.shared_I*3 + 4*T*self.shared_I
            shared_bytes = ((2*self.H + 3*self.shared_I) * T
                            + 3*self.H*self.shared_I) * self.dtype_bytes

        # ---- Stage C: weighted aggregation + scatter ----
        scatter_flops = 2 * T * self.topk * self.H
        scatter_bytes = (T*self.topk*self.H + T*self.H) * self.dtype_bytes

        # ---- Optional residual ----
        residual_flops = 0; residual_bytes = 0
        if self.has_residual:
            residual_flops = T * self.H
            residual_bytes = 3 * T * self.H * self.dtype_bytes

        total_flops = (gate_flops + expert_flops + shared_flops
                       + scatter_flops + residual_flops)
        total_bytes = (gate_bytes + expert_bytes + shared_bytes
                       + scatter_bytes + residual_bytes)

        # Grouped-GEMM SoL vs serial-loop SoL
        t_sol_grouped = max(total_flops/self.peak, total_bytes/PEAK_BW) * 1e6
        per_expert_t = np.maximum(flops_per_expert/self.peak,
                                  bytes_per_expert/PEAK_BW)
        t_sol_serial = (per_expert_t.sum()
                        + max(gate_flops/self.peak, gate_bytes/PEAK_BW)
                        + (max(shared_flops/self.peak, shared_bytes/PEAK_BW)
                           if self.shared_I else 0)) * 1e6

        ai = total_flops / total_bytes if total_bytes else 0.0
        # Regime classification — based on the expert-MLP portion's per-token AI
        # at peak load (more meaningful than whole-row average)
        peak_tokens = max(routing["peak"], 1)
        per_token_ai = (2*self.H*self.I*3) / (
            (2*self.H + 3*self.I) * self.dtype_bytes
            + (3*self.H*self.I*self.dtype_bytes) / peak_tokens
        )
        ridge = self.peak / PEAK_BW
        if t_sol_grouped < LATENCY_FLOOR_US:
            regime = "latency"
        elif per_token_ai > 2*ridge:
            regime = "compute"
        elif per_token_ai < 0.5*ridge:
            regime = "memory"
        else:
            regime = "balanced"

        return dict(
            axes=axes, T=T,
            routing=dict(peak=int(routing["peak"]), mean=routing["mean"],
                         active=int(routing["active"])),
            imbalance=routing["peak"] / max(routing["mean"], 1e-9),
            flops=total_flops, bytes=total_bytes, ai=ai,
            expert_ai_at_peak=per_token_ai,
            t_sol_grouped_us=t_sol_grouped,
            t_sol_serial_us=t_sol_serial,
            serial_overhead=t_sol_serial / max(t_sol_grouped, 1e-9),
            regime=regime,
            mfu_ceiling=min(1.0, PEAK_BW*ai/self.peak),
        )


# ---------------------------------------------------------------------------
# G-route / G-dense: deterministic specs (no simulation)
# ---------------------------------------------------------------------------
@dataclass
class DetSpec:
    """Deterministic MoE spec — routing-only or dense (every expert all tokens)."""
    name: str
    subdir: str
    fn: Callable[[dict], tuple[float, float, float]]
    kind: str            # "route" or "dense"
    note: str = ""

    def analyze_row(self, axes: dict) -> dict:
        flops, bytes_, peak = self.fn(axes)
        ai = flops / bytes_ if bytes_ else 0.0
        t_sol = max(flops/peak, bytes_/PEAK_BW) * 1e6
        regime = classify_regime(flops, bytes_, peak)
        return dict(
            axes=axes, kind=self.kind, flops=flops, bytes=bytes_, ai=ai,
            t_sol_grouped_us=t_sol, t_sol_serial_us=t_sol,
            serial_overhead=1.0,
            regime=regime,
            mfu_ceiling=min(1.0, PEAK_BW*ai/peak),
            routing=None, imbalance=None, expert_ai_at_peak=None, T=None,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_tokens(axes: dict) -> int:
    if "num_tokens" in axes:    return axes["num_tokens"]
    if "batch_seq_len" in axes: return axes["batch_seq_len"]
    if "seq_len" in axes and "batch_size" in axes:
        return axes["batch_size"] * axes["seq_len"]
    if "seq_len" in axes:       return axes["seq_len"]
    raise KeyError(f"no token axis in {axes}")


def load_workload(subdir: str) -> list[dict]:
    return [json.loads(l) for l in (CONTEST_ROOT / subdir / "workload.jsonl"
                                    ).read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Routing-only / dense formulas (G-route, G-dense)
# ---------------------------------------------------------------------------
def routing_gemm_fn(H: int, E: int, dtype_bytes: int = 2,
                    peak: float = PEAK_BF16):
    """Hidden @ gate.T → (T, E) logits + softmax + topk. Pure routing GEMM."""
    def _fn(axes: dict):
        T = _get_tokens(axes)
        # FLOPs: GEMM + softmax (5*T*E) + topk (~T*E for sort)
        flops = 2*T*H*E + 6*T*E
        # Bytes: hidden, weight, logits write
        bytes_ = (T*H + E*H + T*E) * dtype_bytes
        return flops, bytes_, peak
    return _fn


def group_score_only_fn(E: int, n_group: int, dtype_bytes: int = 4):
    """L1/059: scores → group_mask, no GEMM. Just elementwise + topk per group."""
    def _fn(axes: dict):
        T = _get_tokens(axes)
        flops = 10 * T * E   # softmax + group aggregation + topk
        # Read scores [T, E] fp32, write masked_scores [T, E] + group_mask [T, n_group]
        bytes_ = (T*E + T*E + T*n_group) * dtype_bytes
        return flops, bytes_, PEAK_BF16
    return _fn


def dense_moe_fn(H: int, I: int, E: int, dtype_bytes: int = 2):
    """L1/076: every expert processes every token (no sparse dispatch).
    Weights: gate_up_proj [E, H, 2I], down_proj [E, I, H]. Routing weights applied.
    FLOPs dominated by E·T·H·(3I) and aggregation."""
    def _fn(axes: dict):
        T = _get_tokens(axes)
        # Each expert: 2I gate_up GEMM (2*T*H*2I) + silu*up (4*T*I) + down (2*T*I*H)
        per_expert_flops = 2*T*H*(2*I) + 4*T*I + 2*T*I*H
        flops = E * per_expert_flops + T*E*H        # + weighted sum aggregation
        # Bytes: hidden (T*H), routing_weights (T*E), 3 weights per expert (3*I*H each)
        # then per-expert intermediate (E*T*I bf16 if not fused), output (T*H)
        bytes_ = (T*H + T*E + E*3*I*H + E*T*H) * dtype_bytes   # T*H per expert output
        return flops, bytes_, PEAK_BF16
    return _fn


# ===========================================================================
# REGISTRY
# ===========================================================================
SIM_PROBLEMS: list[MoESpec] = [
    MoESpec("L1/044 moe_expert_mlp H=7168 I=2048 E=256 topk=8",
            "L1/044_moe_expert_computation",
            H=7168, I=2048, E=256, topk=8,
            note="given routing — only expert MLP + scatter"),
    MoESpec("L2/008 moe_routing_dispatch H=2048 I=768 E=128 topk=8",
            "L2/008_moe_sparse_routing_and_dispatch",
            H=2048, I=768, E=128, topk=8),
    MoESpec("L2/010 moe_expert_weighted_accum H=2048 I=768 E=128 topk=8",
            "L2/010_moe_expert_computation_with_weighted_accumulation",
            H=2048, I=768, E=128, topk=8,
            note="given routing weights + selected experts"),
    MoESpec("L2/012 moe_batched_capacity H=6144 I=2560 E=160 topk=8",
            "L2/012_moe_expert_batched_execution_with_capacity_factor",
            H=6144, I=2560, E=160, topk=8,
            note="capacity-factor batching — true cost may be lower if cap drops tokens"),
    MoESpec("L2/013 moe_with_shared H=2048 I=512 sh_I=512 E=512 topk=10",
            "L2/013_expert_weighted_aggregation_with_shared_expert",
            H=2048, I=512, E=512, topk=10, shared_I=512),
    MoESpec("L2/029 moe_dispatch_shared H=8192 I=3584 sh_I=7168 E=64 topk=8",
            "L2/029_moe_sparse_routing_and_dispatch",
            H=8192, I=3584, E=64, topk=8, shared_I=7168,
            note="Mixtral-style with HUGE shared expert (I_sh=7168)"),
    MoESpec("L2/048 moe_expert_inference H=4096 I=1024 E=256 topk=8",
            "L2/048_moe_expert_inference_batched_dispatch",
            H=4096, I=1024, E=256, topk=8),
    MoESpec("L2/065 sparse_dispatch_combine H=2880 I=2880 E=128 topk=4",
            "L2/065_sparse_expert_dispatch_and_combine",
            H=2880, I=2880, E=128, topk=4, fused_gate_up=True),
    MoESpec("L2/081 moe_dispatch_shared H=4096 I=1408 sh_I=1408 E=128 topk=8",
            "L2/081_moe_sparse_expert_dispatch",
            H=4096, I=1408, E=128, topk=8, shared_I=1408),
    MoESpec("L2/082 moe_full_with_residual H=5120 I=1536 sh_I=1536 E=160 topk=8",
            "L2/082_moe_layer_complete_forward_with_residual",
            H=5120, I=1536, E=160, topk=8, shared_I=1536, has_residual=True),
    MoESpec("FIB/020 fp8_moe H=7168 I=2048 E=256 local=32 topk=8",
            "FlashInfer-Bench/020_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
            H=7168, I=2048, E=256, topk=8, local_experts=32,
            fused_gate_up=True, dtype_bytes=1, peak=PEAK_FP8,
            note="FP8 expert parallel — only 32/256 experts get tokens locally"),
]

DET_PROBLEMS: list[DetSpec] = [
    DetSpec("L1/059 moe_group_score_agg E=256 n_group=8",
            "L1/059_moe_group_score_aggregation_and_masking",
            group_score_only_fn(E=256, n_group=8),
            kind="route",
            note="no GEMM — just softmax + group masking"),
    DetSpec("L2/049 group_limited_topk_routing H=4096 E=256 n_group=8",
            "L2/049_group_limited_topk_routing",
            routing_gemm_fn(H=4096, E=256),
            kind="route",
            note="hidden @ gate.T → topk over groups"),
    DetSpec("Quant/011 fp8_moe_gate_routing H=7168 E=256 (bf16 in/out)",
            "Quant/011_fp8_moe_gate_routing",
            routing_gemm_fn(H=7168, E=256, dtype_bytes=2, peak=PEAK_BF16),
            kind="route",
            note="bf16 declared in/out, kernel may go fp8 internally"),
    DetSpec("L1/076 batched_dense_moe H=2880 I=2880 E=128",
            "L1/076_batched_expert_forward",
            dense_moe_fn(H=2880, I=2880, E=128, dtype_bytes=4),  # fp32!
            kind="dense",
            note="dense MoE: ALL experts on ALL tokens. fp32 inputs."),
]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _smoke_select(rows: list) -> list:
    n = len(rows)
    return rows if n <= 3 else [rows[0], rows[n // 2], rows[-1]]


def report_sim(spec: MoESpec, smoke: bool = False) -> None:
    rows_in = load_workload(spec.subdir)
    results = [spec.analyze_row(r["axes"]) for r in rows_in]
    results.sort(key=lambda r: r["t_sol_grouped_us"])
    display_rows = _smoke_select(results) if smoke else results

    print("=" * 130)
    print(f"PROBLEM (MoE-sim): {spec.name}")
    print(f"  H={spec.H} I={spec.I} E={spec.E} topk={spec.topk}"
          + (f" shared_I={spec.shared_I}" if spec.shared_I else "")
          + (f" local={spec.local_experts}" if spec.local_experts else "")
          + (f"  dtype={'fp8' if spec.dtype_bytes==1 else 'bf16'}"
             f" peak={spec.peak/1e12:.0f} TFLOPS"))
    if spec.note: print(f"  note: {spec.note}")
    print("=" * 130)
    hdr = (f"{'axes':<25} {'T':>6} {'peak/avg':>9} {'AI':>7} "
           f"{'flops':>9} {'bytes':>9} {'t_sol_g':>9} {'serial':>8} "
           f"{'regime':>10} {'mfu_max':>8}")
    print(hdr)
    print(f"{'':<25} {'':>6} {'(tokens)':>9} {'fl/B':>7} "
          f"{'(G)':>9} {'(MB)':>9} {'(us)':>9} {'(x)':>8} {'':>10} {'':>8}")
    print("-" * 130)
    counts = {}
    for r in display_rows:
        axes = r["axes"]
        ax_str = (f"B={axes['batch_size']} S={axes['seq_len']}"
                  if "batch_size" in axes and "seq_len" in axes else
                  f"T={r['T']}")
        ax_str = ax_str[:23]
        rt = r["routing"]
        print(f"{ax_str:<25} {r['T']:>6} {rt['peak']:>4}/{rt['mean']:>4.0f} "
              f"{r['ai']:>7.1f} {r['flops']/1e9:>8.2f} {r['bytes']/1e6:>8.2f} "
              f"{r['t_sol_grouped_us']:>9.1f} {r['serial_overhead']:>7.1f}x "
              f"{r['regime']:>10} {r['mfu_ceiling']:>8.3f}")
    # regime counts over ALL rows (not just smoke subset)
    for r in results:
        counts[r["regime"]] = counts.get(r["regime"], 0) + 1
    print("-" * 130)
    print(f"regime: {counts}" + (f"  (showing {len(display_rows)}/{len(results)} smoke rows)" if smoke else ""))
    print()


def report_det(spec: DetSpec, smoke: bool = False) -> None:
    rows_in = load_workload(spec.subdir)
    results = [spec.analyze_row(r["axes"]) for r in rows_in]
    results.sort(key=lambda r: r["t_sol_grouped_us"])
    display_rows = _smoke_select(results) if smoke else results
    print("=" * 100)
    print(f"PROBLEM (MoE-{spec.kind}): {spec.name}")
    if spec.note: print(f"  note: {spec.note}")
    print("=" * 100)
    print(f"{'axes':<32} {'AI':>7} {'flops':>9} {'bytes':>9} {'t_sol':>9} {'regime':>10} {'mfu_max':>8}")
    print("-" * 100)
    counts = {}
    for r in display_rows:
        axes_str = ", ".join(f"{k}={v}" for k,v in r["axes"].items())[:30]
        print(f"{axes_str:<32} {r['ai']:>7.1f} {r['flops']/1e9:>8.3f} {r['bytes']/1e6:>9.2f} "
              f"{r['t_sol_grouped_us']:>9.2f} {r['regime']:>10} {r['mfu_ceiling']:>8.3f}")
    for r in results:
        counts[r["regime"]] = counts.get(r["regime"], 0) + 1
    print("-" * 100)
    print(f"regime: {counts}" + (f"  (showing {len(display_rows)}/{len(results)} smoke rows)" if smoke else "") + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MoE batch roofline analyzer")
    import _hardware; _hardware.add_hardware_arg(ap)
    ap.add_argument("--smoke", action="store_true",
                    help="show 3 representative workloads per problem (small/mid/large)")
    ap.add_argument("--problem", help="only run this problem (substring match on name)")
    args = ap.parse_args()
    for s in SIM_PROBLEMS:
        if args.problem and args.problem not in s.name: continue
        report_sim(s, smoke=args.smoke)
    for d in DET_PROBLEMS:
        if args.problem and args.problem not in d.name: continue
        report_det(d, smoke=args.smoke)
