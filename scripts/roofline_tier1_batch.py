"""Tier-1 batch analyzer: templates A (RMSNorm), B (GEMM), C (RoPE/position).

Each problem is one entry in PROBLEMS — a closure that takes axes dict and
returns (flops, bytes, peak_flops_for_dtype). The framework loads workload.jsonl,
classifies each row's regime, and prints the report.

This file is the per-problem analysis 'database' — adding a problem = adding
one entry. The classification and reporting logic is shared.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

CONTEST_ROOT = Path(__file__).resolve().parent.parent / "data" / "benchmark" / "Contest"

# H800 SXM5 peaks (override via env var SOL_LITE_HARDWARE=B200 / H200 / ...)
from _hardware import (PEAK_BF16, PEAK_FP8, PEAK_BW,
                       RIDGE_BF16, RIDGE_FP8, LATENCY_FLOOR_US)


@dataclass
class Problem:
    name: str            # display name
    subdir: str          # path relative to Contest/
    fn: Callable[[dict], tuple[float, float, float]]  # axes -> (flops, bytes, peak)
    note: str = ""


def classify(flops: float, bytes_: float, peak_flops: float) -> dict:
    ai = flops / bytes_ if bytes_ else 0.0
    t_c = flops / peak_flops
    t_m = bytes_ / PEAK_BW
    t_sol = max(t_c, t_m) * 1e6
    ridge = peak_flops / PEAK_BW
    if t_sol < LATENCY_FLOOR_US:
        regime = "latency"
    elif ai > 2 * ridge:
        regime = "compute"
    elif ai < 0.5 * ridge:
        regime = "memory"
    else:
        regime = "balanced"
    mfu_ceiling = min(1.0, PEAK_BW * ai / peak_flops)
    return dict(
        ai=ai, t_compute_us=t_c*1e6, t_memory_us=t_m*1e6,
        t_sol_us=t_sol, regime=regime, mfu_ceiling=mfu_ceiling,
    )


def _smoke_select(rows: list) -> list:
    """Pick small / mid / large representative workloads (borrowed from SOLBench-H800)."""
    n = len(rows)
    if n <= 3:
        return rows
    return [rows[0], rows[n // 2], rows[-1]]


def report(prob: Problem, smoke: bool = False) -> None:
    wpath = CONTEST_ROOT / prob.subdir / "workload.jsonl"
    rows_in = [json.loads(l) for l in wpath.read_text().splitlines() if l.strip()]
    results = []
    for r in rows_in:
        flops, bytes_, peak = prob.fn(r["axes"])
        m = classify(flops, bytes_, peak)
        results.append({"axes": r["axes"], "flops": flops, "bytes": bytes_,
                        "peak": peak, **m})
    results.sort(key=lambda x: x["t_sol_us"])
    display_rows = _smoke_select(results) if smoke else results

    print("=" * 110)
    print(f"PROBLEM: {prob.name}")
    if prob.note:
        print(f"  note: {prob.note}")
    print("=" * 110)
    print(f"{'axes':<32} {'AI':>8} {'flops':>9} {'bytes':>9} {'t_c':>7} {'t_m':>7} {'t_sol':>7} {'regime':>10} {'mfu_max':>8}")
    print(f"{'':<32} {'fl/B':>8} {'(G)':>9} {'(MB)':>9} {'(us)':>7} {'(us)':>7} {'(us)':>7} {'':>10} {'':>8}")
    print("-" * 110)
    counts = {}
    for r in display_rows:
        axes_str = ", ".join(f"{k}={v}" for k, v in r["axes"].items())[:30]
        print(f"{axes_str:<32} {r['ai']:>8.1f} {r['flops']/1e9:>8.3f} {r['bytes']/1e6:>9.2f} "
              f"{r['t_compute_us']:>7.2f} {r['t_memory_us']:>7.2f} {r['t_sol_us']:>7.2f} "
              f"{r['regime']:>10} {r['mfu_ceiling']:>8.3f}")
    # Regime counts are over ALL rows, not just displayed ones — gives the right
    # signal for "recommended metric" even in --smoke mode.
    for r in results:
        counts[r["regime"]] = counts.get(r["regime"], 0) + 1
    print("-" * 110)
    print(f"regime: {counts}" + (f"  (showing {len(display_rows)}/{len(results)} smoke rows)" if smoke else ""))
    _recommend(counts, len(results))
    print()


def _recommend(counts: dict, n: int) -> None:
    pure = next((k for k, v in counts.items() if v == n), None)
    if pure == "compute":
        print("→ ALL rows compute-bound. Report: MFU only.")
    elif pure == "memory":
        print("→ ALL rows memory-bound. Report: BW% only.")
    elif pure == "latency":
        print("→ ALL rows latency-bound. Report: t_us + speedup_vs_reference only.")
    else:
        parts = []
        if counts.get("compute"):  parts.append(f"compute({counts['compute']})→MFU")
        if counts.get("memory"):   parts.append(f"memory({counts['memory']})→BW%")
        if counts.get("balanced"): parts.append(f"balanced({counts['balanced']})→MFU+BW%+ceiling")
        if counts.get("latency"):  parts.append(f"latency({counts['latency']})→time+speedup")
        print("→ MIXED. Per-row metric: " + " | ".join(parts))


# ===========================================================================
# TEMPLATE A: RMSNorm  (5 problems)
#   y = rmsnorm(x [+ residual]) * weight
#   flops ≈ 5 * num_elements  (sum, mean, rsqrt amortized, scale, weight mul)
#   bytes = read x [+ residual] + read weight (small, amortized in L2) + write y
# ===========================================================================
def rmsnorm_fn(H: int, residual: bool, dtype_bytes: int = 2):
    def _fn(axes: dict):
        # axes may use 'batch_size' alone (B-only) or 'batch_size'+'seq_len' (BS).
        if "seq_len" in axes:
            N = axes["batch_size"] * axes["seq_len"]
        else:
            N = axes["batch_size"]
        # Ray-234 convention: pure memory-bound ops report flops=0 to keep
        # MFU cleanly interpretable (would otherwise show ~0.003% for RMSNorm)
        flops = 0
        # read x + (optional residual) + write y; weight is H, amortized but counted once
        in_tensors = 2 if residual else 1
        bytes_ = (in_tensors * N * H + N * H + H) * dtype_bytes
        return flops, bytes_, PEAK_BF16
    return _fn


# ===========================================================================
# TEMPLATE B: Pure GEMM  (3 problems — BF16 and FP8 variants)
#   C[M,N] = A[M,K] @ B[N,K].T
#   flops = 2*M*N*K
#   bytes = (M*K + N*K + M*N) * dtype  (FP8: 1 byte inputs/weights, 2 byte output)
# ===========================================================================
def gemm_fn(N: int, K: int, dtype_in_bytes: int, dtype_out_bytes: int, peak: float):
    def _fn(axes: dict):
        M = axes.get("M") or axes.get("batch_size", 1) * axes.get("seq_len", 1)
        flops = 2 * M * N * K
        bytes_ = (M * K + N * K) * dtype_in_bytes + M * N * dtype_out_bytes
        return flops, bytes_, peak
    return _fn


# ===========================================================================
# TEMPLATE C: RoPE / position encoding  (5 problems)
#   These compute cos/sin tables (small, often latency-bound) OR
#   apply rotation in-place (memory-heavy if writing whole cache).
# ===========================================================================
def rope_table_fn(D: int, dtype_bytes: int = 2):
    """RoPE cos/sin computation: output (..., D, 2). Tiny FLOPs vs output bytes."""
    def _fn(axes: dict):
        # Quant/014 only has seq_len. L1/011 has batch_size+seq_len.
        if "batch_size" in axes and "seq_len" in axes:
            N = axes["batch_size"] * axes["seq_len"]
        elif "seq_len" in axes:
            N = axes["seq_len"]
        elif "batch_size" in axes:
            N = axes["batch_size"]
        else:
            raise KeyError(f"rope_table_fn: no recognized axis in {axes}")
        # FLOPs: pure memory-bound RoPE (trig ops trivial relative to output bytes)
        # Ray-234 convention: report 0 to keep MFU cleanly interpretable.
        flops = 0
        # Bytes: read position_ids (N * 8 int64), read inv_freq (D/2 * 4 fp32),
        #        write cos_sin (N * D * 2 elements * dtype)
        bytes_ = N * 8 + (D // 2) * 4 + N * D * 2 * dtype_bytes
        return flops, bytes_, PEAK_BF16
    return _fn


def kv_cache_update_rope_fn(num_kv_heads: int, head_dim: int, dtype_bytes: int = 2):
    """KV cache update with RoPE.

    Reads: key_states, value_states (new), cos, sin, key_cache, value_cache (old).
    Writes: updated_key_cache, updated_value_cache (which include old + new positions).

    The dominant cost is the full cache copy if implementation isn't in-place.
    Algorithmic minimum: just the new positions need to be touched.
    """
    def _fn(axes: dict):
        B = axes["batch_size"]
        S_new = axes["new_seq_len"]
        S_cur = axes["current_seq_len"]
        S_updated = S_new + S_cur
        # Ray-234 convention: pure memory-bound (RoPE + copy), flops=0
        flops = 0
        # Bytes (algorithmic minimum, in-place cache update):
        #   read new k, v: 2 * B * H_kv * S_new * D
        #   read cos, sin: 2 * B * 1 * S_new * D
        #   write to cache slots: 2 * B * H_kv * S_new * D
        bytes_min = (5 * B * num_kv_heads * S_new * head_dim) * dtype_bytes
        return flops, bytes_min, PEAK_BF16
    return _fn


def multimodal_rope_position_fn():
    """L2/006: multimodal position-id calculation. Indexing-heavy, ~no FLOPs.

    Output is (3, B, S) int64 position_ids — pure memory write.
    """
    def _fn(axes: dict):
        B, S = axes["batch_size"], axes["seq_len"]
        # Ray-234 convention: indexing/lookup ops report 0 FLOPs
        flops = 0
        # Bytes: read input_ids (B*S*8), write position_ids (3*B*S*8)
        bytes_ = (B * S + 3 * B * S) * 8
        return flops, bytes_, PEAK_BF16
    return _fn


def multimodal_grid_rope_fn():
    """L1/023: multimodal 3D RoPE with grid-based indexing. Output is large
    patch_pos_embeds + cos + sin tables. Dominantly memory-bound.
    """
    def _fn(axes: dict):
        # vars: num_images, total_tokens
        T = axes["total_tokens"]
        H = 1536          # hidden_size
        D = 128           # head_dim
        N_pos = 1225      # num_position_embeddings
        # Output bytes: patch_pos_embeds[T, H] fp32 + cos[T, D] fp32 + sin[T, D] fp32
        out_bytes = T * H * 4 + 2 * T * D * 4
        # Input bytes: pos_embed_weight[N_pos, H] fp32 + grid_thw (small)
        in_bytes = N_pos * H * 4 + axes["num_images"] * 3 * 8
        # Ray-234 convention: indexing/gather is memory-bound; flops=0
        flops = 0
        return flops, in_bytes + out_bytes, PEAK_BF16
    return _fn


# ===========================================================================
# TEMPLATE D: MLA / fused QKV  (4 problems, all bf16)
#   Multiple GEMMs in series + one or two RMSNorms. No data dependence.
#   Algorithmic bytes: hidden read once + each weight read once + outputs write once.
#   Intermediate tensors (q_a, compressed_kv) stay in-register if fused.
# ===========================================================================
def mla_l1_043_fn():
    """L1/043: hidden -> q_a_proj -> norm -> q_b_proj -> split (q_nope+q_pe)
                       -> kv_a_proj -> split (compressed_kv + k_pe)
    H=7168, q_lora=1536, q_b_out=24576, kv_a_out=576
    """
    def _fn(axes: dict):
        B, S = axes["batch_size"], axes["seq_len"]
        N = B * S
        H, q_lora, q_b_out, kv_a_out = 7168, 1536, 24576, 576
        flops = (2*N*H*q_lora + 2*N*q_lora*q_b_out + 2*N*H*kv_a_out
                 + 5*N*q_lora)        # q_a RMSNorm
        # Algorithmic bytes (in bf16, in-register intermediates):
        bytes_ = (
            N*H * 2                                  # hidden read once
            + q_lora*H * 2                           # q_a_proj weight
            + q_b_out*q_lora * 2                     # q_b_proj weight
            + kv_a_out*H * 2                         # kv_a_proj weight
            + q_lora * 2                             # q_a norm weight
            + N * (q_b_out + kv_a_out) * 2           # write outputs
        )
        return flops, bytes_, PEAK_BF16
    return _fn


def mla_l1_064_fn():
    """L1/064: RMSNorm on compressed_kv -> kv_b_proj -> split (k_nope, value).
    kv_lora=512, kv_expanded=128*(128+128)=32768
    """
    def _fn(axes: dict):
        B, S = axes["batch_size"], axes["seq_len"]
        N = B * S
        kv_lora, kv_exp = 512, 32768
        flops = 5*N*kv_lora + 2*N*kv_lora*kv_exp
        bytes_ = (
            N*kv_lora * 2 + kv_lora * 2              # read compressed_kv + norm weight
            + kv_exp*kv_lora * 2                     # kv_b_proj weight
            + N*kv_exp * 2                           # write output
        )
        return flops, bytes_, PEAK_BF16
    return _fn


def mla_quant_013_fn():
    """Quant/013: kv_a_proj -> split -> RMSNorm on compressed_kv -> kv_b_proj.
    H=7168, kv_a_out_padded=640, kv_lora=512, kv_b_out=128*(128+128)=32768
    Inputs bf16 (kernel does FP8 internally); peak = BF16."""
    def _fn(axes: dict):
        B, S = axes["batch_size"], axes["seq_len"]
        N = B * S
        H, kv_a_out, kv_lora, kv_b_out = 7168, 640, 512, 32768
        flops = 2*N*H*kv_a_out + 5*N*kv_lora + 2*N*kv_lora*kv_b_out
        bytes_ = (
            N*H * 2                                  # hidden read
            + kv_a_out*H * 2                         # kv_a_proj weight
            + kv_lora * 2                            # norm weight
            + kv_b_out*kv_lora * 2                   # kv_b_proj weight
            + N * (kv_b_out + 64) * 2                # outputs: kv_expanded + k_pe(64)
        )
        return flops, bytes_, PEAK_BF16
    return _fn


def mla_quant_016_fn():
    """Quant/016: full MLA QKV. Q path (q_a->norm->q_b) + KV path (kv_a->norm->kv_b).
    H=7168, q_lora=1536, q_b_out=24576, kv_a_padded=640, kv_lora=512, kv_b_out=32768"""
    def _fn(axes: dict):
        B, S = axes["batch_size"], axes["seq_len"]
        N = B * S
        H = 7168
        q_lora, q_b_out = 1536, 24576
        kv_a, kv_lora, kv_b_out = 640, 512, 32768
        flops = (2*N*H*q_lora + 5*N*q_lora + 2*N*q_lora*q_b_out
                 + 2*N*H*kv_a + 5*N*kv_lora + 2*N*kv_lora*kv_b_out)
        bytes_ = (
            N*H * 2                                  # hidden read
            + q_lora*H * 2 + q_lora * 2              # q_a + norm
            + q_b_out*q_lora * 2                     # q_b
            + kv_a*H * 2 + kv_lora * 2               # kv_a + norm
            + kv_b_out*kv_lora * 2                   # kv_b
            + N * (q_b_out + kv_b_out + 64) * 2      # outputs (incl k_pe 64)
        )
        return flops, bytes_, PEAK_BF16
    return _fn


# ===========================================================================
# TEMPLATE E: MLP / SwiGLU  (5 problems)
#   gate + up + (silu*up) [+ down]. 2 or 3 GEMMs.
#   Algorithmic bytes: input + weights + output (intermediates fused in-register).
# ===========================================================================
def mlp_gate_up_silu_fn(H: int, I: int, has_down: bool,
                       dtype_in_bytes: int = 2, dtype_out_bytes: int = 2,
                       peak: float = PEAK_BF16, extra_inputs_bytes: int = 0):
    """gate+up+silu*up [+down]. extra_inputs_bytes for FP8 scales etc."""
    def _fn(axes: dict):
        M = axes.get("M") or axes.get("num_tokens") or (
            axes.get("batch_size", 1) * axes.get("seq_len", 1))
        # FLOPs: gate+up (2 GEMMs) + silu*up elementwise [+ down GEMM]
        flops = 2 * 2*M*H*I + 4*M*I
        if has_down:
            flops += 2*M*I*H
        # Bytes (algorithmic minimum, fused):
        n_weights = 3 if has_down else 2
        bytes_ = (
            M*H * dtype_in_bytes                                # input
            + n_weights * I*H * dtype_in_bytes                  # weights
            + M * (H if has_down else I) * dtype_out_bytes      # output
            + extra_inputs_bytes
        )
        return flops, bytes_, peak
    return _fn


def quant_004_fn():
    """Quant/004: bf16 in, fp8 internal. gate_up is fused-shape [2I, H]."""
    def _fn(axes: dict):
        N = axes["num_tokens"]
        H, I = 3584, 2048
        # gate_up: 2I*H, then silu*up, then down: I*H
        flops = 2*N*H*(2*I) + 4*N*I + 2*N*I*H + N*H   # +routing weight mul
        bytes_ = (
            N*H * 2                                  # hidden
            + (2*I)*H * 2                            # gate_up_weight
            + I*H * 2                                # down_weight
            + N*1 * 2                                # routing_weight
            + N*H * 2                                # output
        )
        return flops, bytes_, PEAK_BF16
    return _fn


def fp8_internal(base_fn):
    """Wrap a bf16-declared analyzer fn to model an internal FP8 kernel path:
    halve the byte count (activations + weights are quantized to fp8) and
    switch peak to FP8. Matches Ray-234's accounting for Quant/002/004/011/
    012/013/016 which declare bf16 inputs but expect an FP8-internal kernel."""
    def wrapped(axes: dict):
        flops, bytes_, _peak = base_fn(axes)
        return flops, bytes_ / 2.0, PEAK_FP8
    return wrapped


# Additional one-off Tier-1 problems
def lm_head_003_fn():
    """L1/003: hidden_states[B,S,H=2048] -> logits[B, K, V=102400] where
    K = logits_to_keep. Algorithmic FLOPs use K, not full S."""
    def _fn(axes: dict):
        B = axes["batch_size"]
        K = axes["logits_to_keep"]
        H, V = 2048, 102400
        flops = 2 * B * K * H * V
        bytes_ = (B*K*H + V*H + B*K*V) * 2
        return flops, bytes_, PEAK_BF16
    return _fn


def l1_018_fused_rope_qknorm_kv_cache_fn():
    """L1/018: per-head Q/K RMSNorm + RoPE + KV cache slice write.
    H_q=96, H_kv=8, D=128, max_pos=262144.
    Cost dominated by writing new positions into cache (bf16).
    NOTE: max_position_embeddings is for cache allocation; only seq_len new
    positions are actually written (slice update, not full cache rewrite).
    """
    def _fn(axes: dict):
        B = axes["batch_size"]
        S = axes["seq_len"]
        H_q, H_kv, D = 96, 8, 128
        q_elems = B * H_q * S * D
        k_elems = B * H_kv * S * D
        # FLOPs: q_norm (5*B*H_q*S*D) + k_norm (5*B*H_kv*S*D)
        #        + q_rope (6*q_elems) + k_rope (6*k_elems)
        flops = 5*q_elems + 5*k_elems + 6*q_elems + 6*k_elems
        # Bytes (algorithmic minimum, in-place cache slice update):
        #   read q, k, v: q_elems + 2*k_elems
        #   read cos/sin (small, per-position): 2*B*S*D
        #   write rotated q, k (returned): q_elems + k_elems
        #   write new positions in cache (k, v): 2*k_elems
        bytes_ = (q_elems + 3*k_elems + 2*B*S*D + q_elems + 3*k_elems) * 2
        return flops, bytes_, PEAK_BF16
    return _fn


def l1_020_vision_patch_merger_fn():
    """L1/020: 4-patch spatial shuffle -> layernorm(H_exp=6144) -> fc1 -> GELU -> fc2.
    Input H=1536, after 2x2 merge: H_exp=6144, MLP intermediate=6144, out=3584.
    FC1: (M_out, 6144) @ (6144, 6144)^T, FC2: (M_out, 6144) @ (3584, 6144)^T.
    M_in = num_patches, M_out = num_patches // 4.
    """
    def _fn(axes: dict):
        M_out = axes["num_merged_patches"]
        H, H_exp, H_out = 1536, 6144, 3584
        # FC1 and FC2 GEMMs
        f_fc1 = 2 * M_out * H_exp * H_exp
        f_fc2 = 2 * M_out * H_out * H_exp
        # GELU: 8 ops/elem; layernorm: 7 ops/elem
        f_ln  = 7 * M_out * H_exp
        f_gelu = 8 * M_out * H_exp
        flops = f_fc1 + f_fc2 + f_ln + f_gelu
        # Bytes: input read, weights read once, output write
        bytes_ = (
            M_out * H_exp * 2                          # post-shuffle input
            + 2 * H_exp * 2                            # ln weight+bias
            + H_exp * H_exp * 2 + H_exp * 2            # fc1_w + bias
            + H_out * H_exp * 2 + H_out * 2            # fc2_w + bias
            + M_out * H_out * 2                        # output
        )
        return flops, bytes_, PEAK_BF16
    return _fn


# ===========================================================================
# TEMPLATE F: Attention variants  (11 problems)
#
# Conventions (FlashAttention-style IO accounting):
#   - Attention matrix is NEVER materialized in HBM; only Q, K, V, O traffic counts.
#   - QK + AV matmuls: 2 matmuls of effective shape (S_q, D) x (S_k, D).
#   - GQA: K and V are read once (small num_kv_heads); FLOPs use full num_qo_heads.
#   - Causal: divide attention FLOPs by 2 (each q attends to ~S/2 positions).
#   - MLA: KV cache has NO head dim — shared latent across all H_q heads.
# ===========================================================================

# ---- F1: Full attention (QKV proj + RoPE + attn + O proj) ----
def full_attention_fn(H: int, H_q: int, H_kv: int, D: int,
                     causal: bool, has_qk_norm: bool, has_bias: bool):
    """Generic full attention: QKV proj + (optional RoPE/norm) + attn + O proj."""
    def _fn(axes: dict):
        B, S = axes["batch_size"], axes["seq_len"]
        N = B * S
        q_dim = H_q * D
        kv_dim = H_kv * D
        causal_factor = 0.5 if causal else 1.0

        # GEMMs: Q, K, V proj + O proj
        flops_gemm = (
            2*N*H*q_dim                      # q_proj
            + 2 * 2*N*H*kv_dim               # k_proj + v_proj
            + 2*N*q_dim*H                    # o_proj
        )
        # Attention QK + AV (causal halves it)
        flops_attn = 2 * 2 * B * H_q * S * S * D * causal_factor
        # QK norm (per-head over D)
        flops_norm = 5*B*S*q_dim + 5*B*S*kv_dim if has_qk_norm else 0
        flops = flops_gemm + flops_attn + flops_norm

        bytes_ = (
            N*H * 2                          # hidden_states read
            + (q_dim + 2*kv_dim)*H * 2       # weights for QKV
            + q_dim*H * 2                    # o_proj weight
            + (B*S*q_dim + 2*B*S*kv_dim) * 2 # QKV intermediate (write+read)
            + N*H * 2                        # output write
        )
        # Cos/sin (small)
        bytes_ += 2 * B * S * D * 2
        if has_bias:
            bytes_ += (q_dim + 2*kv_dim) * 2
        return flops, bytes_, PEAK_BF16
    return _fn


def qkv_proj_only_fn(H: int, H_q: int, H_kv: int, D: int,
                    has_qk_norm: bool, has_bias: bool):
    """Quant/002: QKV proj + Q/K RMSNorm, NO attention, NO O proj.
    Outputs Q[B, H_q, S, D], K[B, H_kv, S, D], V[B, H_kv, S, D]."""
    def _fn(axes: dict):
        B, S = axes["batch_size"], axes["seq_len"]
        N = B * S
        q_dim = H_q * D
        kv_dim = H_kv * D
        flops = (
            2*N*H*q_dim + 2 * 2*N*H*kv_dim   # 3 GEMMs
            + (5*B*S*q_dim + 5*B*S*kv_dim if has_qk_norm else 0)
        )
        bytes_ = (
            N*H * 2                          # hidden
            + (q_dim + 2*kv_dim)*H * 2       # weights
            + B*S*(q_dim + 2*kv_dim) * 2     # outputs
        )
        if has_bias:
            bytes_ += (q_dim + 2*kv_dim) * 2
        if has_qk_norm:
            bytes_ += (D + D) * 2            # q_norm + k_norm weights
        return flops, bytes_, PEAK_BF16
    return _fn


# ---- F2: Attention single steps ----
def attn_softmax_only_fn(num_heads: int):
    """L1/046: just softmax over [B, H, Sq, Sk]. Pure memory-bound."""
    def _fn(axes: dict):
        B, Sq, Sk = axes["batch_size"], axes["seq_len_q"], axes["seq_len_k"]
        N = B * num_heads * Sq * Sk
        # Ray-234 convention: softmax alone is memory-bound; flops=0
        flops = 0
        bytes_ = 2 * N * 2                   # read + write
        return flops, bytes_, PEAK_BF16
    return _fn


def attn_qk_matmul_only_fn(H_q: int, H_kv: int, D: int):
    """L1/049: just Q @ K.T with GQA repeat + scaling. Output [B, H_q, S, S].
    NOTE: attention matrix IS materialized here (it's the output)."""
    def _fn(axes: dict):
        B, S = axes["batch_size"], axes["seq_len"]
        flops = 2 * B * H_q * S * S * D
        bytes_ = (
            B * H_q * S * D * 2              # Q read
            + B * H_kv * S * D * 2           # K read (small in GQA)
            + B * H_q * S * S * 2            # attn_scores write (huge for large S)
        )
        return flops, bytes_, PEAK_BF16
    return _fn


# ---- F3: Paged / ragged attention ----
def gqa_paged_decode_fn(H_q: int, H_kv: int, D: int):
    """FIB/013: GQA paged decode. Each batch has its own KV slice.

    Per batch: q has 1 token, attends to L_i KV positions. Total KV = num_kv_indices.
    """
    def _fn(axes: dict):
        B = axes["batch_size"]
        L = axes["num_kv_indices"]   # total KV tokens across all batches
        # FLOPs: QK + AV per token. Sum over batches = 2 * H_q * sum(L_i) * D * 2
        flops = 2 * 2 * H_q * L * D
        bytes_ = (
            B * H_q * D * 2                  # Q read (per batch)
            + 2 * L * H_kv * D * 2           # K + V cache reads (total)
            + B * H_q * D * 2                # O write
        )
        return flops, bytes_, PEAK_BF16
    return _fn


def gqa_ragged_prefill_causal_fn(H_q: int, H_kv: int, D: int):
    """FIB/017: GQA ragged prefill causal. Single sequence approx (len_indptr=2).
    For uniform multi-seq: sum(S_i²) is best-case bounded by total_q² / num_seqs."""
    def _fn(axes: dict):
        Tq, Tkv = axes["total_q"], axes["total_kv"]
        num_seqs = axes["len_indptr"] - 1
        # Assume uniform per-seq length (real value depends on cu_seqlens safetensor).
        # FLOPs: causal halves it. sum(S_i^2) ≈ (Tq/num_seqs) * Tq when uniform.
        avg_S = Tq / max(num_seqs, 1)
        flops = 2 * 2 * H_q * num_seqs * avg_S * avg_S * D * 0.5
        bytes_ = (
            Tq * H_q * D * 2                 # Q
            + 2 * Tkv * H_kv * D * 2         # K + V
            + Tq * H_q * D * 2               # O
        )
        return flops, bytes_, PEAK_BF16
    return _fn


def mla_paged_decode_fn(H_q: int, D_ckv: int, D_kpe: int):
    """FIB/018: MLA paged decode. KV cache has NO head dim (latent shared across H_q).
    Cache element = ckv (D_ckv) + kpe (D_kpe) per position.
    Q has two parts: q_nope[B, H_q, D_ckv], q_pe[B, H_q, D_kpe].
    Attention: (Q_nope·K_ckv + Q_pe·K_pe), then softmax, then (probs·V) where V=ckv.
    """
    def _fn(axes: dict):
        B = axes["batch_size"]
        L = axes["num_kv_indices"]
        # QK FLOPs: 2 * H_q * L * (D_ckv + D_kpe)  per batch token
        # AV FLOPs: 2 * H_q * L * D_ckv            (output dim = D_ckv)
        flops = 2 * H_q * L * (D_ckv + D_kpe) + 2 * H_q * L * D_ckv
        bytes_ = (
            B * H_q * (D_ckv + D_kpe) * 2    # q_nope + q_pe
            + L * (D_ckv + D_kpe) * 2        # KV cache (NO H dim — that's MLA's magic)
            + B * H_q * D_ckv * 2            # output
        )
        return flops, bytes_, PEAK_BF16
    return _fn


def mla_paged_prefill_causal_fn(H_q: int, D_ckv: int, D_kpe: int):
    """FIB/019: MLA paged prefill causal. total_q queries, num_kv_indices KV positions."""
    def _fn(axes: dict):
        Tq = axes["total_q"]
        L = axes["num_kv_indices"]
        num_seqs = axes["len_indptr"] - 1
        avg_Sq = Tq / max(num_seqs, 1)
        avg_Sk = L  / max(num_seqs, 1)
        # Causal: factor 0.5 because each q attends to subset of past KV
        flops_qk = 2 * H_q * num_seqs * avg_Sq * avg_Sk * (D_ckv + D_kpe) * 0.5
        flops_av = 2 * H_q * num_seqs * avg_Sq * avg_Sk * D_ckv * 0.5
        flops = flops_qk + flops_av
        bytes_ = (
            Tq * H_q * (D_ckv + D_kpe) * 2
            + L * (D_ckv + D_kpe) * 2
            + Tq * H_q * D_ckv * 2
        )
        return flops, bytes_, PEAK_BF16
    return _fn


# ---- F4: Variable-length vision attention ----
def vision_varlen_attention_fn():
    """L1/021: cu_seqlens variable-length attention. Includes QKV proj + attn + O proj.
    H=1280, num_heads=16, D=80 — note small D. Uniform assumption for per-seq lengths."""
    def _fn(axes: dict):
        T = axes["total_seq_len"]
        n_seqs = axes["num_seqs"]
        H_total, H_q, D = 1280, 16, 80
        avg_S = T / max(n_seqs, 1)
        # QKV proj: T tokens, fused (qkv_out = 3*H_q*D = 3840)
        flops_qkv = 2 * T * H_total * (3 * H_q * D)
        # O proj
        flops_o = 2 * T * H_q * D * H_total
        # Attention (non-causal in vision): sum_i (2 * H_q * S_i^2 * D * 2) ≈ 2 * H_q * n_seqs * avg_S^2 * D * 2
        flops_attn = 2 * 2 * H_q * n_seqs * avg_S * avg_S * D
        flops = flops_qkv + flops_o + flops_attn
        bytes_ = (
            T * H_total * 4                       # hidden_states fp32 (4 bytes!)
            + H_total * 3 * H_q * D * 4           # qkv_weight fp32
            + H_total * H_total * 4               # proj_weight fp32
            + T * 3 * H_q * D * 4                 # QKV intermediates
            + T * H_total * 4                     # output fp32
        )
        return flops, bytes_, PEAK_BF16
    return _fn


# ===========================================================================
# REGISTRY: 13 + 4(D) + 5(E) + 11(F) = 33 problems
# ===========================================================================
PROBLEMS = [
    # ---- Template A: RMSNorm ----
    Problem("L1/069 residual+rmsnorm h8192",
            "L1/069_rms_norm",
            rmsnorm_fn(H=8192, residual=True),
            "fused (residual + rmsnorm)"),
    Problem("FIB/002 fused_add_rmsnorm h4096",
            "FlashInfer-Bench/002_fused_add_rmsnorm_h4096",
            rmsnorm_fn(H=4096, residual=True)),
    Problem("FIB/003 fused_add_rmsnorm h7168",
            "FlashInfer-Bench/003_fused_add_rmsnorm_h7168",
            rmsnorm_fn(H=7168, residual=True)),
    Problem("FIB/023 rmsnorm h1536",
            "FlashInfer-Bench/023_rmsnorm_h1536",
            rmsnorm_fn(H=1536, residual=False)),
    Problem("FIB/026 rmsnorm h7168",
            "FlashInfer-Bench/026_rmsnorm_h7168",
            rmsnorm_fn(H=7168, residual=False)),
    # ---- Template B: GEMM ----
    Problem("FIB/005 gemm N=256 K=7168 fp16",
            "FlashInfer-Bench/005_gemm_n256_k7168",
            gemm_fn(N=256, K=7168, dtype_in_bytes=2, dtype_out_bytes=2, peak=PEAK_BF16),
            "asymptotic AI=247 < ridge 295 → fundamentally memory-bound"),
    Problem("Quant/005 fp8 router N=128 K=3584",
            "Quant/005_fp8_moe_router_projection",
            gemm_fn(N=128, K=3584, dtype_in_bytes=1, dtype_out_bytes=2, peak=PEAK_FP8),
            "FP8 inputs (1B), bf16 output (2B); peak=1979 TFLOPS"),
    Problem("Quant/015 fp8 o_proj N=7168 K=16384",
            "Quant/015_fp8_mla_attention_output_projection",
            gemm_fn(N=7168, K=16384, dtype_in_bytes=1, dtype_out_bytes=2, peak=PEAK_FP8),
            "FP8 MLA output projection; K=num_heads*v_head_dim=128*128=16384"),
    # ---- Template C: RoPE / position ----
    Problem("L1/011 rope_compute D=128",
            "L1/011_rotary_position_embedding",
            rope_table_fn(D=128)),
    Problem("L1/071 kv_cache_update H_kv=10 D=128",
            "L1/071_kv_cache_update_with_rope",
            kv_cache_update_rope_fn(num_kv_heads=10, head_dim=128),
            "algorithmic-minimum bytes (in-place); naive impl writes full cache"),
    Problem("Quant/014 fp8_yarn_rope D=64",
            "Quant/014_fp8_yarn_rope_embedding",
            rope_table_fn(D=64),
            "only computes cos/sin tables — tiny problem, mostly latency"),
    Problem("L1/023 multimodal_3d_rope",
            "L1/023_multimodal_rope_position_computation_with_grid_based_indexing",
            multimodal_grid_rope_fn(),
            "patch_pos_embeds + cos/sin tables, large pos_embed_weight read"),
    Problem("L2/006 multimodal_position_calc",
            "L2/006_multimodal_rope_position_calculation",
            multimodal_rope_position_fn(),
            "pure indexing → memory or latency-bound"),
    # ---- Template D: MLA / fused QKV ----
    Problem("L1/043 mla_fused_qkv_rope_split",
            "L1/043_mla_fused_qkv_rope_split",
            mla_l1_043_fn(),
            "DS-V3 MLA: q_a→norm→q_b + kv_a, all bf16"),
    Problem("L1/064 latent_kv_expansion",
            "L1/064_latent_kv_expansion_with_split",
            mla_l1_064_fn(),
            "compressed_kv→norm→kv_b_proj(expand to 128*256)"),
    Problem("Quant/013 fp8_mla_kv_compression",
            "Quant/013_fp8_mla_kv_compression_projection",
            fp8_internal(mla_quant_013_fn()),
            "bf16 in, FP8 internal: kv_a→split→norm→kv_b"),
    Problem("Quant/016 fp8_mla_qkv_projection",
            "Quant/016_fp8_multi_latent_attention_qkv_projection",
            fp8_internal(mla_quant_016_fn()),
            "bf16 in, FP8 internal: full MLA Q+KV (6 GEMMs+2 norms)"),
    # ---- Template E: MLP / SwiGLU ----
    Problem("L1/048 fused_gate_up_swiglu",
            "L1/048_fused_gate_up_projection_with_swiglu",
            mlp_gate_up_silu_fn(H=3072, I=24576, has_down=False),
            "gate+up+silu, NO down. Output is [M, I]."),
    Problem("L1/063 attn_out_reshape_proj",
            "L1/063_attention_output_reshape_and_projection",
            gemm_fn(N=7168, K=16384, dtype_in_bytes=2, dtype_out_bytes=2, peak=PEAK_BF16),
            "single GEMM after transpose+reshape; K=heads*v_head_dim=128*128"),
    Problem("Quant/003 fp8_mlp_gate_up",
            "Quant/003_fp8_mlp_gate_up_projection",
            mlp_gate_up_silu_fn(H=3584, I=18944, has_down=False,
                                dtype_in_bytes=1, dtype_out_bytes=2,
                                peak=PEAK_FP8),
            "TRUE FP8 inputs (declared fp8), peak=1979 TFLOPS"),
    Problem("Quant/004 fp8_moe_expert_linear",
            "Quant/004_fp8_moe_expert_linear",
            fp8_internal(quant_004_fn()),
            "bf16 in, FP8 internal: gate_up→silu→down (full MLP) + routing_w"),
    Problem("Quant/012 fp8_shared_expert_mlp",
            "Quant/012_fp8_shared_expert_mlp",
            fp8_internal(mlp_gate_up_silu_fn(H=7168, I=2048, has_down=True)),
            "bf16 in, FP8 internal: gate+up+silu+down (3 GEMMs)"),
    # ---- Template F: Attention variants ----
    # F1: full attention
    Problem("L1/015 gqa_rope_qknorm H=4096 H_q=32 H_kv=8 D=128",
            "L1/015_grouped_query_attention_with_rope_and_qk_norm",
            full_attention_fn(H=4096, H_q=32, H_kv=8, D=128,
                              causal=True, has_qk_norm=True, has_bias=False)),
    Problem("L1/067 flash_attn_gqa H=4096 H_q=32 H_kv=8 (fp32!)",
            "L1/067_flash_attention_gqa_ultralong",
            full_attention_fn(H=4096, H_q=32, H_kv=8, D=128,
                              causal=False, has_qk_norm=False, has_bias=False),
            "inputs fp32 — bytes/peak conservative; real kernel likely casts to bf16"),
    Problem("L1/092 gqa_qknorm H=4096 H_q=96 H_kv=8 D=128",
            "L1/092_gqa_attention_with_qk_norm",
            full_attention_fn(H=4096, H_q=96, H_kv=8, D=128,
                              causal=True, has_qk_norm=True, has_bias=True),
            "H_q=96 (12 GQA groups) — high arithmetic intensity attention"),
    Problem("Quant/002 fp8_qkv_proj H=7680 H_q=60 H_kv=8 D=128",
            "Quant/002_fp8_attention_qkv_projection",
            fp8_internal(qkv_proj_only_fn(H=7680, H_q=60, H_kv=8, D=128,
                             has_qk_norm=True, has_bias=True)),
            "QKV proj only (no attention/O), bf16 in/fp8 internal"),
    # F2: attention single steps
    Problem("L1/046 attn_softmax (softcap+dropout)",
            "L1/046_attention_softmax_with_softcapping_and_dropout",
            attn_softmax_only_fn(num_heads=8),
            "softmax over [B, 8, Sq, Sk] — pure memory-bound"),
    Problem("L1/049 attn_qk_matmul H_q=4 H_kv=1 D=256",
            "L1/049_attention_qk_matmul_with_gqa_repeat_and_scaling",
            attn_qk_matmul_only_fn(H_q=4, H_kv=1, D=256),
            "QK^T only; attn_scores [B, H_q, S, S] IS the output (huge)"),
    # F3: paged / ragged
    Problem("FIB/013 gqa_paged_decode H_q=32 H_kv=8 D=128",
            "FlashInfer-Bench/013_gqa_paged_decode_h32_kv8_d128_ps1",
            gqa_paged_decode_fn(H_q=32, H_kv=8, D=128),
            "decode: each batch attends to L_i KV positions; AI ≈ H_q/H_kv = 4"),
    Problem("FIB/017 gqa_ragged_prefill_causal H_q=32 H_kv=8 D=128",
            "FlashInfer-Bench/017_gqa_ragged_prefill_causal_h32_kv8_d128",
            gqa_ragged_prefill_causal_fn(H_q=32, H_kv=8, D=128),
            "uniform-S approximation; real sum(S_i^2) needs cu_seqlens"),
    Problem("FIB/018 mla_paged_decode H_q=16 D_ckv=512 D_kpe=64",
            "FlashInfer-Bench/018_mla_paged_decode_h16_ckv512_kpe64_ps1",
            mla_paged_decode_fn(H_q=16, D_ckv=512, D_kpe=64),
            "MLA: KV cache shared across heads → AI ≈ H_q = 16, better than GQA decode"),
    Problem("FIB/019 mla_paged_prefill_causal H_q=16 D_ckv=512 D_kpe=64",
            "FlashInfer-Bench/019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1",
            mla_paged_prefill_causal_fn(H_q=16, D_ckv=512, D_kpe=64),
            "MLA prefill causal; uniform-S approximation"),
    # F4: variable-length
    Problem("L1/021 vision_cu_seqlens H=1280 H_q=16 D=80 (fp32!)",
            "L1/021_vision_cu_seqlens_variable_length_attention",
            vision_varlen_attention_fn(),
            "fp32 inputs, uniform-S approximation"),
    # ---- Catch-up: 3 L1 problems missed in earlier templates ----
    Problem("L1/003 lm_head_logit_slicing H=2048 V=102400",
            "L1/003_lm_head_projection_with_logit_slicing",
            lm_head_003_fn(),
            "FLOPs use logits_to_keep (algorithmic) — NOT full seq_len"),
    Problem("L1/018 fused_qknorm_rope_kv_cache H_q=96 H_kv=8 D=128",
            "L1/018_fused_rope_with_qk_norm_and_kv_cache_update",
            l1_018_fused_rope_qknorm_kv_cache_fn(),
            "in-place cache slice update; max_pos=262144 only allocs cache"),
    Problem("L1/020 vision_patch_merger H_exp=6144 -> 3584",
            "L1/020_vision_patch_merger_spatial_shuffle_mlp",
            l1_020_vision_patch_merger_fn(),
            "spatial shuffle 4:1 + layernorm + fc1 + GELU + fc2"),
]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Tier-1 batch roofline analyzer")
    import _hardware; _hardware.add_hardware_arg(ap)
    ap.add_argument("--smoke", action="store_true",
                    help="show 3 representative workloads per problem (small/mid/large)")
    ap.add_argument("--problem", help="only run this problem (substring match on name)")
    args = ap.parse_args()
    for p in PROBLEMS:
        if args.problem and args.problem not in p.name:
            continue
        report(p, smoke=args.smoke)
