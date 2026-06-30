"""Tier-3 (L2 fused multi-kernel) batch analyzer.

For each problem, we MANUALLY decompose the layer into individual ops, each
with its own (flops, bytes). Per-op regime classification is shared. The
'row-level' answer is the sum of per-op t_sol — there's no honest single
relative number for a multi-kernel row.

Each problem function takes an axes dict (with bf16 dtype assumed unless the
problem declares fp32) and returns list[Op].
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

CONTEST_ROOT = Path(__file__).resolve().parent.parent / "data" / "benchmark" / "Contest"

PEAK_BF16  = 989e12
PEAK_BW    = 3.35e12
RIDGE_BF16 = PEAK_BF16 / PEAK_BW
LATENCY_FLOOR_US = 5.0


@dataclass
class Op:
    name: str
    flops: float
    bytes: float
    kind: str   # 'gemm' | 'norm' | 'rope' | 'attn' | 'elementwise' | 'moe' | 'router'

    def metrics(self) -> dict:
        ai = self.flops / self.bytes if self.bytes else 0.0
        t_c = self.flops / PEAK_BF16
        t_m = self.bytes / PEAK_BW
        t_sol = max(t_c, t_m) * 1e6
        if t_sol < LATENCY_FLOOR_US:
            regime = "latency"
        elif ai > 2*RIDGE_BF16:
            regime = "compute"
        elif ai < 0.5*RIDGE_BF16:
            regime = "memory"
        else:
            regime = "balanced"
        return dict(name=self.name, kind=self.kind, flops=self.flops,
                    bytes=self.bytes, ai=ai,
                    t_compute_us=t_c*1e6, t_memory_us=t_m*1e6,
                    t_sol_us=t_sol, regime=regime)


# ---------------------------------------------------------------------------
# Helper builders shared across problems
# ---------------------------------------------------------------------------
def _gemm(name: str, M: int, N: int, K: int, db: int = 2) -> Op:
    """Generic GEMM op: (M,K) @ (K,N) -> (M,N), all dtype `db` bytes."""
    return Op(name, flops=2*M*N*K, bytes=(M*K + N*K + M*N)*db, kind="gemm")


def _rmsnorm(name: str, BS: int, H: int, db: int = 2, has_residual: bool = False) -> Op:
    flops = 5 * BS * H
    bytes_ = (BS*H + (BS*H if has_residual else 0) + BS*H + H) * db
    return Op(name, flops=flops, bytes=bytes_, kind="norm")


def _layernorm(name: str, BS: int, H: int, db: int = 2) -> Op:
    # LayerNorm has mean+var: ~7 ops/elem vs RMS ~5
    return Op(name, flops=7*BS*H, bytes=(2*BS*H + 2*H)*db, kind="norm")


def _rope(name: str, B: int, H_heads: int, S: int, D: int, db: int = 2) -> Op:
    n = B * H_heads * S * D
    return Op(name, flops=6*n, bytes=(2*n + 2*B*S*D)*db, kind="rope")


def _attn_qk(name: str, B: int, H_q: int, H_kv: int, S: int, D: int, db: int = 2) -> Op:
    """FlashAttention-style: attn matrix NOT materialized to HBM."""
    flops = 2 * B * H_q * S * S * D
    bytes_ = (B*H_q*S*D + B*H_kv*S*D) * db   # Q + K reads only
    return Op(name, flops=flops, bytes=bytes_, kind="attn")


def _attn_av(name: str, B: int, H_q: int, H_kv: int, S: int, D: int, db: int = 2) -> Op:
    flops = 2 * B * H_q * S * S * D
    bytes_ = (B*H_kv*S*D + B*H_q*S*D) * db   # V read + O write
    return Op(name, flops=flops, bytes=bytes_, kind="attn")


def _softmax(name: str, B: int, H_q: int, S_q: int, S_k: int, db: int = 2) -> Op:
    """Standalone softmax — only used when attn matrix is materialized."""
    n = B * H_q * S_q * S_k
    return Op(name, flops=5*n, bytes=2*n*db, kind="elementwise")


def _silu_mul(name: str, BS: int, I: int, db: int = 2) -> Op:
    return Op(name, flops=4*BS*I, bytes=3*BS*I*db, kind="elementwise")


def _residual_add(name: str, BS: int, H: int, db: int = 2) -> Op:
    return Op(name, flops=BS*H, bytes=3*BS*H*db, kind="elementwise")


# ===========================================================================
# Per-problem decompositions
# ===========================================================================

def decompose_002(axes: dict, db: int = 2) -> list[Op]:
    """L2/002 — Llama-3 decoder full block. H=4096 I=14336 H_q=32 H_kv=8 D=128.
    Note: declared fp32; we use db=2 (bf16) since real solutions cast."""
    B, S = axes["batch_size"], axes["seq_len"]
    H, I, H_q, H_kv, D = 4096, 14336, 32, 8, 128
    qkv_dim, kv_dim = H_q*D, H_kv*D
    BS = B*S
    return [
        _rmsnorm("input_rmsnorm", BS, H, db),
        _gemm("q_proj", BS, qkv_dim, H, db),
        _gemm("k_proj", BS, kv_dim, H, db),
        _gemm("v_proj", BS, kv_dim, H, db),
        _rope("q_rope", B, H_q, S, D, db),
        _rope("k_rope", B, H_kv, S, D, db),
        _attn_qk("attn_qk", B, H_q, H_kv, S, D, db),
        _attn_av("attn_av", B, H_q, H_kv, S, D, db),
        _gemm("o_proj", BS, H, qkv_dim, db),
        _residual_add("attn_residual_add", BS, H, db),
        _rmsnorm("post_attn_rmsnorm", BS, H, db),
        _gemm("gate_proj", BS, I, H, db),
        _gemm("up_proj", BS, I, H, db),
        _silu_mul("silu_mul", BS, I, db),
        _gemm("down_proj", BS, H, I, db),
        _residual_add("mlp_residual_add", BS, H, db),
    ]


def decompose_004(axes: dict, db: int = 2) -> list[Op]:
    """L2/004 — residual + RMSNorm + MLP block. H=16384 I=53248 (huge!)."""
    B, S = axes["batch_size"], axes["seq_len"]
    H, I = 16384, 53248
    BS = B*S
    return [
        _rmsnorm("residual_rmsnorm", BS, H, db, has_residual=True),
        _gemm("gate_proj", BS, I, H, db),
        _gemm("up_proj", BS, I, H, db),
        _silu_mul("silu_mul", BS, I, db),
        _gemm("down_proj", BS, H, I, db),
    ]


def decompose_007(axes: dict, db: int = 2) -> list[Op]:
    """L2/007 — Qwen2VL multimodal RoPE GQA attention.
    H=3584 H_q=28 H_kv=4 D=128, has bias on QKV.
    Note: NO MLP (this is attention-only, fused multi-modal RoPE)."""
    B, S = axes["batch_size"], axes["seq_len"]
    H, H_q, H_kv, D = 3584, 28, 4, 128
    qkv_dim, kv_dim = H_q*D, H_kv*D
    BS = B*S
    ops = [
        _gemm("q_proj", BS, qkv_dim, H, db),
        _gemm("k_proj", BS, kv_dim, H, db),
        _gemm("v_proj", BS, kv_dim, H, db),
        # mRoPE = standard RoPE on 3 axes; net cost ~ same as single
        _rope("q_mrope", B, H_q, S, D, db),
        _rope("k_mrope", B, H_kv, S, D, db),
        _attn_qk("attn_qk", B, H_q, H_kv, S, D, db),
        _attn_av("attn_av", B, H_q, H_kv, S, D, db),
        _gemm("o_proj", BS, H, qkv_dim, db),
    ]
    return ops


def decompose_009(axes: dict, db: int = 2) -> list[Op]:
    """L2/009 — Qwen3-MoE decoder full layer (attention with QK norm + MoE MLP).
    H=2048 H_q=32 H_kv=4 D=128 I_moe=768 E=128 topk=8.
    MoE is approximated with 'expected' load (mean = topk*T/E tokens per expert)."""
    B, S = axes["batch_size"], axes["seq_len"]
    H, H_q, H_kv, D = 2048, 32, 4, 128
    E, topk, I_moe = 128, 8, 768
    qkv_dim, kv_dim = H_q*D, H_kv*D
    BS = B*S
    T = BS
    # MoE expected: each expert sees mean = topk*T/E tokens
    mean_n = topk * T / E
    moe_expert_flops = E * (2*mean_n*H*I_moe*3 + 4*mean_n*I_moe)
    moe_expert_bytes = (
        T*H * db                             # hidden read (full set, dispatched)
        + 3*E*I_moe*H * db                   # all expert weights
        + T*H * db                           # output write
    )
    return [
        _rmsnorm("input_rmsnorm", BS, H, db),
        _gemm("q_proj", BS, qkv_dim, H, db),
        _gemm("k_proj", BS, kv_dim, H, db),
        _gemm("v_proj", BS, kv_dim, H, db),
        # Q/K RMS norm (per-head)
        Op("q_norm", flops=5*B*H_q*S*D,
           bytes=(2*B*H_q*S*D + D)*db, kind="norm"),
        Op("k_norm", flops=5*B*H_kv*S*D,
           bytes=(2*B*H_kv*S*D + D)*db, kind="norm"),
        _rope("q_rope", B, H_q, S, D, db),
        _rope("k_rope", B, H_kv, S, D, db),
        _attn_qk("attn_qk", B, H_q, H_kv, S, D, db),
        _attn_av("attn_av", B, H_q, H_kv, S, D, db),
        _gemm("o_proj", BS, H, qkv_dim, db),
        _residual_add("attn_residual", BS, H, db),
        _rmsnorm("post_attn_rmsnorm", BS, H, db),
        # MoE router
        _gemm("router", T, E, H, db),
        Op("moe_experts(expected)", flops=moe_expert_flops,
           bytes=moe_expert_bytes, kind="moe"),
        _residual_add("mlp_residual", BS, H, db),
    ]


def decompose_019(axes: dict, db: int = 2) -> list[Op]:
    """L2/019 — Qwen2VL decoder full block with mRoPE.
    H=3584 H_q=28 H_kv=4 D=128 I=18944. kv_seq_len differs from seq_len.
    Note: declared fp32; we use bf16 (db=2)."""
    B, S = axes["batch_size"], axes["seq_len"]
    kv_S = axes.get("kv_seq_len", S)
    H, H_q, H_kv, D = 3584, 28, 4, 128
    I = 18944
    qkv_dim, kv_dim = H_q*D, H_kv*D
    BS = B*S
    return [
        _rmsnorm("input_rmsnorm", BS, H, db),
        _gemm("q_proj", BS, qkv_dim, H, db),
        _gemm("k_proj", BS, kv_dim, H, db),
        _gemm("v_proj", BS, kv_dim, H, db),
        _rope("q_mrope", B, H_q, S, D, db),
        _rope("k_mrope", B, H_kv, S, D, db),
        # Attention: Q has S tokens, K/V have kv_S tokens
        Op("attn_qk", flops=2*B*H_q*S*kv_S*D,
           bytes=(B*H_q*S*D + B*H_kv*kv_S*D)*db, kind="attn"),
        Op("attn_av", flops=2*B*H_q*S*kv_S*D,
           bytes=(B*H_kv*kv_S*D + B*H_q*S*D)*db, kind="attn"),
        _gemm("o_proj", BS, H, qkv_dim, db),
        _residual_add("attn_residual", BS, H, db),
        _rmsnorm("post_attn_rmsnorm", BS, H, db),
        _gemm("gate_proj", BS, I, H, db),
        _gemm("up_proj", BS, I, H, db),
        _silu_mul("silu_mul", BS, I, db),
        _gemm("down_proj", BS, H, I, db),
        _residual_add("mlp_residual", BS, H, db),
    ]


def decompose_020(axes: dict, db: int = 2) -> list[Op]:
    """L2/020 — decoder pre/post norm + residual. H=6144 I=19648 H_q=64 H_kv=8 D=96."""
    B, S = axes["batch_size"], axes["seq_len"]
    H, I = 6144, 19648
    H_q, H_kv, D = 64, 8, 96
    qkv_dim, kv_dim = H_q*D, H_kv*D
    BS = B*S
    return [
        _rmsnorm("input_rmsnorm", BS, H, db),
        _gemm("q_proj", BS, qkv_dim, H, db),
        _gemm("k_proj", BS, kv_dim, H, db),
        _gemm("v_proj", BS, kv_dim, H, db),
        # RoPE uses half_head_dim
        _rope("q_rope", B, H_q, S, D, db),
        _rope("k_rope", B, H_kv, S, D, db),
        _attn_qk("attn_qk", B, H_q, H_kv, S, D, db),
        _attn_av("attn_av", B, H_q, H_kv, S, D, db),
        _gemm("o_proj", BS, H, qkv_dim, db),
        _residual_add("attn_residual", BS, H, db),
        _rmsnorm("post_attn_rmsnorm", BS, H, db),
        _gemm("gate_proj", BS, I, H, db),
        _gemm("up_proj", BS, I, H, db),
        _silu_mul("silu_mul", BS, I, db),
        _gemm("down_proj", BS, H, I, db),
        _residual_add("mlp_residual", BS, H, db),
    ]


def decompose_027(axes: dict, db: int = 2) -> list[Op]:
    """L2/027 — GQA attention with YaRN RoPE + Q/K RMSNorm.
    H=5120 H_q=40 H_kv=8 D=128. ATTENTION ONLY (no MLP)."""
    B, S = axes["batch_size"], axes["seq_len"]
    H, H_q, H_kv, D = 5120, 40, 8, 128
    qkv_dim, kv_dim = H_q*D, H_kv*D
    BS = B*S
    return [
        _gemm("q_proj", BS, qkv_dim, H, db),
        _gemm("k_proj", BS, kv_dim, H, db),
        _gemm("v_proj", BS, kv_dim, H, db),
        Op("q_norm", flops=5*B*H_q*S*D,
           bytes=(2*B*H_q*S*D + D)*db, kind="norm"),
        Op("k_norm", flops=5*B*H_kv*S*D,
           bytes=(2*B*H_kv*S*D + D)*db, kind="norm"),
        _rope("q_yarn_rope", B, H_q, S, D, db),
        _rope("k_yarn_rope", B, H_kv, S, D, db),
        _attn_qk("attn_qk", B, H_q, H_kv, S, D, db),
        _attn_av("attn_av", B, H_q, H_kv, S, D, db),
        _gemm("o_proj", BS, H, qkv_dim, db),
    ]


def decompose_053(axes: dict, db: int = 2) -> list[Op]:
    """L2/053 — Mllama text decoder layer. H=4096 H_q=32 H_kv=8 D=128 I=14336.
    (Same shape family as Llama-3; near-identical to 002 but no explicit RoPE here.)"""
    B, S = axes["batch_size"], axes["seq_len"]
    H, I, H_q, H_kv, D = 4096, 14336, 32, 8, 128
    qkv_dim, kv_dim = H_q*D, H_kv*D
    BS = B*S
    return [
        _rmsnorm("input_rmsnorm", BS, H, db),
        _gemm("q_proj", BS, qkv_dim, H, db),
        _gemm("k_proj", BS, kv_dim, H, db),
        _gemm("v_proj", BS, kv_dim, H, db),
        # rope_theta passed but inv_freq computed inside — treat as RoPE op
        _rope("q_rope", B, H_q, S, D, db),
        _rope("k_rope", B, H_kv, S, D, db),
        _attn_qk("attn_qk", B, H_q, H_kv, S, D, db),
        _attn_av("attn_av", B, H_q, H_kv, S, D, db),
        _gemm("o_proj", BS, H, qkv_dim, db),
        _residual_add("attn_residual", BS, H, db),
        _rmsnorm("post_attn_rmsnorm", BS, H, db),
        _gemm("gate_proj", BS, I, H, db),
        _gemm("up_proj", BS, I, H, db),
        _silu_mul("silu_mul", BS, I, db),
        _gemm("down_proj", BS, H, I, db),
        _residual_add("mlp_residual", BS, H, db),
    ]


def decompose_054(axes: dict, db: int = 2) -> list[Op]:
    """L2/054 — vision encoder block (non-causal attention + gated residuals).
    H=1280 num_heads=16 D=80 I=5120. LayerNorm (not RMSNorm). FC1+GELU+FC2."""
    B, S = axes["batch_size"], axes["seq_len"]
    H, I, n_heads, D = 1280, 5120, 16, 80
    BS = B*S
    return [
        _layernorm("input_layernorm", BS, H, db),
        _gemm("q_proj", BS, H, H, db),
        _gemm("k_proj", BS, H, H, db),
        _gemm("v_proj", BS, H, H, db),
        # Non-causal attention (full S²)
        Op("attn_qk", flops=2*B*n_heads*S*S*D,
           bytes=(B*n_heads*S*D + B*n_heads*S*D)*db, kind="attn"),
        Op("attn_av", flops=2*B*n_heads*S*S*D,
           bytes=(B*n_heads*S*D + B*n_heads*S*D)*db, kind="attn"),
        _gemm("o_proj", BS, H, H, db),
        # gated residual: scalar * residual_add ≈ same cost as add
        _residual_add("gated_attn_residual", BS, H, db),
        _layernorm("post_attn_layernorm", BS, H, db),
        _gemm("fc1", BS, I, H, db),
        # GELU
        Op("gelu", flops=8*BS*I, bytes=2*BS*I*db, kind="elementwise"),
        _gemm("fc2", BS, H, I, db),
        _residual_add("gated_ffn_residual", BS, H, db),
    ]


# ===========================================================================
# REGISTRY
# ===========================================================================
@dataclass
class L2Problem:
    name: str
    subdir: str
    decompose: Callable[[dict], list[Op]]
    note: str = ""

PROBLEMS = [
    L2Problem("L2/002 llama3_decoder_full",
              "L2/002_decoder_layer_full_block", decompose_002,
              "H=4096 I=14336 H_q=32 H_kv=8 D=128 (Llama-3 8B)"),
    L2Problem("L2/004 fused_residual_rms_mlp",
              "L2/004_fused_residual_rms_mlp", decompose_004,
              "H=16384 I=53248 (Hermes-4-405B!) — MLP only, no attention"),
    L2Problem("L2/007 mrope_gqa_attn",
              "L2/007_multimodal_rotary_embedding_attention", decompose_007,
              "H=3584 H_q=28 H_kv=4 D=128 (Qwen2VL) — attention only"),
    L2Problem("L2/009 qwen3_decoder_with_moe",
              "L2/009_decoder_layer_with_residual_connections", decompose_009,
              "H=2048 H_q=32 H_kv=4 D=128 + MoE(E=128 topk=8 I=768) — expected load"),
    L2Problem("L2/019 mrope_decoder_full",
              "L2/019_decoder_layer_fused_attention_mlp", decompose_019,
              "H=3584 I=18944 H_q=28 H_kv=4 D=128 (Qwen2VL) — kv_seq_len ≠ seq_len"),
    L2Problem("L2/020 decoder_pre_post_norm",
              "L2/020_decoder_layer_pre_post_norm_residual", decompose_020,
              "H=6144 I=19648 H_q=64 H_kv=8 D=96"),
    L2Problem("L2/027 gqa_yarn_rope_qknorm",
              "L2/027_grouped_query_attention_with_yarn_rope_and_qk_norm", decompose_027,
              "H=5120 H_q=40 H_kv=8 D=128 — attention only"),
    L2Problem("L2/053 mllama_text_decoder",
              "L2/053_text_decoder_layer_with_self_attention_and_mlp", decompose_053,
              "H=4096 I=14336 — same family as L2/002"),
    L2Problem("L2/054 vision_encoder_block",
              "L2/054_vision_encoder_layer_with_gated_residuals", decompose_054,
              "H=1280 I=5120 D=80 (small heads) + LayerNorm + GELU"),
]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_l2(prob: L2Problem, axes: dict) -> None:
    ops = prob.decompose(axes)
    results = [op.metrics() for op in ops]
    sum_t_sol = sum(r["t_sol_us"] for r in results) or 1e-9
    total_flops = sum(r["flops"] for r in results)
    total_bytes = sum(r["bytes"] for r in results)
    regime_t = {"compute":0.0,"memory":0.0,"balanced":0.0,"latency":0.0}
    regime_n = {"compute":0,"memory":0,"balanced":0,"latency":0}
    for r in results:
        regime_t[r["regime"]] += r["t_sol_us"]; regime_n[r["regime"]] += 1

    print(f"--- {prob.name}  axes={axes}")
    print(f"    total {total_flops/1e9:.1f} GFLOPs, {total_bytes/1e6:.1f} MB, sum_t_sol={sum_t_sol:.1f}us")
    # Per-op compact line
    hdr = f"    {'op':<22} {'kind':<8} {'FLOPs(G)':>8} {'B(MB)':>7} {'AI':>6} {'t_sol(us)':>9} {'%':>5} {'regime':>9}"
    print(hdr)
    for r in results:
        pct = 100*r["t_sol_us"]/sum_t_sol
        print(f"    {r['name']:<22} {r['kind']:<8} {r['flops']/1e9:>7.2f} {r['bytes']/1e6:>7.1f} "
              f"{r['ai']:>6.0f} {r['t_sol_us']:>9.2f} {pct:>4.1f}% {r['regime']:>9}")
    # Regime breakdown
    breakdown = " ".join(f"{k}:{regime_n[k]}({100*regime_t[k]/sum_t_sol:.0f}%)"
                         for k in ["compute","balanced","memory","latency"] if regime_n[k])
    print(f"    regime: {breakdown}")
    # Top-3 hot ops
    hot = sorted(results, key=lambda r: -r["t_sol_us"])[:3]
    print(f"    HOT ops: " + " | ".join(
        f"{h['name']}({h['regime'][:3]},{h['t_sol_us']:.0f}us)" for h in hot))
    print()


def load_workload(subdir: str) -> list[dict]:
    return [json.loads(l) for l in (CONTEST_ROOT / subdir / "workload.jsonl"
                                    ).read_text().splitlines() if l.strip()]


def report_problem(prob: L2Problem, smoke: bool = True) -> None:
    """Render decomposition for one or more workload rows of this L2 problem.

    smoke=True (default): show only small/mid/large representatives (3 rows).
    smoke=False: every workload row — useful for long-context analysis where
                 attention dominance shifts dramatically with seq_len.
    """
    print("=" * 130)
    print(f"PROBLEM: {prob.name}")
    if prob.note: print(f"  note: {prob.note}")
    print("=" * 130)
    wl = load_workload(prob.subdir)
    wl.sort(key=lambda r: r["axes"].get("batch_size",1)*r["axes"].get("seq_len",1))
    if smoke and len(wl) > 3:
        picks = [wl[0], wl[len(wl)//2], wl[-1]]
    else:
        picks = wl
    for w in picks:
        report_l2(prob, w["axes"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="L2 multi-kernel batch roofline analyzer")
    ap.add_argument("--full", action="store_true",
                    help="show every workload row (default: smoke 3 reps per problem)")
    ap.add_argument("--problem", help="only run this problem (substring match on name)")
    args = ap.parse_args()
    for p in PROBLEMS:
        if args.problem and args.problem not in p.name:
            continue
        report_problem(p, smoke=not args.full)
