# H800 Roofline Analysis — Methodology

This document explains **why** SOL-Lite picks the metrics it does, why a
uniform "MFU + BW%" report is wrong for ~70% of the 60 Contest problems,
and how each problem ends up in one of four regimes.

The per-problem outcomes are in [`roofline_summary.md`](roofline_summary.md)
and [`roofline_summary.csv`](roofline_summary.csv). This file is the
*reasoning* behind those tables — read it once before reading the tables.

---

## 1. The hardware roofline (default: H800 SXM5)

| Quantity | Value (H800 SXM5) |
|---|---|
| BF16 / FP16 Tensor Core peak | 989 TFLOPS |
| FP8 Tensor Core peak | 1979 TFLOPS |
| HBM3 bandwidth | 3.35 TB/s |
| L2 cache | 50 MB |
| **Ridge point (BF16)** | **295 FLOPs/byte** |
| **Ridge point (FP8)** | **591 FLOPs/byte** |

Ridge = peak_FLOPs / peak_BW. It is the arithmetic intensity above which
a kernel is compute-bound and below which it is memory-bound. The ridge
is purely a hardware property — independent of any kernel implementation.

Other GPUs can be targeted via `--hardware {H800|H100|H200|B200|A100|...}`
on every analyzer / measurement entry point, or via the `SOL_LITE_HARDWARE`
env var. The full preset table is at the bottom of [README.md](README.md).
The chosen GPU only changes the ridge and ceiling constants; the regime
classifier logic (and FLOPs / bytes computation) is identical.

When the hardware running the kernel differs from the hardware the
ceilings are computed for (e.g. measuring on B200 with H800 ceilings),
MFU and BW% will reflect the *target* hardware's saturation, not the
*measurement* hardware's. Set `--hardware` to match the measurement GPU
to get directly-interpretable [0, 1] ratios.

---

## 2. The four regimes

Each workload row of each problem is classified into exactly one regime:

| Regime | Trigger | Primary metric |
|---|---|---|
| **compute** | AI > 2 × ridge, t_sol ≥ 5μs | MFU |
| **memory** | AI < 0.5 × ridge, t_sol ≥ 5μs | BW% |
| **balanced** | AI within 0.5×..2× of ridge, t_sol ≥ 5μs | MFU **and** BW% |
| **latency** | t_sol < 5μs (regardless of AI) | time_us + speedup_vs_reference |

Where:

```
flops    = algorithmic FLOPs at this workload (closed-form for dense ops;
           realized via routing simulation for MoE; per-op decomposition for L2)
bytes    = algorithmic HBM bytes (each input read once, each output written once)
peak     = peak FLOPs for the dtype in question (BF16 989, FP8 1979)
AI       = flops / bytes
t_sol    = max(flops / peak, bytes / peak_BW)   ← speed-of-light time
```

### Why these specific thresholds

**Why 2× / 0.5× around ridge?** Right at the ridge, a real kernel might
be limited by either roof depending on micro-architectural details
(occupancy, register pressure, L2 hits). The 2× band gives margin so that
"compute" and "memory" labels actually predict where headroom is.

**Why 5μs latency floor?** H800 has ~5–8μs of CUDA launch + warm-up
overhead per kernel. If `t_sol < 5μs`, the kernel cannot reach steady-state
throughput — its wall time is dominated by launch and SM occupancy ramp,
not roofline. Reporting MFU/BW% in this regime is noise: any number you
compute is below the actual ceiling because the kernel never got going.

---

## 3. Why "MFU + BW% for everyone" is wrong

There are four distinct failure modes. The summary CSV tags every problem
with which failure applies.

### Failure 1 — MFU is physically bounded far below 1.0

For memory-bound problems, even a *perfect* kernel cannot reach high MFU,
because it would have to sustain more FLOPs than the memory bandwidth can
feed.

The actual upper bound on MFU is:

```
mfu_ceiling = min(1.0, peak_BW · AI / peak_FLOPs)
```

A small sample:

| Problem | AI | mfu_ceiling | What happens if you report MFU |
|---|---:|---:|---|
| L1/069 residual+rmsnorm | 0.8 | **0.003** | "MFU = 0.001" reads as terrible; it is actually 33% of physical limit |
| FIB/013 GQA paged decode | 4 | **0.013** | MFU column is uninterpretable; BW% is the real signal |
| FIB/018 MLA paged decode | 29 | **0.10** | MLA's compressed KV doubles AI vs GQA, but still memory-bound |
| Quant/005 FP8 router | 230 | **0.40** | FP8 doubles peak compute, so even AI=230 doesn't reach the FP8 ridge of 591 |
| L1/044 MoE expert | 80 | **0.58** | Capped at 58% by the sparse weight load |

Note that MFU's *definition* is in [0, 1]. The 0.003 / 0.013 / 0.10 / 0.40
above are **physical achievability ceilings for this problem**, not a
redefinition of MFU. The two are different concepts that get confused
because both live on the [0, 1] axis.

The harm of reporting MFU alone on memory-bound problems: readers
unconsciously interpret `MFU = 0.003` as "kernel achieves 0.3% of best
possible" when in fact it might be achieving 100% of the physically
possible.

### Failure 2 — BW% is symmetrically bounded for compute-bound problems

The dual problem: when `AI >> ridge`, even a perfect kernel can't drive
HBM hard because most of its time is spent computing.

```
bw_ceiling = min(1.0, peak_FLOPs / (peak_BW · AI))
```

| Problem | AI | bw_ceiling |
|---|---:|---:|
| L2/004 fused MLP (H=16384) | 1764 | 0.17 |
| Quant/003 FP8 MLP | 3000 | 0.20 |
| Quant/015 MLA o_proj | 5000 | 0.12 |

A perfect kernel on Quant/003 sustains 12% of HBM. Reporting BW% there
mirrors Failure 1 in the opposite direction.

### Failure 3 — Roofline assumptions fail for tiny problems

When `t_sol < 5μs`, the kernel cannot reach steady-state throughput. Both
MFU and BW% will be artificially low, but the cause is *not* poor
implementation — it's that the problem is too small to fill the GPU.

| Problem | Latency rows | Total rows |
|---|---:|---:|
| L2/006 multimodal position calc | 16 | 16 (all) |
| L1/071 KV cache update | 15 | 16 |
| L1/011 RoPE compute | 15 | 16 |
| FIB/013 GQA paged decode | 29 | 48 |
| FIB/018 MLA paged decode | 30 | 47 |

For these rows the meaningful metric is **wall time + speedup vs.
reference**: did the solution kernel actually do the operation faster
than the reference? Roofline is irrelevant.

### Failure 4 — One problem can cross multiple regimes across its workloads

This is the most consequential failure mode by row count: **34/60 problems
have at least two regimes across their workload rows**.

The two clearest examples:

**FIB/005 GEMM N=256 K=7168** (17 workload rows):

| M | AI | t_sol | regime | mfu_ceiling |
|---:|---:|---:|---|---:|
| 1 | 1 | 1.10 μs | latency | 0.003 |
| 80 | 60 | 1.45 μs | latency | 0.205 |
| 901 | 194 | 5.09 μs | balanced | 0.66 |
| 14104 | 243 | 63.6 μs | balanced | 0.82 |

Reporting one MFU number for this problem averages a latency-bound row
with `mfu_ceiling=0.003` against a balanced row with `mfu_ceiling=0.82`.
The average has no physical meaning.

**L1/067 flash attention ultra-long** (18 rows, AI spans 126 → 10271):
spans latency / memory / balanced / compute regimes within one problem.

Solution: **report per-row regime** and let aggregates be per-regime, not
per-problem.

---

## 4. The four metrics, and when to actually report them

| Metric | When to report | When NOT to report |
|---|---|---|
| **MFU** | compute, balanced | memory (mfu_ceiling too small), latency (roofline invalid) |
| **BW%** | memory, balanced | compute (bw_ceiling too small), latency (roofline invalid) |
| **SoL efficiency** (`t_sol / t_measured`) | optional unifier across compute/memory/balanced — equals MFU in compute region, equals BW% in memory region | latency |
| **time_us + speedup_vs_reference** | always available; primary for latency regime | — (always safe to include) |

The summary CSV's `metric` column picks one of:

- `MFU` (compute regime is ≥80% of rows)
- `BW%` (memory regime is ≥80% of rows)
- `MFU + BW% (both)` (balanced is ≥80% of rows)
- `time + speedup_vs_reference` (latency is ≥80% of rows)
- `per-row regime` (mixed: pick per row using its regime tag)

Out of 60 problems, the distribution is:

| metric | count | share |
|---|---:|---:|
| per-row regime | 34 | 57% |
| time + speedup_vs_reference | 8 | 13% |
| MFU | 8 | 13% |
| BW% | 6 | 10% |
| MFU + BW% (both) | 4 | 7% |

**Only 18/60 (30%)** admit a single MFU-or-BW% metric. The remaining 70%
need either per-row classification (34) or non-roofline metrics (8).

---

## 5. Why three analyzer tiers

A single `flops_bytes_fn(axes) -> (flops, bytes, peak)` works for some
problems but not others. The 60 problems decompose into three groups by
how their FLOPs and bytes depend on inputs:

### Tier 1 — dense, data-independent, single-kernel (36 problems)

FLOPs and bytes are closed-form expressions in the workload axes.

Examples: GEMM (2·M·N·K), RMSNorm (5·B·S·H), RoPE (6 ops/element),
attention (with FlashAttention-style IO accounting).

Sub-templates inside Tier 1:

- **A** RMSNorm-family (5 problems)
- **B** dense GEMM (3 problems)
- **C** RoPE / position encoding (5 problems)
- **D** MLA / fused QKV chains (4 problems)
- **E** MLP / SwiGLU (5 problems)
- **F** attention variants (11 problems)
- **Catch-up** 3 odds: LM head, fused RoPE+norm+cache, vision patch merger

### Tier 2 — data-dependent (MoE, 15 problems)

FLOPs and bytes depend on token-to-expert routing, which is determined
at runtime by softmax over random gate logits. A closed-form expression
gives only an expected value; reality has load imbalance that affects
wall time.

The analyzer simulates routing on random logits and reports:

- realized per-expert token counts
- imbalance ratio (peak / mean)
- serial vs. grouped-GEMM `t_sol` (the gap is the optimization headroom)

Without simulation, MoE FLOPs/bytes are off by 10–50% (depending on T,
topk, E). With simulation, they match a real run's expected cost — the
remaining gap is wall-clock launch/imbalance overhead.

### Tier 3 — multi-kernel fused (L2, 9 problems)

A single decoder layer is 10–17 ops, each with a different regime:

- input RMSNorm: memory
- q/k/v_proj GEMMs: depends on M (typically compute for medium-large M)
- RoPE: memory
- attention Q·K^T: balanced for prefill, memory for decode
- softmax: memory
- attention A·V: same as Q·K^T
- o_proj GEMM: compute or balanced
- residual add: memory
- post-attention RMSNorm: memory
- gate/up/down MLP GEMMs: compute (large M)
- silu_mul: memory

A single MFU for the whole row would average all of these — meaningless.
The Tier-3 analyzer hand-decomposes each L2 problem into ops, classifies
each op independently, and reports:

- per-op `(flops, bytes, regime, t_sol)`
- regime distribution by `t_sol`-weighted share
- hot ops (top-3 by `t_sol`) — the optimization priority list

Critical empirical findings from the Tier-3 analyzer:

1. **gate/up/down MLP GEMMs dominate (60–80% of t_sol)** in every L2
   layer that has an MLP. Optimizing them first is the right call.
2. **silu_mul is a hidden BW killer**: 5–8% of t_sol because of the 3·B·S·I
   element traffic (larger than all norms combined).
3. **Attention's S² ops can dominate at long context**: L2/002 at B=2
   S=8192, softmax alone takes 28% of total t_sol.
4. **RMSNorm is 1–2% of t_sol**: fused-rmsnorm optimization caps out at
   ~2% layer-level improvement.

---

## 6. Special cases by category

### L1 (20 problems)

Mostly Tier-1, with a few Tier-2 (MoE) entries.

- **003 lm_head** — `logits_to_keep` is the *algorithmic* slice length, not
  `seq_len`. Reference implementation computes the full S then slices; the
  semantic FLOPs use `logits_to_keep`. A correct kernel can show >100% MFU
  against the naive `2·B·S·H·V` formula — we use `logits_to_keep` instead.
- **011 / 071 / 014 / 023 / 020 (rope+grid+vision-patch)** — small, dominantly
  latency-bound. Report time + speedup.
- **021 vision cu_seqlens** — variable-length attention. `sum(S_i²)` cannot
  be derived from axes alone; we approximate via uniform-S assumption and
  flag the caveat. Real FLOPs need `cu_seqlens` from a safetensors file.
- **046 attention softmax (softcap+dropout)** — just softmax, no matmul.
  AI ≈ 1.2, pure memory regime. MFU is meaningless here.
- **049 QK matmul only** — output is the `[B, H_q, S, S]` attention matrix,
  which IS materialized (it's the requested output). Different bytes
  formula from a fused attention kernel.

### L2 (20 problems, mixed Tier-2 / Tier-3)

- **9 Tier-3 fused blocks**: 002, 004, 007, 009, 019, 020, 027, 053, 054
- **9 Tier-2 MoE forwards**: 008, 010, 012, 013, 029, 048, 065, 081, 082
- **1 Tier-2 MoE routing-only**: 049
- **1 Tier-1 indexing**: 006 (multimodal position calculation, no GEMMs)

L2/006 is 100% latency for all 16 workloads. L2/009 mixes Tier-3
attention with Tier-2 MoE in one layer — the analyzer models the MoE
section with expected per-expert load.

### FlashInfer-Bench (10 problems)

Production kernels from real frameworks. Key facts:

- **005 GEMM N=256 K=7168** — asymptotic AI = `N·K/(N+K) = 247 < ridge
  295`. Even M → ∞ is memory-bound. mfu_ceiling caps at 0.82.
- **013 / 018 paged decode** — decode has only 1 query token per batch,
  AI = `H_q / H_kv` (GQA) or `H_q` (MLA). All workloads fall into
  latency/memory regimes; MFU < 0.013 (GQA) or < 0.10 (MLA).
- **020 FP8 MoE block-scale** — `num_local_experts=32` of 256 total, so
  expert parallel limits the local view. Routing simulation accounts for
  this.

### Quant (10 problems, all FP8 variants)

A critical distinction: **declared dtype matters**.

- **003 / 005 / 015** — `float8_e4m3fn` declared inputs → bytes use 1
  byte/element, peak = FP8 1979 TFLOPS
- **002 / 004 / 011 / 012 / 013 / 014 / 016** — `bfloat16` declared
  inputs; the kernel may internally use FP8 paths but the HBM-level
  read cost is bf16. Bytes use 2 byte/element, peak = BF16 989 TFLOPS

Reporting MFU against the wrong dtype's peak distorts the metric by 2×.
The summary CSV picks the right peak based on declared dtype.

FP8 changes the ridge point: 591 FLOPs/byte vs. 295 for BF16. Most FP8
GEMM problems land *below* their FP8 ridge despite being above the BF16
ridge — i.e., switching to FP8 reduces effective AI relative to ridge.

---

## 7. The unified registry (roofline_measure.py)

All 60 problems are registered in `scripts/roofline_measure.py` under a
single `problem_name → handler` mapping. The handler abstracts over the
three tiers:

```
Handler.evaluate(axes) -> (flops, bytes, peak, ai, t_sol_us, regime, mfu_ceiling)
```

Given a sol-execbench trace JSONL with measured `latency_ms`, the
measurement loop produces:

```
mfu                   = achieved_flops_per_s / peak_flops
mfu_ceiling           = analytical upper bound (often << 1!)
bandwidth_utilization = achieved_bytes_per_s / peak_BW
sol_efficiency        = t_sol_us / t_measured_us
speedup_vs_reference  = ref_latency_ms / latency_ms   (already in trace)
below_latency_floor   = True when t_sol < 5μs
```

The CLI reports geometric means per problem and identifies the dominant
regime. Per-row detail is in the augmented JSONL output.

---

## 8. When to re-classify

Regime is a function of `(axes, dtype, peak_FLOPs, peak_BW)`. It changes
when:

- **The implementation uses a different dtype than declared.** A kernel
  that takes bf16 input but does FP8 GEMM internally is *not* a Quant/003
  problem; its declared peak is still BF16. The achievable MFU
  is the BF16 peak, but the achieved throughput can exceed the BF16 ridge
  if the FP8 path is effective — leading to MFU > mfu_ceiling. This is
  unusual but legal; flag it.
- **The reference is asymptotically wasteful.** L1/003 lm_head's reference
  computes all logits before slicing. The algorithmic minimum (only the
  kept slice) is what determines the regime. We use the minimum.
- **The hardware changes.** Switching to H100 (3.0 TB/s, 989 TFLOPS) or
  B200 (8 TB/s, 4500 TFLOPS) shifts the ridge. Tier-1 / Tier-2 regimes can
  flip. Tier-3 per-op regimes shift too.

---

## 9. Recommended reading

1. [`roofline_summary.md`](roofline_summary.md) — per-problem regime table
   (the *outcome* of the analysis in this file)
2. [`roofline_summary.csv`](roofline_summary.csv) — same, machine-readable
3. `scripts/roofline_tier1_batch.py`, `roofline_moe.py`, `roofline_l2.py` —
   the three analyzers, where you can read the FLOPs/bytes formulas for
   any specific problem
4. `scripts/roofline_measure.py` — the trace-augmentation tool
5. `scripts/roofline_bench.py` — standalone timing engine (back-to-back
   launches; methodology borrowed from
   [SOLBench-H800](https://github.com/runboo-fly/SOLBench-H800))

---

## 10. Summary in one paragraph

MFU and BW% are the right metric *types* for a roofline analysis, but
**they cannot be applied uniformly** to all 60 problems. Each problem
has a physical MFU ceiling and a physical BW ceiling, both ≤ 1.0, and
typically one of them is far below 1.0 (the metric on the *other* roof).
70% of problems either cross regimes within their workload set or are
dominantly latency-bound and should not use roofline at all. The right
report is per-regime: MFU for compute, BW% for memory, both for balanced,
and time + speedup for latency. This document, the three analyzers, and
the measurement loop together implement this regime-aware report.
