# FLOPs/Bytes Disagreements: SOL-Lite Analyzer vs Ray-234 Costs

Cross-check of SOL-Lite's per-axes analytical `(flops, bytes)` formulas
against Ray-234's per-UUID `workload_costs.jsonl` (imported from
[SoL-Contest-InfiniAI](https://github.com/qhy991/SoL-Contest-InfiniAI)).

All 1019 workload UUIDs across 60 problems were checked. Summary:

- **20/60 problems agree completely** (RMSNorm, GEMM, MLP, MLA QKV, dense attention)
- **40/60 problems have at least one disagreement**
- **236/1019 rows** have FLOPs off by ≥2×
- **455/1019 rows** have bytes off by ≥2×

The root causes group into **5 categories**, listed below with the fix
approach.

---

## Root cause 1: Convention — elementwise/norm/RoPE FLOPs

**Problems affected**: `069_rms_norm`, `002/003/023/026_rmsnorm`,
`011/014/018_rope_variants`, `046_attention_softmax`,
`059_moe_group_score`, `006/023_multimodal_position`.

**SOL-Lite**: computes true FLOPs (`5·B·S·H` for RMSNorm,
`6·B·H·S·D` for RoPE, etc.).

**Ray-234**: sets `flops = 0` for these ops by convention.

**Why both are defensible**:
- Ray-234's convention: these ops are always memory-bound. Reporting
  MFU=0.003% would be misleading — better to zero it out and only
  report BW%. This mirrors what SOLBench-H800's `bench_spec.py`
  protocol asks authors to do.
- SOL-Lite's convention: FLOPs are algorithmic ground truth,
  independent of whether they're the bottleneck. Downstream code can
  decide to hide "insignificant MFU" if desired.

**Fix**: adopt Ray-234's per-UUID number. Our regime classifier still
sees the same "memory" answer either way (AI ≈ 1 with FLOPs, still
< ridge; AI = 0 without FLOPs, trivially < ridge). No behavioral
regression.

## Root cause 2: Paged decode/prefill bytes (SOL-Lite bug)

**Problems affected**: `013_gqa_paged_decode`,
`018_mla_paged_decode`, `019_mla_paged_prefill` (3 problems, ~130 rows).

**SOL-Lite formula** (`roofline_tier1_batch.py:gqa_paged_decode_fn`):
```
bytes = B·H_q·D·2       # Q
      + 2·L·H_kv·D·2    # K + V cache reads (L = num_kv_indices)
      + B·H_q·D·2       # O
```
where `L = num_kv_indices` is the sum of KV pages across the batch.

**Bug**: this counts the KV cache reads as `L · H_kv · D` bytes, but
in a paged decode kernel each batch element reads only its *own* KV
pages, not the union across the batch. The correct formula depends on
per-batch page counts (which vary per workload).

**Ratio observed**: SOL-Lite over-counts by **30–70×** on 013 GQA
paged decode, 35× on 018 MLA paged decode.

**Fix**: adopt Ray-234's per-UUID bytes (which appear to be measured
during actual paged-kernel execution).

## Root cause 3: Quant FP8 dtype byte-size (SOL-Lite bug)

**Problems affected**: `002/004/011/012/013/016_fp8_*` in `Quant/`
(6 problems, ~90 rows).

**SOL-Lite reasoning**: the problems declare `dtype=bfloat16` for
inputs/weights in `definition.json`, so bytes are counted at 2
bytes/element.

**Ray-234 reasoning**: even though inputs are declared bf16, an
optimal kernel casts them to fp8 internally to use the FP8 tensor
core. Bytes should be counted at **1 byte/element** to reflect what
the FP8 path actually moves through HBM.

**Ratio observed**: SOL-Lite over-counts bytes by exactly **2.00×**
across all these rows (FLOPs match to 4 decimals).

**Fix**: adopt Ray-234's per-UUID number. Alternative: detect FP8
intent from problem name and switch to 1-byte accounting in the
analyzer.

## Root cause 4: MoE bytes (SOL-Lite bug — L2 cache oblivious)

**Problems affected**: `008/009/010_moe_*`, `044/076_moe_expert*`,
`029_moe_shared`, `013_expert_shared`, and 5 more L2 MoE problems
(~150 rows).

**SOL-Lite formula** (`roofline_moe.py:MoESpec.analyze_row`, expert
MLP stage):
```
bytes_per_expert = (2H + 3I) · n_e · dtype_bytes  # activations
                 + 3 · H · I · dtype_bytes         # weights (cold read)
```
Summed across all active experts.

**Bug**: this assumes each active expert's weight is read **cold from
HBM**. In practice, each expert's weight is ~9 MB (H=2048, I=768),
which fits comfortably in the H800/B200 L2 cache (50–100 MB). A
grouped-GEMM kernel that batches all tokens for an expert into one
launch reads that weight from HBM **once** and reuses it in L2 for
subsequent tokens of the same expert.

**Ray-234 reasoning**: L2-cache-aware — each expert's weight counted
as one HBM read regardless of token count.

**Ratio observed**: SOL-Lite over-counts by **30–150×** for L2 MoE
problems.

**Fix**: adopt Ray-234's per-UUID number. Correcting the formula
would require modeling L2 residency, which is subtle for MoE (depends
on expert sequential order, cache eviction, etc.). Not worth the
effort when the per-UUID data is available.

## Root cause 5: Missing edge FLOPs in paged prefill (minor)

**Problems affected**: `017_gqa_ragged_prefill`, `019_mla_paged_prefill`.

**SOL-Lite**: attention QK + AV only.

**Ray-234**: includes cos/sin lookup and mask-related ops.

**Ratio**: SOL-Lite under-counts FLOPs by ~2× (median 0.5). Minor
because these problems are memory-bound anyway, so FLOPs don't
determine `t_sol`.

**Fix**: adopt Ray-234's per-UUID number.

---

## Fix implementation

`scripts/_costs.py` loads Ray-234's precomputed per-UUID costs
(`data/costs/ray234_h800.jsonl`, 1019 entries, ~250 KB). The
`Handler.evaluate(axes, uuid=None)` in `roofline_measure.py` looks up
UUID first; if found, uses Ray-234's `(flops, bytes_moved, precision)`
directly; otherwise falls back to the analytical formula.

Per-row output gains a `cost_source ∈ {"ray234", "analytic"}` field
so downstream consumers know the provenance.

Impact on regime classification: MoE problems that were misclassified
as "memory" (because of over-counted bytes) now correctly show as
"balanced" or "compute" once L2-cache-aware bytes are used. Paged
decode problems that were "memory" stay "memory" but with much higher
`mfu_ceiling` (since actual bytes moved is ~30× smaller).

## Cross-check going forward

To regenerate the per-problem disagreement table:

```bash
uv run python scripts/diagnose_ray234.py
```

To re-check after adding a new problem or updating Ray-234 data, this
diagnostic will re-scan all 1019 rows and highlight new discrepancies.

## Attribution

Ray-234 per-UUID costs are the work of Team Fudan for the SOL-Contest
submission `Ray-234/ray234-h800-analytical-tsol`. See
[SoL-Contest-InfiniAI docs/RAY234_H800_TSOL_TBASE.md](https://github.com/qhy991/SoL-Contest-InfiniAI/blob/main/docs/RAY234_H800_TSOL_TBASE.md).
