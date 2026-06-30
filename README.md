# SOL-Lite

Curated **60-problem Contest subset** extracted from [NVIDIA SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench).

This repo contains benchmark definitions only. Evaluation uses the upstream [sol-execbench](https://github.com/NVIDIA/SOL-ExecBench) CLI.

## Problem breakdown

| Subset | Count | Source |
|--------|------:|--------|
| L1 | 20 | SOL-ExecBench L1 picks |
| L2 | 20 | SOL-ExecBench L2 picks |
| Quant | 10 | SOL-ExecBench Quant picks |
| FlashInfer-Bench | 10 | SOL-ExecBench FlashInfer-Bench picks |
| **Total** | **60** | |

Problem list: [`data/benchmark/problem-lists/Contest.txt`](data/benchmark/problem-lists/Contest.txt)

## Layout

```
data/benchmark/
├── Contest/
│   ├── L1/
│   ├── L2/
│   ├── Quant/
│   └── FlashInfer-Bench/
└── problem-lists/
    └── Contest.txt
```

Each problem directory contains `definition.json`, `reference.py`, and `workload.jsonl`.

## Prerequisites

1. Install [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) and its dependencies (Python 3.12+, CUDA GPU).
2. Ensure `sol-execbench` is on your `PATH` (or run from that repo's virtualenv).

## Run

From this repo root, using the bundled batch script:

```bash
# All 60 Contest problems (reference implementation)
uv run scripts/run_dataset.py data/benchmark --category Contest -o ./out

# One subset
uv run scripts/run_dataset.py data/benchmark/Contest/L1 -o ./out

# Single problem
uv run scripts/run_dataset.py data/benchmark/Contest/L1/069_rms_norm -o ./out

# Custom solution
uv run scripts/run_dataset.py data/benchmark/Contest/L1/069_rms_norm \
  --solution-name solution.py -o ./out
```

Quick smoke test:

```bash
uv run scripts/run_dataset.py data/benchmark/Contest/L1/069_rms_norm \
  --max-workloads 1 -o ./out
```

## Upstream links

- [SOL-ExecBench (GitHub)](https://github.com/NVIDIA/SOL-ExecBench)
- [Full dataset (HuggingFace)](https://huggingface.co/datasets/nvidia/SOL-ExecBench)
- [Leaderboard](https://research.nvidia.com/benchmarks/sol-execbench)

## Roofline analysis (H800)

Beyond raw `speedup_factor`, this repo includes an offline roofline analyzer
that classifies each workload row's regime against the H800 SXM5 peaks
(BF16 989 TFLOPS, FP8 1979 TFLOPS, HBM3 3.35 TB/s) and tells you **which
metric to report** for that row. The supported regimes are:

| regime | when | primary metric |
|---|---|---|
| compute | AI > 2× ridge, t_sol ≥ 5μs | **MFU** |
| memory | AI < 0.5× ridge, t_sol ≥ 5μs | **BW%** |
| balanced | AI within 0.5×..2× of ridge | MFU **and** BW% (both) |
| latency | t_sol < 5μs (too small for roofline) | time + speedup_vs_reference |

Why per-row: across the 60 problems only ~18 admit a single MFU-or-BW%
metric. The rest either cross regimes within one problem (34), or are
dominantly latency-bound (8). See [`roofline_summary.md`](roofline_summary.md)
for the full per-problem regime table.

### Workflow

```bash
# 1. Run sol-execbench, save trace JSONL
uv run scripts/run_dataset.py data/benchmark/Contest/L1/069_rms_norm \
    -o ./out

# 2. Feed the trace into the roofline measurement tool
uv run python scripts/roofline_measure.py process ./out/traces.jsonl \
    -o ./out/traces.with_roofline.jsonl --report
```

The output adds a `roofline` block to each PASSED row with:

```
regime, ai, t_sol_us, t_measured_us,
achieved_tflops, achieved_gbps,
mfu, mfu_ceiling, bandwidth_utilization,
sol_efficiency, speedup_vs_reference, below_latency_floor
```

The summary table reports geometric means per problem, with the dominant
regime so you know which column to actually read.

### Standalone (offline-only) tools

```bash
# Per-problem regime tables (no measurements needed)
uv run python scripts/roofline_tier1_batch.py   # 36 dense problems
uv run python scripts/roofline_moe.py           # 15 MoE problems (with routing simulation)
uv run python scripts/roofline_l2.py            # 9 fused multi-kernel L2 blocks

# Cross-cutting summary CSV + markdown
uv run python scripts/roofline_summary.py
# -> roofline_summary.csv, roofline_summary.md

# Quick analytical lookup for one (problem, axes) without a trace
uv run python scripts/roofline_measure.py offline 069_rms_norm \
    '{"batch_size":1,"seq_len":2048}'

# List all 60 registered problems and which analyzer handles each
uv run python scripts/roofline_measure.py list
```

### Three analyzers, one registry

| script | tier | covers |
|---|---|---|
| `roofline_tier1_batch.py` | dense single-kernel / multi-GEMM chains | RMSNorm, GEMM, RoPE, MLA, MLP, attention variants |
| `roofline_moe.py` | data-dependent MoE | routing simulated via random gate logits → realized FLOPs/bytes |
| `roofline_l2.py` | L2 fused multi-kernel blocks | per-op decomposition (10-17 ops/layer) |

`roofline_measure.py` builds a unified `definition-name → handler` registry
across all three so any trace can be processed regardless of tier.

## License

Benchmark problem definitions inherit the license terms of NVIDIA SOL-ExecBench (Apache-2.0).
