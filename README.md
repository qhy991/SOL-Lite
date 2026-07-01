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

**Minimum (offline analysis only, no GPU needed)**:
- Python 3.10+
- `numpy` (pulled in by `uv sync`)

**Full analysis (timing + measurement)**:
- CUDA GPU (H800 / H100 / H200 / B200 / A100 supported by presets)
- PyTorch 2.x with matching CUDA build (pulled by `uv sync --extra bench`)
- If timing sol-baseline kernels: `flashinfer`, `flash-attn`,
  `liger-kernel`, `causal-conv1d` per each baseline's requirements
  (typically already present in a sol-execbench environment)
- `safetensors` (for problems whose workload inputs point at safetensors
  blobs, e.g. FlashInfer-Bench paged/ragged attention)

## Environment variables

SOL-Lite scripts do not hardcode any paths. Set these before running the
GPU-bound tooling:

| Variable | Default | Used by | What it points at |
|---|---|---|---|
| `SOL_LITE_HARDWARE` | `H800` | all analyzers | GPU preset name (see [Hardware presets](#hardware-presets)); can also be passed as `--hardware {H800\|H100\|H200\|B200\|A100\|H800_PCIE\|H100_PCIE}` |
| `SOL_BASELINE_ROOT` | *(none — required)* | `bench_baselines.py` | Local checkout of [sol-baseline](https://github.com/qhy991/SOL-Baseline), for reading each baseline's `solution.json` |
| `SOL_EXECBENCH_ROOT` | *(none — required only if a workload uses safetensors inputs)* | `roofline_bench.py`, `bench_baselines.py` | Local checkout of [sol-execbench](https://github.com/NVIDIA/SOL-ExecBench), because paged/ragged attention workloads reference safetensors blobs from its `data/` tree |
| `SOL_LITE_RAY234_JSONL` | `data/costs/ray234_h800.jsonl` (bundled) | `_costs.py`, `diagnose_ray234.py` | Path to Ray-234's precomputed per-UUID `workload_costs.jsonl`. The bundled copy is authoritative for the 60 Contest problems; only override if you have newer data |

CLI flags always override env vars.

### One-shot setup

```bash
# Adjust paths for your machine
export SOL_BASELINE_ROOT=$HOME/sol-baseline
export SOL_EXECBENCH_ROOT=$HOME/sol-execbench
export SOL_LITE_HARDWARE=B200            # or omit to default to H800

# From this repo root
uv sync                         # numpy only (offline analysis)
uv sync --extra bench           # add torch (needed for scripts/roofline_bench.py
                                #  and scripts/bench_baselines.py)
```

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
that classifies each workload row's regime against configurable GPU peaks
(default H800; switch via `--hardware`) and tells you **which metric to
report** for that row. The supported regimes are:

| regime | when | primary metric |
|---|---|---|
| compute | AI > 2× ridge, t_sol ≥ 5μs | **MFU** |
| memory | AI < 0.5× ridge, t_sol ≥ 5μs | **BW%** |
| balanced | AI within 0.5×..2× of ridge | MFU **and** BW% (both) |
| latency | t_sol < 5μs (too small for roofline) | time + speedup_vs_reference |

Why per-row: across the 60 problems only ~18 admit a single MFU-or-BW%
metric. The rest either cross regimes within one problem (34), or are
dominantly latency-bound (8). See [`roofline_summary.md`](roofline_summary.md)
for the full per-problem regime table, and
[`ROOFLINE_ANALYSIS.md`](ROOFLINE_ANALYSIS.md) for the methodology
(why these metrics, why per-regime, why three analyzer tiers).

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

### Standalone timing engine (no sol-execbench needed)

For quick kernel iteration without going through the full sol-execbench
CLI, use `roofline_bench.py`. It times your kernel with launch-overhead
amortisation (back-to-back launches between cuda.Events with no sync
between them — methodology borrowed from
[SOLBench-H800](https://github.com/runboo-fly/SOLBench-H800)) and reports
regime-aware metrics.

```bash
# Install the optional bench extra (pulls in torch)
uv sync --extra bench

# Time the reference impl of one problem
uv run python scripts/roofline_bench.py L1/069_rms_norm --smoke

# Time a user solution
uv run python scripts/roofline_bench.py L1/069_rms_norm \
    --solution my_kernel.py --smoke

# CSV output
uv run python scripts/roofline_bench.py L1/069_rms_norm \
    --smoke -o results.csv
```

Output (per row): `regime, latency_us, MFU, mfu_ceiling, BW%, SoL_eff`.
Below the table: regime distribution + recommended primary metric.

### Standalone (offline-only) tools

```bash
# Per-problem regime tables (no measurements needed)
uv run python scripts/roofline_tier1_batch.py   # 36 dense problems
uv run python scripts/roofline_moe.py           # 15 MoE problems (with routing simulation)
uv run python scripts/roofline_l2.py            # 9 fused multi-kernel L2 blocks

# --smoke (3 reps per problem) and --problem <substring> filtering
uv run python scripts/roofline_tier1_batch.py --smoke --problem 069
uv run python scripts/roofline_moe.py --smoke
uv run python scripts/roofline_l2.py --problem L2/002    # L2 default IS smoke; use --full for all rows

# Cross-cutting summary CSV + markdown
uv run python scripts/roofline_summary.py
# -> roofline_summary.csv, roofline_summary.md

# Quick analytical lookup for one (problem, axes) without a trace
uv run python scripts/roofline_measure.py offline 069_rms_norm \
    '{"batch_size":1,"seq_len":2048}'

# List all 60 registered problems and which analyzer handles each
uv run python scripts/roofline_measure.py list
```

### Benchmarking sol-baseline solutions

`bench_baselines.py` walks the sol-baseline checkout, times each kernel,
and produces a regime-aware CSV + markdown plus optional v3 submission
directories:

```bash
# Env vars set as shown above; also need to be in the sol-execbench venv
# for flashinfer / flash-attn / liger / causal-conv1d imports.
uv run --project "$SOL_EXECBENCH_ROOT" \
    python scripts/bench_baselines.py --hardware B200 --smoke \
    -o baseline_roofline

# Export SoL-Contest-InfiniAI schema-v3 submission dirs at the same time
uv run --project "$SOL_EXECBENCH_ROOT" \
    python scripts/bench_baselines.py --hardware B200 --smoke \
    --emit-v3-submissions data/submissions \
    --submission-user alice \
    -o baseline_roofline
```

Each v3 submission is a self-contained directory:

```
data/submissions/<user>/<task_id>/r<round>/
  manifest.json           # SoL-Contest-InfiniAI schema v3
  solution/README.md
  solution/baseline.py    # extracted kernel source
  results/workloads.json  # per-workload t_sol_ms / t_base_ms / mfu / bw
```

Directly importable via [SoL-Contest-InfiniAI](https://github.com/qhy991/SoL-Contest-InfiniAI)'s
`lb import-trace`.

### Three analyzers, one registry

| script | tier | covers |
|---|---|---|
| `roofline_tier1_batch.py` | dense single-kernel / multi-GEMM chains | RMSNorm, GEMM, RoPE, MLA, MLP, attention variants |
| `roofline_moe.py` | data-dependent MoE | routing simulated via random gate logits → realized FLOPs/bytes |
| `roofline_l2.py` | L2 fused multi-kernel blocks | per-op decomposition (10-17 ops/layer) |
| `roofline_measure.py` | trace augmentation | join sol-execbench traces with the analyzer registry |
| `roofline_bench.py` | standalone timing | back-to-back launch (borrowed from SOLBench-H800) + regime-aware metrics |
| `bench_baselines.py` | batch benchmark | walks sol-baseline, times each solution, exports v3 submissions |
| `diagnose_ray234.py` | cross-check | compare analyzer's per-axes formulas vs Ray-234's per-UUID costs |

`roofline_measure.py` and `roofline_bench.py` share a unified
`definition-name → handler` registry across all three analyzer tiers,
so any trace or solution can be processed regardless of tier.

## Acknowledgements

- The benchmark problem set itself is from
  [NVIDIA SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench).
- The back-to-back-launch timing methodology in `scripts/roofline_bench.py`
  is adapted from [SOLBench-H800](https://github.com/runboo-fly/SOLBench-H800)
  (`harness.py`).
- Per-workload `(flops, bytes_moved)` in `data/costs/ray234_h800.jsonl`
  comes from Team Fudan's Ray-234 submission to
  [SoL-Contest-InfiniAI](https://github.com/qhy991/SoL-Contest-InfiniAI).
  Our analytical formulas over-count bytes by up to 150× for MoE
  problems (they don't model L2 cache reuse), so the per-UUID Ray-234
  data is preferred when available. See [DISAGREEMENTS.md](DISAGREEMENTS.md)
  for the full root-cause analysis.

## Hardware presets

All analyzers and the timing engine accept `--hardware {H800|H100|H200|B200|A100|H800_PCIE|H100_PCIE}`
(or set `SOL_LITE_HARDWARE` env var). The default is **H800 SXM5**.
Presets are dense Tensor-Core peaks (no sparsity assumptions):

| GPU | BF16 TFLOPS | FP8 TFLOPS | HBM TB/s | ridge_BF16 | ridge_FP8 |
|---|---:|---:|---:|---:|---:|
| H800 / H100 SXM | 989 | 1979 | 3.35 | 295 | 591 |
| H800 / H100 PCIe | 756 | 1513 | 2.00 | 378 | 756 |
| H200 SXM | 989 | 1979 | 4.80 | 206 | 412 |
| B200 SXM | 2250 | 4500 | 8.00 | 281 | 562 |
| A100 SXM | 312 | — | 2.04 | 153 | — |

Inspect: `uv run python scripts/_hardware.py`

## License

Benchmark problem definitions inherit the license terms of NVIDIA SOL-ExecBench (Apache-2.0).
