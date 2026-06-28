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

## License

Benchmark problem definitions inherit the license terms of NVIDIA SOL-ExecBench (Apache-2.0).
