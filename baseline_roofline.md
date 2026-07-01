# sol-baseline × SOL-Lite roofline report (B200, Ray-234 costs + safetensors)

Tested 52 baselines: **44 successfully timed, 8 errored** on
`ModuleNotFoundError: flash_attn / liger_kernel` in the current
sol-execbench venv state (unrelated to analyzer changes).

Since the previous run:
- **017_gqa_ragged_prefill_causal flashinfer** now succeeds (was 0
  rows) — safetensors `qo_indptr`/`kv_indptr` are loaded from disk.
- **44 v3 submission directories** are exported under
  `data/submissions/sol-lite/` matching SoL-Contest-InfiniAI schema v3
  (manifest.json + solution/ + results/workloads.json).
- Analytical fallback improved: 25/60 problems now agree perfectly
  with Ray-234 (was 20/60). Elementwise/norm/RoPE FLOPs report 0
  per Ray-234 convention; Quant FP8 problems use fp8 byte accounting.

All 129 successful rows use `cost_source=ray234` (Ray-234 per-UUID
data covers everything in this benchmark set).

## Setup

- Hardware: **NVIDIA B200** (`--hardware B200`)
- ridge_BF16 = 281 FLOPs/byte, ridge_FP8 = 562
- Cost source: `data/costs/ray234_h800.jsonl` (1019 per-UUID entries)
  from [SoL-Contest-InfiniAI](https://github.com/qhy991/SoL-Contest-InfiniAI)
- Timing: back-to-back cuda.Event launches, 5 groups × 30 launches, median

All numbers measured on this machine. Columns:
- `geomean_us` — geometric mean latency across workload rows (smoke = 3 reps)
- `peak_MFU` / `peak_BW%` — max achievable values across rows (read against `mfu_ceiling`!)
- `recommended` — primary metric to read for this baseline given its regime mix

| definition | lib | n_rows | regime | recommended | geomean_us | peak_MFU | peak_BW% |
|---|---|---:|---|---|---:|---:|---:|
| 002_fp8_attention_qkv_projection | flashinfer | 3 | compute:3 | MFU | 1702.30 | 5.5% | 0.9% |
| 002_fused_add_rmsnorm_h4096 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 19.32 | 0.0% | 0.3% |
| 003_fp8_mlp_gate_up_projection | flashinfer | 3 | compute:3 | MFU | 420.35 | 30.6% | 7.4% |
| 003_fused_add_rmsnorm_h7168 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 19.60 | 0.0% | 14.6% |
| 004_fp8_moe_expert_linear | flashinfer | 3 | compute:3 | MFU | 610.04 | 5.2% | 0.9% |
| 004_fused_residual_rms_mlp | flashinfer | 3 | compute:3 | MFU | 16023.07 | 60.9% | 10.6% |
| 005_fp8_moe_router_projection | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 61.05 | 1.8% | 4.1% |
| 007_multimodal_rotary_embedding_attention | flash_attn | — | — | ERROR: ModuleNotFoundError: No module named 'flash_attn' | — | — | — |
| 007_multimodal_rotary_embedding_attention | torch | 3 | compute:2 balanced:1 | per-row regime | 288.41 | 5.1% | 1.5% |
| 008_moe_sparse_routing_and_dispatch | torch | 3 | compute:3 | MFU | 10758.73 | 1.5% | 0.1% |
| 009_decoder_layer_with_residual_connections | flash_attn | 3 | compute:3 | MFU | 21197.51 | 0.7% | 0.0% |
| 010_moe_expert_computation_with_weighted_accumulation | torch | 3 | compute:3 | MFU | 19239.98 | 0.7% | 0.0% |
| 011_fp8_moe_gate_routing | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 425.31 | 0.7% | 0.8% |
| 011_rotary_position_embedding | torch | 3 | latency:3 | time + speedup_vs_reference | 76.07 | 0.0% | 0.4% |
| 012_fp8_shared_expert_mlp | flashinfer | 3 | compute:3 | MFU | 971.20 | 4.4% | 0.9% |
| 013_expert_weighted_aggregation_with_shared_expert | torch | 3 | memory:3 | BW% | 36074.25 | 0.3% | 1.2% |
| 013_fp8_mla_kv_compression_projection | flashinfer | 3 | balanced:3 | MFU + BW% (both) | 950.77 | 5.9% | 3.3% |
| 013_gqa_paged_decode_h32_kv8_d128_ps1 | flash_attn | — | — | ERROR: ModuleNotFoundError: No module named 'flash_attn' | — | — | — |
| 013_gqa_paged_decode_h32_kv8_d128_ps1 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 147.05 | 0.2% | 0.0% |
| 015_fp8_mla_attention_output_projection | flashinfer | 3 | compute:3 | MFU | 539.28 | 36.1% | 8.2% |
| 015_grouped_query_attention_with_rope_and_qk_norm | flash_attn | — | — | ERROR: ModuleNotFoundError: No module named 'flash_attn' | — | — | — |
| 015_grouped_query_attention_with_rope_and_qk_norm | torch | 3 | compute:2 balanced:1 | per-row regime | 554.43 | 27.3% | 5.2% |
| 016_fp8_multi_latent_attention_qkv_projection | flashinfer | 3 | compute:2 balanced:1 | per-row regime | 2576.21 | 5.5% | 1.6% |
| 017_gqa_ragged_prefill_causal_h32_kv8_d128 | flash_attn | — | — | ERROR: ModuleNotFoundError: No module named 'flash_attn' | — | — | — |
| 017_gqa_ragged_prefill_causal_h32_kv8_d128 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 82.84 | 0.0% | 0.3% |
| 018_fused_rope_with_qk_norm_and_kv_cache_update | flashinfer | 3 | latency:2 memory:1 | per-row regime | 216.54 | 0.0% | 3.0% |
| 018_mla_paged_decode_h16_ckv512_kpe64_ps1 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 129.74 | 0.5% | 0.1% |
| 019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | flashinfer | 3 | latency:2 compute:1 | per-row regime | 626.40 | 7.0% | 1.4% |
| 020_decoder_layer_pre_post_norm_residual | flash_attn | — | — | ERROR: ModuleNotFoundError: No module named 'flash_attn' | — | — | — |
| 020_decoder_layer_pre_post_norm_residual | torch | 3 | compute:2 memory:1 | per-row regime | 974.56 | 43.6% | 22.0% |
| 020_vision_patch_merger_spatial_shuffle_mlp | flashinfer | 3 | compute:3 | MFU | 364.37 | 39.1% | 1.8% |
| 021_vision_cu_seqlens_variable_length_attention | torch | 3 | latency:3 | time + speedup_vs_reference | 268.69 | 1.1% | 0.1% |
| 023_multimodal_rope_position_computation_with_grid_based_indexing | torch | 3 | latency:3 | time + speedup_vs_reference | 3747.84 | 0.0% | 0.1% |
| 023_rmsnorm_h1536 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 10.00 | 0.0% | 4.3% |
| 026_rmsnorm_h7168 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 6.90 | 0.0% | 52.5% |
| 027_grouped_query_attention_with_yarn_rope_and_qk_norm | torch | 3 | compute:3 | MFU | 1325.29 | 26.6% | 3.0% |
| 029_moe_sparse_routing_and_dispatch | torch | 3 | compute:3 | MFU | 18076.62 | 33.5% | 12.0% |
| 043_mla_fused_qkv_rope_split | flashinfer | 3 | compute:3 | MFU | 374.26 | 31.9% | 6.0% |
| 044_moe_expert_computation | torch | 3 | memory:2 balanced:1 | per-row regime | 21616.27 | 8.8% | 13.3% |
| 048_fused_gate_up_projection_with_swiglu | flashinfer | 3 | compute:3 | MFU | 1636.24 | 47.7% | 6.9% |
| 048_fused_gate_up_projection_with_swiglu | liger | — | — | ERROR: ModuleNotFoundError: No module named 'liger_kernel | — | — | — |
| 049_attention_qk_matmul_with_gqa_repeat_and_scaling | torch | 3 | latency:3 | time + speedup_vs_reference | 47.25 | 10.2% | 13.0% |
| 049_group_limited_topk_routing | torch | 3 | latency:3 | time + speedup_vs_reference | 197.15 | 1.1% | 1.4% |
| 053_text_decoder_layer_with_self_attention_and_mlp | torch | 3 | memory:1 compute:2 | per-row regime | 1347.34 | 31.9% | 8.2% |
| 054_vision_encoder_layer_with_gated_residuals | flash_attn | — | — | ERROR: ModuleNotFoundError: No module named 'flash_attn' | — | — | — |
| 054_vision_encoder_layer_with_gated_residuals | flashinfer | 3 | compute:3 | MFU | 381.96 | 32.4% | 2.6% |
| 064_latent_kv_expansion_with_split | flashinfer | 3 | compute:3 | MFU | 409.47 | 12.8% | 6.1% |
| 065_sparse_expert_dispatch_and_combine | torch | 3 | memory:3 | BW% | 27340.17 | 0.8% | 3.5% |
| 069_rms_norm | flashinfer | 3 | memory:3 | BW% | 30.42 | 0.0% | 52.4% |
| 081_moe_sparse_expert_dispatch | torch | 3 | compute:1 memory:2 | per-row regime | 4130.14 | 17.1% | 19.9% |
| 082_moe_layer_complete_forward_with_residual | torch | 3 | balanced:2 memory:1 | per-row regime | 5250.04 | 15.7% | 29.7% |
| 092_gqa_attention_with_qk_norm | flash_attn | — | — | ERROR: ModuleNotFoundError: No module named 'flash_attn' | — | — | — |

## Errors

### 007_multimodal_rotary_embedding_attention (flash_attn)
```
ModuleNotFoundError: No module named 'flash_attn'
```
<details><summary>traceback</summary>

```
Traceback (most recent call last):
  File "/home/qinhaiyan/SOL-Lite/scripts/bench_baselines.py", line 106, in benchmark_one
    rows, summary = rb.bench_problem(
                    ^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 267, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmpbv6osjz6.py", line 3, in <module>
    from flash_attn import flash_attn_func
ModuleNotFoundError: No module named 'flash_attn'

```
</details>

### 013_gqa_paged_decode_h32_kv8_d128_ps1 (flash_attn)
```
ModuleNotFoundError: No module named 'flash_attn'
```
<details><summary>traceback</summary>

```
Traceback (most recent call last):
  File "/home/qinhaiyan/SOL-Lite/scripts/bench_baselines.py", line 106, in benchmark_one
    rows, summary = rb.bench_problem(
                    ^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 267, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmpu7ii3ahx.py", line 3, in <module>
    from flash_attn import flash_attn_varlen_func
ModuleNotFoundError: No module named 'flash_attn'

```
</details>

### 015_grouped_query_attention_with_rope_and_qk_norm (flash_attn)
```
ModuleNotFoundError: No module named 'flash_attn'
```
<details><summary>traceback</summary>

```
Traceback (most recent call last):
  File "/home/qinhaiyan/SOL-Lite/scripts/bench_baselines.py", line 106, in benchmark_one
    rows, summary = rb.bench_problem(
                    ^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 267, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmpwp4iflqs.py", line 3, in <module>
    from flash_attn import flash_attn_func
ModuleNotFoundError: No module named 'flash_attn'

```
</details>

### 017_gqa_ragged_prefill_causal_h32_kv8_d128 (flash_attn)
```
ModuleNotFoundError: No module named 'flash_attn'
```
<details><summary>traceback</summary>

```
Traceback (most recent call last):
  File "/home/qinhaiyan/SOL-Lite/scripts/bench_baselines.py", line 106, in benchmark_one
    rows, summary = rb.bench_problem(
                    ^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 267, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmp_9mfmclk.py", line 3, in <module>
    from flash_attn import flash_attn_varlen_func
ModuleNotFoundError: No module named 'flash_attn'

```
</details>

### 020_decoder_layer_pre_post_norm_residual (flash_attn)
```
ModuleNotFoundError: No module named 'flash_attn'
```
<details><summary>traceback</summary>

```
Traceback (most recent call last):
  File "/home/qinhaiyan/SOL-Lite/scripts/bench_baselines.py", line 106, in benchmark_one
    rows, summary = rb.bench_problem(
                    ^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 267, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmpjm308r_7.py", line 3, in <module>
    from flash_attn import flash_attn_func
ModuleNotFoundError: No module named 'flash_attn'

```
</details>

### 048_fused_gate_up_projection_with_swiglu (liger)
```
ModuleNotFoundError: No module named 'liger_kernel'
```
<details><summary>traceback</summary>

```
Traceback (most recent call last):
  File "/home/qinhaiyan/SOL-Lite/scripts/bench_baselines.py", line 106, in benchmark_one
    rows, summary = rb.bench_problem(
                    ^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 267, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmpc6nl3807.py", line 2, in <module>
    from liger_kernel.ops.geglu import LigerGELUMulFunction
ModuleNotFoundError: No module named 'liger_kernel'

```
</details>

### 054_vision_encoder_layer_with_gated_residuals (flash_attn)
```
ModuleNotFoundError: No module named 'flash_attn'
```
<details><summary>traceback</summary>

```
Traceback (most recent call last):
  File "/home/qinhaiyan/SOL-Lite/scripts/bench_baselines.py", line 106, in benchmark_one
    rows, summary = rb.bench_problem(
                    ^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 267, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmpx7zbmcp9.py", line 3, in <module>
    from flash_attn import flash_attn_func
ModuleNotFoundError: No module named 'flash_attn'

```
</details>

### 092_gqa_attention_with_qk_norm (flash_attn)
```
ModuleNotFoundError: No module named 'flash_attn'
```
<details><summary>traceback</summary>

```
Traceback (most recent call last):
  File "/home/qinhaiyan/SOL-Lite/scripts/bench_baselines.py", line 106, in benchmark_one
    rows, summary = rb.bench_problem(
                    ^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 267, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmpbtr0sffk.py", line 3, in <module>
    from flash_attn import flash_attn_func
ModuleNotFoundError: No module named 'flash_attn'

```
</details>

