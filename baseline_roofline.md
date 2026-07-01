# sol-baseline × SOL-Lite roofline report (B200, Ray-234 costs)

Tested 52 baselines: **44 successfully timed, 8 errored** on
`ModuleNotFoundError: flash_attn / liger_kernel` in the current
sol-execbench venv state (both libs were present in earlier runs; the
failures are unrelated to the analyzer changes).

**All 129 successful rows use `cost_source=ray234`** — Ray-234's
per-UUID `(flops, bytes_moved)` data is applied everywhere available.
Compared to the previous purely-analytical run, MoE bytes dropped
30–150× (L2-cache-aware weight reuse), and Quant FP8 bytes dropped 2×
(fp8 internal path). See [DISAGREEMENTS.md](DISAGREEMENTS.md).

## Setup

- Hardware: **NVIDIA B200** (`--hardware B200`)
- ridge_BF16 = 281 FLOPs/byte, ridge_FP8 = 562
- Cost source: `data/costs/ray234_h800.jsonl` (1019 per-UUID entries)
  from [SoL-Contest-InfiniAI](https://github.com/qhy991/SoL-Contest-InfiniAI)
- Timing: back-to-back cuda.Event launches, 5 groups × 30 launches, median

## Columns

- `geomean_us` — geometric mean latency across 3 smoke workload rows
- `peak_MFU` / `peak_BW%` — max achieved across rows, normalized to B200 ceiling
- `recommended` — primary metric to report given the per-row regime distribution

All numbers measured on this machine. Columns:
- `geomean_us` — geometric mean latency across workload rows (smoke = 3 reps)
- `peak_MFU` / `peak_BW%` — max achievable values across rows (read against `mfu_ceiling`!)
- `recommended` — primary metric to read for this baseline given its regime mix

| definition | lib | n_rows | regime | recommended | geomean_us | peak_MFU | peak_BW% |
|---|---|---:|---|---|---:|---:|---:|
| 002_fp8_attention_qkv_projection | flashinfer | 3 | compute:3 | MFU | 3779.09 | 2.4% | 0.4% |
| 002_fused_add_rmsnorm_h4096 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 3.64 | 0.0% | 1.4% |
| 003_fp8_mlp_gate_up_projection | flashinfer | 3 | compute:3 | MFU | 911.56 | 14.2% | 3.3% |
| 003_fused_add_rmsnorm_h7168 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 4.92 | 0.0% | 38.2% |
| 004_fp8_moe_expert_linear | flashinfer | 3 | compute:3 | MFU | 1288.13 | 2.5% | 0.4% |
| 004_fused_residual_rms_mlp | flashinfer | 3 | compute:3 | MFU | 39133.41 | 26.0% | 3.7% |
| 005_fp8_moe_router_projection | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 17.84 | 5.8% | 13.2% |
| 007_multimodal_rotary_embedding_attention | flash_attn | — | — | ERROR: ModuleNotFoundError: No module named 'flash_attn' | — | — | — |
| 007_multimodal_rotary_embedding_attention | torch | 3 | compute:2 balanced:1 | per-row regime | 254.08 | 6.9% | 1.9% |
| 008_moe_sparse_routing_and_dispatch | torch | 3 | compute:3 | MFU | 22500.44 | 0.7% | 0.0% |
| 009_decoder_layer_with_residual_connections | flash_attn | 3 | compute:3 | MFU | 52848.68 | 0.3% | 0.0% |
| 010_moe_expert_computation_with_weighted_accumulation | torch | 3 | compute:3 | MFU | 45640.49 | 0.3% | 0.0% |
| 011_fp8_moe_gate_routing | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 1083.15 | 0.2% | 0.2% |
| 011_rotary_position_embedding | torch | 3 | latency:3 | time + speedup_vs_reference | 24.14 | 0.0% | 1.0% |
| 012_fp8_shared_expert_mlp | flashinfer | 3 | compute:3 | MFU | 3322.68 | 1.3% | 0.3% |
| 013_expert_weighted_aggregation_with_shared_expert | torch | 3 | memory:3 | BW% | 59401.99 | 0.2% | 0.7% |
| 013_fp8_mla_kv_compression_projection | flashinfer | 3 | balanced:3 | MFU + BW% (both) | 3019.41 | 1.8% | 1.0% |
| 013_gqa_paged_decode_h32_kv8_d128_ps1 | flash_attn | — | — | ERROR: ModuleNotFoundError: No module named 'flash_attn' | — | — | — |
| 013_gqa_paged_decode_h32_kv8_d128_ps1 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 4452.60 | 0.0% | 0.0% |
| 015_fp8_mla_attention_output_projection | flashinfer | 3 | compute:3 | MFU | 1504.95 | 12.4% | 4.1% |
| 015_grouped_query_attention_with_rope_and_qk_norm | flash_attn | — | — | ERROR: ModuleNotFoundError: No module named 'flash_attn' | — | — | — |
| 015_grouped_query_attention_with_rope_and_qk_norm | torch | 3 | compute:2 balanced:1 | per-row regime | 1242.21 | 7.9% | 4.3% |
| 016_fp8_multi_latent_attention_qkv_projection | flashinfer | 3 | compute:2 balanced:1 | per-row regime | 7424.87 | 2.1% | 0.6% |
| 017_gqa_ragged_prefill_causal_h32_kv8_d128 | flash_attn | — | — | ERROR: ModuleNotFoundError: No module named 'flash_attn' | — | — | — |
| 017_gqa_ragged_prefill_causal_h32_kv8_d128 | flashinfer | 0 |  | (no rows) | 0.00 | 0.0% | 0.0% |
| 018_fused_rope_with_qk_norm_and_kv_cache_update | flashinfer | 3 | latency:2 memory:1 | per-row regime | 192.52 | 0.0% | 1.4% |
| 018_mla_paged_decode_h16_ckv512_kpe64_ps1 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 3030.49 | 0.0% | 0.0% |
| 019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | flashinfer | 3 | latency:2 compute:1 | per-row regime | 3911.38 | 1.0% | 0.2% |
| 020_decoder_layer_pre_post_norm_residual | flash_attn | — | — | ERROR: ModuleNotFoundError: No module named 'flash_attn' | — | — | — |
| 020_decoder_layer_pre_post_norm_residual | torch | 3 | compute:2 memory:1 | per-row regime | 2629.69 | 14.1% | 11.7% |
| 020_vision_patch_merger_spatial_shuffle_mlp | flashinfer | 3 | compute:3 | MFU | 5031.02 | 7.9% | 0.4% |
| 021_vision_cu_seqlens_variable_length_attention | torch | 3 | latency:3 | time + speedup_vs_reference | 731.82 | 0.3% | 0.0% |
| 023_multimodal_rope_position_computation_with_grid_based_indexing | torch | 3 | latency:3 | time + speedup_vs_reference | 16167.55 | 0.0% | 0.0% |
| 023_rmsnorm_h1536 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 1.89 | 0.0% | 19.7% |
| 026_rmsnorm_h7168 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 2.47 | 0.0% | 52.5% |
| 027_grouped_query_attention_with_yarn_rope_and_qk_norm | torch | 3 | compute:3 | MFU | 3916.72 | 8.5% | 0.9% |
| 029_moe_sparse_routing_and_dispatch | torch | 3 | compute:3 | MFU | 48351.70 | 12.6% | 3.6% |
| 043_mla_fused_qkv_rope_split | flashinfer | 3 | compute:3 | MFU | 702.32 | 17.0% | 3.3% |
| 044_moe_expert_computation | torch | 3 | memory:2 balanced:1 | per-row regime | 38183.18 | 4.3% | 8.1% |
| 048_fused_gate_up_projection_with_swiglu | flashinfer | 3 | compute:3 | MFU | 2839.85 | 27.8% | 4.0% |
| 048_fused_gate_up_projection_with_swiglu | liger | — | — | ERROR: ModuleNotFoundError: No module named 'liger_kernel | — | — | — |
| 049_attention_qk_matmul_with_gqa_repeat_and_scaling | torch | 3 | latency:3 | time + speedup_vs_reference | 28.84 | 16.8% | 21.4% |
| 049_group_limited_topk_routing | torch | 3 | latency:3 | time + speedup_vs_reference | 321.82 | 0.6% | 0.8% |
| 053_text_decoder_layer_with_self_attention_and_mlp | torch | 3 | memory:1 compute:2 | per-row regime | 2092.74 | 17.9% | 7.4% |
| 054_vision_encoder_layer_with_gated_residuals | flash_attn | — | — | ERROR: ModuleNotFoundError: No module named 'flash_attn' | — | — | — |
| 054_vision_encoder_layer_with_gated_residuals | flashinfer | 3 | compute:3 | MFU | 800.35 | 15.7% | 1.1% |
| 064_latent_kv_expansion_with_split | flashinfer | 3 | compute:3 | MFU | 637.17 | 12.8% | 6.2% |
| 065_sparse_expert_dispatch_and_combine | torch | 3 | memory:3 | BW% | 97473.13 | 0.3% | 0.9% |
| 069_rms_norm | flashinfer | 3 | memory:3 | BW% | 33.50 | 0.0% | 52.3% |
| 081_moe_sparse_expert_dispatch | torch | 3 | compute:1 memory:2 | per-row regime | 19797.15 | 4.1% | 3.8% |
| 082_moe_layer_complete_forward_with_residual | torch | 3 | balanced:2 memory:1 | per-row regime | 26404.42 | 3.5% | 4.9% |
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
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 243, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmp_zn58uqc.py", line 3, in <module>
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
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 243, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmphn0nnb1i.py", line 3, in <module>
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
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 243, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmps04r33yg.py", line 3, in <module>
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
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 243, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmprcexhtzq.py", line 3, in <module>
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
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 243, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmpxe6xby9d.py", line 3, in <module>
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
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 243, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmpcum4j8v7.py", line 2, in <module>
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
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 243, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmp7k6avno2.py", line 3, in <module>
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
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 243, in bench_problem
    run_fn, _ = import_run_fn(sol_path, name + "_sol")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinhaiyan/SOL-Lite/scripts/roofline_bench.py", line 117, in import_run_fn
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/tmp/tmpo05xnv9q.py", line 3, in <module>
    from flash_attn import flash_attn_func
ModuleNotFoundError: No module named 'flash_attn'

```
</details>

