# sol-baseline × SOL-Lite roofline report (B200)

Tested 52 baselines, 0 errored. **1 baseline yielded 0 rows**
(017_gqa_ragged_prefill flashinfer — input generator does not yet
materialize the safetensors-loaded `cu_seqlens` / `qo_indptr` arrays).

## Setup

- Hardware: **NVIDIA B200** (HBM3e 8.00 TB/s, BF16 dense 2250 TFLOPS,
  FP8 4500 TFLOPS).
- Analyzer ceilings: `--hardware B200` (matched to measurement GPU).
- ridge_BF16 = 281 FLOPs/byte, ridge_FP8 = 562.
- For each baseline, 3 representative workloads (small / mid / large)
  are timed via back-to-back launches between cuda.Events, median of
  5 groups of 30 launches each.

To re-run on a different GPU: `--hardware {H800|H100|H200|A100|...}`.
See [README.md § Hardware presets](README.md#hardware-presets) for full
preset table.

## Caveat: 019_mla_paged_prefill outlier

`019_mla_paged_prefill_causal flashinfer MFU=542%` is still anomalous.
The analyzer's flops/bytes estimate for paged-MLA prefill with causal
mask + variable sequence lengths is likely wrong — paged decode/prefill
problems use `num_kv_indices` as their bytes proxy but the actual access
pattern can be very different from this estimate. Treat that row with
suspicion.

## Columns

- `geomean_us` — geometric mean latency across the 3 smoke workload rows
- `peak_MFU` / `peak_BW%` — max achieved across rows, normalized against
   the B200 ceiling; should normally be ≤ 100%
- `recommended` — primary metric to read for this baseline given its
   per-row regime distribution (always read in conjunction with
   `mfu_ceiling` from the per-row CSV)

| definition | lib | n_rows | regime | recommended | geomean_us | peak_MFU | peak_BW% |
|---|---|---:|---|---|---:|---:|---:|
| 002_fp8_attention_qkv_projection | flashinfer | 3 | compute:3 | MFU | 1929.27 | 10.9% | 1.8% |
| 002_fused_add_rmsnorm_h4096 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 21.11 | 0.0% | 0.2% |
| 003_fp8_mlp_gate_up_projection | flashinfer | 3 | compute:3 | MFU | 418.19 | 30.5% | 8.6% |
| 003_fused_add_rmsnorm_h7168 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 20.64 | 0.0% | 14.0% |
| 004_fp8_moe_expert_linear | flashinfer | 3 | compute:3 | MFU | 747.44 | 8.8% | 1.5% |
| 004_fused_residual_rms_mlp | flashinfer | 3 | compute:3 | MFU | 17551.14 | 56.9% | 12.7% |
| 005_fp8_moe_router_projection | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 63.01 | 1.8% | 4.2% |
| 007_multimodal_rotary_embedding_attention | flash_attn | 3 | balanced:3 | MFU + BW% (both) | 347.01 | 4.3% | 3.6% |
| 007_multimodal_rotary_embedding_attention | torch | 3 | balanced:3 | MFU + BW% (both) | 257.62 | 5.8% | 4.9% |
| 008_moe_sparse_routing_and_dispatch | torch | 3 | memory:2 balanced:1 | per-row regime | 10555.71 | 1.6% | 2.2% |
| 009_decoder_layer_with_residual_connections | flash_attn | 3 | balanced:3 | MFU + BW% (both) | 23172.95 | 0.6% | 0.9% |
| 010_moe_expert_computation_with_weighted_accumulation | torch | 3 | balanced:1 memory:2 | per-row regime | 18730.34 | 0.6% | 1.1% |
| 011_fp8_moe_gate_routing | flashinfer | 3 | balanced:2 latency:1 | per-row regime | 569.01 | 1.2% | 1.5% |
| 011_rotary_position_embedding | torch | 3 | latency:3 | time + speedup_vs_reference | 90.93 | 0.0% | 0.3% |
| 012_fp8_shared_expert_mlp | flashinfer | 3 | compute:3 | MFU | 1244.10 | 6.3% | 1.7% |
| 013_expert_weighted_aggregation_with_shared_expert | torch | 3 | memory:3 | BW% | 34600.57 | 0.3% | 1.4% |
| 013_fp8_mla_kv_compression_projection | flashinfer | 3 | balanced:3 | MFU + BW% (both) | 932.22 | 11.9% | 6.7% |
| 013_gqa_paged_decode_h32_kv8_d128_ps1 | flash_attn | 3 | latency:1 memory:2 | per-row regime | 99.72 | 0.3% | 22.7% |
| 013_gqa_paged_decode_h32_kv8_d128_ps1 | flashinfer | 3 | latency:1 memory:2 | per-row regime | 112.72 | 0.3% | 24.6% |
| 015_fp8_mla_attention_output_projection | flashinfer | 3 | compute:3 | MFU | 585.25 | 37.3% | 6.8% |
| 015_grouped_query_attention_with_rope_and_qk_norm | flash_attn | 3 | compute:2 balanced:1 | per-row regime | 571.89 | 25.7% | 4.2% |
| 015_grouped_query_attention_with_rope_and_qk_norm | torch | 3 | compute:2 balanced:1 | per-row regime | 667.95 | 16.8% | 5.2% |
| 016_fp8_multi_latent_attention_qkv_projection | flashinfer | 3 | compute:2 balanced:1 | per-row regime | 2924.89 | 9.9% | 3.3% |
| 017_gqa_ragged_prefill_causal_h32_kv8_d128 | flash_attn | 3 | latency:3 | time + speedup_vs_reference | 454.85 | 0.0% | 0.1% |
| 017_gqa_ragged_prefill_causal_h32_kv8_d128 | flashinfer | 0 |  | (no rows) | 0.00 | 0.0% | 0.0% |
| 018_fused_rope_with_qk_norm_and_kv_cache_update | flashinfer | 3 | latency:2 memory:1 | per-row regime | 208.03 | 0.0% | 3.5% |
| 018_mla_paged_decode_h16_ckv512_kpe64_ps1 | flashinfer | 3 | latency:2 memory:1 | per-row regime | 106.68 | 0.9% | 8.3% |
| 019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | flashinfer | 3 | latency:1 balanced:1 compute:1 | per-row regime | 85.05 | 542.2% | 57.9% |
| 020_decoder_layer_pre_post_norm_residual | flash_attn | 3 | compute:2 memory:1 | per-row regime | 930.55 | 48.1% | 22.8% |
| 020_decoder_layer_pre_post_norm_residual | torch | 3 | compute:2 memory:1 | per-row regime | 906.18 | 46.1% | 27.2% |
| 020_vision_patch_merger_spatial_shuffle_mlp | flashinfer | 3 | compute:2 memory:1 | per-row regime | 501.45 | 32.1% | 7.3% |
| 021_vision_cu_seqlens_variable_length_attention | torch | 3 | latency:3 | time + speedup_vs_reference | 280.59 | 1.1% | 1.8% |
| 023_multimodal_rope_position_computation_with_grid_based_indexing | torch | 3 | latency:3 | time + speedup_vs_reference | 3381.91 | 0.0% | 0.1% |
| 023_rmsnorm_h1536 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 10.68 | 0.0% | 3.9% |
| 026_rmsnorm_h7168 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 10.75 | 0.1% | 16.7% |
| 027_grouped_query_attention_with_yarn_rope_and_qk_norm | torch | 3 | compute:3 | MFU | 1165.04 | 29.8% | 10.5% |
| 029_moe_sparse_routing_and_dispatch | torch | 3 | balanced:1 compute:2 | per-row regime | 16796.15 | 33.4% | 17.9% |
| 043_mla_fused_qkv_rope_split | flashinfer | 3 | compute:3 | MFU | 308.40 | 36.8% | 12.1% |
| 044_moe_expert_computation | torch | 3 | memory:2 balanced:1 | per-row regime | 21817.22 | 8.6% | 14.2% |
| 048_fused_gate_up_projection_with_swiglu | flashinfer | 3 | compute:3 | MFU | 1555.50 | 58.3% | 9.2% |
| 048_fused_gate_up_projection_with_swiglu | liger | 3 | compute:3 | MFU | 1683.19 | 55.2% | 8.4% |
| 049_attention_qk_matmul_with_gqa_repeat_and_scaling | torch | 3 | latency:3 | time + speedup_vs_reference | 48.11 | 11.1% | 14.0% |
| 049_group_limited_topk_routing | torch | 3 | latency:3 | time + speedup_vs_reference | 212.35 | 1.0% | 1.3% |
| 053_text_decoder_layer_with_self_attention_and_mlp | torch | 3 | memory:1 compute:2 | per-row regime | 1145.86 | 39.7% | 14.9% |
| 054_vision_encoder_layer_with_gated_residuals | flash_attn | 3 | balanced:3 | MFU + BW% (both) | 446.88 | 34.8% | 25.0% |
| 054_vision_encoder_layer_with_gated_residuals | flashinfer | 3 | balanced:3 | MFU + BW% (both) | 377.57 | 35.0% | 25.1% |
| 064_latent_kv_expansion_with_split | flashinfer | 3 | balanced:3 | MFU + BW% (both) | 418.18 | 12.9% | 13.4% |
| 065_sparse_expert_dispatch_and_combine | torch | 3 | memory:3 | BW% | 24496.60 | 1.0% | 3.5% |
| 069_rms_norm | flashinfer | 3 | memory:3 | BW% | 32.62 | 0.1% | 49.6% |
| 081_moe_sparse_expert_dispatch | torch | 3 | balanced:1 memory:2 | per-row regime | 4186.01 | 18.0% | 22.3% |
| 082_moe_layer_complete_forward_with_residual | torch | 3 | balanced:2 memory:1 | per-row regime | 5135.65 | 17.8% | 32.4% |
| 092_gqa_attention_with_qk_norm | flash_attn | 3 | balanced:1 compute:2 | per-row regime | 518.15 | 25.0% | 9.5% |
