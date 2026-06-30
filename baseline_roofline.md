# sol-baseline × SOL-Lite roofline report

Tested 51 baselines, 0 errored. **1 baseline yielded 0 rows**
(017_gqa_ragged_prefill flashinfer — input generator does not yet
materialize the safetensors-loaded `cu_seqlens` / `qo_indptr` arrays).

## Hardware caveat

Timing was collected on **NVIDIA B200** (HBM3e ~8 TB/s, BF16 peak
~2250 TFLOPS), but the analyzer's roofline ceilings are for **H800**
(HBM3 3.35 TB/s, BF16 peak 989 TFLOPS). When MFU or BW% exceeds 100%
in the table below, the kernel is beating the *H800* ceiling because
the underlying GPU is faster — divide by ~2-2.5× to get the
B200-relative ratio. The regime classification and `mfu_ceiling` are
still valid because they are properties of the algorithmic AI vs ridge,
not of wall time.

## Outlier

`019_mla_paged_prefill_causal flashinfer MFU=1234%` is too extreme even
for the B200 mismatch — the analyzer's flops/bytes estimate for
paged-MLA prefill with causal mask + variable sequence lengths is
likely wrong. Treat that one row with suspicion until the analyzer is
refined.

## Columns

- `geomean_us` — geometric mean latency across the 3 smoke workload rows
- `peak_MFU` / `peak_BW%` — max achievable values across rows
   (always read together with `mfu_ceiling` from the per-row CSV!)
- `recommended` — primary metric to read for this baseline given its
   per-row regime distribution

| definition | lib | n_rows | regime | recommended | geomean_us | peak_MFU | peak_BW% |
|---|---|---:|---|---|---:|---:|---:|
| 002_fp8_attention_qkv_projection | flashinfer | 3 | compute:3 | MFU | 1702.59 | 24.9% | 4.2% |
| 002_fused_add_rmsnorm_h4096 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 21.26 | 0.0% | 0.6% |
| 003_fp8_mlp_gate_up_projection | flashinfer | 3 | compute:3 | MFU | 413.82 | 70.5% | 20.8% |
| 003_fused_add_rmsnorm_h7168 | flashinfer | 3 | latency:2 memory:1 | per-row regime | 19.39 | 0.1% | 35.7% |
| 004_fp8_moe_expert_linear | flashinfer | 3 | compute:3 | MFU | 605.07 | 23.6% | 4.2% |
| 004_fused_residual_rms_mlp | flashinfer | 3 | compute:3 | MFU | 14762.66 | 138.0% | 30.1% |
| 005_fp8_moe_router_projection | flashinfer | 3 | latency:2 memory:1 | per-row regime | 59.46 | 4.2% | 10.7% |
| 007_multimodal_rotary_embedding_attention | flash_attn | 3 | balanced:3 | MFU + BW% (both) | 356.81 | 10.1% | 8.8% |
| 007_multimodal_rotary_embedding_attention | torch | 3 | balanced:3 | MFU + BW% (both) | 274.52 | 12.2% | 11.5% |
| 008_moe_sparse_routing_and_dispatch | torch | 3 | memory:2 balanced:1 | per-row regime | 9845.50 | 3.7% | 5.6% |
| 009_decoder_layer_with_residual_connections | flash_attn | 3 | balanced:3 | MFU + BW% (both) | 19824.93 | 1.7% | 2.4% |
| 010_moe_expert_computation_with_weighted_accumulation | torch | 3 | balanced:1 memory:2 | per-row regime | 17369.81 | 1.6% | 2.7% |
| 011_fp8_moe_gate_routing | flashinfer | 3 | balanced:3 | MFU + BW% (both) | 427.54 | 3.2% | 4.0% |
| 011_rotary_position_embedding | torch | 3 | latency:3 | time + speedup_vs_reference | 72.81 | 0.0% | 0.9% |
| 012_fp8_shared_expert_mlp | flashinfer | 3 | compute:3 | MFU | 969.32 | 20.0% | 4.3% |
| 013_expert_weighted_aggregation_with_shared_expert | torch | 3 | memory:3 | BW% | 35380.33 | 0.8% | 3.3% |
| 013_fp8_mla_kv_compression_projection | flashinfer | 3 | balanced:3 | MFU + BW% (both) | 969.75 | 27.1% | 15.9% |
| 013_gqa_paged_decode_h32_kv8_d128_ps1 | flash_attn | 3 | latency:1 memory:2 | per-row regime | 101.47 | 0.7% | 53.9% |
| 013_gqa_paged_decode_h32_kv8_d128_ps1 | flashinfer | 3 | latency:1 memory:2 | per-row regime | 112.52 | 0.8% | 59.3% |
| 015_fp8_mla_attention_output_projection | flashinfer | 3 | compute:3 | MFU | 547.33 | 82.8% | 20.5% |
| 015_grouped_query_attention_with_rope_and_qk_norm | flash_attn | 3 | compute:2 balanced:1 | per-row regime | 614.36 | 58.2% | 9.9% |
| 015_grouped_query_attention_with_rope_and_qk_norm | torch | 3 | compute:2 balanced:1 | per-row regime | 583.71 | 57.1% | 10.9% |
| 016_fp8_multi_latent_attention_qkv_projection | flashinfer | 3 | compute:2 balanced:1 | per-row regime | 2433.81 | 27.8% | 8.6% |
| 017_gqa_ragged_prefill_causal_h32_kv8_d128 | flash_attn | 3 | latency:3 | time + speedup_vs_reference | 155.49 | 0.0% | 0.4% |
| 017_gqa_ragged_prefill_causal_h32_kv8_d128 | flashinfer | 0 |  | (no rows) | 0.00 | 0.0% | 0.0% |
| 018_fused_rope_with_qk_norm_and_kv_cache_update | flashinfer | 3 | latency:1 memory:2 | per-row regime | 212.45 | 0.1% | 8.1% |
| 018_mla_paged_decode_h16_ckv512_kpe64_ps1 | flashinfer | 3 | latency:2 memory:1 | per-row regime | 105.15 | 2.0% | 19.9% |
| 019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | flashinfer | 3 | latency:1 balanced:1 compute:1 | per-row regime | 84.58 | 1233.8% | 138.4% |
| 020_decoder_layer_pre_post_norm_residual | flash_attn | 3 | compute:2 memory:1 | per-row regime | 979.19 | 108.6% | 47.9% |
| 020_decoder_layer_pre_post_norm_residual | torch | 3 | compute:2 memory:1 | per-row regime | 967.31 | 102.4% | 55.5% |
| 020_vision_patch_merger_spatial_shuffle_mlp | flashinfer | 3 | compute:2 memory:1 | per-row regime | 333.53 | 90.7% | 26.7% |
| 021_vision_cu_seqlens_variable_length_attention | torch | 3 | balanced:2 memory:1 | per-row regime | 271.64 | 2.5% | 4.2% |
| 023_multimodal_rope_position_computation_with_grid_based_indexing | torch | 3 | latency:2 memory:1 | per-row regime | 3622.55 | 0.0% | 0.2% |
| 023_rmsnorm_h1536 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 13.80 | 0.0% | 6.3% |
| 026_rmsnorm_h7168 | flashinfer | 3 | latency:3 | time + speedup_vs_reference | 10.81 | 0.2% | 39.9% |
| 027_grouped_query_attention_with_yarn_rope_and_qk_norm | torch | 3 | compute:3 | MFU | 1168.96 | 67.7% | 25.0% |
| 029_moe_sparse_routing_and_dispatch | torch | 3 | balanced:1 compute:2 | per-row regime | 15403.96 | 76.7% | 43.0% |
| 043_mla_fused_qkv_rope_split | flashinfer | 3 | compute:3 | MFU | 310.96 | 81.5% | 29.5% |
| 044_moe_expert_computation | torch | 3 | memory:2 balanced:1 | per-row regime | 22582.48 | 18.1% | 32.5% |
| 048_fused_gate_up_projection_with_swiglu | flashinfer | 3 | compute:3 | MFU | 1297.66 | 136.1% | 21.5% |
| 048_fused_gate_up_projection_with_swiglu | liger | 3 | compute:3 | MFU | 1374.33 | 129.9% | 19.6% |
| 049_attention_qk_matmul_with_gqa_repeat_and_scaling | torch | 3 | latency:2 balanced:1 | per-row regime | 51.98 | 22.6% | 30.2% |
| 053_text_decoder_layer_with_self_attention_and_mlp | torch | 3 | memory:1 compute:2 | per-row regime | 1366.44 | 89.5% | 35.2% |
| 054_vision_encoder_layer_with_gated_residuals | flash_attn | 3 | balanced:3 | MFU + BW% (both) | 456.20 | 76.9% | 58.0% |
| 054_vision_encoder_layer_with_gated_residuals | flashinfer | 3 | balanced:3 | MFU + BW% (both) | 361.34 | 77.4% | 58.3% |
| 064_latent_kv_expansion_with_split | flashinfer | 3 | balanced:3 | MFU + BW% (both) | 373.57 | 32.6% | 31.1% |
| 065_sparse_expert_dispatch_and_combine | torch | 3 | memory:3 | BW% | 24469.10 | 2.3% | 8.6% |
| 069_rms_norm | flashinfer | 3 | memory:3 | BW% | 40.11 | 0.3% | 118.3% |
| 081_moe_sparse_expert_dispatch | torch | 3 | balanced:1 memory:2 | per-row regime | 4840.91 | 38.8% | 40.6% |
| 082_moe_layer_complete_forward_with_residual | torch | 3 | balanced:2 memory:1 | per-row regime | 5470.67 | 40.0% | 72.3% |
| 092_gqa_attention_with_qk_norm | flash_attn | 3 | balanced:1 compute:2 | per-row regime | 571.20 | 56.7% | 17.1% |
