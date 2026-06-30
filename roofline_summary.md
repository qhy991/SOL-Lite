# SOL ExecBench — 60-problem roofline summary

Target hardware: **NVIDIA H800 SXM5** (BF16 peak 989 TFLOPS, FP8 peak 1979 TFLOPS, HBM3 3.35 TB/s).

Each row's regime is determined by per-workload arithmetic intensity vs. the ridge
(295 FLOPs/byte for BF16, 591 for FP8) and an absolute `t_sol < 5 μs` latency floor.

Columns:
  - `C/B/M/L`: rows classified as compute / balanced / memory / latency
  - `AI`: arithmetic-intensity range across this problem's workloads
  - `MFU_max`: max achievable MFU across workloads (physical ceiling, not 1.0!)
  - `metric`: recommended primary metric to report

## L1 (20 problems)

| # | problem | dtype | rows | C/B/M/L | AI range | MFU_max | metric |
|---|---|---|---:|---|---|---:|---|
| `003` | lm_head_projection_with_logit_slicing | bf16 | 16 | 11/4/1/0 | 120–1613 | 1.00 | **per-row regime** |
| `011` | rotary_position_embedding | bf16 | 16 | 0/0/1/15 | 2–2 | 0.01 | **time + speedup_vs_reference** |
| `015` | grouped_query_attention_with_rope_and_qk_norm | bf16 | 16 | 11/4/1/0 | 127–3831 | 1.00 | **per-row regime** |
| `018` | fused_rope_with_qk_norm_and_kv_cache_update | bf16 | 13 | 0/0/8/5 | 2–2 | 0.01 | **per-row regime** |
| `020` | vision_patch_merger_spatial_shuffle_mlp | bf16 | 15 | 9/4/2/0 | 125–4471 | 1.00 | **per-row regime** |
| `021` | vision_cu_seqlens_variable_length_attention | bf16 | 16 | 0/12/4/0 | 58–459 | 1.00 | **per-row regime** |
| `023` | multimodal_rope_position_computation_with_grid_based_indexing | bf16 | 16 | 0/0/2/14 | 0–0 | 0.00 | **time + speedup_vs_reference** |
| `043` | mla_fused_qkv_rope_split | bf16 | 16 | 13/2/1/0 | 121–1364 | 1.00 | **MFU** |
| `044` | moe_expert_computation | bf16 | 16 | 0/2/14/0 | 62–171 | 0.58 | **BW%** |
| `046` | attention_softmax_with_softcapping_and_dropout | bf16 | 16 | 0/0/10/6 | 1–1 | 0.00 | **per-row regime** |
| `048` | fused_gate_up_projection_with_swiglu | bf16 | 16 | 11/3/2/0 | 125–3278 | 1.00 | **per-row regime** |
| `049` | attention_qk_matmul_with_gqa_repeat_and_scaling | bf16 | 16 | 0/7/1/8 | 73–246 | 0.83 | **per-row regime** |
| `059` | moe_group_score_aggregation_and_masking | bf16 | 16 | 0/0/3/13 | 1–1 | 0.00 | **time + speedup_vs_reference** |
| `063` | attention_output_reshape_and_projection | bf16 | 16 | 12/3/1/0 | 128–2249 | 1.00 | **per-row regime** |
| `064` | latent_kv_expansion_with_split | bf16 | 16 | 0/14/2/0 | 102–475 | 1.00 | **MFU + BW% (both)** |
| `067` | flash_attention_gqa_ultralong | bf16 | 18 | 11/5/2/0 | 126–10271 | 1.00 | **per-row regime** |
| `069` | rms_norm | bf16 | 16 | 0/0/14/2 | 1–1 | 0.00 | **BW%** |
| `071` | kv_cache_update_with_rope | bf16 | 16 | 0/0/1/15 | 1–1 | 0.00 | **time + speedup_vs_reference** |
| `076` | batched_expert_forward | bf16 | 14 | 5/6/3/0 | 63–1386 | 1.00 | **per-row regime** |
| `092` | gqa_attention_with_qk_norm | bf16 | 16 | 12/3/1/0 | 126–3109 | 1.00 | **per-row regime** |

## L2 (20 problems)

| # | problem | dtype | rows | C/B/M/L | AI range | MFU_max | metric |
|---|---|---|---:|---|---|---:|---|
| `002` | decoder_layer_full_block | bf16 | 18 | 11/6/1/0 | 118–1425 | 1.00 | **per-row regime** |
| `004` | fused_residual_rms_mlp | bf16 | 16 | 15/1/0/0 | 246–3550 | 1.00 | **MFU** |
| `006` | multimodal_rope_position_calculation | bf16 | 16 | 0/0/0/16 | 0–0 | 0.00 | **time + speedup_vs_reference** |
| `007` | multimodal_rotary_embedding_attention | bf16 | 16 | 11/4/1/0 | 113–1261 | 1.00 | **per-row regime** |
| `008` | moe_sparse_routing_and_dispatch | bf16 | 16 | 0/6/10/0 | 57–261 | 0.89 | **per-row regime** |
| `009` | decoder_layer_with_residual_connections | bf16 | 16 | 2/4/10/0 | 85–836 | 1.00 | **per-row regime** |
| `010` | moe_expert_computation_with_weighted_accumulation | bf16 | 16 | 0/7/9/0 | 57–244 | 0.83 | **per-row regime** |
| `012` | moe_expert_batched_execution_with_capacity_factor | bf16 | 16 | 0/4/12/0 | 62–331 | 1.00 | **per-row regime** |
| `013` | expert_weighted_aggregation_with_shared_expert | bf16 | 16 | 0/1/15/0 | 62–126 | 0.43 | **BW%** |
| `019` | decoder_layer_fused_attention_mlp | bf16 | 16 | 12/2/2/0 | 116–1279 | 1.00 | **per-row regime** |
| `020` | decoder_layer_pre_post_norm_residual | bf16 | 16 | 14/1/1/0 | 119–1413 | 1.00 | **MFU** |
| `027` | grouped_query_attention_with_yarn_rope_and_qk_norm | bf16 | 16 | 13/2/1/0 | 117–1093 | 1.00 | **MFU** |
| `029` | moe_sparse_routing_and_dispatch | bf16 | 9 | 3/6/0/0 | 218–842 | 1.00 | **per-row regime** |
| `048` | moe_expert_inference_batched_dispatch | bf16 | 16 | 0/0/16/0 | 60–111 | 0.38 | **BW%** |
| `049` | group_limited_topk_routing | bf16 | 16 | 0/16/0/0 | 216–238 | 0.80 | **MFU + BW% (both)** |
| `053` | text_decoder_layer_with_self_attention_and_mlp | bf16 | 16 | 11/4/1/0 | 118–1336 | 1.00 | **per-row regime** |
| `054` | vision_encoder_layer_with_gated_residuals | bf16 | 16 | 11/3/0/2 | 98–576 | 1.00 | **per-row regime** |
| `065` | sparse_expert_dispatch_and_combine | bf16 | 16 | 0/1/15/0 | 61–371 | 1.00 | **BW%** |
| `081` | moe_sparse_expert_dispatch | bf16 | 16 | 0/5/11/0 | 67–421 | 1.00 | **per-row regime** |
| `082` | moe_layer_complete_forward_with_residual | bf16 | 16 | 0/4/12/0 | 67–292 | 0.99 | **per-row regime** |

## FlashInfer-Bench (10 problems)

| # | problem | dtype | rows | C/B/M/L | AI range | MFU_max | metric |
|---|---|---|---:|---|---|---:|---|
| `002` | fused_add_rmsnorm_h4096 | bf16 | 14 | 0/0/5/9 | 1–1 | 0.00 | **per-row regime** |
| `003` | fused_add_rmsnorm_h7168 | bf16 | 8 | 0/0/3/5 | 1–1 | 0.00 | **per-row regime** |
| `005` | gemm_n256_k7168 | bf16 | 17 | 0/3/0/14 | 1–243 | 0.82 | **time + speedup_vs_reference** |
| `013` | gqa_paged_decode_h32_kv8_d128_ps1 | bf16 | 48 | 0/0/19/29 | 1–4 | 0.01 | **per-row regime** |
| `017` | gqa_ragged_prefill_causal_h32_kv8_d128 | bf16 | 21 | 0/2/2/17 | 0–503 | 1.00 | **time + speedup_vs_reference** |
| `018` | mla_paged_decode_h16_ckv512_kpe64_ps1 | bf16 | 47 | 0/0/17/30 | 6–30 | 0.10 | **per-row regime** |
| `019` | mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 | bf16 | 38 | 2/6/6/24 | 2–7931 | 1.00 | **per-row regime** |
| `020` | moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 | fp8 | 19 | 0/2/16/1 | 2–482 | 0.82 | **BW%** |
| `023` | rmsnorm_h1536 | bf16 | 8 | 0/0/2/6 | 1–1 | 0.00 | **per-row regime** |
| `026` | rmsnorm_h7168 | bf16 | 8 | 0/0/2/6 | 1–1 | 0.00 | **per-row regime** |

## Quant (10 problems)

| # | problem | dtype | rows | C/B/M/L | AI range | MFU_max | metric |
|---|---|---|---:|---|---|---:|---|
| `002` | fp8_attention_qkv_projection | bf16 | 16 | 11/4/1/0 | 124–3402 | 1.00 | **per-row regime** |
| `003` | fp8_mlp_gate_up_projection | fp8 | 16 | 14/2/0/0 | 688–4760 | 1.00 | **MFU** |
| `004` | fp8_moe_expert_linear | bf16 | 16 | 14/2/0/0 | 341–1756 | 1.00 | **MFU** |
| `005` | fp8_moe_router_projection | fp8 | 16 | 0/0/5/11 | 163–235 | 0.40 | **per-row regime** |
| `011` | fp8_moe_gate_routing | bf16 | 16 | 0/13/0/3 | 126–240 | 0.81 | **MFU + BW% (both)** |
| `012` | fp8_shared_expert_mlp | bf16 | 16 | 13/3/0/0 | 236–1756 | 1.00 | **MFU** |
| `013` | fp8_mla_kv_compression_projection | bf16 | 16 | 0/16/0/0 | 173–517 | 1.00 | **MFU + BW% (both)** |
| `014` | fp8_yarn_rope_embedding | bf16 | 16 | 0/0/3/13 | 2–2 | 0.01 | **time + speedup_vs_reference** |
| `015` | fp8_mla_attention_output_projection | fp8 | 16 | 14/1/1/0 | 248–6199 | 1.00 | **MFU** |
| `016` | fp8_multi_latent_attention_qkv_projection | bf16 | 16 | 9/6/1/0 | 115–1018 | 1.00 | **per-row regime** |

## Cross-cutting summary

Total problems: **60**.

Recommended primary metric distribution:

| metric | count | share |
|---|---:|---:|
| per-row regime | 34 | 57% |
| time + speedup_vs_reference | 8 | 13% |
| MFU | 8 | 13% |
| BW% | 6 | 10% |
| MFU + BW% (both) | 4 | 7% |
