import torch
import flashinfer.norm


@torch.no_grad()
def run(compressed_kv, kv_a_layernorm_weight, kv_b_proj_weight, eps):
    num_heads = 128
    qk_nope_head_dim = 128
    v_head_dim = 128

    bsz, seq_len, kv_lora_rank = compressed_kv.shape

    # RMSNorm via FlashInfer
    x_flat = compressed_kv.reshape(-1, kv_lora_rank).contiguous()
    normalized = flashinfer.norm.rmsnorm(x_flat, kv_a_layernorm_weight, eps).reshape(
        bsz, seq_len, kv_lora_rank
    )

    # Linear projection
    expanded = torch.matmul(normalized, kv_b_proj_weight.t())

    # Reshape and split
    kv = expanded.view(bsz, seq_len, num_heads, qk_nope_head_dim + v_head_dim)
    kv = kv.transpose(1, 2)
    k_nope = kv[:, :, :, :qk_nope_head_dim].contiguous()
    value_states = kv[:, :, :, qk_nope_head_dim:].contiguous()

    return k_nope, value_states
