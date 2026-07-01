import torch
import torch.nn.functional as F
import flashinfer.norm


@torch.no_grad()
def run(hidden_states, q_a_proj_weight, q_a_layernorm_weight, q_b_proj_weight, kv_a_proj_weight, rms_norm_eps):
    num_heads = 128
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    q_head_dim = 192
    kv_lora_rank = 512

    bsz, seq_len, hidden_size = hidden_states.shape

    # Q latent projection
    q_latent = F.linear(hidden_states, q_a_proj_weight)
    q_latent_dim = q_latent.shape[-1]

    # RMSNorm on q_latent via FlashInfer
    q_flat = q_latent.reshape(-1, q_latent_dim).contiguous()
    q_latent = flashinfer.norm.rmsnorm(q_flat, q_a_layernorm_weight, rms_norm_eps).reshape(
        bsz, seq_len, q_latent_dim
    )

    # Q projection
    q = F.linear(q_latent, q_b_proj_weight)
    q = q.view(bsz, seq_len, num_heads, q_head_dim)
    q_nope = q[..., :qk_nope_head_dim].contiguous()
    q_pe = q[..., qk_nope_head_dim:].contiguous()

    # KV projection
    kv_combined = F.linear(hidden_states, kv_a_proj_weight)
    compressed_kv = kv_combined[..., :kv_lora_rank].contiguous()
    k_pe = kv_combined[..., kv_lora_rank:].contiguous()
    k_pe = k_pe.view(bsz, seq_len, 1, qk_rope_head_dim)

    return q_nope, q_pe, compressed_kv, k_pe
