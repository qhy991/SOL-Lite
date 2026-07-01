import torch
import flashinfer.norm


@torch.no_grad()
def run(query, key, value, position_ids, key_cache, value_cache, cache_position,
        q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
    batch_size, num_q_heads, seq_len, head_dim = query.shape
    num_kv_heads = key.shape[1]

    # QK RMSNorm via FlashInfer (need 2D input)
    q_flat = query.reshape(-1, head_dim).contiguous()
    query_norm = flashinfer.norm.rmsnorm(q_flat, q_norm_weight, rms_norm_eps).reshape(query.shape)
    k_flat = key.reshape(-1, head_dim).contiguous()
    key_norm = flashinfer.norm.rmsnorm(k_flat, k_norm_weight, rms_norm_eps).reshape(key.shape)

    # RoPE computation (preserve reference)
    inv_freq_expanded = inv_freq[None, None, :].expand(batch_size, seq_len, -1)
    position_ids_expanded = position_ids[:, :, None].float()
    freqs = position_ids_expanded * inv_freq_expanded
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(query.dtype).unsqueeze(1)
    sin = emb.sin().to(query.dtype).unsqueeze(1)

    def apply_rope(x):
        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2:]
        x_rotated = torch.cat([-x2, x1], dim=-1)
        return (x * cos) + (x_rotated * sin)

    query_rotated = apply_rope(query_norm)
    key_rotated = apply_rope(key_norm)

    # Update KV cache
    key_cache[:, :, cache_position] = key_rotated
    value_cache[:, :, cache_position] = value

    return query_rotated, key_rotated, key_cache, value_cache
