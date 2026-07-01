import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states, position_ids, attention_mask,
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
    q_norm_weight, k_norm_weight,
    inv_freq,
    rms_norm_eps, attention_factor, scaling,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 40
    num_key_value_heads = 8
    head_dim = 128
    kv_hidden_size = num_key_value_heads * head_dim

    # Project Q, K, V
    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)

    # Q/K RMSNorm on full hidden dim (not per-head)
    query_states = F.rms_norm(query_states, (hidden_size,), q_norm_weight, eps=rms_norm_eps)
    key_states = F.rms_norm(key_states, (kv_hidden_size,), k_norm_weight, eps=rms_norm_eps)

    # Reshape
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    # YARN RoPE
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_factor).unsqueeze(1).to(query_states.dtype)
    sin = (emb.sin() * attention_factor).unsqueeze(1).to(query_states.dtype)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (rotate_half(key_states) * sin)

    # SDPA with arbitrary mask + GQA
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states,
        attn_mask=attention_mask,
        scale=scaling,
        enable_gqa=True,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    output = F.linear(attn_output, o_proj_weight)
    return output
