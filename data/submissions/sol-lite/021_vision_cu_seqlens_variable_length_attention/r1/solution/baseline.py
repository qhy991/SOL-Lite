import torch
import torch.nn.functional as F

# Threshold based on benchmark: block-diag is faster for total_seq_len < 2500,
# reference loop is faster beyond (mask becomes O(N^2) memory-bound).
_BLOCK_DIAG_THRESHOLD = 2500


@torch.no_grad()
def run(
    hidden_states, cu_seqlens, cos, sin,
    qkv_weight, qkv_bias, proj_weight, proj_bias,
):
    hidden_size = 1280
    num_heads = 16
    head_dim = 80
    scaling = head_dim ** -0.5
    total_seq_len = hidden_states.shape[0]
    device = hidden_states.device
    dtype = hidden_states.dtype

    # QKV + RoPE (common)
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.reshape(total_seq_len, 3, num_heads, head_dim).permute(1, 0, 2, 3)
    q, k, v = qkv.unbind(0)

    cos_e = cos.unsqueeze(1); sin_e = sin.unsqueeze(1)
    q1, q2 = q[..., :head_dim // 2], q[..., head_dim // 2:]
    q = (q * cos_e) + (torch.cat((-q2, q1), dim=-1) * sin_e)
    k1, k2 = k[..., :head_dim // 2], k[..., head_dim // 2:]
    k = (k * cos_e) + (torch.cat((-k2, k1), dim=-1) * sin_e)

    if total_seq_len < _BLOCK_DIAG_THRESHOLD:
        # Short sequences: block-diag SDPA is ~2x faster
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        positions = torch.arange(total_seq_len, device=device)
        segment_ids = torch.searchsorted(cu_seqlens, positions, right=True) - 1
        allowed = segment_ids.unsqueeze(0) == segment_ids.unsqueeze(1)
        mask = torch.where(allowed, 0.0, float('-inf')).to(dtype)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask.unsqueeze(0).unsqueeze(0),
            scale=scaling,
        )
        attn_output = attn_output.transpose(1, 2).reshape(total_seq_len, hidden_size).contiguous()
    else:
        # Long sequences: per-sequence SDPA (O(seg_len^2) instead of O(total^2))
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        num_seqs = cu_seqlens.shape[0]
        attn_outputs = []
        for i in range(num_seqs - 1):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            if end <= start:
                continue
            q_seq = q[:, :, start:end, :]
            k_seq = k[:, :, start:end, :]
            v_seq = v[:, :, start:end, :]
            out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, scale=scaling)
            attn_outputs.append(out.transpose(1, 2))

        attn_output = torch.cat(attn_outputs, dim=1).reshape(total_seq_len, hidden_size).contiguous()

    return F.linear(attn_output, proj_weight, proj_bias)
