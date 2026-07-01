import torch
import torch.nn.functional as F
import flashinfer.norm as fn

NQ = 32
NKV = 8
HD = 128


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@torch.no_grad()
def run(hidden_states, cos, sin, attention_mask,
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        q_norm_weight, k_norm_weight, rms_norm_eps, scaling):
    b, s, h = hidden_states.shape

    q = F.linear(hidden_states, q_proj_weight).view(b, s, NQ, HD)
    k = F.linear(hidden_states, k_proj_weight).view(b, s, NKV, HD)
    v = F.linear(hidden_states, v_proj_weight).view(b, s, NKV, HD)

    # Per-head RMSNorm on Q/K via flashinfer (flatten all heads as M, HD)
    q = fn.rmsnorm(q.reshape(-1, HD).contiguous(), q_norm_weight, rms_norm_eps).view(b, s, NQ, HD)
    k = fn.rmsnorm(k.reshape(-1, HD).contiguous(), k_norm_weight, rms_norm_eps).view(b, s, NKV, HD)

    # Transpose to (B, H, S, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    cos_e = cos.unsqueeze(1)
    sin_e = sin.unsqueeze(1)
    q = q * cos_e + _rotate_half(q) * sin_e
    k = k * cos_e + _rotate_half(k) * sin_e

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attention_mask,
        scale=scaling,
        enable_gqa=True,
    )
    out = out.transpose(1, 2).contiguous().reshape(b, s, NQ * HD)
    return F.linear(out, o_proj_weight)
