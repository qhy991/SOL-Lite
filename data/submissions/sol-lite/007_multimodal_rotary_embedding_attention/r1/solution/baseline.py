import torch
import torch.nn.functional as F

NQ = 28
NKV = 4
HD = 128
H = 3584
SCALING = HD ** -0.5
MROPE_SEC = [16, 24, 24]


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _combine_mrope(cs):
    splits = cs.split([s * 2 for s in MROPE_SEC], dim=-1)
    return torch.cat([m[i % 3] for i, m in enumerate(splits)], dim=-1).unsqueeze(1)


@torch.no_grad()
def run(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias,
        o_weight, cos, sin, attention_mask):
    b, s, _ = hidden_states.shape
    q = F.linear(hidden_states, q_weight, q_bias).view(b, s, NQ, HD).transpose(1, 2)
    k = F.linear(hidden_states, k_weight, k_bias).view(b, s, NKV, HD).transpose(1, 2)
    v = F.linear(hidden_states, v_weight, v_bias).view(b, s, NKV, HD).transpose(1, 2)

    cos_c = _combine_mrope(cos)
    sin_c = _combine_mrope(sin)
    q = q * cos_c + _rotate_half(q) * sin_c
    k = k * cos_c + _rotate_half(k) * sin_c

    mask = attention_mask[..., :s]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=SCALING, enable_gqa=True)
    out = out.transpose(1, 2).contiguous().reshape(b, s, H)
    return F.linear(out, o_weight)
