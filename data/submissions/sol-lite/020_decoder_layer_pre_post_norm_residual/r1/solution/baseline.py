import torch
import torch.nn.functional as F
import flashinfer.norm as fn
import flashinfer.activation as fa

NQ = 64
NKV = 8
HD = 96
HALF = 48
H = 6144
SCALING = HD ** -0.5


def _apply_reka_rope(x, cos_e, sin_e):
    x1 = x[..., :HALF]
    x2 = x[..., HALF:]
    return torch.cat([x1 * cos_e - x2 * sin_e, x1 * sin_e + x2 * cos_e], dim=-1)


@torch.no_grad()
def run(hidden_states, cos, sin, attention_mask,
        input_layernorm_weight, q_proj_weight, k_proj_weight, v_proj_weight,
        o_proj_weight, post_attention_layernorm_weight,
        gate_proj_weight, up_proj_weight, down_proj_weight, rms_norm_eps):
    b, s, _ = hidden_states.shape
    residual = hidden_states

    x_flat = hidden_states.reshape(-1, H).contiguous()
    normed = fn.rmsnorm(x_flat, input_layernorm_weight, rms_norm_eps).view(b, s, H)

    q = F.linear(normed, q_proj_weight).view(b, s, NQ, HD).transpose(1, 2)
    k = F.linear(normed, k_proj_weight).view(b, s, NKV, HD).transpose(1, 2)
    v = F.linear(normed, v_proj_weight).view(b, s, NKV, HD).transpose(1, 2)

    cos_e = cos.unsqueeze(1)  # (b, 1, s, half)
    sin_e = sin.unsqueeze(1)
    q = _apply_reka_rope(q, cos_e, sin_e)
    k = _apply_reka_rope(k, cos_e, sin_e)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=SCALING, enable_gqa=True)
    out = out.transpose(1, 2).contiguous().reshape(b, s, NQ * HD)
    hidden_states = residual + F.linear(out, o_proj_weight)

    residual = hidden_states
    x_flat = hidden_states.reshape(-1, H).contiguous()
    normed = fn.rmsnorm(x_flat, post_attention_layernorm_weight, rms_norm_eps).view(b, s, H)
    gated = torch.cat([F.linear(normed, gate_proj_weight), F.linear(normed, up_proj_weight)], dim=-1)
    inter_dim = gate_proj_weight.shape[0]
    intermediate = fa.silu_and_mul(gated.reshape(-1, 2 * inter_dim).contiguous()).view(b, s, inter_dim)
    return residual + F.linear(intermediate, down_proj_weight)
