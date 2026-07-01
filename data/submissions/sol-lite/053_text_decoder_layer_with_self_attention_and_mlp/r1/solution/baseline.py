import torch
import torch.nn.functional as F
import flashinfer.norm as fn
import flashinfer.activation as fa

NQ = 32
NKV = 8
HD = 128
H = 4096
SCALING = HD ** -0.5


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@torch.no_grad()
def run(hidden_states, attention_mask,
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        gate_proj_weight, up_proj_weight, down_proj_weight,
        input_layernorm_weight, post_attention_layernorm_weight,
        rms_norm_eps, rope_theta):
    b, s, _ = hidden_states.shape
    dev = hidden_states.device
    residual = hidden_states

    x_flat = hidden_states.reshape(-1, H).contiguous()
    normed = fn.rmsnorm(x_flat, input_layernorm_weight, rms_norm_eps).view(b, s, H)

    q = F.linear(normed, q_proj_weight).view(b, s, NQ, HD).transpose(1, 2)
    k = F.linear(normed, k_proj_weight).view(b, s, NKV, HD).transpose(1, 2)
    v = F.linear(normed, v_proj_weight).view(b, s, NKV, HD).transpose(1, 2)

    # Compute cos/sin from rope_theta
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, HD, 2, dtype=torch.float32, device=dev) / HD))
    pos = torch.arange(s, device=dev).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(q.dtype)[None, None]
    sin = emb.sin().to(q.dtype)[None, None]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin

    mask = attention_mask[:, :, :s, :s]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=SCALING, enable_gqa=True)
    out = out.transpose(1, 2).contiguous().reshape(b, s, NQ * HD)
    hidden_states = residual + F.linear(out, o_proj_weight)

    residual = hidden_states
    x_flat = hidden_states.reshape(-1, H).contiguous()
    normed = fn.rmsnorm(x_flat, post_attention_layernorm_weight, rms_norm_eps).view(b, s, H)
    gated = torch.cat([F.linear(normed, gate_proj_weight), F.linear(normed, up_proj_weight)], dim=-1)
    inter_dim = gate_proj_weight.shape[0]
    intermediate = fa.silu_and_mul(gated.reshape(-1, 2 * inter_dim).contiguous()).view(b, s, inter_dim)
    return residual + F.linear(intermediate, down_proj_weight)
