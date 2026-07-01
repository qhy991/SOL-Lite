import torch
import torch.nn.functional as F
import flashinfer.norm as fn

NQ = 32
NKV = 4
HD = 128
H = 2048
E = 128
TOP_K = 8
SCALING = HD ** -0.5


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@torch.no_grad()
def run(hidden_states, cos, sin, attention_mask,
        input_layernorm_weight, q_proj_weight, q_proj_bias,
        k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias,
        q_norm_weight, k_norm_weight, o_proj_weight, o_proj_bias,
        post_attention_layernorm_weight,
        router_weight, expert_gate_weights, expert_up_weights, expert_down_weights,
        rms_norm_eps):
    b, s, _ = hidden_states.shape
    dt = hidden_states.dtype
    dev = hidden_states.device
    residual = hidden_states

    x_flat = hidden_states.reshape(-1, H).contiguous()
    normed = fn.rmsnorm(x_flat, input_layernorm_weight, rms_norm_eps).view(b, s, H)

    q = F.linear(normed, q_proj_weight, q_proj_bias).view(b, s, NQ, HD)
    k = F.linear(normed, k_proj_weight, k_proj_bias).view(b, s, NKV, HD)
    v = F.linear(normed, v_proj_weight, v_proj_bias).view(b, s, NKV, HD)

    q = fn.rmsnorm(q.reshape(-1, HD).contiguous(), q_norm_weight, rms_norm_eps).view(b, s, NQ, HD)
    k = fn.rmsnorm(k.reshape(-1, HD).contiguous(), k_norm_weight, rms_norm_eps).view(b, s, NKV, HD)

    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    cos_e = cos.unsqueeze(1)
    sin_e = sin.unsqueeze(1)
    q_t = q_t * cos_e + _rotate_half(q_t) * sin_e
    k_t = k_t * cos_e + _rotate_half(k_t) * sin_e

    v_t = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attention_mask, scale=SCALING, enable_gqa=True)
    out = out.transpose(1, 2).contiguous().reshape(b, s, NQ * HD)
    hidden_states = residual + F.linear(out, o_proj_weight, o_proj_bias)

    residual = hidden_states
    x_flat = hidden_states.reshape(-1, H).contiguous()
    normed = fn.rmsnorm(x_flat, post_attention_layernorm_weight, rms_norm_eps)  # (b*s, H)

    # MoE
    M = normed.shape[0]
    logits = F.linear(normed, router_weight)
    probs = F.softmax(logits.float(), dim=1)
    w_topk, e_topk = torch.topk(probs, TOP_K, dim=-1)
    w_topk = (w_topk / w_topk.sum(dim=-1, keepdim=True)).to(dt)

    flat_e = e_topk.reshape(-1)
    flat_tok = torch.arange(M, device=dev).repeat_interleave(TOP_K)
    flat_w = w_topk.reshape(-1)
    sort_idx = torch.argsort(flat_e)
    s_e = flat_e[sort_idx]
    s_tok = flat_tok[sort_idx]
    s_w = flat_w[sort_idx]

    counts = torch.bincount(s_e, minlength=E)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), counts.cumsum(0)])
    counts_cpu = counts.tolist()
    offsets_cpu = offsets.tolist()

    x_sorted = normed.index_select(0, s_tok)
    expert_out = torch.empty_like(x_sorted)
    for e in range(E):
        n = counts_cpu[e]
        if n == 0:
            continue
        st = offsets_cpu[e]
        ed = st + n
        xe = x_sorted[st:ed]
        gw = expert_gate_weights[e]
        uw = expert_up_weights[e]
        dw = expert_down_weights[e]
        gate = F.silu(F.linear(xe, gw))
        up = F.linear(xe, uw)
        expert_out[st:ed] = F.linear(gate * up, dw)

    weighted = (expert_out * s_w.unsqueeze(-1)).to(dt)
    moe_out = torch.zeros(M, H, dtype=dt, device=dev)
    moe_out.index_add_(0, s_tok, weighted)
    return residual + moe_out.view(b, s, H)
