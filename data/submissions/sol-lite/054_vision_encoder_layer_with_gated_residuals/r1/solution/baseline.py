import torch
import torch.nn.functional as F
import flashinfer.norm as fn

NH = 16
H = 1280
HD = H // NH
SCALING = HD ** -0.5


@torch.no_grad()
def run(hidden_state, input_layernorm_weight, input_layernorm_bias,
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        post_attention_layernorm_weight, post_attention_layernorm_bias,
        fc1_weight, fc1_bias, fc2_weight, fc2_bias, gate_attn, gate_ffn, norm_eps):
    b, s, _ = hidden_state.shape
    residual = hidden_state

    x_flat = hidden_state.reshape(-1, H).contiguous()
    normed = fn.layernorm(x_flat, input_layernorm_weight.float().contiguous(),
                           input_layernorm_bias.float().contiguous(), norm_eps).view(b, s, H)

    q = F.linear(normed, q_proj_weight).view(b, s, NH, HD).transpose(1, 2)
    k = F.linear(normed, k_proj_weight).view(b, s, NH, HD).transpose(1, 2)
    v = F.linear(normed, v_proj_weight).view(b, s, NH, HD).transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, scale=SCALING)
    out = out.transpose(1, 2).contiguous().reshape(b, s, H)
    attn_out = F.linear(out, o_proj_weight)
    hidden_state = residual + torch.tanh(gate_attn) * attn_out

    residual = hidden_state
    x_flat = hidden_state.reshape(-1, H).contiguous()
    normed = fn.layernorm(x_flat, post_attention_layernorm_weight.float().contiguous(),
                           post_attention_layernorm_bias.float().contiguous(), norm_eps).view(b, s, H)

    mlp = F.linear(F.gelu(F.linear(normed, fc1_weight, fc1_bias)), fc2_weight, fc2_bias)
    return residual + torch.tanh(gate_ffn) * mlp
