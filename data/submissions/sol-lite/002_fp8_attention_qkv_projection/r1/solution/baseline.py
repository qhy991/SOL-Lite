import torch
import flashinfer.gemm as fg
import flashinfer.norm as fn

E4M3_MAX = 448.0


def _quant_act_1x128(x_bf16):
    M, K = x_bf16.shape
    x_f32 = x_bf16.to(torch.float32).reshape(M, K // 128, 128)
    amax = x_f32.abs().amax(dim=2)
    scale = (amax / E4M3_MAX).clamp(min=1e-12)
    qx = (x_f32 / scale.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX)
    qx = qx.reshape(M, K).to(torch.float8_e4m3fn)
    return qx, scale.t().contiguous()


def _quant_weight_128x128(w_bf16):
    N, K = w_bf16.shape
    blocks = w_bf16.to(torch.float32).reshape(N // 128, 128, K // 128, 128)
    amax = blocks.abs().amax(dim=(1, 3))
    scale = (amax / E4M3_MAX).clamp(min=1e-12)
    qw = (blocks / scale[:, None, :, None]).clamp(-E4M3_MAX, E4M3_MAX)
    qw = qw.reshape(N, K).to(torch.float8_e4m3fn)
    return qw, scale.t().contiguous()


def _fp8_linear(x_bf16, weight_bf16, bias_bf16):
    qx, sx = _quant_act_1x128(x_bf16)
    qw, sw = _quant_weight_128x128(weight_bf16)
    out = fg.gemm_fp8_nt_groupwise(
        qx, qw, sx, sw,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode='MN',
        out_dtype=torch.bfloat16,
    )
    if bias_bf16 is not None and bias_bf16.numel():
        out = out + bias_bf16
    return out


@torch.no_grad()
def run(hidden_states, q_weight, k_weight, v_weight,
        q_bias, k_bias, v_bias, q_norm_weight, k_norm_weight, rms_norm_eps):
    NQ = 60; NKV = 8; HD = 128
    b, s, h = hidden_states.shape
    x_flat = hidden_states.reshape(-1, h).contiguous()

    q = _fp8_linear(x_flat, q_weight, q_bias)
    k = _fp8_linear(x_flat, k_weight, k_bias)
    v = _fp8_linear(x_flat, v_weight, v_bias)

    q_normed = fn.rmsnorm(q.reshape(-1, HD).contiguous(), q_norm_weight, rms_norm_eps)
    q = q_normed.reshape(b, s, NQ, HD).permute(0, 2, 1, 3).contiguous()

    k_normed = fn.rmsnorm(k.reshape(-1, HD).contiguous(), k_norm_weight, rms_norm_eps)
    k = k_normed.reshape(b, s, NKV, HD).permute(0, 2, 1, 3).contiguous()

    v = v.reshape(b, s, NKV, HD).permute(0, 2, 1, 3).contiguous()
    return q, k, v
