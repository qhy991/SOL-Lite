import torch
import flashinfer.gemm as fg
import flashinfer.norm as fn

E4M3_MAX = 448.0
KV_LORA_RANK = 512
QK_ROPE = 64
NUM_HEADS = 128
QK_NOPE = 128
V_HEAD = 128


def _q_act_1x128(x_bf16):
    M, K = x_bf16.shape
    x_f32 = x_bf16.to(torch.float32).reshape(M, K // 128, 128)
    amax = x_f32.abs().amax(dim=2)
    scale = (amax / E4M3_MAX).clamp(min=1e-12)
    qx = (x_f32 / scale.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX)
    qx = qx.reshape(M, K).to(torch.float8_e4m3fn)
    return qx, scale.t().contiguous()


def _q_w_128x128(w_bf16):
    N, K = w_bf16.shape
    blocks = w_bf16.to(torch.float32).reshape(N // 128, 128, K // 128, 128)
    amax = blocks.abs().amax(dim=(1, 3))
    scale = (amax / E4M3_MAX).clamp(min=1e-12)
    qw = (blocks / scale[:, None, :, None]).clamp(-E4M3_MAX, E4M3_MAX)
    qw = qw.reshape(N, K).to(torch.float8_e4m3fn)
    return qw, scale.t().contiguous()


def _fp8_matmul(x_bf16, w_bf16):
    qx, sx = _q_act_1x128(x_bf16)
    qw, sw = _q_w_128x128(w_bf16)
    return fg.gemm_fp8_nt_groupwise(
        qx, qw, sx, sw,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode='MN',
        out_dtype=torch.bfloat16,
    )


@torch.no_grad()
def run(hidden_states, kv_a_proj_weight, kv_a_layernorm_weight, kv_b_proj_weight, rms_norm_eps):
    b, s, h = hidden_states.shape
    x = hidden_states.reshape(-1, h).contiguous()
    a = _fp8_matmul(x, kv_a_proj_weight)               # (M, 640)
    compressed = a[:, :KV_LORA_RANK].contiguous()       # (M, 512)
    k_pe = a[:, KV_LORA_RANK:KV_LORA_RANK + QK_ROPE]   # (M, 64)
    normed = fn.rmsnorm(compressed, kv_a_layernorm_weight, rms_norm_eps)
    expanded = _fp8_matmul(normed, kv_b_proj_weight)    # (M, num_heads*(qk_nope+v_head))
    kv_expanded = expanded.reshape(b, s, NUM_HEADS, QK_NOPE + V_HEAD)
    k_pe = k_pe.reshape(b, s, 1, QK_ROPE)
    return kv_expanded.contiguous(), k_pe.contiguous()
