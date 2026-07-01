import torch
import torch.nn.functional as F
import flashinfer.gemm as fg

E4M3_MAX = 448.0


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
def run(hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight):
    gate = _fp8_matmul(hidden_states, gate_proj_weight)
    up = _fp8_matmul(hidden_states, up_proj_weight)
    intermediate = (F.silu(gate) * up).contiguous()
    return _fp8_matmul(intermediate, down_proj_weight)
