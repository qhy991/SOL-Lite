import torch
import flashinfer.gemm as fg
import torch.nn.functional as F


@torch.no_grad()
def run(x, scale_x, gate_proj_weight, scale_gate, up_proj_weight, scale_up):
    sx = scale_x.t().contiguous()
    sg = scale_gate.t().contiguous()
    su = scale_up.t().contiguous()
    gate = fg.gemm_fp8_nt_groupwise(
        x, gate_proj_weight, sx, sg,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode='MN',
        out_dtype=torch.bfloat16,
    )
    up = fg.gemm_fp8_nt_groupwise(
        x, up_proj_weight, sx, su,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode='MN',
        out_dtype=torch.bfloat16,
    )
    return F.silu(gate) * up
