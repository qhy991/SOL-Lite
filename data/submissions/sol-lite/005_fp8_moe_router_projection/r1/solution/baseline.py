import torch
import flashinfer.gemm as fg

NUM_EXPERTS = 64


@torch.no_grad()
def run(hidden_states, gate_weight, scale_hidden, scale_weight):
    sx = scale_hidden.t().contiguous()
    sw = scale_weight.t().contiguous()
    out = fg.gemm_fp8_nt_groupwise(
        hidden_states, gate_weight, sx, sw,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode='MN',
        out_dtype=torch.bfloat16,
    )
    return out[:, :NUM_EXPERTS].contiguous()
