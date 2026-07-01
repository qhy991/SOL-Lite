import torch
import flashinfer.gemm as fg


@torch.no_grad()
def run(hidden_states, weight, scale_x, scale_w):
    b, s, d = hidden_states.shape
    x = hidden_states.view(-1, d)
    sx = scale_x.view(-1, scale_x.shape[-1]).t().contiguous()
    sw = scale_w.contiguous()
    out = fg.gemm_fp8_nt_groupwise(
        x, weight, sx, sw,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode='MN',
        out_dtype=torch.bfloat16,
    )
    return out.view(b, s, -1)
