import torch
import flashinfer.gemm as fg

E4M3_MAX = 448.0
N_EXPERTS = 256
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
EXPERTS_PER_GROUP = N_EXPERTS // N_GROUP


def _quant_with_scale_1x128(x_bf16, scale_x):
    M, K = x_bf16.shape
    x_f32 = x_bf16.to(torch.float32).reshape(M, K // 128, 128)
    qx = (x_f32 / scale_x.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX)
    return qx.reshape(M, K).to(torch.float8_e4m3fn)


def _quant_w_with_scale_128x128(w_bf16, scale_w):
    N, K = w_bf16.shape
    blocks = w_bf16.to(torch.float32).reshape(N // 128, 128, K // 128, 128)
    # scale_w shape per axes: (hidden_blocks, expert_blocks) = (K//128, N//128). Transpose to (N//128, K//128).
    scale_w_nk = scale_w.t().contiguous()
    qw = (blocks / scale_w_nk[:, None, :, None]).clamp(-E4M3_MAX, E4M3_MAX)
    return qw.reshape(N, K).to(torch.float8_e4m3fn)


@torch.no_grad()
def run(hidden_states, weight, e_score_correction_bias,
        scale_x, scale_w, routed_scaling_factor):
    M, K = hidden_states.shape
    qx = _quant_with_scale_1x128(hidden_states, scale_x)
    qw = _quant_w_with_scale_128x128(weight, scale_w)
    sx_mn = scale_x.t().contiguous()           # (K//128, M)
    sw_mn = scale_w.contiguous()                # already (K//128, N//128)
    logits = fg.gemm_fp8_nt_groupwise(
        qx, qw, sx_mn, sw_mn,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode='MN',
        out_dtype=torch.bfloat16,
    )
    scores = torch.sigmoid(logits.float())
    scores_for_choice = scores + e_score_correction_bias.float().unsqueeze(0)
    grp = scores_for_choice.view(M, N_GROUP, EXPERTS_PER_GROUP)
    group_scores = grp.topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(M, N_GROUP, EXPERTS_PER_GROUP).reshape(M, N_EXPERTS)
    tmp = scores_for_choice.masked_fill(~score_mask.bool(), float('-inf'))
    _, topk_idx = torch.topk(tmp, k=TOP_K, dim=-1, sorted=False)
    topk_weight = scores.gather(1, topk_idx)
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * routed_scaling_factor
    return topk_idx, topk_weight
