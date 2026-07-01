import torch
import torch.nn.functional as F

N_EXPERTS = 256
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
EXPERTS_PER_GROUP = 32


@torch.no_grad()
def run(hidden_states, weight, expert_bias, routed_scaling_factor):
    M = hidden_states.shape[0]
    logits = F.linear(hidden_states, weight).float()      # bf16 GEMM, fp32 result
    scores = torch.sigmoid(logits)
    scores_for_routing = scores + expert_bias.float()

    grp = scores_for_routing.view(M, N_GROUP, EXPERTS_PER_GROUP)
    group_scores = torch.topk(grp, k=2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(-1).expand(M, N_GROUP, EXPERTS_PER_GROUP).reshape(M, N_EXPERTS).bool()

    neg_inf = torch.finfo(torch.float32).min
    masked = scores_for_routing.masked_fill(~score_mask, neg_inf)
    topk_idx = torch.topk(masked, k=TOP_K, dim=-1, sorted=False)[1]
    selected = torch.gather(scores, 1, topk_idx)
    topk_weight = selected / (selected.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_idx, topk_weight * routed_scaling_factor
