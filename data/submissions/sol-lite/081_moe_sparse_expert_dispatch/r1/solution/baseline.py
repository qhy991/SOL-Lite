import torch
import torch.nn.functional as F

TOP_K = 8
N_EXPERTS = 128


@torch.no_grad()
def run(hidden_states, router_weight,
        expert_gate_weights, expert_up_weights, expert_down_weights,
        shared_gate_weight, shared_up_weight, shared_down_weight,
        e_score_correction_bias, routed_scaling_factor):
    M, H = hidden_states.shape
    dt = hidden_states.dtype
    dev = hidden_states.device

    logits = F.linear(hidden_states.float(), router_weight.float())
    scores = torch.sigmoid(logits)
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    topk_w, topk_idx = torch.topk(scores_for_choice, k=TOP_K, dim=-1, sorted=False)
    topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
    topk_w = (topk_w * routed_scaling_factor).to(dt)

    flat_e = topk_idx.reshape(-1)
    flat_tok = torch.arange(M, device=dev).repeat_interleave(TOP_K)
    flat_w = topk_w.reshape(-1)

    sort_idx = torch.argsort(flat_e)
    s_e = flat_e[sort_idx]
    s_tok = flat_tok[sort_idx]
    s_w = flat_w[sort_idx]

    counts = torch.bincount(s_e, minlength=N_EXPERTS)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), counts.cumsum(0)])
    counts_cpu = counts.tolist()
    offsets_cpu = offsets.tolist()

    x_sorted = hidden_states.index_select(0, s_tok)
    expert_out = torch.empty_like(x_sorted)

    for e in range(N_EXPERTS):
        n = counts_cpu[e]
        if n == 0:
            continue
        st = offsets_cpu[e]
        ed = st + n
        xe = x_sorted[st:ed]
        g = F.silu(F.linear(xe, expert_gate_weights[e]))
        u = F.linear(xe, expert_up_weights[e])
        inter = g * u
        expert_out[st:ed] = F.linear(inter, expert_down_weights[e])

    weighted = expert_out * s_w.unsqueeze(-1)
    routed = torch.zeros(M, H, dtype=dt, device=dev)
    routed.index_add_(0, s_tok, weighted)

    sg = F.silu(F.linear(hidden_states, shared_gate_weight))
    su = F.linear(hidden_states, shared_up_weight)
    shared_out = F.linear(sg * su, shared_down_weight)

    return routed + shared_out
