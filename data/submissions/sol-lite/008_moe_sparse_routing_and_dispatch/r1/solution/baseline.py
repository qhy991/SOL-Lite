import torch
import torch.nn.functional as F

TOP_K = 8


@torch.no_grad()
def run(hidden_states, gate_weight,
        expert_gate_proj, expert_up_proj, expert_down_proj, norm_topk_prob):
    b, s, H = hidden_states.shape
    x = hidden_states.view(-1, H)
    M = x.shape[0]
    E = gate_weight.shape[0]
    dt = x.dtype
    dev = x.device

    logits = x @ gate_weight.t()
    probs = F.softmax(logits.float(), dim=1).to(dt)
    w_topk, e_topk = torch.topk(probs, TOP_K, dim=-1)
    if norm_topk_prob:
        w_topk = w_topk / (w_topk.sum(dim=-1, keepdim=True) + 1e-9)

    flat_e = e_topk.reshape(-1)
    flat_tok = torch.arange(M, device=dev).repeat_interleave(TOP_K)
    flat_w = w_topk.reshape(-1)

    sort_idx = torch.argsort(flat_e)
    s_e = flat_e[sort_idx]
    s_tok = flat_tok[sort_idx]
    s_w = flat_w[sort_idx]

    counts = torch.bincount(s_e, minlength=E)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), counts.cumsum(0)])
    counts_cpu = counts.tolist()
    offsets_cpu = offsets.tolist()

    x_sorted = x.index_select(0, s_tok)
    expert_out = torch.empty_like(x_sorted)

    for e in range(E):
        n = counts_cpu[e]
        if n == 0:
            continue
        st = offsets_cpu[e]
        ed = st + n
        xe = x_sorted[st:ed]
        g = xe @ expert_gate_proj[e].t()
        u = xe @ expert_up_proj[e].t()
        inter = F.silu(g) * u
        expert_out[st:ed] = inter @ expert_down_proj[e].t()

    weighted = (expert_out * s_w.unsqueeze(-1)).to(dt)
    out = torch.zeros(M, H, dtype=dt, device=dev)
    out.index_add_(0, s_tok, weighted)
    return out.view(b, s, H)
