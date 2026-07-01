import torch
import torch.nn.functional as F

TOP_K = 8
N_EXPERTS = 64


@torch.no_grad()
def run(hidden_states, gate_weight, e_score_correction_bias,
        expert_gate_proj, expert_up_proj, expert_down_proj,
        shared_gate_proj, shared_up_proj, shared_down_proj, norm_min):
    b, s, H = hidden_states.shape
    x = hidden_states.view(-1, H)
    M = x.shape[0]
    dt = hidden_states.dtype
    dev = x.device

    # Shared expert
    sg = F.silu(x @ shared_gate_proj.t())
    su = x @ shared_up_proj.t()
    shared_out = (sg * su) @ shared_down_proj.t()

    # Routing
    router_logits = x.float() @ gate_weight.t()
    probs = F.softmax(router_logits, dim=1, dtype=torch.float32) + e_score_correction_bias
    _, sel = torch.topk(probs, TOP_K, dim=-1)
    rw = torch.gather(probs, -1, sel)
    rw = rw / torch.clamp(rw.sum(dim=-1, keepdim=True), min=norm_min)
    rw = rw.to(dt)

    flat_e = sel.reshape(-1)
    flat_tok = torch.arange(M, device=dev).repeat_interleave(TOP_K)
    flat_w = rw.reshape(-1)

    sort_idx = torch.argsort(flat_e)
    s_e = flat_e[sort_idx]
    s_tok = flat_tok[sort_idx]
    s_w = flat_w[sort_idx]

    counts = torch.bincount(s_e, minlength=N_EXPERTS)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), counts.cumsum(0)])
    counts_cpu = counts.tolist()
    offsets_cpu = offsets.tolist()

    x_sorted = x.index_select(0, s_tok)
    expert_out = torch.empty_like(x_sorted)

    for e in range(N_EXPERTS):
        n = counts_cpu[e]
        if n == 0:
            continue
        st = offsets_cpu[e]
        ed = st + n
        xe = x_sorted[st:ed]
        g = F.silu(xe @ expert_gate_proj[e].t())
        u = xe @ expert_up_proj[e].t()
        inter = g * u
        expert_out[st:ed] = inter @ expert_down_proj[e].t()

    weighted = (expert_out * s_w.unsqueeze(-1)).to(dt)
    routed = torch.zeros(M, H, dtype=dt, device=dev)
    routed.index_add_(0, s_tok, weighted)

    output = (routed + shared_out).view(b, s, H)
    return output, router_logits
