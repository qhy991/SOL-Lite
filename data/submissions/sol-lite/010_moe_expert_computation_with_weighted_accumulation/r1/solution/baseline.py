import torch
import torch.nn.functional as F


@torch.no_grad()
def run(hidden_states, routing_weights, selected_experts,
        gate_proj_weights, up_proj_weights, down_proj_weights):
    M, H = hidden_states.shape
    E = gate_proj_weights.shape[0]
    K = selected_experts.shape[1]
    dev = hidden_states.device
    dt = hidden_states.dtype

    flat_e = selected_experts.reshape(-1)
    flat_tok = torch.arange(M, device=dev).repeat_interleave(K)
    flat_w = routing_weights.float().reshape(-1)

    sort_idx = torch.argsort(flat_e)
    s_e = flat_e[sort_idx]
    s_tok = flat_tok[sort_idx]
    s_w = flat_w[sort_idx]

    counts = torch.bincount(s_e, minlength=E)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), counts.cumsum(0)])
    counts_cpu = counts.tolist()
    offsets_cpu = offsets.tolist()

    x_sorted = hidden_states.index_select(0, s_tok).float()
    expert_out = torch.empty_like(x_sorted)

    for e in range(E):
        n = counts_cpu[e]
        if n == 0:
            continue
        st = offsets_cpu[e]
        ed = st + n
        x = x_sorted[st:ed]
        gw = gate_proj_weights[e].float()
        uw = up_proj_weights[e].float()
        dw = down_proj_weights[e].float()
        g = x @ gw.t()
        u = x @ uw.t()
        inter = F.silu(g) * u
        expert_out[st:ed] = inter @ dw.t()

    weighted = (expert_out * s_w.unsqueeze(-1)).to(dt)
    output = torch.zeros(M, H, dtype=dt, device=dev)
    output.index_add_(0, s_tok, weighted)
    return output
