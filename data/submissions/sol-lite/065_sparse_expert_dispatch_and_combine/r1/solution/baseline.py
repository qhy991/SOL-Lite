import torch

ALPHA = 1.702
LIMIT = 7.0


@torch.no_grad()
def run(hidden_states, router_indices, routing_weights,
        gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias):
    M, H = hidden_states.shape
    E = gate_up_proj.shape[0]
    K = router_indices.shape[1]
    dev = hidden_states.device

    flat_e = router_indices.reshape(-1)
    flat_tok = torch.arange(M, device=dev).repeat_interleave(K)

    # Drop masking-class assignments (index == E)
    keep = flat_e < E
    flat_e = flat_e[keep]
    flat_tok = flat_tok[keep]

    sort_idx = torch.argsort(flat_e)
    s_e = flat_e[sort_idx]
    s_tok = flat_tok[sort_idx]
    # routing weight for each (token, expert) pair
    s_w = routing_weights[s_tok, s_e]

    counts = torch.bincount(s_e, minlength=E)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), counts.cumsum(0)])
    counts_cpu = counts.tolist()
    offsets_cpu = offsets.tolist()

    x_sorted = hidden_states.index_select(0, s_tok)
    output = torch.zeros_like(hidden_states)

    for e in range(E):
        n = counts_cpu[e]
        if n == 0:
            continue
        st = offsets_cpu[e]
        ed = st + n
        xe = x_sorted[st:ed]
        gu = xe @ gate_up_proj[e] + gate_up_proj_bias[e]
        gate = gu[..., ::2].clamp(max=LIMIT)
        up = gu[..., 1::2].clamp(min=-LIMIT, max=LIMIT)
        glu = gate * torch.sigmoid(gate * ALPHA)
        gated = (up + 1) * glu
        expert_out = gated @ down_proj[e] + down_proj_bias[e]
        weighted = expert_out * s_w[st:ed].unsqueeze(-1)
        output.index_add_(0, s_tok[st:ed], weighted)
    return output
