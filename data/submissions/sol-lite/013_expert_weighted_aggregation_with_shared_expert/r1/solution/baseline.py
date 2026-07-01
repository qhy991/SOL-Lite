import torch
import torch.nn.functional as F


@torch.no_grad()
def run(hidden_states, routing_weights, selected_experts,
        expert_gate_proj_weights, expert_up_proj_weights, expert_down_proj_weights,
        shared_expert_gate_proj_weight, shared_expert_up_proj_weight,
        shared_expert_down_proj_weight, shared_expert_gate_weight):
    M, H = hidden_states.shape
    E = expert_gate_proj_weights.shape[0]
    K = selected_experts.shape[1]
    dev = hidden_states.device
    dt = hidden_states.dtype

    # ---- routed experts via sorted dispatch ----
    flat_e = selected_experts.reshape(-1)
    flat_tok = torch.arange(M, device=dev).repeat_interleave(K)
    flat_w = routing_weights.reshape(-1)

    sort_idx = torch.argsort(flat_e)
    s_e = flat_e[sort_idx]
    s_tok = flat_tok[sort_idx]
    s_w = flat_w[sort_idx]

    counts = torch.bincount(s_e, minlength=E)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), counts.cumsum(0)])
    counts_cpu = counts.tolist()
    offsets_cpu = offsets.tolist()

    x_sorted = hidden_states.index_select(0, s_tok)
    expert_out = torch.empty_like(x_sorted)

    for e in range(E):
        n = counts_cpu[e]
        if n == 0:
            continue
        st = offsets_cpu[e]
        ed = st + n
        x = x_sorted[st:ed]
        g = x @ expert_gate_proj_weights[e].t()
        u = x @ expert_up_proj_weights[e].t()
        inter = F.silu(g) * u
        expert_out[st:ed] = inter @ expert_down_proj_weights[e].t()

    weighted = expert_out * s_w.unsqueeze(-1).to(dt)
    routed_out = torch.zeros(M, H, dtype=dt, device=dev)
    routed_out.index_add_(0, s_tok, weighted)

    # ---- shared expert ----
    sg = hidden_states @ shared_expert_gate_proj_weight.t()
    su = hidden_states @ shared_expert_up_proj_weight.t()
    shared_inter = F.silu(sg) * su
    shared_out = shared_inter @ shared_expert_down_proj_weight.t()
    gate_logits = hidden_states @ shared_expert_gate_weight.t()
    shared_gate = torch.sigmoid(gate_logits.float()).to(dt)
    shared_out = shared_gate * shared_out

    return routed_out + shared_out
