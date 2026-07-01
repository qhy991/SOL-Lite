import torch
import torch.nn.functional as F


@torch.no_grad()
def run(hidden_states, expert_ids, expert_weights,
        gate_proj_weights, up_proj_weights, down_proj_weights):
    num_tokens, hidden_size = hidden_states.shape
    num_experts = gate_proj_weights.shape[0]
    top_k = expert_ids.shape[1]

    # Flatten (num_tokens, top_k) into a single sortable list.
    flat_experts = expert_ids.reshape(-1)
    flat_token_idx = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(top_k)
    flat_weights = expert_weights.reshape(-1)

    sort_idx = torch.argsort(flat_experts)
    sorted_experts = flat_experts[sort_idx]
    sorted_token_idx = flat_token_idx[sort_idx]
    sorted_weights = flat_weights[sort_idx]

    expert_counts = torch.bincount(sorted_experts, minlength=num_experts)
    offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=hidden_states.device),
        expert_counts.cumsum(0),
    ])
    counts_cpu = expert_counts.tolist()
    offsets_cpu = offsets.tolist()

    sorted_inputs = hidden_states.index_select(0, sorted_token_idx)
    expert_out = torch.empty_like(sorted_inputs)

    for e in range(num_experts):
        n = counts_cpu[e]
        if n == 0:
            continue
        start = offsets_cpu[e]
        end = start + n
        x = sorted_inputs[start:end]
        gate = x @ gate_proj_weights[e].t()
        up = x @ up_proj_weights[e].t()
        inter = F.silu(gate) * up
        expert_out[start:end] = inter @ down_proj_weights[e].t()

    weighted = expert_out * sorted_weights.unsqueeze(-1).to(expert_out.dtype)
    output = torch.zeros_like(hidden_states)
    output.index_add_(0, sorted_token_idx, weighted)
    return output
