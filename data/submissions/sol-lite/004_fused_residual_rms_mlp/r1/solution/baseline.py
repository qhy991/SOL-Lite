import torch
import torch.nn.functional as F
import flashinfer.norm


@torch.no_grad()
def run(hidden_states, residual, norm_weight, gate_proj_weight, up_proj_weight, down_proj_weight, eps):
    # Residual + RMSNorm via FlashInfer
    x = residual + hidden_states
    batch, seq_len, hidden_size = x.shape
    x_flat = x.reshape(-1, hidden_size).contiguous()
    normed = flashinfer.norm.rmsnorm(x_flat, norm_weight, eps).reshape(batch, seq_len, hidden_size)

    # SwiGLU MLP
    gate = F.linear(normed, gate_proj_weight)
    up = F.linear(normed, up_proj_weight)
    intermediate = F.silu(gate) * up
    output = F.linear(intermediate, down_proj_weight)
    return output
