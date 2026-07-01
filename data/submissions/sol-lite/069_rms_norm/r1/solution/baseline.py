import torch
import flashinfer.norm


@torch.no_grad()
def run(hidden_states, residual, weight, eps):
    x = residual + hidden_states
    return flashinfer.norm.rmsnorm(x, weight, eps)
