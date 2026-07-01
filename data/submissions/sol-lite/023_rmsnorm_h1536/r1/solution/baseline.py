import torch
import flashinfer.norm


@torch.no_grad()
def run(hidden_states, weight):
    eps = 1e-6
    return flashinfer.norm.rmsnorm(hidden_states, weight, eps)
