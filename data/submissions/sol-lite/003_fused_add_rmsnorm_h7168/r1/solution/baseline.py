import torch
import flashinfer.norm


@torch.no_grad()
def run(hidden_states, residual, weight):
    # FlashInfer fused_add_rmsnorm does residual += input, then input = rmsnorm(residual)
    # But returns None (in-place). Reference does: (x+residual).rmsnorm() * weight
    # We use rmsnorm on the sum instead
    x = residual + hidden_states
    eps = 1e-6
    return flashinfer.norm.rmsnorm(x, weight, eps)
