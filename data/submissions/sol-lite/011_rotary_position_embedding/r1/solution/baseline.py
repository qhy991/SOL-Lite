import torch


@torch.no_grad()
def run(position_ids, inv_freq, attention_scaling):
    pos = position_ids.float()
    freqs = pos.unsqueeze(-1) * inv_freq.float()  # (B, S, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)        # (B, S, head_dim)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return torch.stack([cos, sin], dim=-1).to(torch.bfloat16)
