import torch
import torch.nn.functional as F
import flashinfer.norm as fn

MERGE = 2
HIDDEN = 1536
EXPANDED = 6144


@torch.no_grad()
def run(hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias,
        fc2_weight, fc2_bias, eps):
    ln_w = ln_weight.float().contiguous()
    ln_b = ln_bias.float().contiguous()
    normed = fn.layernorm(hidden.contiguous(), ln_w, ln_b, eps)

    grids = grid_thw.tolist()
    pieces = []
    offset = 0
    for t, h, w in grids:
        n = t * h * w
        chunk = normed[offset:offset + n]
        offset += n
        hm = h // MERGE
        wm = w // MERGE
        chunk = chunk.view(t, hm, MERGE, wm, MERGE, HIDDEN).permute(0, 1, 3, 2, 4, 5)
        chunk = chunk.reshape(t * hm * wm, EXPANDED)
        pieces.append(chunk)
    shuffled = torch.cat(pieces, dim=0)

    h1 = F.linear(shuffled, fc1_weight, fc1_bias)
    h2 = F.gelu(h1)
    return F.linear(h2, fc2_weight, fc2_bias)
