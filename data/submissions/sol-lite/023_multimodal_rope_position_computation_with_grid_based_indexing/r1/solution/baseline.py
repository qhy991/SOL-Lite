import torch

MERGE = 2
NUM_GRID = 35
MROPE_SECTION = [24, 20, 20]


@torch.no_grad()
def run(grid_thw, pos_embed_weight, inv_freq):
    device = grid_thw.device
    dtype = pos_embed_weight.dtype
    grid_ts = grid_thw[:, 0].tolist()
    grid_hs = grid_thw[:, 1].tolist()
    grid_ws = grid_thw[:, 2].tolist()
    head_dim = inv_freq.shape[0] * 4

    # ---- Part 1: per-grid bilinear interp on GPU, then torch.cat ----
    idx_lists = [[], [], [], []]
    w_lists = [[], [], [], []]
    for h, w in zip(grid_hs, grid_ws):
        h_idxs = torch.linspace(0, NUM_GRID - 1, h, device=device)
        w_idxs = torch.linspace(0, NUM_GRID - 1, w, device=device)
        h_f = h_idxs.long()
        w_f = w_idxs.long()
        h_c = (h_f + 1).clamp(max=NUM_GRID - 1)
        w_c = (w_f + 1).clamp(max=NUM_GRID - 1)
        dh = h_idxs - h_f.float()
        dw = w_idxs - w_f.float()
        base_f = h_f * NUM_GRID
        base_c = h_c * NUM_GRID
        idx_lists[0].append((base_f[:, None] + w_f[None, :]).flatten())
        idx_lists[1].append((base_f[:, None] + w_c[None, :]).flatten())
        idx_lists[2].append((base_c[:, None] + w_f[None, :]).flatten())
        idx_lists[3].append((base_c[:, None] + w_c[None, :]).flatten())
        w_lists[0].append(((1 - dh)[:, None] * (1 - dw)[None, :]).flatten().to(dtype))
        w_lists[1].append(((1 - dh)[:, None] * dw[None, :]).flatten().to(dtype))
        w_lists[2].append((dh[:, None] * (1 - dw)[None, :]).flatten().to(dtype))
        w_lists[3].append((dh[:, None] * dw[None, :]).flatten().to(dtype))

    idx_tensor = torch.stack([torch.cat(idx_lists[i]) for i in range(4)], dim=0)
    weight_tensor = torch.stack([torch.cat(w_lists[i]) for i in range(4)], dim=0)
    pos_embeds = pos_embed_weight[idx_tensor] * weight_tensor[:, :, None]
    patch_sum = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    splits = [h * w for h, w in zip(grid_hs, grid_ws)]
    patch_parts = patch_sum.split(splits)

    permuted = []
    for pos_embed, t, h, w in zip(patch_parts, grid_ts, grid_hs, grid_ws):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = pos_embed.view(t, h // MERGE, MERGE, w // MERGE, MERGE, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
        permuted.append(pos_embed)
    patch_pos_embeds = torch.cat(permuted)

    # ---- Part 2: MRoPE ----
    max_hw = max(max(grid_hs), max(grid_ws))
    seq = torch.arange(max_hw, device=device, dtype=torch.float32)
    freqs = torch.outer(seq, inv_freq)

    total_tokens = sum(t * h * w for t, h, w in zip(grid_ts, grid_hs, grid_ws))
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
    offset = 0
    for t, h, w in zip(grid_ts, grid_hs, grid_ws):
        mh = h // MERGE
        mw = w // MERGE
        br = torch.arange(mh, device=device)
        bc = torch.arange(mw, device=device)
        ir = torch.arange(MERGE, device=device)
        ic = torch.arange(MERGE, device=device)
        row = br[:, None, None, None] * MERGE + ir[None, None, :, None]
        col = bc[None, :, None, None] * MERGE + ic[None, None, None, :]
        row = row.expand(mh, mw, MERGE, MERGE).reshape(-1)
        col = col.expand(mh, mw, MERGE, MERGE).reshape(-1)
        coords = torch.stack((row, col), dim=-1)
        if t > 1:
            coords = coords.repeat(t, 1)
        n = coords.shape[0]
        pos_ids[offset:offset + n] = coords
        offset += n

    embeddings = freqs[pos_ids].flatten(1)
    freqs_3d = embeddings.unsqueeze(0).expand(3, -1, -1).clone()
    freqs_t = freqs_3d[0]
    for dim, off_dim in enumerate((1, 2), start=1):
        length = MROPE_SECTION[dim] * 3
        idx = slice(off_dim, length, 3)
        freqs_t[..., idx] = freqs_3d[dim, ..., idx]
    emb = torch.cat((freqs_t, freqs_t), dim=-1)
    return patch_pos_embeds, emb.cos(), emb.sin()
