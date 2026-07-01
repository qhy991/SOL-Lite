import torch

NQ = 4
NKV = 1


@torch.no_grad()
def run(query, key, scaling):
    b, _, slen, d = key.shape
    n_rep = NQ // NKV
    k_expanded = key[:, :, None, :, :].expand(b, NKV, n_rep, slen, d).reshape(b, NQ, slen, d)
    return (torch.matmul(query, k_expanded.transpose(2, 3)) * scaling).to(query.dtype)
