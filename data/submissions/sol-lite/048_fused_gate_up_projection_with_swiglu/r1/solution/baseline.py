import torch
import flashinfer.activation as fia

_BUF = {}


def _gated_buf(rows, inter, dtype, device):
    key = (rows, inter, dtype, device)
    buf = _BUF.get(key)
    if buf is None or buf.shape != (rows, 2 * inter):
        buf = torch.empty(rows, 2 * inter, dtype=dtype, device=device)
        _BUF[key] = buf
    return buf


@torch.no_grad()
def run(x, gate_proj, up_proj):
    b, s, h = x.shape
    inter = gate_proj.shape[0]
    x_flat = x.reshape(-1, h)
    rows = x_flat.shape[0]
    gated = _gated_buf(rows, inter, x.dtype, x.device)
    torch.matmul(x_flat, gate_proj.t(), out=gated[:, :inter])
    torch.matmul(x_flat, up_proj.t(), out=gated[:, inter:])
    out = fia.gelu_tanh_and_mul(gated)
    return out.reshape(b, s, inter)
