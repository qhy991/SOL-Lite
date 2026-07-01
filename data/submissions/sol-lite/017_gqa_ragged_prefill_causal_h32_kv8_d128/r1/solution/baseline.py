import torch
import flashinfer

_WS = None
_W = None


def _get_wrapper(device):
    global _WS, _W
    if _W is None:
        _WS = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        _W = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(_WS, 'NHD')
    return _W


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    w = _get_wrapper(q.device)
    w.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim_qk=128,
        head_dim_vo=128,
        causal=True,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        sm_scale=sm_scale,
    )
    output, lse = w.run(q, k, v, return_lse=True)
    return output, lse.float()
