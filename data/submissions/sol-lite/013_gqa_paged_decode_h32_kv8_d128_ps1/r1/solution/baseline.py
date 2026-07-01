import math
import torch
import flashinfer

_WS = None
_W = None


def _get_wrapper(device):
    global _WS, _W
    if _W is None:
        _WS = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        _W = flashinfer.BatchDecodeWithPagedKVCacheWrapper(_WS, 'NHD')
    return _W


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    b = q.shape[0]
    w = _get_wrapper(q.device)
    last_page_len = torch.ones(b, dtype=torch.int32, device=q.device)
    w.plan(
        indptr=kv_indptr,
        indices=kv_indices,
        last_page_len=last_page_len,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=1,
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
        sm_scale=sm_scale,
    )
    output, lse = w.run(q, (k_cache, v_cache), return_lse=True)
    return output, lse.float()
