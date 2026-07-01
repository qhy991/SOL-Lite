import torch
import flashinfer

_WS = None
_W = None


def _get_wrapper(device):
    global _WS, _W
    if _W is None:
        _WS = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        _W = flashinfer.mla.BatchMLAPagedAttentionWrapper(_WS)
    return _W


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    dev = q_nope.device
    w = _get_wrapper(dev)
    kv_len_arr = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    w.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=kv_len_arr,
        num_heads=16,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=1,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=q_nope.dtype,
        kv_data_type=ckv_cache.dtype,
    )
    output, lse = w.run(q_nope, q_pe, ckv_cache, kpe_cache, return_lse=True)
    return output, lse.float()
