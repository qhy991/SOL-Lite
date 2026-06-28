"""
High-performance Triton GEMM: C = A @ B.T
  A: [M, 7168] float16, M in 1..14104
  B: [256, 7168] float16
  C: [M, 256]   float16

Design
------
Two kernels + one reduce kernel:
  _gemm_direct_kernel  -- SPLIT_K=1, writes fp16 directly to output.
  _gemm_splitk_kernel  -- SPLIT_K>1, each split writes its fp32 partial
                          sum to a private slice of a pre-allocated workspace
                          (no atomics, no workspace zeroing needed).
  _reduce_workspace_kernel -- reduces the [SPLIT_K, M, N] fp32 workspace
                               into the fp16 output.

Dispatch table (tuned for A800 / sm_80, 108 SMs):
  M=1:          BM=1,  SK=64,  nw=4, ns=2
  M=2..8:       BM=M,  SK=56,  nw=4, ns=2
  M=9..16:      BM=8,  SK=64,  nw=4, ns=2
  M=17..64:     BM=8,  SK=28,  nw=8, ns=3
  M=65..128:    BM=16, SK=14,  nw=8, ns=3
  M=129..256:   BM=32, SK=8,   nw=8, ns=3
  M=257..512:   BM=32, SK=4,   nw=8, ns=3
  M=513..1024:  BM=16, SK=1,   nw=8, ns=4  (direct)
  M=1025..2048: BM=32, SK=1,   nw=8, ns=4  (direct)
  M=2049..4096: BM=64, SK=1,   nw=8, ns=4  (direct)
  M=4097..7168: BM=128,SK=1,   nw=16,ns=3  (direct)
  M>7168:       BM=64, SK=1,   nw=8, ns=3  (direct)

All accumulators run in fp32; outputs stored as fp16.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A: no split-K, writes fp16 directly to output
# ---------------------------------------------------------------------------

@triton.jit
def _gemm_direct_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Standard tiled GEMM (SPLIT_K=1).  Accumulates fp32, stores fp16."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_off < M
    n_mask = n_off < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    A_base = A_ptr + m_off[:, None] * stride_am
    B_base = B_ptr + n_off[:, None] * stride_bn

    k_off = tl.arange(0, BLOCK_K)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = k_off < K
        a = tl.load(
            A_base + k_off[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            B_base + k_off[None, :] * stride_bk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, tl.trans(b), acc=acc, out_dtype=tl.float32)
        k_off += BLOCK_K

    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(
        C_ptr + m_off[:, None] * stride_cm + n_off[None, :] * stride_cn,
        acc.to(tl.float16),
        mask=out_mask,
    )


# ---------------------------------------------------------------------------
# Kernel B: split-K GEMM, writes fp32 partial sums (non-overlapping slices)
# ---------------------------------------------------------------------------

@triton.jit
def _gemm_splitk_kernel(
    A_ptr, B_ptr, Workspace_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Split-K GEMM.  pid_sk owns K[k_start .. k_end) exclusively.
    Writes to Workspace[pid_sk, m_tile, n_tile] -- no atomic needed.
    """
    pid_m  = tl.program_id(0)
    pid_n  = tl.program_id(1)
    pid_sk = tl.program_id(2)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_off < M
    n_mask = n_off < N

    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_sk * k_per_split
    k_end   = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    A_base = A_ptr + m_off[:, None] * stride_am
    B_base = B_ptr + n_off[:, None] * stride_bn

    k_off = k_start + tl.arange(0, BLOCK_K)

    for _ in range(0, tl.cdiv(k_end - k_start, BLOCK_K)):
        k_mask = k_off < k_end
        a = tl.load(
            A_base + k_off[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            B_base + k_off[None, :] * stride_bk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, tl.trans(b), acc=acc, out_dtype=tl.float32)
        k_off += BLOCK_K

    # Write to private split-K slice (non-overlapping -> no atomic)
    out_mask = m_mask[:, None] & n_mask[None, :]
    ws_ptr = Workspace_ptr + pid_sk * M * N + m_off[:, None] * N + n_off[None, :]
    tl.store(ws_ptr, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Kernel C: reduce fp32 workspace -> fp16 output
# ---------------------------------------------------------------------------

@triton.jit
def _reduce_workspace_kernel(
    Workspace_ptr,   # [SPLIT_K, M, N] fp32
    C_ptr,           # [M, N]          fp16
    M, N,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One CTA per output tile; static_range unrolls the reduction."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (m_off[:, None] < M) & (n_off[None, :] < N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for sk in tl.static_range(0, SPLIT_K):
        ws = tl.load(
            Workspace_ptr + sk * M * N + m_off[:, None] * N + n_off[None, :],
            mask=out_mask,
            other=0.0,
        )
        acc += ws

    tl.store(
        C_ptr + m_off[:, None] * N + n_off[None, :],
        acc.to(tl.float16),
        mask=out_mask,
    )


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

# Contiguous fp32 workspace; allocated once per call and reused by reduce.
# Size = SPLIT_K * M * N * 4 bytes.  For all M values in 1..14104, this
# fits inside A800 L2 (40 MB), so the reduce kernel hits L2.

def _get_config(M: int):
    """Return (BLOCK_M, SPLIT_K, num_warps, num_stages) for given M.

    BLOCK_M must be a power of 2 (required by tl.arange).
    Tuned for A800 / sm_80 (108 SMs, 164 KB SMEM per SM).
    num_warps=8 selects faster MMA primitives vs num_warps=4 for BM>=16.
    """
    if M == 1:
        return   1,  64,  4, 2   #   1 tile, SK=64  ->  64 CTAs
    if M <=  2:
        return   2,  56,  4, 2   #   1 tile, SK=56  ->  56 CTAs
    if M <=  4:
        return   4,  56,  4, 2   #   1 tile, SK=56  ->  56 CTAs
    if M <=  8:
        return   8,  56,  4, 2   #   1 tile, SK=56  ->  56 CTAs
    if M <= 16:
        return   8,  64,  4, 2   #   2 tiles, SK=64 -> 128 CTAs
    if M <= 64:
        return   8,  28,  8, 3   # 2-8 tiles, SK=28 ->  56-224 CTAs
    if M <= 128:
        return  16,  14,  8, 3   #   8 tiles, SK=14 -> 112 CTAs
    if M <= 256:
        return  32,   8,  8, 3   #   8 tiles, SK=8  ->  64 CTAs
    if M <= 512:
        return  32,   4,  8, 3   #  16 tiles, SK=4  ->  64 CTAs
    if M <= 1024:
        return  16,   1,  8, 4   #  64 tiles, direct (SK=1)
    if M <= 2048:
        return  32,   1,  8, 4   #  64 tiles, direct
    if M <= 4096:
        return  64,   1,  8, 4   #  64 tiles, direct
    if M <= 7168:
        return 128,   1, 16, 3   #  56 tiles, direct, BM=128 nw=16
    return       64,  1,  8, 3   # 221 tiles, direct


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    C = A @ B.T
      A: [M, K] fp16
      B: [N, K] fp16
      returns C: [M, N] fp16
    """
    assert A.dtype == torch.float16
    assert B.dtype == torch.float16
    assert A.is_cuda and B.is_cuda

    M, K  = A.shape
    N, KB = B.shape
    assert K == KB, f"K mismatch: {A.shape} vs {B.shape}"

    BLOCK_N = N      # always 256; one tile covers all of N
    BLOCK_K = 64

    BLOCK_M, SPLIT_K, num_warps, num_stages = _get_config(M)

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)   # = 1

    C = torch.empty(M, N, dtype=torch.float16, device=A.device)

    if SPLIT_K == 1:
        _gemm_direct_kernel[(grid_m, grid_n)](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        # fp32 workspace: each split writes to its own [M, N] slice
        workspace = torch.empty(SPLIT_K, M, N, dtype=torch.float32, device=A.device)

        _gemm_splitk_kernel[(grid_m, grid_n, SPLIT_K)](
            A, B, workspace,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            SPLIT_K=SPLIT_K,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        _reduce_workspace_kernel[(grid_m, grid_n)](
            workspace, C,
            M, N,
            SPLIT_K=SPLIT_K,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=1,
        )

    return C


# ---------------------------------------------------------------------------
# Correctness smoke-test + timing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    test_cases = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 7168, 14104]
    print("Correctness check:")
    for M in test_cases:
        A = torch.randn(M, 7168, dtype=torch.float16, device=device)
        B = torch.randn(256,  7168, dtype=torch.float16, device=device)

        C_ref = (A.float() @ B.float().T).half()
        C_tri = run(A, B)

        max_err = (C_ref.float() - C_tri.float()).abs().max().item()
        rel_err = max_err / (C_ref.float().abs().max().item() + 1e-6)
        status  = "PASS" if rel_err < 0.01 else "FAIL"
        print(f"  M={M:6d}  max_abs={max_err:.4f}  rel={rel_err:.4f}  [{status}]")

    import time
    print("\nPerformance (100-iter average after warmup):")
    for M in [1, 8, 64, 256, 1024, 7168, 14104]:
        A = torch.randn(M, 7168, dtype=torch.float16, device=device)
        B = torch.randn(256,  7168, dtype=torch.float16, device=device)

        for _ in range(20):
            run(A, B)
        torch.cuda.synchronize()

        N_ITER = 100
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            run(A, B)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        ms     = (t1 - t0) / N_ITER * 1e3
        flops  = 2 * M * 256 * 7168
        tflops = flops / (ms * 1e-3) / 1e12
        BM, SK, nw, ns = _get_config(M)
        gm = triton.cdiv(M, BM)
        print(f"  M={M:6d}  {ms:.3f} ms  {tflops:.2f} TFLOPS  [BM={BM} SK={SK} nw={nw} ns={ns} CTAs={gm*SK}]")
