"""ShiftingMemoryPoolAllocator — avoid cudaMalloc inside timing loops.

Ported from sol-execbench timing.py. Pre-allocates a pool of identically-shaped
tensors and rotates data_ptr each iteration so the kernel writes to fresh
memory without calling cudaMalloc.
"""
from __future__ import annotations


class ShiftingMemoryPoolAllocator:
    """Pre-allocate N identically-shaped tensors, rotate through them.

    Usage::

        pool = ShiftingMemoryPoolAllocator(shape, dtype, device, n=3)
        for i in range(50):
            inputs = pool.next()  # returns tensors pointing to pool slot i % n
            kernel(**inputs)
    """

    def __init__(self, shape, dtype, device, n: int = 3):
        import torch

        self._n = n
        self._idx = 0
        self._pool = [
            torch.empty(shape, dtype=dtype, device=device)
            for _ in range(n)
        ]
        self._base_ptr = self._pool[0].data_ptr()

    def next(self) -> "torch.Tensor":
        """Return a tensor view into the next pool slot (no allocation)."""
        import torch
        t = self._pool[self._idx % self._n]
        self._idx += 1
        return t

    def reset(self):
        self._idx = 0


class MultiTensorPool:
    """Manage a pool for each of several named tensors.

    Usage::

        pool = MultiTensorPool({
            "hidden_states": (shape, dtype),
            "residual": (shape, dtype),
        }, device="cuda", n=3)
        for i in range(50):
            kwargs = pool.next()
            kernel(**kwargs)
    """

    def __init__(self, specs: dict[str, tuple], device: str, n: int = 3):
        self._pools = {
            name: ShiftingMemoryPoolAllocator(*spec, device, n=n)
            for name, spec in specs.items()
        }
        self._keys = list(specs.keys())

    def next(self) -> dict:
        return {k: self._pools[k].next() for k in self._keys}

    def reset(self):
        for p in self._pools.values():
            p.reset()