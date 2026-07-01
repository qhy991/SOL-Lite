"""Per-workload cost lookup for authoritative FLOPs/bytes data.

Loads Ray-234's precomputed workload_costs.jsonl (imported from
SoL-Contest-InfiniAI). Falls back to analytical formulas in the
analyzers when a UUID is missing.

Why per-UUID beats per-axes formulas:
  - MoE: routing distribution matters, and Ray-234's numbers assume
    L2-cache-aware weight reuse across tokens (analyzer formula
    over-counts by 30-150x for these problems)
  - Paged decode: bytes depend on which pages are actually indexed,
    not on the whole KV cache (analyzer over-counts by 30-70x)
  - Quant FP8: cost model assumes fp8 activations/weights internally
    even though declared dtype is bf16 (analyzer over-counts bytes 2x)
  - Elementwise/norm/RoPE: Ray-234 sets FLOPs=0 by convention (these
    are memory-bound and reporting a nonzero MFU is misleading);
    our analyzer computes them but they never affect regime

See DISAGREEMENTS.md for the full analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

# Data lives here in the SOL-Lite tree
_COSTS_ROOT = Path(__file__).resolve().parent.parent / "data" / "costs"
_DEFAULT_COSTS = _COSTS_ROOT / "ray234_h800.jsonl"


def _load(path: Path) -> dict[str, dict]:
    """UUID -> {kernel, flops, bytes_moved, precision, ...}"""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        uuid = d.get("workload_uuid")
        if uuid:
            out[uuid] = d
    return out


# Loaded once at import; small enough (~250 KB for 1019 entries) to keep in memory
COSTS: dict[str, dict] = _load(_DEFAULT_COSTS)


def lookup(uuid: str | None) -> dict | None:
    """Return the Ray-234 cost row for this workload UUID, or None."""
    if not uuid:
        return None
    return COSTS.get(uuid)


def has_costs() -> bool:
    return len(COSTS) > 0


def summary() -> str:
    return f"loaded {len(COSTS)} per-UUID cost rows from {_DEFAULT_COSTS.name}"


if __name__ == "__main__":
    print(summary())
