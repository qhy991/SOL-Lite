"""Hardware-ceiling constants. Selected via env var SOL_LITE_HARDWARE.

Default is H800. Override with:
    SOL_LITE_HARDWARE=B200 uv run python scripts/roofline_tier1_batch.py
or pass --hardware B200 to any CLI entry point (which sets the env var
before importing the analyzers).

Numbers are dense Tensor-Core peaks (no sparsity) for the listed dtypes,
plus the HBM peak bandwidth. Sources are vendor datasheets / Wikipedia GPU
spec pages as of 2026-06.

  HBM_PEAK   : peak DRAM bandwidth (bytes/s)
  TC_BF16    : peak BF16 / FP16 Tensor Core throughput (FLOPs/s)
  TC_FP8     : peak FP8 Tensor Core throughput (FLOPs/s)
  RIDGE_BF16 : TC_BF16 / HBM_PEAK   (FLOPs/byte)
  RIDGE_FP8  : TC_FP8  / HBM_PEAK   (FLOPs/byte)

The "compute / memory / balanced / latency" regime thresholds use these.
"""
from __future__ import annotations

import os
import sys

# Pre-parse --hardware from sys.argv so the env var is set BEFORE the
# constants below are evaluated. This means a CLI can simply
#   `from _hardware import PEAK_BF16, ...`
# and any user-passed `--hardware B200` flag is honoured at import time —
# even if the importing module declares --hardware in its own argparse
# (the argparse step happens later, after import).
for _i, _a in enumerate(sys.argv):
    if _a == "--hardware" and _i + 1 < len(sys.argv):
        os.environ["SOL_LITE_HARDWARE"] = sys.argv[_i + 1]
        break
    if _a.startswith("--hardware="):
        os.environ["SOL_LITE_HARDWARE"] = _a.split("=", 1)[1]
        break

# Preset table — dense Tensor-Core peaks, no sparsity assumptions
# Keys mirror common GPU names; SOL_LITE_HARDWARE matches by uppercase.
PRESETS: dict[str, dict[str, float]] = {
    "H800":     dict(TC_BF16=989e12,  TC_FP8=1979e12, HBM=3.35e12, L2=50e6),   # SXM5, the SOL-Lite default
    "H800_PCIE":dict(TC_BF16=756e12,  TC_FP8=1513e12, HBM=2.00e12, L2=50e6),
    "H100":     dict(TC_BF16=989e12,  TC_FP8=1979e12, HBM=3.35e12, L2=50e6),   # SXM5
    "H100_PCIE":dict(TC_BF16=756e12,  TC_FP8=1513e12, HBM=2.00e12, L2=50e6),
    "H200":     dict(TC_BF16=989e12,  TC_FP8=1979e12, HBM=4.80e12, L2=60e6),   # SXM5, HBM3e
    "B200":     dict(TC_BF16=2250e12, TC_FP8=4500e12, HBM=8.00e12, L2=100e6),  # SXM5, HBM3e (dense)
    "A100":     dict(TC_BF16=312e12,  TC_FP8=0,       HBM=2.04e12, L2=40e6),   # SXM4 80GB
}

LATENCY_FLOOR_US = 5.0


def _resolve() -> tuple[str, dict[str, float]]:
    name = os.environ.get("SOL_LITE_HARDWARE", "H800").upper().replace("-", "_")
    if name not in PRESETS:
        raise ValueError(
            f"Unknown SOL_LITE_HARDWARE={name!r}. Known: {sorted(PRESETS)}"
        )
    return name, PRESETS[name]


HARDWARE_NAME, _peaks = _resolve()
PEAK_BF16 = _peaks["TC_BF16"]
PEAK_FP8  = _peaks["TC_FP8"]
PEAK_BW   = _peaks["HBM"]
L2_SIZE   = _peaks.get("L2", 50e6)   # bytes; for L2-cache-aware bytes accounting

# Derived ridges
RIDGE_BF16 = PEAK_BF16 / PEAK_BW if PEAK_BW else 0.0
RIDGE_FP8  = PEAK_FP8  / PEAK_BW if (PEAK_BW and PEAK_FP8) else 0.0


def banner() -> str:
    """One-line summary suitable for printing at CLI start."""
    return (f"hardware={HARDWARE_NAME}  "
            f"BF16={PEAK_BF16/1e12:.0f} TFLOPS  "
            f"FP8={PEAK_FP8/1e12:.0f} TFLOPS  "
            f"HBM={PEAK_BW/1e12:.2f} TB/s  "
            f"ridge_bf16={RIDGE_BF16:.0f}  ridge_fp8={RIDGE_FP8:.0f}")


# CLI helper: add --hardware to a parser and apply BEFORE importing analyzers.
def add_hardware_arg(parser):
    parser.add_argument("--hardware", default=None, choices=list(PRESETS),
                        help=f"override H800 default (env: SOL_LITE_HARDWARE). "
                             f"Available: {' '.join(PRESETS)}")


def apply_hardware_from_args(args) -> None:
    """Call BEFORE importing roofline_* analyzers if --hardware was passed.

    Sets the env var and reloads this module's constants. Because the
    analyzers do `from _hardware import PEAK_*`, they take the value at
    THEIR import time — so this must run before they are imported.
    """
    hw = getattr(args, "hardware", None)
    if hw is None:
        return
    os.environ["SOL_LITE_HARDWARE"] = hw
    # Refresh this module's module-level constants for callers that hold a
    # reference to it (e.g. CLI that imports add_hardware_arg).
    global HARDWARE_NAME, PEAK_BF16, PEAK_FP8, PEAK_BW, RIDGE_BF16, RIDGE_FP8, L2_SIZE
    HARDWARE_NAME, _p = _resolve()
    PEAK_BF16 = _p["TC_BF16"]
    PEAK_FP8  = _p["TC_FP8"]
    PEAK_BW   = _p["HBM"]
    L2_SIZE   = _p.get("L2", 50e6)
    RIDGE_BF16 = PEAK_BF16 / PEAK_BW if PEAK_BW else 0.0
    RIDGE_FP8  = PEAK_FP8  / PEAK_BW if (PEAK_BW and PEAK_FP8) else 0.0


if __name__ == "__main__":
    # Print the resolved hardware constants for the current SOL_LITE_HARDWARE.
    print(banner())
    print()
    print("All presets:")
    for k in sorted(PRESETS):
        p = PRESETS[k]
        ridge_bf16 = p["TC_BF16"] / p["HBM"] if p["HBM"] else 0
        ridge_fp8 = p["TC_FP8"] / p["HBM"] if p["HBM"] and p["TC_FP8"] else 0
        print(f"  {k:<10} BF16={p['TC_BF16']/1e12:>6.0f} TFLOPS  "
              f"FP8={p['TC_FP8']/1e12:>6.0f} TFLOPS  "
              f"HBM={p['HBM']/1e12:>5.2f} TB/s  "
              f"ridge_bf16={ridge_bf16:>4.0f}  ridge_fp8={ridge_fp8:>4.0f}")
