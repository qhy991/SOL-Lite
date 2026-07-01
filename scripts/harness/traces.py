"""Trace formatting — produce sol-execbench-compatible JSONL output.

Each trace line is a JSON object with:
  - definition, workload, evaluation (compatible with roofline_measure.py)
  - roofline block (regime-aware metrics)
"""
from __future__ import annotations

import json


def format_trace(
    definition_name: str,
    workload: dict,
    correctness: dict,
    timing: dict | None,
    roofline: dict | None,
) -> dict:
    """Build a single trace dict.

    Args:
        definition_name: e.g. "069_rms_norm"
        workload: the raw workload row dict (uuid, axes, tolerance, inputs)
        correctness: result from compute_error_stats()
        timing: result from time_kernel() or None if correctness failed
        roofline: result from Handler.evaluate() + measure() or None
    """
    trace = {
        "definition": definition_name,
        "workload": {
            "uuid": workload.get("uuid"),
            "axes": workload.get("axes", {}),
            "tolerance": workload.get("tolerance", {}),
        },
        "evaluation": {
            "status": correctness["status"],
            "error": correctness if correctness["status"] != "PASSED" else None,
        },
    }

    if timing is not None and correctness["status"] == "PASSED":
        latency_ms = timing["latency_us"] / 1000.0
        perf = {
            "latency_ms": latency_ms,
            "timing_mode": timing["mode"],
        }
        # Reference latency: if we're timing the reference itself, same value
        trace["evaluation"]["performance"] = perf

    if roofline is not None and correctness["status"] == "PASSED":
        trace["roofline"] = roofline

    return trace


def format_traces_jsonl(traces: list[dict]) -> str:
    """Serialize a list of trace dicts to JSONL string."""
    return "\n".join(json.dumps(t, ensure_ascii=False) for t in traces) + "\n"


def build_roofline_block(
    handler,
    axes: dict,
    t_measured_us: float,
    uuid: str | None = None,
    speedup_factor: float | None = None,
) -> dict:
    """Compute per-row roofline metrics from measured latency.

    Matches the schema produced by roofline_measure.py:measure().
    """
    from pathlib import Path
    import sys
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "scripts"))
    from _hardware import PEAK_BW, LATENCY_FLOOR_US

    a = handler.evaluate(axes, uuid=uuid)
    flops, bytes_, peak = a["flops"], a["bytes"], a["peak"]
    achieved_flops_per_s = flops / (t_measured_us * 1e-6) if t_measured_us > 0 else 0.0
    achieved_bytes_per_s = bytes_ / (t_measured_us * 1e-6) if t_measured_us > 0 else 0.0

    return {
        "regime": a["regime"],
        "cost_source": a.get("cost_source", "analytic"),
        "flops": flops,
        "bytes": bytes_,
        "ai": a["ai"],
        "peak_flops": peak,
        "peak_bw": PEAK_BW,
        "t_sol_us": a["t_sol_us"],
        "t_measured_us": t_measured_us,
        "achieved_tflops": achieved_flops_per_s / 1e12,
        "achieved_gbps": achieved_bytes_per_s / 1e9,
        "mfu": achieved_flops_per_s / peak if peak else 0.0,
        "mfu_ceiling": a["mfu_ceiling"],
        "bandwidth_utilization": achieved_bytes_per_s / PEAK_BW,
        "sol_efficiency": a["t_sol_us"] / t_measured_us if t_measured_us > 0 else 0.0,
        "speedup_vs_reference": speedup_factor,
        "below_latency_floor": a["t_sol_us"] < LATENCY_FLOOR_US,
    }