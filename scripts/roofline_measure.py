"""Roofline measurement feedback: join sol-execbench traces with the offline
roofline analysis, compute achieved MFU / BW% / SoL efficiency per row.

Usage:
    # Process a trace JSONL produced by sol-execbench --output:
    uv run scripts/roofline_measure.py traces.jsonl -o measured.jsonl

    # Print per-problem summary table:
    uv run scripts/roofline_measure.py traces.jsonl --report

Trace schema (one JSON object per line):
    {
      "definition": "069_rms_norm",
      "workload": {"uuid": "...", "axes": {...}, ...},
      "evaluation": {
        "status": "PASSED",
        "performance": {"latency_ms": 3.21, "reference_latency_ms": 8.50,
                        "speedup_factor": 2.65},
        ...
      }
    }

Output JSONL adds a `roofline` block per row:
    {
      ...,
      "roofline": {
        "regime": "memory",
        "flops": 4.29e9, "bytes": 5.37e8,
        "ai": 7.99,
        "peak_flops": 9.89e14, "peak_bw": 3.35e12,
        "t_sol_us": 160.30,
        "achieved_tflops": 13.4, "achieved_gbps": 1672,
        "mfu": 0.0136, "mfu_ceiling": 0.024,
        "bandwidth_utilization": 0.499,
        "sol_efficiency": 0.501,
        "speedup_vs_reference": 2.65   # echoed from trace
      }
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import roofline_tier1_batch as t1
import roofline_moe as t2
import roofline_l2 as t3


from _hardware import PEAK_BW, LATENCY_FLOOR_US
import _costs


# ---------------------------------------------------------------------------
# Build a name -> handler registry from all three analyzers.
# Each problem's `subdir` ends in the full problem-name (e.g. "L1/069_rms_norm").
# The trace's `definition` field is the bare name without category prefix.
# ---------------------------------------------------------------------------
def _name(subdir: str) -> str:
    return subdir.split("/", 1)[1]


class Handler:
    """Wraps a problem's flops/bytes computation regardless of tier."""

    def __init__(self, kind: str, subdir: str, compute_fn, peak: float | None = None,
                 note: str = ""):
        self.kind = kind          # 'tier1' | 'moe-sim' | 'moe-det' | 'l2'
        self.subdir = subdir
        self.compute_fn = compute_fn
        self.peak = peak
        self.note = note

    def evaluate(self, axes: dict, uuid: str | None = None) -> dict:
        """Return (flops, bytes, peak, regime, t_sol_us, mfu_ceiling, cost_source).

        If a workload UUID is supplied and Ray-234 per-UUID costs are
        loaded, use those authoritative (flops, bytes, precision) values —
        our analytical formulas are known to over-count for MoE (30-150x,
        L2-cache-aware), Quant FP8 (2x, wrong dtype byte-size), and paged
        decode (30-70x, whole-cache vs indexed pages). See DISAGREEMENTS.md.

        Falls back to the analytical formula when uuid is missing or has
        no cost entry.
        """
        cost = _costs.lookup(uuid)
        if cost is not None:
            # Authoritative Ray-234 numbers. Resolve peak from declared precision.
            flops  = float(cost.get("flops") or 0.0)
            bytes_ = float(cost.get("bytes_moved") or 0.0)
            prec   = (cost.get("precision") or "bf16").lower()
            # Import lazily so hardware overrides propagate correctly
            from _hardware import PEAK_BF16, PEAK_FP8
            peak = PEAK_FP8 if prec.startswith("fp8") or prec.startswith("float8") else PEAK_BF16
            cost_source = "ray234"
        elif self.kind == "tier1":
            flops, bytes_, peak = self.compute_fn(axes); cost_source = "analytic"
        elif self.kind == "moe-sim":
            r = self.compute_fn.analyze_row(axes)
            flops, bytes_, peak = r["flops"], r["bytes"], self.compute_fn.peak
            cost_source = "analytic"
        elif self.kind == "moe-det":
            r = self.compute_fn.analyze_row(axes)
            flops, bytes_, _peak = self.compute_fn.fn(axes)
            peak = _peak; cost_source = "analytic"
        elif self.kind == "l2":
            ops = self.compute_fn.decompose(axes)
            results = [op.metrics() for op in ops]
            flops = sum(x["flops"] for x in results)
            bytes_ = sum(x["bytes"] for x in results)
            peak = t3.PEAK_BF16; cost_source = "analytic"
        else:
            raise ValueError(f"unknown kind: {self.kind}")

        ai = flops / bytes_ if bytes_ else 0.0
        t_sol_us = max(flops/peak, bytes_/PEAK_BW) * 1e6
        ridge = peak / PEAK_BW
        if t_sol_us < LATENCY_FLOOR_US:
            regime = "latency"
        elif ai > 2*ridge:    regime = "compute"
        elif ai < 0.5*ridge:  regime = "memory"
        else:                 regime = "balanced"
        mfu_ceiling = min(1.0, PEAK_BW * ai / peak) if peak else 0.0
        return dict(flops=flops, bytes=bytes_, peak=peak, ai=ai,
                    t_sol_us=t_sol_us, regime=regime, mfu_ceiling=mfu_ceiling,
                    cost_source=cost_source)


def build_registry() -> dict[str, Handler]:
    reg: dict[str, Handler] = {}
    for prob in t1.PROBLEMS:
        reg[_name(prob.subdir)] = Handler("tier1", prob.subdir, prob.fn, note=prob.note)
    for spec in t2.SIM_PROBLEMS:
        reg[_name(spec.subdir)] = Handler("moe-sim", spec.subdir, spec, note=spec.note)
    for spec in t2.DET_PROBLEMS:
        reg[_name(spec.subdir)] = Handler("moe-det", spec.subdir, spec, note=spec.note)
    for prob in t3.PROBLEMS:
        reg[_name(prob.subdir)] = Handler("l2", prob.subdir, prob, note=prob.note)
    return reg


# ---------------------------------------------------------------------------
# Per-row measurement: take t_measured_us from trace, compute achieved metrics
# ---------------------------------------------------------------------------
def measure(handler: Handler, axes: dict, t_measured_us: float,
            speedup_factor: float | None = None,
            uuid: str | None = None) -> dict:
    a = handler.evaluate(axes, uuid=uuid)
    flops, bytes_, peak = a["flops"], a["bytes"], a["peak"]
    achieved_flops_per_s = flops / (t_measured_us * 1e-6) if t_measured_us > 0 else 0.0
    achieved_bytes_per_s = bytes_ / (t_measured_us * 1e-6) if t_measured_us > 0 else 0.0
    return {
        "regime":               a["regime"],
        "cost_source":          a.get("cost_source", "analytic"),
        "flops":                flops,
        "bytes":                bytes_,
        "ai":                   a["ai"],
        "peak_flops":           peak,
        "peak_bw":              PEAK_BW,
        "t_sol_us":             a["t_sol_us"],
        "t_measured_us":        t_measured_us,
        "achieved_tflops":      achieved_flops_per_s / 1e12,
        "achieved_gbps":        achieved_bytes_per_s / 1e9,
        "mfu":                  achieved_flops_per_s / peak if peak else 0.0,
        "mfu_ceiling":          a["mfu_ceiling"],
        "bandwidth_utilization": achieved_bytes_per_s / PEAK_BW,
        "sol_efficiency":       a["t_sol_us"] / t_measured_us if t_measured_us > 0 else 0.0,
        "speedup_vs_reference": speedup_factor,
        "below_latency_floor":  a["t_sol_us"] < LATENCY_FLOOR_US,
    }


# ---------------------------------------------------------------------------
# Trace processing
# ---------------------------------------------------------------------------
def process_trace(trace_path: Path, registry: dict[str, Handler]):
    """Yield (trace_dict, roofline_dict_or_None, error_message_or_None) per line."""
    for line in trace_path.read_text().splitlines():
        line = line.strip()
        if not line: continue
        try:
            trace = json.loads(line)
        except json.JSONDecodeError as e:
            yield None, None, f"bad JSON: {e}"
            continue
        defn = trace.get("definition", "")
        evaluation = trace.get("evaluation") or {}
        status = evaluation.get("status")
        perf = (evaluation.get("performance") or {})
        latency_ms = perf.get("latency_ms")
        speedup = perf.get("speedup_factor")
        workload = trace.get("workload") or {}
        axes = workload.get("axes")
        uuid = workload.get("uuid")

        if status != "PASSED" or latency_ms is None or axes is None:
            yield trace, None, f"skip ({status})"
            continue
        handler = registry.get(defn)
        if handler is None:
            yield trace, None, f"no analyzer for definition '{defn}'"
            continue
        try:
            r = measure(handler, axes, latency_ms * 1000, speedup, uuid=uuid)
        except Exception as e:
            yield trace, None, f"analyzer error on '{defn}': {e}"
            continue
        yield trace, r, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_process(args):
    registry = build_registry()
    out_lines = []
    counts = {"ok": 0, "skip": 0, "error": 0}
    per_problem: dict[str, list] = defaultdict(list)

    for trace, r, err in process_trace(Path(args.trace), registry):
        if trace is None:
            counts["error"] += 1
            continue
        if err is not None and r is None:
            counts["skip"] += 1
            if args.verbose:
                print(f"[skip] {trace.get('definition','?')}: {err}", file=sys.stderr)
            if args.output:
                out_lines.append(json.dumps(trace))
            continue
        counts["ok"] += 1
        trace["roofline"] = r
        per_problem[trace["definition"]].append(r)
        if args.output:
            out_lines.append(json.dumps(trace))

    if args.output:
        Path(args.output).write_text("\n".join(out_lines) + "\n")
        print(f"Wrote {args.output}  (ok={counts['ok']} skip={counts['skip']} error={counts['error']})")

    if args.report or not args.output:
        _print_report(per_problem)


def _print_report(per_problem: dict[str, list]) -> None:
    print()
    print("=" * 110)
    print(f"{'problem':<48} {'rows':>5} {'regime':>10} "
          f"{'mfu':>8} {'bw%':>8} {'SoL':>6} {'speedup':>8}")
    print("=" * 110)
    for name in sorted(per_problem):
        rows = per_problem[name]
        regime_counts = defaultdict(int)
        for r in rows:
            regime_counts[r["regime"]] += 1
        dom = max(regime_counts, key=regime_counts.get)
        # Aggregate: geomean is more robust than mean for ratios
        def gmean(xs):
            xs = [x for x in xs if x and x > 0]
            if not xs: return 0.0
            import math
            return math.exp(sum(math.log(x) for x in xs) / len(xs))
        mfu = gmean([r["mfu"] for r in rows])
        bw  = gmean([r["bandwidth_utilization"] for r in rows])
        sol = gmean([r["sol_efficiency"] for r in rows])
        spd = gmean([r["speedup_vs_reference"] or 0 for r in rows])
        print(f"{name[:48]:<48} {len(rows):>5} {dom:>10} "
              f"{mfu:>8.3f} {bw:>8.3f} {sol:>6.2f} {spd:>8.2f}x")
    print("=" * 110)
    print()
    print("Note: aggregates are geometric means across workload rows.")
    print("  - 'regime' = dominant per-row regime")
    print("  - 'mfu', 'bw%', 'SoL' are achieved fractions in [0, 1]")
    print("  - 'speedup' = solution latency vs. reference latency")
    print("  - Read per-row JSONL for full breakdown including mfu_ceiling per row.")


def cmd_offline_one(args):
    """Standalone: compute analytical (flops, bytes, regime, t_sol) for one
    (problem-name, axes) without needing a trace. Useful for sanity-checking
    the analyzer output before running sol-execbench."""
    registry = build_registry()
    h = registry.get(args.problem)
    if h is None:
        print(f"ERROR: unknown problem '{args.problem}'. Try --list.", file=sys.stderr)
        sys.exit(2)
    try:
        axes = json.loads(args.axes)
    except json.JSONDecodeError as e:
        print(f"ERROR: --axes must be valid JSON: {e}", file=sys.stderr)
        sys.exit(2)
    r = h.evaluate(axes)
    print(json.dumps(r, indent=2))


def cmd_list(args):
    registry = build_registry()
    for name, h in sorted(registry.items()):
        print(f"{h.kind:<8} {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    import _hardware; _hardware.add_hardware_arg(p)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("process", help="Process a sol-execbench trace JSONL")
    pp.add_argument("trace", help="Input trace JSONL (one Trace JSON per line)")
    pp.add_argument("-o", "--output", help="Output augmented JSONL")
    pp.add_argument("--report", action="store_true", help="Print summary table")
    pp.add_argument("-v", "--verbose", action="store_true")
    pp.set_defaults(func=cmd_process)

    po = sub.add_parser("offline", help="Compute analytical roofline for one row")
    po.add_argument("problem", help="Problem name e.g. '069_rms_norm'")
    po.add_argument("axes", help='JSON axes e.g. \'{"batch_size":1,"seq_len":256}\'')
    po.set_defaults(func=cmd_offline_one)

    pl = sub.add_parser("list", help="List known problem names")
    pl.set_defaults(func=cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
