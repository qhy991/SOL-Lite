"""Standalone roofline benchmark — time a solution kernel directly without
going through sol-execbench, then classify regime and report achieved metrics.

Timing methodology borrowed from SOLBench-H800 (qhy991/runboo-fly):
  - `--warmup` untimed calls to absorb JIT / autotune / cuBLAS-algo cold cost
  - `--groups` timed groups; each group enqueues `--n-batch` back-to-back
    kernel launches between two cuda.Events with NO sync in between.
    Host dispatch overlaps GPU execution, so elapsed/n_batch is the
    launch-amortised per-call latency. Median across groups.

What we add on top of that timing engine:
  - Joins each row to its analytical (flops, bytes, peak, regime) via the
    same registry as roofline_measure.py — so the report is regime-aware
    (MFU only when meaningful; BW% only when meaningful; mfu_ceiling shown
    when the physical bound is < 1.0).
  - Reads the problem's workload.jsonl + reference.py from the SOL-Lite
    benchmark tree (no need for the user to write a bench_spec.py).
  - The 'solution' is either the reference (default — sanity check) or a
    user-provided Python file with a `run(...)` function matching the
    reference signature.

Usage:
    # Time the reference impl across all workloads of one problem
    uv run python scripts/roofline_bench.py L1/069_rms_norm

    # Smoke test (small/mid/large only)
    uv run python scripts/roofline_bench.py L1/069_rms_norm --smoke

    # Time a custom solution.py
    uv run python scripts/roofline_bench.py L1/069_rms_norm \
        --solution path/to/my_kernel.py

    # CSV output
    uv run python scripts/roofline_bench.py L1/069_rms_norm \
        --smoke -o results.csv

Requires CUDA + PyTorch. Falls back to a clean error if CUDA unavailable.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Lazy torch import so `--help` works on a CPU-only machine.
def _need_torch():
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install per https://pytorch.org/get-started",
              file=sys.stderr)
        sys.exit(2)
    return torch


# ---------------------------------------------------------------------------
# Timing engine (back-to-back launch — borrowed from SOLBench-H800/harness.py)
# ---------------------------------------------------------------------------
def time_kernel(call_fn, n_batch: int, warmup: int, groups: int) -> float:
    """Returns median per-call latency in microseconds."""
    torch = _need_torch()
    for _ in range(warmup):
        call_fn()
    torch.cuda.synchronize()
    per_call_us = []
    for _ in range(groups):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_batch):
            call_fn()
        e.record()
        torch.cuda.synchronize()
        # elapsed_time returns ms; convert to us and divide by n_batch
        per_call_us.append(s.elapsed_time(e) * 1e3 / n_batch)
    return statistics.median(per_call_us)


# ---------------------------------------------------------------------------
# Problem loading: workload.jsonl + reference.py (or user solution)
# ---------------------------------------------------------------------------
CONTEST_ROOT = ROOT / "data" / "benchmark" / "Contest"
DTYPE_MAP = {
    "float32": "float32", "float16": "float16", "bfloat16": "bfloat16",
    "float8_e4m3fn": "float8_e4m3fn", "float8_e5m2": "float8_e5m2",
    "int8": "int8", "int32": "int32", "int64": "int64", "bool": "bool",
}


def load_problem(path: str) -> tuple[Path, dict]:
    """Resolve a problem path (relative to data/benchmark/Contest/ or absolute).
    Returns (problem_dir, definition_dict)."""
    candidates = [
        Path(path),
        CONTEST_ROOT / path,
    ]
    for c in candidates:
        if c.is_dir() and (c / "definition.json").exists():
            defn = json.loads((c / "definition.json").read_text())
            return c, defn
    raise FileNotFoundError(
        f"could not find problem at {path}. Tried: {[str(c) for c in candidates]}")


def import_run_fn(py_path: Path, label: str):
    """Import a Python file and return its `run` function plus optional get_inputs."""
    spec = importlib.util.spec_from_file_location(f"_runner_{label}", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run"):
        raise AttributeError(f"{py_path} does not export `run`")
    return mod.run, getattr(mod, "get_inputs", None)


def generate_inputs(defn: dict, axes: dict, get_inputs_fn, device: str):
    """Produce a kwargs dict ready to pass to run().

    If the definition declares `custom_inputs_entrypoint`, use it. Otherwise
    synthesize random tensors per declared shape + dtype.
    """
    torch = _need_torch()
    if get_inputs_fn is not None and defn.get("custom_inputs_entrypoint"):
        # Custom inputs entrypoint - merge axes with consts + exprs
        axes_and_scalars = dict(axes)
        for k, v in defn["axes"].items():
            if v.get("type") == "const":
                axes_and_scalars.setdefault(k, v["value"])
        # Evaluate expression axes (often used by custom get_inputs)
        for k, v in defn["axes"].items():
            if v.get("type") == "expr" and k not in axes_and_scalars:
                try:
                    axes_and_scalars[k] = eval(v["expression"], {}, axes_and_scalars)
                except Exception:
                    pass
        return get_inputs_fn(axes_and_scalars, torch.device(device))

    # Synthesize random inputs from the spec.
    # Build the full axis namespace: var (from row) + const + expr (evaluated)
    namespace = dict(axes)
    for k, v in defn["axes"].items():
        if v.get("type") == "const":
            namespace.setdefault(k, v["value"])
    # Iteratively resolve expressions (one expr may reference another)
    pending = {k: v["expression"] for k, v in defn["axes"].items()
               if v.get("type") == "expr"}
    progress = True
    while pending and progress:
        progress = False
        for k in list(pending):
            try:
                namespace[k] = eval(pending[k], {}, namespace)
                del pending[k]
                progress = True
            except (NameError, SyntaxError):
                continue
    consts = namespace

    out = {}
    for k, v in defn["inputs"].items():
        shape_spec = v.get("shape")
        dtype_str = v.get("dtype")
        if shape_spec is None:
            # scalar — must come from workload's "inputs" section
            continue
        # Resolve symbolic shape against the merged namespace
        shape = []
        for s in shape_spec:
            if isinstance(s, int): shape.append(s); continue
            if s in consts: shape.append(consts[s]); continue
            if s.isdigit(): shape.append(int(s));    continue
            # Last attempt: eval expression in namespace
            try:
                shape.append(int(eval(s, {}, consts)))
            except Exception:
                raise KeyError(f"unresolved shape axis '{s}' for input '{k}' "
                               f"(known: {sorted(consts)})")
        dtype = getattr(torch, DTYPE_MAP.get(dtype_str, dtype_str))
        # Use randn for floats, zeros for ints, ones for bool
        if dtype_str.startswith("float") or dtype_str.startswith("bfloat"):
            if dtype_str.startswith("float8"):
                out[k] = torch.randn(*shape, dtype=torch.float32,
                                     device=device).to(dtype)
            else:
                out[k] = torch.randn(*shape, dtype=dtype, device=device)
        elif dtype_str.startswith("int"):
            out[k] = torch.zeros(*shape, dtype=dtype, device=device)
        elif dtype_str == "bool":
            out[k] = torch.ones(*shape, dtype=dtype, device=device)
        else:
            out[k] = torch.zeros(*shape, dtype=dtype, device=device)
    return out


def materialize_scalars(workload_inputs: dict, kwargs: dict,
                        safetensors_root: Path | None = None) -> dict:
    """Pull scalar values from the workload's inputs section into kwargs.
    Also load safetensors-typed inputs from disk if `safetensors_root` is set."""
    torch = _need_torch()
    for k, v in workload_inputs.items():
        if not isinstance(v, dict):
            continue
        t = v.get("type")
        if t == "scalar":
            kwargs[k] = v["value"]
        elif t == "safetensors" and safetensors_root is not None:
            path = v.get("path")
            key  = v.get("tensor_key")
            if not path or not key:
                continue
            # Try a few candidate roots in order
            for root in (safetensors_root, safetensors_root.parent,
                         Path("/home/qinhaiyan/sol-execbench")):
                candidate = root / path if not Path(path).is_absolute() else Path(path)
                if candidate.exists():
                    try:
                        from safetensors.torch import load_file
                        loaded = load_file(str(candidate))
                        if key in loaded:
                            kwargs[k] = loaded[key].to("cuda")
                            break
                    except Exception:
                        pass
    return kwargs


# ---------------------------------------------------------------------------
# Roofline-aware report (regime-aware metric selection from roofline_measure)
# ---------------------------------------------------------------------------
def get_handler(problem_name: str):
    import roofline_measure as rm
    registry = rm.build_registry()
    h = registry.get(problem_name)
    if h is None:
        raise KeyError(f"no analyzer registered for '{problem_name}'. "
                       f"Run scripts/roofline_measure.py list to see options.")
    return h


# ---------------------------------------------------------------------------
# Main per-problem benchmark
# ---------------------------------------------------------------------------
def bench_problem(problem_dir: Path, defn: dict, solution_path: Path | None,
                  smoke: bool, n_batch: int, warmup: int, groups: int,
                  device: str = "cuda") -> tuple[list[dict], dict]:
    torch = _need_torch()
    if not torch.cuda.is_available():
        print("ERROR: CUDA device required for timing.", file=sys.stderr)
        sys.exit(2)

    name = defn["name"]
    workload_path = problem_dir / "workload.jsonl"
    rows = [json.loads(l) for l in workload_path.read_text().splitlines() if l.strip()]
    if smoke and len(rows) > 3:
        rows = [rows[0], rows[len(rows) // 2], rows[-1]]

    sol_path = solution_path or (problem_dir / "reference.py")
    if not sol_path.exists():
        raise FileNotFoundError(f"solution file not found: {sol_path}")
    run_fn, _ = import_run_fn(sol_path, name + "_sol")

    # For input generation, always pull get_inputs from the REFERENCE — the
    # solution often only exports run() and would not know how to materialize
    # 'custom' input types declared by the workload.
    ref_path = problem_dir / "reference.py"
    _, get_inputs_fn = import_run_fn(ref_path, name + "_ref")

    handler = get_handler(name)

    print(f"# device={torch.cuda.get_device_name(0)}  problem={name}  "
          f"solution={sol_path.name}  n_batch={n_batch} warmup={warmup} groups={groups}"
          f"{'  [smoke]' if smoke else ''}")
    print(f"{'axes':<32} {'regime':>10} {'lat_us':>10} {'MFU':>8} "
          f"{'mfu_ceil':>9} {'BW%':>8} {'SoL':>8}")
    print("-" * 96)

    out_rows = []
    for w in rows:
        axes = w["axes"]
        uuid = w.get("uuid")
        torch.manual_seed(0)
        kwargs = generate_inputs(defn, axes, get_inputs_fn, device)
        if isinstance(kwargs, dict):
            kwargs = materialize_scalars(w.get("inputs", {}), kwargs,
                                          safetensors_root=Path("/home/qinhaiyan/sol-execbench"))
        else:
            kwargs = list(kwargs)   # tuples too

        # Build the per-call invocation
        if isinstance(kwargs, dict):
            call_fn = lambda kw=kwargs: run_fn(**kw)
        else:
            call_fn = lambda kw=kwargs: run_fn(*kw)

        # Warmup once outside the timing loop so failures surface clearly
        try:
            _ = call_fn()
        except Exception as e:
            print(f"  ERROR on axes={axes}: {type(e).__name__}: {e}")
            continue

        lat_us = time_kernel(call_fn, n_batch, warmup, groups)

        # Roofline classification + achieved metrics; uses Ray-234 per-UUID
        # costs when available (accurate for MoE / paged / Quant FP8),
        # otherwise falls back to the analytical formula.
        a = handler.evaluate(axes, uuid=uuid)
        flops, bytes_, peak = a["flops"], a["bytes"], a["peak"]
        regime = a["regime"]
        mfu_ceiling = a["mfu_ceiling"]
        cost_source = a.get("cost_source", "analytic")
        t_sec = lat_us * 1e-6
        mfu = (flops / t_sec) / peak if (flops and peak) else 0.0
        from _hardware import PEAK_BW
        bw  = (bytes_ / t_sec) / PEAK_BW if bytes_ else 0.0
        sol = a["t_sol_us"] / lat_us if lat_us > 0 else 0.0

        axes_str = ", ".join(f"{k}={v}" for k, v in axes.items())[:30]
        src_tag = " [r234]" if cost_source == "ray234" else ""
        print(f"{axes_str:<32} {regime:>10} {lat_us:>10.2f} "
              f"{mfu:>7.1%} {mfu_ceiling:>9.3f} {bw:>7.1%} {sol:>7.1%}{src_tag}")

        out_rows.append({
            "problem":      name,
            "uuid":         uuid,
            "axes":         json.dumps(axes),
            "regime":       regime,
            "cost_source":  cost_source,
            "latency_us":   lat_us,
            "mfu":          mfu,
            "mfu_ceiling":  mfu_ceiling,
            "bandwidth_utilization": bw,
            "sol_efficiency":        sol,
            "ai":           a["ai"],
            "t_sol_us":     a["t_sol_us"],
            "flops":        flops,
            "bytes":        bytes_,
            "peak":         peak,
        })

    # Summary
    summary = _summarize(out_rows)
    print("-" * 96)
    print(f"regime distribution: {summary['regime_counts']}")
    print(f"primary recommended metric: {summary['recommended_metric']}")
    if summary["geomean_us"]:
        print(f"geomean latency = {summary['geomean_us']:.2f} us   "
              f"peak MFU = {summary['peak_mfu']:.1%}   "
              f"peak BW% = {summary['peak_bw']:.1%}   "
              f"PASS {len(out_rows)}/{len(rows)}")
    print()
    return out_rows, summary


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return dict(regime_counts={}, recommended_metric="(no rows)",
                    geomean_us=0.0, peak_mfu=0.0, peak_bw=0.0)
    regime_counts = {}
    for r in rows:
        regime_counts[r["regime"]] = regime_counts.get(r["regime"], 0) + 1
    n = len(rows)
    dom, dom_count = max(regime_counts.items(), key=lambda x: x[1])
    if dom_count / n >= 0.8:
        metric = {"compute": "MFU", "memory": "BW%",
                  "balanced": "MFU + BW% (both)",
                  "latency": "time + speedup_vs_reference"}.get(dom, "per-row regime")
    else:
        metric = "per-row regime"
    lats = [r["latency_us"] for r in rows if r["latency_us"] > 0]
    geo = math.exp(sum(math.log(x) for x in lats) / len(lats)) if lats else 0.0
    return dict(regime_counts=regime_counts,
                recommended_metric=metric,
                geomean_us=geo,
                peak_mfu=max(r["mfu"] for r in rows),
                peak_bw=max(r["bandwidth_utilization"] for r in rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Standalone roofline benchmark (no sol-execbench needed).",
        epilog="Timing engine borrowed from SOLBench-H800; metrics layered with our regime classifier."
    )
    import _hardware; _hardware.add_hardware_arg(ap)
    ap.add_argument("problem", help="problem path or subdir, e.g. L1/069_rms_norm")
    ap.add_argument("--solution", help="path to a .py with a run() function "
                                       "(default: the problem's reference.py)")
    ap.add_argument("--smoke", action="store_true",
                    help="small/mid/large workloads only (3 rows)")
    ap.add_argument("--n-batch", type=int, default=30,
                    help="back-to-back launches per timed group (default 30)")
    ap.add_argument("--warmup", type=int, default=25,
                    help="untimed warmup calls (default 25)")
    ap.add_argument("--groups", type=int, default=5,
                    help="timed groups; median taken (default 5)")
    ap.add_argument("-o", "--out", help="write per-row CSV here")
    args = ap.parse_args()

    problem_dir, defn = load_problem(args.problem)
    sol = Path(args.solution) if args.solution else None
    rows, summary = bench_problem(problem_dir, defn, sol,
                                  smoke=args.smoke,
                                  n_batch=args.n_batch,
                                  warmup=args.warmup,
                                  groups=args.groups)
    if args.out and rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"# wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
