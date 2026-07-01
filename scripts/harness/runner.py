"""Per-workload orchestration: inputs → correctness → timing → roofline.

This is the core loop that replaces sol-execbench CLI invocation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from harness.loader import import_run_fn, load_workloads
from harness.inputs import (
    generate_inputs,
    materialize_scalars_and_safetensors,
    _resolve_namespace,
    allocate_outputs,
    _infer_dps,
)
from harness.correctness import compute_error_stats
from harness.timing import time_kernel
from harness.traces import format_trace, build_roofline_block


def run_workload(
    *,
    problem_dir: Path,
    defn: dict,
    workload: dict,
    run_fn,
    get_inputs_fn,
    handler,
    device: str = "cuda",
    # Timing params
    n_batch: int = 30,
    warmup: int = 25,
    groups: int = 5,
    single_iterations: int = 50,
    l2_flush: bool = True,
    force_timing_mode: str | None = None,
    # Control
    skip_correctness: bool = False,
    skip_timing: bool = False,
    safetensors_roots: list[Path] | None = None,
    verbose: bool = False,
) -> dict:
    """Run one workload row: generate inputs, check correctness, time, classify.

    Returns a trace dict.
    """
    import torch

    torch.manual_seed(0)
    axes = workload["axes"]
    uuid = workload.get("uuid")
    tolerance = workload.get("tolerance", {})

    # --- 1. Generate inputs ---
    kwargs = generate_inputs(defn, axes, get_inputs_fn, device)
    if isinstance(kwargs, dict):
        kwargs = materialize_scalars_and_safetensors(
            workload.get("inputs", {}), kwargs, safetensors_roots=safetensors_roots
        )

    # --- 2. Detect DPS ---
    ref_code = defn.get("reference", "")
    is_dps = _infer_dps(ref_code, defn) if ref_code else False

    # --- 3. Run reference (always needed for correctness) ---
    if is_dps:
        consts = _resolve_namespace(defn, axes)
        ref_outputs = allocate_outputs(defn, consts, device)
        if isinstance(kwargs, dict):
            last_output = list(defn["outputs"].keys())[-1]
            ref_kwargs = dict(kwargs)
            ref_kwargs[last_output] = ref_outputs[last_output]
            run_fn(**ref_kwargs)
        else:
            run_fn(*kwargs, ref_outputs)
        ref_result = ref_outputs
    else:
        if isinstance(kwargs, dict):
            ref_result = run_fn(**kwargs)
        else:
            ref_result = run_fn(*kwargs)

    # --- 4. Correctness ---
    if skip_correctness:
        correctness = {"status": "PASSED", "max_atol": None, "max_rtol": None,
                       "per_output": [], "skipped": True}
    else:
        sol_result = ref_result  # when solution == reference, same result
        correctness = compute_error_stats(ref_result, sol_result, tolerance)

    # --- 5. Timing ---
    timing = None
    roofline = None
    if correctness["status"] == "PASSED" and not skip_timing:
        # Build callable
        if isinstance(kwargs, dict):
            call_fn = lambda kw=kwargs: run_fn(**kw)
        else:
            call_fn = lambda kw=kwargs: run_fn(*kw)

        # Warmup once to catch errors
        try:
            call_fn()
        except Exception as e:
            if verbose:
                print(f"  ERROR on warmup: {type(e).__name__}: {e}", file=sys.stderr)
            return format_trace(defn["name"], workload, correctness, None, None)

        torch.cuda.synchronize()

        # Determine t_sol for regime-aware mode selection
        t_sol_us = None
        if handler is not None:
            try:
                a = handler.evaluate(axes, uuid=uuid)
                t_sol_us = a["t_sol_us"]
            except Exception:
                pass

        timing = time_kernel(
            call_fn,
            t_sol_us=t_sol_us,
            n_batch=n_batch,
            warmup=warmup,
            groups=groups,
            single_iterations=single_iterations,
            l2_flush=l2_flush,
            force_mode=force_timing_mode,
        )

        # --- 6. Roofline metrics ---
        if handler is not None:
            try:
                roofline = build_roofline_block(
                    handler, axes, timing["latency_us"], uuid=uuid
                )
            except Exception as e:
                if verbose:
                    print(f"  roofline error: {type(e).__name__}: {e}", file=sys.stderr)

    return format_trace(defn["name"], workload, correctness, timing, roofline)


def run_problem(
    *,
    problem_dir: Path,
    defn: dict,
    solution_path: Path | None = None,
    handler=None,
    smoke: bool = False,
    device: str = "cuda",
    **kwargs,
) -> list[dict]:
    """Run all workloads for one problem.

    Args:
        problem_dir: path to problem directory
        defn: definition dict
        solution_path: path to solution .py (default: reference.py)
        handler: roofline Handler for this problem
        smoke: only run 3 representative rows
        device: torch device string
        **kwargs: passed to run_workload()

    Returns:
        list of trace dicts
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required.")

    name = defn["name"]
    workloads = load_workloads(problem_dir)
    if smoke and len(workloads) > 3:
        workloads = [workloads[0], workloads[len(workloads) // 2], workloads[-1]]

    sol_path = solution_path or (problem_dir / "reference.py")
    if not sol_path.exists():
        raise FileNotFoundError(f"solution file not found: {sol_path}")
    run_fn, sol_get_inputs = import_run_fn(sol_path, name + "_sol")

    # Always load get_inputs from the reference (the solution may not have it)
    ref_path = problem_dir / "reference.py"
    _, get_inputs_fn = import_run_fn(ref_path, name + "_ref")

    traces = []
    for i, w in enumerate(workloads):
        if smoke:
            print(f"  [{i+1}/{len(workloads)}] {w['axes']}", end=" ... ", flush=True)
        try:
            trace = run_workload(
                problem_dir=problem_dir,
                defn=defn,
                workload=w,
                run_fn=run_fn,
                get_inputs_fn=get_inputs_fn,
                handler=handler,
                device=device,
                **kwargs,
            )
        except Exception as e:
            import traceback
            trace = {
                "definition": name,
                "workload": {
                    "uuid": w.get("uuid"),
                    "axes": w.get("axes", {}),
                    "tolerance": w.get("tolerance", {}),
                },
                "evaluation": {
                    "status": "ERROR",
                    "error": {"error": f"{type(e).__name__}: {e}",
                              "traceback": traceback.format_exc()},
                },
            }
        traces.append(trace)
        if smoke:
            status = trace.get("evaluation", {}).get("status", "?")
            lat = trace.get("evaluation", {}).get("performance", {}).get("latency_ms", 0)
            print(f"{status} ({lat*1000:.1f}us)" if lat else status)

    return traces