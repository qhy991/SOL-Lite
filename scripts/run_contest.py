#!/usr/bin/env python3
"""Run SOL-Lite Contest problems with the independent harness (no sol-execbench).

Usage:
    # Full 60 problems (correctness + timing + roofline)
    uv sync --extra bench
    uv run python scripts/run_contest.py --all -o ./out

    # Smoke test (3 rows per problem)
    uv run python scripts/run_contest.py --all --smoke

    # Single category
    uv run python scripts/run_contest.py --category L1

    # Single problem
    uv run python scripts/run_contest.py L1/069_rms_norm

    # Custom solution
    uv run python scripts/run_contest.py L1/069_rms_norm --solution my_kernel.py

    # Skip correctness (just time)
    uv run python scripts/run_contest.py --all --no-correctness

    # Force timing mode
    uv run python scripts/run_contest.py --all --timing-mode single

Output:
    Per-problem traces written to <out>/<category>/<problem>/traces.jsonl
    Summary written to <out>/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from harness.loader import (
    discover_problems,
    load_problem,
    problem_category_label,
    CONTEST_ROOT,
    CATEGORIES,
)
from harness.runner import run_problem


def build_registry():
    """Lazy-import roofline_measure to build the handler registry."""
    import roofline_measure as rm
    return rm.build_registry()


def main():
    ap = argparse.ArgumentParser(
        description="Run SOL-Lite Contest problems with the independent harness.",
    )
    import _hardware
    _hardware.add_hardware_arg(ap)

    ap.add_argument(
        "problem",
        nargs="?",
        help="Single problem path (e.g. L1/069_rms_norm). "
             "Omit to use --all or --category.",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Run all 60 problems.",
    )
    ap.add_argument(
        "--category",
        type=str,
        nargs="+",
        default=None,
        choices=sorted(CATEGORIES),
        help="Restrict to one or more categories.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of problems to evaluate.",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="3 representative workloads per problem (small/mid/large).",
    )
    ap.add_argument(
        "-o", "--output",
        type=Path,
        default=ROOT / "out",
        help="Output directory for traces and summary. Default: <repo_root>/out.",
    )
    ap.add_argument(
        "--solution",
        help="Path to a custom solution .py file (single-problem mode only).",
    )
    ap.add_argument(
        "--rerun",
        action="store_true",
        help="Re-evaluate problems that already have results.",
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output.",
    )

    # Timing controls
    timing = ap.add_argument_group("Timing")
    timing.add_argument("--n-batch", type=int, default=30,
                        help="Back-to-back launches per group (Mode B, default 30).")
    timing.add_argument("--warmup", type=int, default=25,
                        help="Untimed warmup calls (Mode B, default 25).")
    timing.add_argument("--groups", type=int, default=5,
                        help="Timed groups for Mode B (default 5).")
    timing.add_argument("--single-iterations", type=int, default=50,
                        help="Timed iterations for Mode A (default 50).")
    timing.add_argument("--no-l2-flush", action="store_true",
                        help="Disable L2 cache flush in Mode A.")
    timing.add_argument("--timing-mode", choices=["auto", "single", "batch"],
                        default="auto",
                        help="Force timing mode (default: auto, based on t_sol).")

    # Control flags
    ctrl = ap.add_argument_group("Control")
    ctrl.add_argument("--no-correctness", action="store_true",
                      help="Skip correctness checks (just time).")
    ctrl.add_argument("--no-timing", action="store_true",
                      help="Skip timing (just check correctness).")

    args = ap.parse_args()

    # Apply hardware override early (before analyzers are imported)
    if args.hardware:
        _hardware.apply_hardware_from_args(args)

    # --- Resolve problem list ---
    if args.problem:
        problem_dir, defn = load_problem(args.problem)
        problems = [problem_dir]
        if args.solution:
            args.solution = Path(args.solution)
        print(f"Single problem: {defn['name']}")
    elif args.all or args.category:
        cats = args.category if args.category else None
        problems = discover_problems(CONTEST_ROOT, cats)
        if args.limit:
            problems = problems[: args.limit]
        print(f"Discovered {len(problems)} problems")
    else:
        ap.print_help()
        sys.exit(1)

    if not problems:
        print("No problems found.")
        sys.exit(1)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build roofline registry (lazy: needs analyzers)
    registry = build_registry()
    print(f"Roofline registry: {len(registry)} problems")

    # Per-problem loop
    summaries = []
    n_passed = 0
    n_failed = 0

    for i, problem_dir in enumerate(problems):
        defn = json.loads((problem_dir / "definition.json").read_text())
        name = defn["name"]
        category = problem_category_label(problem_dir)

        print(f"\n[{i+1}/{len(problems)}] {category}/{name}")

        problem_out = output_dir / category / name
        traces_path = problem_out / "traces.jsonl"

        # Skip already-passed unless --rerun
        if not args.rerun and traces_path.exists():
            existing = traces_path.read_text().splitlines()
            existing_traces = [json.loads(l) for l in existing if l.strip()]
            failed = sum(
                1 for t in existing_traces
                if t.get("evaluation", {}).get("status") != "PASSED"
            )
            if failed == 0:
                print("  Skipping (already passed). Use --rerun to re-evaluate.")
                summaries.append({
                    "problem": f"{category}/{name}",
                    "total": len(existing_traces),
                    "passed": len(existing_traces),
                    "failed": 0,
                })
                n_passed += 1
                continue
            print(f"  Re-running (previous run had {failed} failures).")

        # Clear previous run
        if problem_out.exists():
            shutil.rmtree(problem_out)
        problem_out.mkdir(parents=True, exist_ok=True)

        # Get handler
        handler = registry.get(name)
        if handler is None:
            print(f"  WARNING: no roofline analyzer for '{name}' — "
                  f"roofline metrics will be skipped.")

        solution_path = getattr(args, "solution", None)
        if solution_path and not isinstance(solution_path, Path):
            solution_path = Path(solution_path)

        try:
            traces = run_problem(
                problem_dir=problem_dir,
                defn=defn,
                solution_path=solution_path,
                handler=handler,
                smoke=args.smoke,
                n_batch=args.n_batch,
                warmup=args.warmup,
                groups=args.groups,
                single_iterations=args.single_iterations,
                l2_flush=not args.no_l2_flush,
                force_timing_mode=None if args.timing_mode == "auto" else args.timing_mode,
                skip_correctness=args.no_correctness,
                skip_timing=args.no_timing,
                verbose=args.verbose,
            )
        except Exception as e:
            import traceback
            print(f"  ERROR: {type(e).__name__}: {e}")
            if args.verbose:
                traceback.print_exc()
            summaries.append({
                "problem": f"{category}/{name}",
                "total": 0, "passed": 0, "failed": 1,
                "error": str(e),
            })
            n_failed += 1
            continue

        # Save traces
        from harness.traces import format_traces_jsonl
        traces_path.write_text(format_traces_jsonl(traces))

        # Summary
        passed = sum(
            1 for t in traces
            if t.get("evaluation", {}).get("status") == "PASSED"
        )
        failed = len(traces) - passed

        status = "OK" if failed == 0 else "FAIL"
        print(f"  {status}: {passed}/{len(traces)} passed")

        if failed > 0:
            for t in traces:
                if t.get("evaluation", {}).get("status") != "PASSED":
                    err = t.get("evaluation", {}).get("error", {})
                    msg = err.get("error", str(err)[:120]) if isinstance(err, dict) else str(err)[:120]
                    print(f"    FAIL: {msg}")
                    break

        summaries.append({
            "problem": f"{category}/{name}",
            "total": len(traces),
            "passed": passed,
            "failed": failed,
        })
        if failed == 0:
            n_passed += 1
        else:
            n_failed += 1

    # Print overall summary
    print("\n" + "=" * 70)
    print(f"Total: {len(problems)} problems | OK: {n_passed} | FAIL: {n_failed}")
    print("=" * 70)

    # Save summary JSON
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nSummary saved to {summary_path}")
    print(f"Per-problem traces saved under {output_dir}")


if __name__ == "__main__":
    main()