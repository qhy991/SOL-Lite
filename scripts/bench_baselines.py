"""Batch benchmark sol-baseline solutions against the SOL-Lite regime analyzer.

For each solution.json in sol-baseline/baselines/, extract the baseline.py
source, time it on every workload of the matching SOL-Lite problem, and
report regime-aware metrics (MFU / mfu_ceiling / BW% / SoL_eff / speedup
vs. the reference). Aggregates a per-baseline summary CSV + markdown.

Requires the sol-execbench environment (torch + cuda + flashinfer +
flash_attn + liger + causal_conv1d). Run from inside that env, and point
--sol-baseline at your sol-baseline checkout (env var: SOL_BASELINE_ROOT):

    export SOL_BASELINE_ROOT=/path/to/sol-baseline
    uv run --project /path/to/sol-execbench \\
        python scripts/bench_baselines.py --smoke -o baseline_roofline

(The SOL-Lite venv itself only declares torch as optional; the
sol-execbench venv already has every baseline library installed.)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

# Resolve SOL-Lite root regardless of where we're invoked from
SOL_LITE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOL_LITE / "scripts"))


def discover_baselines(sol_baseline_root: Path) -> list[dict]:
    """Walk sol-baseline/baselines/<lib>/<cat>/<task>/solution.json.
    Returns list of dicts: {definition, lib, category, source_path, run_source}.
    De-duplicates: each (definition, lib) appears once (prefers the entry under
    L1/L2/Quant/FlashInfer-Bench over the Contest/ alias)."""
    seen = {}
    for sj in sol_baseline_root.rglob("solution.json"):
        d = json.loads(sj.read_text())
        defn = d.get("definition", "")
        parts = sj.parts
        if "baselines" not in parts:
            continue
        i = parts.index("baselines")
        try:
            lib = parts[i + 1]
            category = parts[i + 2]
        except IndexError:
            continue
        # Prefer the non-Contest path
        key = (defn, lib)
        if key in seen and category != "Contest":
            seen[key] = (sj, d, lib, category)
        elif key not in seen:
            seen[key] = (sj, d, lib, category)

    out = []
    for (defn, lib), (sj, d, lib_, cat) in seen.items():
        # Extract baseline.py source from the inline sources field
        src = None
        for s in d.get("sources", []):
            if s.get("path") in ("baseline.py", "solution.py"):
                src = s["content"]; break
        if src is None and d.get("sources"):
            src = d["sources"][0].get("content")
        if src is None:
            continue
        out.append(dict(
            definition=defn, lib=lib_, category=cat,
            source_path=str(sj), run_source=src,
        ))
    out.sort(key=lambda r: (r["definition"], r["lib"]))
    return out


def benchmark_one(b: dict, smoke: bool, n_batch: int, warmup: int, groups: int,
                  registry: dict) -> dict:
    """Time one baseline. Returns a summary dict (geomean lat + peak metrics)."""
    import roofline_bench as rb     # lazy: needs torch
    import torch                     # noqa: F401

    handler = registry.get(b["definition"])
    if handler is None:
        return {"definition": b["definition"], "lib": b["lib"],
                "error": "no analyzer registered"}

    # Resolve the SOL-Lite problem dir from the handler.subdir (e.g. "L1/069_...").
    pdir = SOL_LITE / "data" / "benchmark" / "Contest" / handler.subdir
    if not pdir.is_dir():
        return {"definition": b["definition"], "lib": b["lib"],
                "error": f"problem dir not found: {pdir}"}
    defn = json.loads((pdir / "definition.json").read_text())

    # Write baseline source to a temp file (roofline_bench imports a .py)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                     dir=tempfile.gettempdir()) as f:
        f.write(b["run_source"])
        sol_path = Path(f.name)

    try:
        rows, summary = rb.bench_problem(
            problem_dir=pdir, defn=defn, solution_path=sol_path,
            smoke=smoke, n_batch=n_batch, warmup=warmup, groups=groups,
        )
    except Exception as e:
        return {"definition": b["definition"], "lib": b["lib"],
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}
    finally:
        try:
            os.unlink(sol_path)
        except OSError:
            pass

    return {
        "definition": b["definition"], "lib": b["lib"], "category": b["category"],
        "n_rows": len(rows),
        "regime_counts": summary["regime_counts"],
        "recommended_metric": summary["recommended_metric"],
        "geomean_us": summary["geomean_us"],
        "peak_mfu": summary["peak_mfu"],
        "peak_bw": summary["peak_bw"],
        "per_row_rows": rows,        # the full per-row data
    }


def write_summary(results: list[dict], csv_path: Path, md_path: Path) -> None:
    # Per-row CSV
    fieldnames = ["definition","lib","category","uuid","axes","regime",
                  "cost_source",
                  "latency_us","mfu","mfu_ceiling","bandwidth_utilization",
                  "sol_efficiency","ai","t_sol_us","flops","bytes","peak"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            if r.get("error"):
                continue
            for row in r["per_row_rows"]:
                w.writerow({"definition": r["definition"], "lib": r["lib"],
                            "category": r["category"],
                            **{k: row.get(k, "") for k in fieldnames if k not in (
                                "definition","lib","category")}})
    # Per-baseline markdown
    lines = [
        "# sol-baseline × SOL-Lite roofline report",
        "",
        f"Tested {sum(1 for r in results if not r.get('error'))} baselines, "
        f"{sum(1 for r in results if r.get('error'))} errored.",
        "",
        "All numbers measured on this machine. Columns:",
        "- `geomean_us` — geometric mean latency across workload rows (smoke = 3 reps)",
        "- `peak_MFU` / `peak_BW%` — max achievable values across rows (read against `mfu_ceiling`!)",
        "- `recommended` — primary metric to read for this baseline given its regime mix",
        "",
        "| definition | lib | n_rows | regime | recommended | geomean_us | peak_MFU | peak_BW% |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['definition']} | {r['lib']} | — | — | "
                         f"ERROR: {r['error'][:50]} | — | — | — |")
            continue
        rc = r["regime_counts"]
        regime_str = " ".join(f"{k}:{v}" for k, v in rc.items())
        lines.append(
            f"| {r['definition']} | {r['lib']} | {r['n_rows']} | {regime_str} | "
            f"{r['recommended_metric']} | {r['geomean_us']:.2f} | "
            f"{r['peak_mfu']:.1%} | {r['peak_bw']:.1%} |"
        )
    # Errors detail
    err_rows = [r for r in results if r.get("error")]
    if err_rows:
        lines += ["", "## Errors", ""]
        for r in err_rows:
            lines.append(f"### {r['definition']} ({r['lib']})")
            lines.append(f"```\n{r['error']}\n```")
            if "traceback" in r:
                lines.append(f"<details><summary>traceback</summary>\n\n"
                             f"```\n{r['traceback']}\n```\n</details>\n")
    md_path.write_text("\n".join(lines) + "\n")


def _emit_v3_submission(result: dict, root: Path, user: str, experiment: str,
                        round_id: int, baseline_source: str) -> Path:
    """Write one baseline's v3 submission dir under `root/<user>/<task_id>/r<round>/`.

    Matches SoL-Contest-InfiniAI schema v3 (see schema/submission.v3.schema.json
    at that repo). Produces:
        <root>/<user>/<task_id>/r<round>/
          manifest.json
          solution/{README.md, baseline.py}
          results/workloads.json
    """
    import datetime
    task_id = result["definition"]
    lib = result["lib"]
    tag = f"{user}-{task_id}-{lib}-r{round_id}"
    outdir = root / user / task_id / f"r{round_id}"
    (outdir / "solution").mkdir(parents=True, exist_ok=True)
    (outdir / "results").mkdir(parents=True, exist_ok=True)

    # solution/ dir: README + baseline source
    (outdir / "solution" / "README.md").write_text(
        f"# {task_id} — {lib} baseline\n\n"
        f"Extracted from sol-baseline.\n"
        f"Timed via SOL-Lite `scripts/bench_baselines.py`.\n"
    )
    (outdir / "solution" / "baseline.py").write_text(baseline_source)

    # results/workloads.json — Ray-234 format
    wl_rows = []
    for row in result["per_row_rows"]:
        axes = json.loads(row["axes"]) if isinstance(row.get("axes"), str) else row.get("axes", {})
        axes_label = ",".join(f"{k}={v}" for k, v in sorted(axes.items()))
        t_base_ms = row["latency_us"] / 1000.0
        t_sol_ms = row["t_sol_us"] / 1000.0
        wl_rows.append({
            "workload_uuid": row.get("uuid"),
            "status": "passed",
            "axes": axes,
            "axes_label": axes_label,
            "t_sol_ms": t_sol_ms,
            "t_base_ms": t_base_ms,
            "mfu_pct":  row["mfu"] * 100.0,
            "bw_util_pct": row["bandwidth_utilization"] * 100.0,
            "t_base_over_t_sol": t_base_ms / t_sol_ms if t_sol_ms > 0 else None,
            "regime": row["regime"],
            "mfu_ceiling": row.get("mfu_ceiling"),
            "cost_source": row.get("cost_source", "analytic"),
        })
    (outdir / "results" / "workloads.json").write_text(
        json.dumps(wl_rows, indent=2) + "\n")

    # manifest.json (v3 schema)
    import math
    def gmean(xs):
        xs = [x for x in xs if x is not None and x > 0]
        if not xs: return 0.0
        return math.exp(sum(math.log(x) for x in xs) / len(xs))
    from _hardware import HARDWARE_NAME
    manifest = {
        "schema_version": 3,
        "submitter": {"user": user, "team": "SOL-Lite"},
        "run": {
            "run_id": tag,
            "experiment_id": experiment,
            "experiment_label": f"SOL-Lite roofline + {lib} baseline",
            "competition_id": "fi-bench-v1",
            "task_id": task_id,
            "round": round_id,
            "submitted_at": datetime.datetime.now(datetime.timezone.utc)
                                            .isoformat(timespec="seconds"),
            "solution_name": f"sol-baseline/{lib}",
        },
        "artifacts": {
            "solution": {"dir": "solution", "entrypoint": "baseline.py::run",
                         "files": ["README.md", "baseline.py"]},
            "results":  {"dir": "results", "trace": "", "workloads": "workloads.json"},
        },
        "provenance": {
            "tool": "SOL-Lite",
            "tool_version": "bench_baselines.py",
            "notes": "Timed via back-to-back cuda.Event launches; regime + t_sol "
                     "from SOL-Lite analyzer with Ray-234 per-UUID cost preference.",
        },
        "environment": {"gpu": HARDWARE_NAME, "hardware": HARDWARE_NAME},
        "token_usage": None,
        "summary": {
            "status": "passed",
            "passed_workloads": len(wl_rows),
            "total_workloads":  len(wl_rows),
            "geomean_t_sol_ms": gmean([r["t_sol_ms"] for r in wl_rows]),
            "geomean_t_base_ms": gmean([r["t_base_ms"] for r in wl_rows]),
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return outdir


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    import _hardware; _hardware.add_hardware_arg(ap)
    ap.add_argument("--sol-baseline",
                    default=os.environ.get("SOL_BASELINE_ROOT"),
                    help="path to sol-baseline checkout "
                         "(default: $SOL_BASELINE_ROOT)")
    ap.add_argument("--smoke", action="store_true",
                    help="3 representative workloads per problem (recommended)")
    ap.add_argument("--n-batch", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--emit-v3-submissions", metavar="ROOT",
                    help="also write SoL-Contest-InfiniAI schema-v3 submission "
                         "directories under ROOT/{user}/{task_id}/r{round}/")
    ap.add_argument("--submission-user", default="sol-lite",
                    help="submitter user name for v3 submissions (default: sol-lite)")
    ap.add_argument("--submission-experiment", default="sol-lite-baselines",
                    help="experiment_id for v3 submissions")
    ap.add_argument("--submission-round", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N baselines (for testing)")
    ap.add_argument("--definition", help="filter to one definition")
    ap.add_argument("--lib", help="filter to one library (flashinfer/flash_attn/...)")
    ap.add_argument("-o", "--out-prefix", default="baseline_roofline",
                    help="output file prefix (writes <prefix>.csv and <prefix>.md)")
    args = ap.parse_args()

    sol_baseline = Path(args.sol_baseline)
    if not sol_baseline.is_dir():
        print(f"ERROR: sol-baseline not found at {sol_baseline}", file=sys.stderr)
        sys.exit(2)

    import roofline_measure as rm
    registry = rm.build_registry()

    baselines = discover_baselines(sol_baseline)
    if args.definition:
        baselines = [b for b in baselines if b["definition"] == args.definition]
    if args.lib:
        baselines = [b for b in baselines if b["lib"] == args.lib]
    matched = [b for b in baselines if b["definition"] in registry]
    skipped = [b for b in baselines if b["definition"] not in registry]
    if args.limit:
        matched = matched[: args.limit]

    print(f"# discovered {len(baselines)} baselines  "
          f"({len(matched)} match SOL-Lite registry, {len(skipped)} skipped)")
    if skipped:
        print(f"# skipped (not in SOL-Lite 60-problem set):")
        for b in skipped:
            print(f"#   {b['definition']} [{b['lib']}]")
    print()

    results = []
    for i, b in enumerate(matched, 1):
        print(f"=== [{i}/{len(matched)}] {b['definition']} [{b['lib']}] ===")
        r = benchmark_one(b, smoke=args.smoke,
                          n_batch=args.n_batch, warmup=args.warmup, groups=args.groups,
                          registry=registry)
        results.append(r)
        if r.get("error"):
            print(f"  ERROR: {r['error']}\n")
        elif args.emit_v3_submissions and r.get("per_row_rows"):
            outdir = _emit_v3_submission(
                r,
                root=Path(args.emit_v3_submissions),
                user=args.submission_user,
                experiment=args.submission_experiment,
                round_id=args.submission_round,
                baseline_source=b["run_source"],
            )
            print(f"  v3 → {outdir}")
        sys.stdout.flush()

    csv_path = Path(f"{args.out_prefix}.csv")
    md_path = Path(f"{args.out_prefix}.md")
    write_summary(results, csv_path, md_path)
    n_ok = sum(1 for r in results if not r.get("error"))
    print(f"\n# done: {n_ok}/{len(results)} ran successfully")
    print(f"# wrote {csv_path}")
    print(f"# wrote {md_path}")
    if args.emit_v3_submissions:
        n_v3 = sum(1 for r in results if not r.get("error") and r.get("per_row_rows"))
        print(f"# wrote {n_v3} v3 submissions under {args.emit_v3_submissions}/{args.submission_user}/")


if __name__ == "__main__":
    main()
