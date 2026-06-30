"""Batch benchmark sol-baseline solutions against the SOL-Lite regime analyzer.

For each solution.json in sol-baseline/baselines/, extract the baseline.py
source, time it on every workload of the matching SOL-Lite problem, and
report regime-aware metrics (MFU / mfu_ceiling / BW% / SoL_eff / speedup
vs. the reference). Aggregates a per-baseline summary CSV + markdown.

Requires the sol-execbench environment (torch + cuda + flashinfer +
flash_attn + liger + causal_conv1d). Run from inside that env:

    cd /home/qinhaiyan/sol-execbench
    uv run python /home/qinhaiyan/SOL-Lite/scripts/bench_baselines.py \\
        --sol-baseline /home/qinhaiyan/sol-baseline \\
        --smoke -o baseline_roofline.csv

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sol-baseline", default="/home/qinhaiyan/sol-baseline",
                    help="path to sol-baseline checkout")
    ap.add_argument("--smoke", action="store_true",
                    help="3 representative workloads per problem (recommended)")
    ap.add_argument("--n-batch", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--groups", type=int, default=5)
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
        sys.stdout.flush()

    csv_path = Path(f"{args.out_prefix}.csv")
    md_path = Path(f"{args.out_prefix}.md")
    write_summary(results, csv_path, md_path)
    n_ok = sum(1 for r in results if not r.get("error"))
    print(f"\n# done: {n_ok}/{len(results)} ran successfully")
    print(f"# wrote {csv_path}")
    print(f"# wrote {md_path}")


if __name__ == "__main__":
    main()
