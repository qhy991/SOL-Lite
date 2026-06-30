"""Generate unified summary CSV + markdown of all 60 Contest problems.

Pulls from the three analyzers:
  - roofline_tier1_batch.PROBLEMS  (33 Tier-1 problems)
  - roofline_moe.SIM_PROBLEMS + DET_PROBLEMS  (15 MoE problems)
  - roofline_l2.PROBLEMS  (9 L2 fused-block problems)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import roofline_tier1_batch as t1
import roofline_moe as t2
import roofline_l2 as t3


def _category_for(subdir: str) -> str:
    return subdir.split("/")[0]


def _recommend(regime_counts: Counter, total: int, mfu_max: float) -> str:
    if total == 0: return "(no rows)"
    dom_name, dom_count = regime_counts.most_common(1)[0]
    dom_pct = dom_count / total
    if dom_pct >= 0.8:
        if dom_name == "compute":  return "MFU"
        if dom_name == "memory":   return "BW%"
        if dom_name == "latency":  return "time + speedup_vs_reference"
        if dom_name == "balanced": return "MFU + BW% (both)"
    return "per-row regime"


def _row_summary(name: str, subdir: str, regime_counts: Counter,
                 ai_min: float, ai_max: float, mfu_max: float,
                 dtype_note: str = "", caveat: str = "") -> dict:
    total = sum(regime_counts.values())
    return {
        "category": _category_for(subdir), "name": name, "subdir": subdir,
        "rows": total, "regime": dict(regime_counts),
        "ai_min": ai_min, "ai_max": ai_max, "mfu_ceiling": mfu_max,
        "metric": _recommend(regime_counts, total, mfu_max),
        "dtype": dtype_note, "caveat": caveat,
    }


def collect_tier1() -> list[dict]:
    out = []
    for prob in t1.PROBLEMS:
        rows = [json.loads(l) for l in (t1.CONTEST_ROOT / prob.subdir /
                "workload.jsonl").read_text().splitlines() if l.strip()]
        regimes = Counter()
        ai_min, ai_max, mfu_max = float("inf"), 0.0, 0.0
        for r in rows:
            flops, bytes_, peak = prob.fn(r["axes"])
            m = t1.classify(flops, bytes_, peak)
            regimes[m["regime"]] += 1
            ai_min = min(ai_min, m["ai"]); ai_max = max(ai_max, m["ai"])
            mfu_max = max(mfu_max, m["mfu_ceiling"])
        dtype_note = "fp8" if (rows and prob.fn(rows[0]["axes"])[2] == t1.PEAK_FP8) else "bf16"
        out.append(_row_summary(prob.name, prob.subdir, regimes,
                                ai_min, ai_max, mfu_max, dtype_note, prob.note))
    return out


def collect_moe() -> list[dict]:
    out = []
    for spec in t2.SIM_PROBLEMS:
        rows = t2.load_workload(spec.subdir)
        regimes = Counter()
        ai_min, ai_max, mfu_max = float("inf"), 0.0, 0.0
        for r in rows:
            m = spec.analyze_row(r["axes"])
            regimes[m["regime"]] += 1
            ai_min = min(ai_min, m["ai"]); ai_max = max(ai_max, m["ai"])
            mfu_max = max(mfu_max, m["mfu_ceiling"])
        cav = ((spec.note or "") + " ; routing simulated").strip(" ;")
        out.append(_row_summary(spec.name, spec.subdir, regimes,
                                ai_min, ai_max, mfu_max,
                                "fp8" if spec.dtype_bytes == 1 else "bf16",
                                cav))
    for spec in t2.DET_PROBLEMS:
        rows = t2.load_workload(spec.subdir)
        regimes = Counter()
        ai_min, ai_max, mfu_max = float("inf"), 0.0, 0.0
        for r in rows:
            m = spec.analyze_row(r["axes"])
            regimes[m["regime"]] += 1
            ai_min = min(ai_min, m["ai"]); ai_max = max(ai_max, m["ai"])
            mfu_max = max(mfu_max, m["mfu_ceiling"])
        out.append(_row_summary(spec.name, spec.subdir, regimes,
                                ai_min, ai_max, mfu_max, "bf16",
                                f"({spec.kind}-only) " + (spec.note or "")))
    return out


def collect_l2() -> list[dict]:
    out = []
    for prob in t3.PROBLEMS:
        rows = t3.load_workload(prob.subdir)
        regimes = Counter()
        ai_min, ai_max, mfu_max = float("inf"), 0.0, 0.0
        for r in rows:
            ops = prob.decompose(r["axes"])
            results = [op.metrics() for op in ops]
            tot_t = sum(x["t_sol_us"] for x in results) or 1e-9
            tot_f = sum(x["flops"] for x in results)
            tot_b = sum(x["bytes"] for x in results)
            ai = tot_f / tot_b if tot_b else 0.0
            mfu_ceiling = min(1.0, t3.PEAK_BW * ai / t3.PEAK_BF16)
            regime_t = {"compute":0.0,"memory":0.0,"balanced":0.0,"latency":0.0}
            for x in results:
                regime_t[x["regime"]] += x["t_sol_us"]
            dom = max(regime_t, key=regime_t.get)
            regimes[dom] += 1
            ai_min = min(ai_min, ai); ai_max = max(ai_max, ai)
            mfu_max = max(mfu_max, mfu_ceiling)
        out.append(_row_summary(prob.name, prob.subdir, regimes,
                                ai_min, ai_max, mfu_max, "bf16",
                                "multi-kernel — recommended: per-op breakdown"))
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category","subdir","name","dtype","rows",
                    "compute","balanced","memory","latency",
                    "ai_min","ai_max","mfu_ceiling","metric","caveat"])
        for r in rows:
            reg = r["regime"]
            w.writerow([r["category"], r["subdir"], r["name"], r["dtype"], r["rows"],
                        reg.get("compute",0), reg.get("balanced",0),
                        reg.get("memory",0), reg.get("latency",0),
                        f"{r['ai_min']:.1f}", f"{r['ai_max']:.1f}",
                        f"{r['mfu_ceiling']:.3f}", r["metric"], r["caveat"]])


def write_markdown(rows: list[dict], path: Path) -> None:
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    L = [
        "# SOL ExecBench — 60-problem roofline summary",
        "",
        "Target hardware: **NVIDIA H800 SXM5** (BF16 peak 989 TFLOPS, FP8 peak 1979 TFLOPS, HBM3 3.35 TB/s).",
        "",
        "Each row's regime is determined by per-workload arithmetic intensity vs. the ridge",
        "(295 FLOPs/byte for BF16, 591 for FP8) and an absolute `t_sol < 5 μs` latency floor.",
        "",
        "Columns:",
        "  - `C/B/M/L`: rows classified as compute / balanced / memory / latency",
        "  - `AI`: arithmetic-intensity range across this problem's workloads",
        "  - `MFU_max`: max achievable MFU across workloads (physical ceiling, not 1.0!)",
        "  - `metric`: recommended primary metric to report",
        "",
    ]
    for cat in ["L1","L2","FlashInfer-Bench","Quant"]:
        if cat not in by_cat: continue
        items = sorted(by_cat[cat], key=lambda r: r["subdir"])
        L.append(f"## {cat} ({len(items)} problems)")
        L.append("")
        L.append("| # | problem | dtype | rows | C/B/M/L | AI range | MFU_max | metric |")
        L.append("|---|---|---|---:|---|---|---:|---|")
        for r in items:
            reg = r["regime"]
            cbml = (f"{reg.get('compute',0)}/{reg.get('balanced',0)}/"
                    f"{reg.get('memory',0)}/{reg.get('latency',0)}")
            short = r["subdir"].split("/", 1)[1]
            L.append(f"| `{short[:3]}` | {short[4:]} | {r['dtype']} | {r['rows']} | {cbml} | "
                     f"{r['ai_min']:.0f}–{r['ai_max']:.0f} | "
                     f"{r['mfu_ceiling']:.2f} | **{r['metric']}** |")
        L.append("")
    total = len(rows)
    metric_counts = Counter(r["metric"] for r in rows)
    L += ["## Cross-cutting summary", "", f"Total problems: **{total}**.", "",
          "Recommended primary metric distribution:", "",
          "| metric | count | share |", "|---|---:|---:|"]
    for m, c in metric_counts.most_common():
        L.append(f"| {m} | {c} | {100*c/total:.0f}% |")
    path.write_text("\n".join(L) + "\n")


def main():
    rows = collect_tier1() + collect_moe() + collect_l2()
    seen = set(); unique = []
    for r in rows:
        if r["subdir"] in seen: continue
        seen.add(r["subdir"]); unique.append(r)
    csv_path = ROOT / "roofline_summary.csv"
    md_path = ROOT / "roofline_summary.md"
    write_csv(unique, csv_path)
    write_markdown(unique, md_path)
    print(f"Wrote {csv_path}  ({len(unique)} problems)")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
