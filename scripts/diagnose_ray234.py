"""Diagnose where SOL-Lite analyzer disagrees with Ray-234 per-UUID costs.

Outputs a per-problem disagreement table sorted by total disagreement.
Does NOT modify any analyzer code — purely diagnostic.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roofline_measure as rm

# Ray-234 workload costs source. Default: the copy imported into SOL-Lite at
# data/costs/ray234_h800.jsonl. Override with:
#   SOL_LITE_RAY234_JSONL=/path/to/workload_costs.jsonl
RAY234 = os.environ.get(
    "SOL_LITE_RAY234_JSONL",
    str(Path(__file__).resolve().parent.parent / "data" / "costs" / "ray234_h800.jsonl"),
)

ray234 = {}
for line in open(RAY234):
    d = json.loads(line)
    ray234[d["workload_uuid"]] = (
        d["kernel"],
        float(d.get("flops") or 0),
        float(d.get("bytes_moved") or 0),
        d.get("precision", "bf16"),
    )

registry = rm.build_registry()
SOL_LITE = Path(__file__).resolve().parent.parent

buckets = defaultdict(lambda: {
    "agree_flop": 0, "agree_byte": 0,
    "flop_disagree": 0, "flop_zero_mine": 0, "flop_zero_theirs": 0,
    "byte_disagree": 0,
    "byte_ratios": [], "flop_ratios": [],
    "total_rows": 0,
})

for name, handler in registry.items():
    pdir = SOL_LITE / "data" / "benchmark" / "Contest" / handler.subdir
    wfile = pdir / "workload.jsonl"
    if not wfile.exists():
        continue
    for line in open(wfile):
        w = json.loads(line)
        uuid = w.get("uuid")
        if uuid not in ray234:
            continue
        kname, their_f, their_b, prec = ray234[uuid]
        try:
            mine = handler.evaluate(w["axes"])
        except Exception:
            continue
        my_f, my_b = mine["flops"], mine["bytes"]
        b = buckets[name]
        b["total_rows"] += 1

        # FLOPs check
        if their_f == 0 and my_f == 0:
            b["agree_flop"] += 1
        elif their_f == 0 and my_f > 0:
            b["flop_zero_theirs"] += 1
        elif my_f == 0 and their_f > 0:
            b["flop_zero_mine"] += 1
        else:
            ratio = my_f / their_f
            b["flop_ratios"].append(ratio)
            if 0.5 < ratio < 2.0:
                b["agree_flop"] += 1
            else:
                b["flop_disagree"] += 1

        # Bytes
        if their_b > 0 and my_b > 0:
            ratio = my_b / their_b
            b["byte_ratios"].append(ratio)
            if 0.5 < ratio < 2.0:
                b["agree_byte"] += 1
            else:
                b["byte_disagree"] += 1


def total_dis(b):
    return b["flop_disagree"] + b["flop_zero_mine"] + b["flop_zero_theirs"] + b["byte_disagree"]


def med(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[len(s) // 2]


rows = sorted(buckets.items(), key=lambda kv: -total_dis(kv[1]))
hdr = ("problem", "n", "flop_dis", "mine=0", "thr=0", "byte_dis", "byte_med", "flop_med")
print(f"{hdr[0]:<55} {hdr[1]:>4} {hdr[2]:>8} {hdr[3]:>6} {hdr[4]:>6} {hdr[5]:>8} {hdr[6]:>9} {hdr[7]:>9}")
print("-" * 115)
for name, b in rows:
    if total_dis(b) == 0:
        continue
    bm = med(b["byte_ratios"])
    fm = med(b["flop_ratios"])
    print(f"{name[:55]:<55} {b['total_rows']:>4} "
          f"{b['flop_disagree']:>8} {b['flop_zero_mine']:>6} {b['flop_zero_theirs']:>6} "
          f"{b['byte_disagree']:>8} {bm:>9.2f} {fm:>9.2f}")

print()
print(f"{'Problems perfectly agreeing:':<55}", sum(1 for _, b in buckets.items() if total_dis(b) == 0))
print(f"{'Problems with disagreement:':<55}", sum(1 for _, b in buckets.items() if total_dis(b) > 0))
