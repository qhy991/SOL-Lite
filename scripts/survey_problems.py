"""Survey all 60 Contest problems: print axes, dtypes, shapes for classification."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "benchmark" / "Contest"
CATEGORIES = ["L1", "L2", "FlashInfer-Bench", "Quant"]

def summarize(p: Path) -> dict:
    d = json.loads((p / "definition.json").read_text())
    name = d["name"]
    desc = d.get("description", "").split(".")[0][:80]
    # axes
    consts = {k: v.get("value") for k, v in d["axes"].items() if v.get("type") == "const"}
    var_axes = [k for k, v in d["axes"].items() if v.get("type") == "var"]
    # inputs
    inputs = []
    for k, v in d["inputs"].items():
        shape = v.get("shape")
        dtype = v.get("dtype")
        inputs.append(f"{k}:{shape}@{dtype}")
    outputs = []
    for k, v in d["outputs"].items():
        outputs.append(f"{k}:{v.get('shape')}@{v.get('dtype')}")
    return {
        "category": p.parent.name,
        "name": name,
        "desc": desc,
        "vars": var_axes,
        "consts": consts,
        "inputs": inputs,
        "outputs": outputs,
    }

if __name__ == "__main__":
    rows = []
    for cat in CATEGORIES:
        cat_dir = ROOT / cat
        for prob in sorted(cat_dir.iterdir()):
            if prob.is_dir():
                rows.append(summarize(prob))
    print(f"Total: {len(rows)} problems\n")
    for r in rows:
        print(f"[{r['category']}] {r['name']}")
        print(f"   {r['desc']}")
        print(f"   vars: {r['vars']}")
        print(f"   consts: {r['consts']}")
        for i in r["inputs"]:
            print(f"   in:  {i}")
        for o in r["outputs"]:
            print(f"   out: {o}")
        print()
