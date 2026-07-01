"""Problem discovery and code loading.

Resolves problem directories from the Contest benchmark tree, imports
reference/solution Python files, and returns parsed definition + workload data.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CONTEST_ROOT = ROOT / "data" / "benchmark" / "Contest"

CATEGORIES = {"L1", "L2", "FlashInfer-Bench", "Quant"}


def is_problem_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "definition.json").exists()
        and (path / "workload.jsonl").exists()
    )


def _collect_problems(parent: Path) -> list[Path]:
    return sorted(child for child in parent.iterdir() if is_problem_dir(child))


def problem_category_label(problem_dir: Path) -> str:
    """Return a stable category label for output paths and summaries."""
    parent = problem_dir.parent
    if parent.parent.name == "Contest":
        return f"Contest/{parent.name}"
    return parent.name


def discover_problems(
    benchmark_dir: Path, categories: list[str] | None = None
) -> list[Path]:
    """Return a sorted list of problem directories under *benchmark_dir*.

    If *categories* is given (e.g. ["L1", "L2"]), only those sub-directories
    are searched.
    """
    if categories:
        roots = [benchmark_dir / c for c in categories]
    else:
        direct = _collect_problems(benchmark_dir)
        if direct:
            return direct
        roots = sorted(
            p for p in benchmark_dir.iterdir()
            if p.is_dir() and p.name in CATEGORIES
        )

    problems = []
    for root in roots:
        if not root.is_dir():
            continue
        problems.extend(_collect_problems(root))
    return problems


def load_problem(problem_path: str | Path) -> tuple[Path, dict]:
    """Resolve a problem path and return (problem_dir, definition_dict).

    Accepts paths relative to data/benchmark/Contest/ or absolute paths.
    """
    candidates = [
        Path(problem_path),
        CONTEST_ROOT / problem_path,
    ]
    for c in candidates:
        if c.is_dir() and (c / "definition.json").exists():
            defn = json.loads((c / "definition.json").read_text())
            return c, defn
    raise FileNotFoundError(
        f"could not find problem at {problem_path}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def load_workloads(problem_dir: Path) -> list[dict]:
    """Load and parse workload.jsonl rows."""
    path = problem_dir / "workload.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def import_module_from_path(py_path: Path, label: str):
    """Import a Python file and return the module object.

    label is used for the module name (must be unique per file).
    """
    spec = importlib.util.spec_from_file_location(f"_harness_{label}", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_run_fn(py_path: Path, label: str):
    """Import a Python file and return its `run` function and optional `get_inputs`."""
    mod = import_module_from_path(py_path, label)
    if not hasattr(mod, "run"):
        raise AttributeError(f"{py_path} does not export `run`")
    return mod.run, getattr(mod, "get_inputs", None)