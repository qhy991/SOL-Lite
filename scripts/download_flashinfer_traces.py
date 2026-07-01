#!/usr/bin/env python3
"""Download FlashInfer trace safetensors from HuggingFace.

The 5 FlashInfer-Bench problems (013, 017, 018, 019, 020) require
safetensors files for indptr/page-table inputs. These are hosted on
HuggingFace at:

    https://huggingface.co/datasets/nvidia/flashinfer-trace

Usage:
    uv run python scripts/download_flashinfer_traces.py

This will download all required safetensors files into:
    data/flashinfer-trace/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "flashinfer-trace"
CONTEST_ROOT = ROOT / "data" / "benchmark" / "Contest"

HF_REPO = "https://huggingface.co/datasets/nvidia/flashinfer-trace/resolve/main"


def collect_safetensors_paths() -> set[str]:
    """Scan all FlashInfer workload files and collect unique safetensors paths."""
    seen = set()
    flashinfer_dir = CONTEST_ROOT / "FlashInfer-Bench"
    if not flashinfer_dir.is_dir():
        print(f"WARNING: {flashinfer_dir} not found", file=sys.stderr)
        return seen

    for problem_dir in flashinfer_dir.iterdir():
        wl_path = problem_dir / "workload.jsonl"
        if not wl_path.exists():
            continue
        for line in wl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            wl = json.loads(line)
            for v in wl.get("inputs", {}).values():
                if isinstance(v, dict) and v.get("type") == "safetensors":
                    path = v.get("path")
                    if path:
                        seen.add(path)

    return seen


def download_file(url: str, dest: Path) -> bool:
    """Download a single file. Returns True on success."""
    try:
        import urllib.request
    except ImportError:
        print("ERROR: urllib.request not available", file=sys.stderr)
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} (already exists)")
        return True

    print(f"  downloading {dest.name} ... ", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"{size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Download FlashInfer trace safetensors from HuggingFace."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded without downloading.",
    )
    args = ap.parse_args()

    paths = collect_safetensors_paths()
    if not paths:
        print("No safetensors paths found in workload files.")
        return

    print(f"Found {len(paths)} unique safetensors paths:\n")
    for p in sorted(paths):
        print(f"  {p}")

    if args.dry_run:
        print("\nDry run — no files downloaded.")
        return

    print(f"\nDownloading to {DATA_DIR} ...\n")
    ok = 0
    failed = 0
    for p in sorted(paths):
        # Paths are like: data/flashinfer-trace/blob/workloads/...
        # The HF repo has the blob/ prefix removed
        dest = DATA_DIR / p
        url = f"{HF_REPO}/{p}"
        if download_file(url, dest):
            ok += 1
        else:
            failed += 1

    print(f"\nDone: {ok} downloaded, {failed} failed")
    if ok > 0:
        print(f"Files saved to {DATA_DIR}")
        print("You can now run FlashInfer-Bench problems with:")
        print("  uv run python scripts/run_contest.py --category FlashInfer-Bench --smoke")


if __name__ == "__main__":
    main()