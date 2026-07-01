"""Correctness checking: compare solution output against reference.

Handles tuple/list outputs, single tensor outputs, and special tolerance
flags like allow_negative_inf (used by FlashInfer attention LSE).
"""
from __future__ import annotations


def _to_tuple(output):
    """Normalize output to a tuple for uniform handling."""
    if isinstance(output, (tuple, list)):
        return tuple(output)
    return (output,)


def compute_error_stats(
    ref_output,
    sol_output,
    tolerance: dict,
) -> dict:
    """Compute error statistics between reference and solution outputs.

    Returns a dict with:
        status: "PASSED" | "FAILED"
        max_atol: float
        max_rtol: float
        per_output: list of per-output error details
    """
    import torch

    ref_tuple = _to_tuple(ref_output)
    sol_tuple = _to_tuple(sol_output)

    max_atol = tolerance.get("max_atol")
    max_rtol = tolerance.get("max_rtol")
    allow_negative_inf = tolerance.get("allow_negative_inf", False)

    n_outputs = len(ref_tuple)
    if len(sol_tuple) != n_outputs:
        return {
            "status": "FAILED",
            "error": (
                f"output count mismatch: reference={n_outputs}, solution={len(sol_tuple)}"
            ),
            "max_atol": None,
            "max_rtol": None,
            "per_output": [],
        }

    per_output = []
    all_passed = True

    for i, (ref, sol) in enumerate(zip(ref_tuple, sol_tuple)):
        if not isinstance(ref, type(sol)):
            per_output.append({
                "index": i,
                "status": "FAILED",
                "error": f"type mismatch: ref={type(ref).__name__}, sol={type(sol).__name__}",
            })
            all_passed = False
            continue

        # Handle tensors
        if hasattr(ref, "shape"):
            r = ref.float()
            s = sol.float().to(r.device)

            if r.shape != s.shape:
                per_output.append({
                    "index": i,
                    "status": "FAILED",
                    "error": f"shape mismatch: ref={tuple(r.shape)}, sol={tuple(s.shape)}",
                })
                all_passed = False
                continue

            diff = (r - s).abs()
            denom_tensor = r

            # Special handling for allow_negative_inf: mask out positions
            # where both reference and solution are -inf BEFORE computing
            # error stats (NaN from inf-inf would poison the comparison).
            if allow_negative_inf:
                ref_inf = torch.isinf(r) & (r < 0)
                sol_inf = torch.isinf(s) & (s < 0)
                both_inf = ref_inf & sol_inf
                if both_inf.any():
                    valid = ~both_inf
                    if valid.any():
                        diff = diff[valid]
                        denom_tensor = r[valid]
                    else:
                        # All values are -inf in both — trivially correct
                        per_output.append({
                            "index": i,
                            "status": "PASSED",
                            "max_abs_diff": 0.0,
                            "max_rel_diff": 0.0,
                            "atol_threshold": max_atol,
                            "rtol_threshold": max_rtol,
                            "atol_ok": True,
                            "rtol_ok": True,
                        })
                        continue

            max_abs = diff.max().item()
            denom = denom_tensor.abs().max().item()
            rel = max_abs / denom if denom > 1e-12 else max_abs

            atol_ok = True
            rtol_ok = True

            if max_atol is not None:
                atol_ok = max_abs <= max_atol
            if max_rtol is not None:
                rtol_ok = rel <= max_rtol

            passed = atol_ok and rtol_ok

            per_output.append({
                "index": i,
                "status": "PASSED" if passed else "FAILED",
                "max_abs_diff": max_abs,
                "max_rel_diff": rel,
                "atol_threshold": max_atol,
                "rtol_threshold": max_rtol,
                "atol_ok": atol_ok,
                "rtol_ok": rtol_ok,
            })

            if not passed:
                all_passed = False

        elif isinstance(ref, (int, float, bool)):
            diff = abs(ref - sol)
            rel = diff / max(abs(ref), 1e-12) if abs(ref) > 1e-12 else diff
            atol_ok = max_atol is None or diff <= max_atol
            rtol_ok = max_rtol is None or rel <= max_rtol
            passed = atol_ok and rtol_ok
            per_output.append({
                "index": i,
                "status": "PASSED" if passed else "FAILED",
                "max_abs_diff": diff,
                "max_rel_diff": rel,
                "atol_ok": atol_ok,
                "rtol_ok": rtol_ok,
            })
            if not passed:
                all_passed = False
        else:
            per_output.append({
                "index": i,
                "status": "FAILED",
                "error": f"unsupported type: {type(ref).__name__}",
            })
            all_passed = False

    return {
        "status": "PASSED" if all_passed else "FAILED",
        "max_atol": max_atol,
        "max_rtol": max_rtol,
        "per_output": per_output,
    }