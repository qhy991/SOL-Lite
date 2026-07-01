"""Input generation for Contest problems.

Supports three modes:
1. Random tensor synthesis from definition.json axes + inputs specs
2. Custom get_inputs() when definition declares custom_inputs_entrypoint
3. Safetensors loading for FlashInfer-Bench problems

Also handles DPS (destination-passing style) detection and output allocation.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

DTYPE_MAP = {
    "float32": "float32", "float16": "float16", "bfloat16": "bfloat16",
    "float8_e4m3fn": "float8_e4m3fn", "float8_e5m2": "float8_e5m2",
    "int8": "int8", "int32": "int32", "int64": "int64", "bool": "bool",
}


def _resolve_namespace(defn: dict, axes: dict) -> dict:
    """Build a full namespace from workload axes + consts + exprs in the definition.

    Returns a dict mapping every axis name to its resolved value.
    """
    namespace = dict(axes)
    for k, v in defn["axes"].items():
        if v.get("type") == "const":
            namespace.setdefault(k, v["value"])

    # Iteratively resolve expressions (one expr may reference another)
    pending = {
        k: v["expression"]
        for k, v in defn["axes"].items()
        if v.get("type") == "expr"
    }
    progress = True
    while pending and progress:
        progress = False
        for k in list(pending):
            try:
                namespace[k] = eval(pending[k], {}, namespace)
                del pending[k]
                progress = True
            except (NameError, SyntaxError):
                continue
    return namespace


def generate_inputs(
    defn: dict,
    axes: dict,
    get_inputs_fn,
    device: str,
) -> dict:
    """Produce a kwargs dict ready to pass to run().

    If the definition declares ``custom_inputs_entrypoint``, use the
    reference's get_inputs() function. Otherwise synthesize random tensors
    from the declared shapes + dtypes.
    """
    import torch

    if get_inputs_fn is not None and defn.get("custom_inputs_entrypoint"):
        axes_and_scalars = _resolve_namespace(defn, axes)
        return get_inputs_fn(axes_and_scalars, torch.device(device))

    # Synthesize random inputs from the spec
    consts = _resolve_namespace(defn, axes)

    out = {}
    for k, v in defn["inputs"].items():
        shape_spec = v.get("shape")
        dtype_str = v.get("dtype")
        if shape_spec is None:
            # scalar — must come from workload's "inputs" section
            continue

        shape = []
        for s in shape_spec:
            if isinstance(s, int):
                shape.append(s)
                continue
            if s in consts:
                shape.append(consts[s])
                continue
            if isinstance(s, str) and s.isdigit():
                shape.append(int(s))
                continue
            # Last attempt: eval expression in namespace
            try:
                shape.append(int(eval(s, {}, consts)))
            except Exception:
                raise KeyError(
                    f"unresolved shape axis '{s}' for input '{k}' "
                    f"(known: {sorted(consts)})"
                )

        dtype = getattr(torch, DTYPE_MAP.get(dtype_str, dtype_str))

        if dtype_str.startswith("float") or dtype_str.startswith("bfloat"):
            if dtype_str.startswith("float8"):
                out[k] = torch.randn(*shape, dtype=torch.float32, device=device).to(dtype)
            else:
                out[k] = torch.randn(*shape, dtype=dtype, device=device)
        elif dtype_str.startswith("int"):
            out[k] = torch.zeros(*shape, dtype=dtype, device=device)
        elif dtype_str == "bool":
            out[k] = torch.ones(*shape, dtype=dtype, device=device)
        else:
            out[k] = torch.zeros(*shape, dtype=dtype, device=device)

    return out


def materialize_scalars_and_safetensors(
    workload_inputs: dict,
    kwargs: dict,
    safetensors_roots: list[Path] | None = None,
) -> dict:
    """Pull scalar values and safetensors from workload inputs into kwargs.

    - Scalar inputs: inject the value directly
    - Safetensors inputs: load from disk using safetensors.torch.load_file

    safetensors_roots is a list of directories to search for .safetensors files.
    Defaults to [data/flashinfer-trace/, $SOL_EXECBENCH_ROOT].
    """
    import torch

    if safetensors_roots is None:
        _roots: list[Path] = []
        local = ROOT / "data" / "flashinfer-trace"
        if local.exists():
            _roots.append(local)
        env_root = os.environ.get("SOL_EXECBENCH_ROOT")
        if env_root:
            _roots.append(Path(env_root))
        safetensors_roots = _roots

    for k, v in workload_inputs.items():
        if not isinstance(v, dict):
            continue
        t = v.get("type")
        if t == "scalar":
            kwargs[k] = v["value"]
        elif t == "safetensors":
            path = v.get("path")
            key = v.get("tensor_key")
            if not path or not key:
                continue
            loaded = False
            for root in safetensors_roots:
                candidate = root / path if not Path(path).is_absolute() else Path(path)
                if candidate.exists():
                    try:
                        from safetensors.torch import load_file
                        loaded_tensors = load_file(str(candidate))
                        if key in loaded_tensors:
                            kwargs[k] = loaded_tensors[key].to("cuda")
                            loaded = True
                            break
                    except Exception:
                        pass
            if not loaded:
                raise FileNotFoundError(
                    f"safetensors input '{k}': could not load {path}/{key} "
                    f"from any of {safetensors_roots}"
                )
    return kwargs


# ---------------------------------------------------------------------------
# DPS (destination-passing style) detection and output allocation
# ---------------------------------------------------------------------------


def _infer_dps(code: str, definition: dict) -> bool:
    """Infer destination-passing style by checking the ``run()`` signature.

    If the last parameter of ``run()`` matches the last output name in the
    definition, the solution writes into pre-allocated output buffers (DPS).
    """
    output_names = list(definition.get("outputs", {}).keys())
    if not output_names:
        return False

    last_output = output_names[-1]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ):
            args = node.args
            if args.args:
                last_param = args.args[-1].arg
                return last_param == last_output
            break

    return False


def allocate_outputs(defn: dict, consts: dict, device: str) -> dict:
    """Allocate output tensors for DPS solutions.

    Returns a dict mapping output name to an empty tensor of the right shape+dtype.
    """
    import torch

    outputs = {}
    for k, v in defn["outputs"].items():
        shape_spec = v.get("shape")
        dtype_str = v.get("dtype", "bfloat16")
        if shape_spec is None:
            continue
        shape = []
        for s in shape_spec:
            if isinstance(s, int):
                shape.append(s)
            elif s in consts:
                shape.append(consts[s])
            elif isinstance(s, str) and s.isdigit():
                shape.append(int(s))
            else:
                try:
                    shape.append(int(eval(s, {}, consts)))
                except Exception:
                    raise KeyError(
                        f"unresolved shape axis '{s}' for output '{k}' "
                        f"(known: {sorted(consts)})"
                    )
        dtype = getattr(torch, DTYPE_MAP.get(dtype_str, dtype_str))
        outputs[k] = torch.empty(*shape, dtype=dtype, device=device)
    return outputs