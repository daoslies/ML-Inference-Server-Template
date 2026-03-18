import re
from gguf import GGUFReader
from typing import Dict, Any, List
import numpy as np

GGUF_FILE_TYPES = {
    0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 7: "MOSTLY_Q2_K",
    15: "MOSTLY_Q8_0", 17: "MOSTLY_Q4_K_M", 18: "MOSTLY_Q5_K_M",
    # add more as needed from gguf spec
}

def try_decode(val):
    # Unwrap numpy arrays/memmaps to a plain list first
    if isinstance(val, (np.ndarray, np.memmap)):
        val = val.tolist()
    if isinstance(val, (list, tuple)) and all(isinstance(x, int) for x in val):
        # Single-element int lists are likely enum values, not strings
        if len(val) == 1:
            return val[0]
        try:
            return bytes(val).decode('utf-8')
        except Exception:
            return str(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    return val

def to_jsonable(obj):
    # Recursively convert numpy types to Python types
    if isinstance(obj, dict):
        return {to_jsonable(k): to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.memmap):
        return obj.tolist()
    else:
        return obj

def parse_fields(reader) -> Dict[str, Any]:
    fields = {}
    for name, field in reader.fields.items():
        try:
            if not field.data:
                fields[name] = None
                continue
            # For single-element fields
            if len(field.data) == 1:
                raw = field.parts[field.data[0]]
                fields[name] = try_decode(raw)
            else:
                # Multi-part fields (e.g. string arrays)
                fields[name] = [try_decode(field.parts[i]) for i in field.data]
        except Exception:
            fields[name] = "<unreadable>"
    return fields

def summarize_tensor_patterns(reader) -> Dict[str, int]:
    patterns = {}
    for t in reader.tensors:
        pattern = re.sub(r'\d+', 'N', t.name)
        patterns[pattern] = patterns.get(pattern, 0) + 1
    return patterns

def infer_lora_ranks(reader) -> Dict[str, int]:
    import numpy as np
    ranks = {}
    for t in reader.tensors:
        if 'lora_a' in t.name:
            # Ensure t.shape is a tuple of ints
            shape = tuple(int(x) for x in t.shape)
            ranks[t.name] = int(min(shape)) if shape else None
    return ranks

def lora_coverage(reader) -> List[str]:
    layers = []
    for t in reader.tensors:
        if 'lora_a' in t.name:
            layers.append(t.name)
    return layers

def get_relevant_tensor_shapes(reader):
    shapes = {}
    key_patterns = ["lora_", "attn_k", "attn_q", "attn_v", "attn_output", "ffn", "output.weight"]
    for t in reader.tensors:
        if any(p in t.name for p in key_patterns):
            shapes[t.name] = [int(x) for x in t.shape]
    return shapes

def check_warnings(fields: Dict[str, Any], ranks: Dict[str, int], patterns: Dict[str, int]) -> List[str]:
    warnings = []
    if 'adapter.type' not in fields or fields['adapter.type'] != 'lora':
        warnings.append("'adapter.type' field missing or not 'lora' — may not load in llama.cpp")
    if 'general.architecture' not in fields:
        warnings.append("'general.architecture' field missing")
    if len(set(ranks.values())) > 1:
        warnings.append("Inconsistent LoRA rank across layers")
    if not any('lora_a' in k for k in patterns):
        warnings.append("No LoRA tensors found (no 'lora_a' patterns)")
    return warnings

def inspect_gguf(path: str) -> Any:
    reader = GGUFReader(path)
    fields = parse_fields(reader)
    patterns = summarize_tensor_patterns(reader)
    ranks = infer_lora_ranks(reader)
    lora_layers = lora_coverage(reader)
    warnings = check_warnings(fields, ranks, patterns)
    tensor_shapes = get_relevant_tensor_shapes(reader)
    result = {
        "file": path,
        "metadata": fields,
        "tensor_summary": {
            "total": len(reader.tensors),
            "patterns": patterns
        },
        "tensor_shapes": tensor_shapes,
        "lora_info": {
            "rank": list(set(ranks.values()))[0] if ranks else None,
            "covered_layers": lora_layers,
            "missing_layers": []  # For future: compare to reference
        },
        "warnings": warnings
    }
    return to_jsonable(result)

def inspect_gguf_compare(path1: str, path2: str, layer: int = 0) -> dict:
    result1 = inspect_gguf(path1)
    result2 = inspect_gguf(path2)
    shape_cmp = compare_tensor_shapes_struct(path1, path2, layer=layer)
    return {
        'file1': result1,
        'file2': result2,
        'tensor_shape_comparison': shape_cmp
    }

def compare_tensor_shapes(path_a, path_b, layer=0):
    reader_a = GGUFReader(path_a)
    reader_b = GGUFReader(path_b)
    tensors_a = {t.name: tuple(int(x) for x in t.shape) for t in reader_a.tensors}
    tensors_b = {t.name: tuple(int(x) for x in t.shape) for t in reader_b.tensors}
    prefix = f"blk.{layer}."
    keys = sorted(set(
        k for k in list(tensors_a.keys()) + list(tensors_b.keys())
        if k.startswith(prefix)
    ))
    print(f"{'Tensor':<45} {'Reference':>20} {'Yours':>20} {'Match':>8}")
    print("-" * 95)
    for k in keys:
        a = tensors_a.get(k, "MISSING")
        b = tensors_b.get(k, "MISSING")
        match = "✓" if a == b else "✗ MISMATCH"
        print(f"{k:<45} {str(a):>20} {str(b):>20} {match:>8}")

def compare_tensor_shapes_struct(path_a, path_b, layer=0):
    reader_a = GGUFReader(path_a)
    reader_b = GGUFReader(path_b)
    tensors_a = {t.name: tuple(int(x) for x in t.shape) for t in reader_a.tensors}
    tensors_b = {t.name: tuple(int(x) for x in t.shape) for t in reader_b.tensors}
    prefix = f"blk.{layer}."
    keys = sorted(set(
        k for k in list(tensors_a.keys()) + list(tensors_b.keys())
        if k.startswith(prefix)
    ))
    results = []
    for k in keys:
        a = tensors_a.get(k, "MISSING")
        b = tensors_b.get(k, "MISSING")
        # If either is missing, that's a real mismatch
        if a == "MISSING" or b == "MISSING":
            match = False
            info = None
        elif a == b:
            match = True
            info = None
        else:
            # Try to detect rank-only difference for lora_a and lora_b
            if (
                isinstance(a, tuple) and isinstance(b, tuple) and
                len(a) == 2 and len(b) == 2 and
                ("lora_a" in k or "lora_b" in k)
            ):
                if "lora_a" in k:
                    # [out_features, rank_N]
                    if a[0] == b[0]:
                        match = True
                        info = f"rank differs: {a[1]} vs {b[1]}"
                    else:
                        match = False
                        info = "STRUCTURAL MISMATCH"
                elif "lora_b" in k:
                    # [rank_N, in_features]
                    if a[1] == b[1]:
                        match = True
                        info = f"rank differs: {a[0]} vs {b[0]}"
                    else:
                        match = False
                        info = "STRUCTURAL MISMATCH"
                else:
                    match = False
                    info = None
            else:
                match = False
                info = None
        results.append({
            "tensor": k,
            "reference": a,
            "yours": b,
            "match": match,
            "info": info
        })
    return results
