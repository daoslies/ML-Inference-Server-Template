#!/usr/bin/env python3
"""
ml — a CLI for interacting with the ML inference server.

Usage:
    ml health
        Check server health and loaded model.
    ml load <model_path>
        Load a model.

    ml infer <prompt> [--max-tokens N] [--temperature F]
        Run inference on a single prompt.

    ml batch <prompt1> <prompt2> ... [--max-tokens N]
        Run inference on multiple prompts.

    ml extract <text> <prompts_path>
        Extract names from text.

    ml shell
        Interactive REPL mode.

    ml unload
        Unload the currently loaded model.

    ml info <model_1.gguf> <model_2_optional_.gguf>
        Show model architecture info. 
        use two model args to compare structures.
"""

import sys
import json
import argparse
import urllib.request
import urllib.error

# -------------------------------------------------------
# Config
# -------------------------------------------------------
def get_port(default: int = 27776) -> int:
    try:
        import yaml
        with open("config.yaml") as f:
            c = yaml.safe_load(f)
        return c.get("server", {}).get("port", default)
    except Exception:
        return default

# -------------------------------------------------------
# Output helpers
# -------------------------------------------------------
CYAN  = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW= "\033[1;33m"
RED   = "\033[0;31m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
NC    = "\033[0m"

def _supports_colour() -> bool:
    import os
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty() \
           and os.environ.get("NO_COLOR") is None

USE_COLOUR = _supports_colour()

def c(code: str, text: str) -> str:
    return f"{code}{text}{NC}" if USE_COLOUR else text

def print_response(label: str, value: str) -> None:
    print(f"\n  {c(BOLD+CYAN, label)}\n  {value}\n")

def print_error(msg: str) -> None:
    print(f"\n  {c(RED, '✗')} {msg}\n", file=sys.stderr)

def print_ok(msg: str) -> None:
    print(f"  {c(GREEN, '✓')} {msg}")

def print_warn(msg: str) -> None:
    print(f"  {c(YELLOW, '!')} {msg}")

# -------------------------------------------------------
# HTTP
# -------------------------------------------------------
def _request(method: str, url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not connect to server — is it running? ({e.reason})") from e

def get(url: str) -> dict:
    return _request("GET", url)

def post(url: str, payload: dict) -> dict:
    return _request("POST", url, payload)

# -------------------------------------------------------
# Commands
# -------------------------------------------------------
def cmd_health(base: str, _args) -> int:
    data = get(f"{base}/health")
    model = data.get("model_path") or "none"
    status = data.get("status", "unknown")
    print_ok(f"status: {c(BOLD, status)}   model: {c(CYAN, model)}")
    return 0

def cmd_load(base: str, args) -> int:
    print(f"  {c(DIM, 'Loading')} {args.model_path} …")
    payload = {"model_path": args.model_path}
    if getattr(args, "lora_path", None):
        payload["lora_path"] = args.lora_path
    data = post(f"{base}/load_model", payload)
    print_ok(data.get("message") or "Model loaded.")
    return 0

def cmd_infer(base: str, args) -> int:
    payload = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    data = post(f"{base}/inference", payload)
    response = data.get("results", {}).get("response", "")
    if not response:
        print_warn(f"Empty response. Raw: {json.dumps(data)}")
        return 1
    print_response("Response", response)
    return 0

def cmd_batch(base: str, args) -> int:
    payload = {"prompt": args.prompts, "max_tokens": args.max_tokens}
    data = post(f"{base}/inference", payload)
    results = data.get("results", [])
    if not results:
        print_warn(f"Empty response. Raw: {json.dumps(data)}")
        return 1
    for i, r in enumerate(results, 1):
        response = r.get("response", "") if isinstance(r, dict) else r
        print_response(f"[{i}]", response)
    return 0

def cmd_extract(base: str, args) -> int:
    payload = {"text": args.text, "prompts_path": args.prompts_path}
    data = post(f"{base}/extract_names", payload)
    names = data.get("names") or data.get("results") or data
    if isinstance(names, list):
        print_ok("Extracted names:")
        for name in names:
            print(f"    • {name}")
    else:
        print_response("Result", json.dumps(names, indent=2))
    return 0

def cmd_unload(base: str, _args) -> int:
    data = post(f"{base}/unload_model", {})
    print_ok(data.get("message") or "Model unloaded.")
    return 0

def print_model_info_block(info, label_prefix=""):
    print_response(f"{label_prefix}File", info.get("file", ""))
    meta = info.get("metadata", {})
    print(f"\n  {c(BOLD, f'{label_prefix}Metadata (LoRA-relevant fields):')}")
    for k in sorted(meta):
        if any(x in k for x in ['lora', 'adapter', 'training', 'general']):
            print(f"    {k:30s}: {meta[k]}")
    tensor_summary = info.get("tensor_summary", {})
    print(f"\n  {c(BOLD, f'{label_prefix}Tensor patterns:')}")
    for p, count in sorted(tensor_summary.get("patterns", {}).items()):
        print(f"    {count:3d}x  {p}")
    lora_info = info.get("lora_info", {})
    print(f"\n  {c(BOLD, f'{label_prefix}Inferred LoRA ranks:')}")
    ranks = lora_info.get('ranks')
    if ranks and isinstance(ranks, dict):
        for layer, rank in sorted(ranks.items()):
            print(f"    {layer}: rank={rank}")
    else:
        for layer in lora_info.get("covered_layers", []):
            print(f"    {layer}: rank={lora_info.get('rank')}")
    if info.get("warnings"):
        print(f"\n  {c(RED, f'{label_prefix}Warnings:')}")
        for w in info["warnings"]:
            print(f"    ⚠ {w}")
    # If lora_adapter present, print its info as well
    if "lora_adapter" in info:
        print(f"\n  {c(BOLD+CYAN, f'{label_prefix}LoRA Adapter Info:')}")
        lora = info["lora_adapter"]
        print(f"    File: {lora.get('file','')}")
        meta = lora.get("metadata", {})
        print(f"    Metadata:")
        for k in sorted(meta):
            if any(x in k for x in ['lora', 'adapter', 'training', 'general']):
                print(f"      {k:28s}: {meta[k]}")
        tensor_summary = lora.get("tensor_summary", {})
        print(f"    Tensor patterns:")
        for p, count in sorted(tensor_summary.get("patterns", {}).items()):
            print(f"      {count:3d}x  {p}")
        lora_info = lora.get("lora_info", {})
        print(f"    Inferred LoRA ranks:")
        ranks = lora_info.get('ranks')
        if ranks and isinstance(ranks, dict):
            for layer, rank in sorted(ranks.items()):
                print(f"      {layer}: rank={rank}")
        else:
            for layer in lora_info.get("covered_layers", []):
                print(f"      {layer}: rank={lora_info.get('rank')}")
        if lora.get("warnings"):
            print(f"    {c(RED, 'Warnings:')}")
            for w in lora["warnings"]:
                print(f"      ⚠ {w}")


def cmd_info(base: str, args) -> int:
    payload = {"model_path": args.model_path}
    if getattr(args, "lora_path", None):
        payload["lora_path"] = args.lora_path
    if getattr(args, "compare_path", None):
        payload["compare_path"] = args.compare_path
    if getattr(args, "json", False):
        payload["json"] = True
    data = post(f"{base}/model_info", payload)
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    if data.get("status") == "success":
        info = data.get("info", {})
        # Handle comparison response
        if "file1" in info and "file2" in info:
            print_model_info_block(info["file1"], label_prefix="Reference: ")
            print_model_info_block(info["file2"], label_prefix="Yours:    ")
            # Tensor shape comparison
            if "tensor_shape_comparison" in info:
                print(f"\n  {c(BOLD+YELLOW, 'Tensor shape comparison for blk.0.*:')}")
                print(f"    {'Tensor':<45} {'Reference':>20} {'Yours':>20} {'Match':>8}  Info")
                print("    " + "-" * 100)
                for row in info["tensor_shape_comparison"]:
                    a = row['reference']
                    b = row['yours']
                    if row.get('info'):
                        match = f"✓ ({row['info']})" if row['match'] else f"✗ {row['info']}"
                    else:
                        match = "✓" if row['match'] else "✗ MISMATCH"
                    print(f"    {row['tensor']:<45} {str(a):>20} {str(b):>20} {match:>12}")
            return 0
        # Single model info
        print_model_info_block(info)
        return 0
    else:
        print_error(data.get("error", "Unknown error"))
        if data.get("traceback"):
            print_warn(data["traceback"])
        return 1

# -------------------------------------------------------
# REPL shell
# -------------------------------------------------------
SHELL_HELP = f"""
  {BOLD}Commands:{NC}
    health
    load <model_path>
    infer <prompt>  [--max-tokens N]  [--temperature F]
    batch <p1> <p2> ...  [--max-tokens N]
    extract <text> <prompts_path>
    unload
    info <model_path> [--lora-path <lora_path>]
    help
    exit
"""

def _parse_shell_line(line: str) -> list[str]:
    """Simple shell-style tokeniser — respects single/double quotes."""
    import shlex
    try:
        return shlex.split(line)
    except ValueError:
        return line.split()

def cmd_shell(base: str, _args) -> int:
    print(f"\n  {c(BOLD+CYAN, 'ml shell')} — type {c(BOLD, 'help')} for commands, {c(BOLD, 'exit')} to quit.\n")
    while True:
        try:
            raw = input(c(CYAN + BOLD, "ml> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        if raw in ("exit", "quit", "q"):
            break
        if raw in ("help", "?"):
            print(SHELL_HELP)
            continue

        tokens = _parse_shell_line(raw)
        # Re-use the same argument parser — just prepend a dummy prog name.
        try:
            parsed, rc = _dispatch(tokens, base)
        except SystemExit:
            # argparse calls sys.exit on bad args; swallow it in the REPL.
            pass
        except RuntimeError as e:
            print_error(str(e))

    return 0

# -------------------------------------------------------
# Argument parsing & dispatch
# -------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml",
        description="CLI for the ML inference server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=None,
                        help="Server port (default: read from config.yaml)")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # health
    sub.add_parser("health", help="Check server health and loaded model.")

    # load
    p_load = sub.add_parser("load", help="Load a model.")
    p_load.add_argument("model_path", help="HuggingFace model path or local path.")
    p_load.add_argument("--lora-path", help="Optional: Path to LoRA adapter (GGUF)", default=None)

    # infer
    p_infer = sub.add_parser("infer", help="Run inference on a single prompt.")
    p_infer.add_argument("prompt")
    p_infer.add_argument("--max-tokens", type=int, default=150, dest="max_tokens")
    p_infer.add_argument("--temperature", type=float, default=0.7)

    # batch
    p_batch = sub.add_parser("batch", help="Run inference on multiple prompts.")
    p_batch.add_argument("prompts", nargs="+")
    p_batch.add_argument("--max-tokens", type=int, default=150, dest="max_tokens")

    # extract
    p_extract = sub.add_parser("extract", help="Extract names from text.")
    p_extract.add_argument("text")
    p_extract.add_argument("prompts_path")

    # shell
    sub.add_parser("shell", help="Interactive REPL mode.")

    # unload
    sub.add_parser("unload", help="Unload the currently loaded model.")

    # info
    p_info = sub.add_parser("info", help="Show model info (GGUF only).")
    p_info.add_argument("model_path", help="Path to GGUF model file.")
    p_info.add_argument("compare_path", nargs="?", help="Optional: Path to second GGUF file for comparison.")
    p_info.add_argument("--lora-path", help="Optional: Path to LoRA adapter (GGUF)", default=None)
    p_info.add_argument("--json", action="store_true", help="Output raw JSON from the server.")

    return parser

COMMANDS = {
    "health":  cmd_health,
    "load":    cmd_load,
    "infer":   cmd_infer,
    "batch":   cmd_batch,
    "extract": cmd_extract,
    "shell":   cmd_shell,
    "unload":  cmd_unload,
    "info":    cmd_info,
}

def _dispatch(argv: list[str], base: str) -> tuple[argparse.Namespace, int]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return args, 0
    fn = COMMANDS[args.command]
    rc = fn(base, args)
    return args, rc

def main() -> None:
    # Peek at --port before full parse so we can build the base URL.
    port = get_port()
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--port" and i + 2 < len(sys.argv):
            try:
                port = int(sys.argv[i + 2])
            except ValueError:
                pass

    base = f"http://localhost:{port}"

    try:
        _, rc = _dispatch(sys.argv[1:], base)
        sys.exit(rc)
    except RuntimeError as e:
        print_error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
