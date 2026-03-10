#!/usr/bin/env python3
"""
ml — a CLI for interacting with the ML inference server.

Usage:
    ml health
    ml load <model_path>
    ml infer <prompt> [--max-tokens N] [--temperature F]
    ml batch <prompt1> <prompt2> ... [--max-tokens N]
    ml extract <text> <prompts_path>
    ml shell
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
    data = post(f"{base}/load_model", {"model_path": args.model_path})
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

    return parser

COMMANDS = {
    "health":  cmd_health,
    "load":    cmd_load,
    "infer":   cmd_infer,
    "batch":   cmd_batch,
    "extract": cmd_extract,
    "shell":   cmd_shell,
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
