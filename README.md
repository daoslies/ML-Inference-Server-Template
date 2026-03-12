# ML-Inference-Server-Template

## To Run

```

bash Run_ML_Server.sh

```

this'll make a venv and deal with your dependencies


## To Uninstall

```

rm -rf ~/venvs/ml_server_template

```


## Command-line client (ml.py)

A simple CLI for interacting with the server is provided as `ml.py`:

```bash
chmod +x ml.py
alias ml="python ml.py"  # (optional, add to your .bashrc)
```

Example usage:

- `ml health` — check server status
- `ml load Qwen/Qwen2.5-7B-Instruct` — load a model
- `ml infer "Summarize this..."` — run inference
- `ml batch "Prompt one" "Prompt two"` — batch inference
- `ml extract "Alice and Bob met..." prompts.yaml` — extract names
- `ml shell` — interactive REPL

The CLI reads the port from `config.yaml` by default, and prints clean, colorized output.


## llama-server Backend (Qwen3.5 and future models)

This server now uses the official [llama-server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server) binary for all llama.cpp-based models (including Qwen3.5). The legacy `llama-cpp-python` and custom .so logic has been removed.

### Configuration

Edit `config.yaml` (or copy from `config.yaml.example`) and set:

```yaml
llama_cpp:
  server_bin: /absolute/path/to/llama-server  # Path to the llama-server binary (required)
  server_port: 8000                           # Port for llama-server to listen on (required)
```

- The server will launch `llama-server` as a subprocess and connect to it for inference.
- `LD_LIBRARY_PATH` is set automatically based on the binary location.
- No Python bindings or custom .so builds are required.

### Requirements
- `llama-cpp-python` is no longer needed and has been removed from `requirements.txt`.
- Only `vllm` and other core dependencies remain.


## Quick API Examples (curl & Python)

### curl

```bash
# 1) Health
curl -s http://localhost:{PORT}/health

# 2) Load model
curl -X POST http://localhost:{PORT}/load_model \
  -H "Content-Type: application/json" \
  -d '{"model_path":"Qwen/Qwen2.5-7B-Instruct"}'

# 3) Inference
curl -X POST http://localhost:{PORT}/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Summarize this...","max_tokens":150}'

# 4) Batch inference
curl -X POST http://localhost:{PORT}/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt":["Prompt one","Prompt two"],"max_tokens":150}'

# 5) Extract names
curl -X POST http://localhost:{PORT}/extract_names \
  -H "Content-Type: application/json" \
  -d '{"text":"Alice and Bob...","prompts_path":"textflow/prompts.yaml"}'
```

### Python (requests)

```python
import requests

PORT = 27776  # or your configured port

# 1) Health
resp = requests.get(f"http://localhost:{PORT}/health")
print(resp.json())

# 2) Load model
resp = requests.post(f"http://localhost:{PORT}/load_model", json={"model_path": "Qwen/Qwen2.5-7B-Instruct"})
print(resp.json())

# 3) Inference
resp = requests.post(f"http://localhost:{PORT}/inference", json={"prompt": "Summarize this...", "max_tokens": 150})
print(resp.json())

# 4) Batch inference
resp = requests.post(f"http://localhost:{PORT}/inference", json={"prompt": ["Prompt one", "Prompt two"], "max_tokens": 150})
print(resp.json())

# 5) Extract names
resp = requests.post(f"http://localhost:{PORT}/extract_names", json={"text": "Alice and Bob...", "prompts_path": "textflow/prompts.yaml"})
print(resp.json())
```


## TO_DO

- Sort out gittable venv setup so the server can run models reliably in any environment.
    - Document how to create and activate the venv, and install dependencies.

- You just came back here to write this exact to do ;) great minds.

-- Right, but the thought extended - we should have a setup script that makes the venv if it ent there.


-- Ok i think both of the above are kinda done. Next is this test suite.

-- It's gone a bit messy in the run_ml_server testing section. 

-- Sort that out and you'll be good.

-- run_ml_server is now clean and works with both vllm and llamacpp


--- Hmmm, if you try and load a model and you run out of vram, there is currently no warning that that's what happened. The server just fails to load the model and doesn't give a clear error message.
