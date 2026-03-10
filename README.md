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


## TO_DO

- Sort out gittable venv setup so the server can run models reliably in any environment.
    - Document how to create and activate the venv, and install dependencies.

- You just came back here to write this exact to do ;) great minds.

-- Right, but the thought extended - we should have a setup script that makes the venv if it ent there.


-- Ok i think both of the above are kinda done. Next is this test suite.

-- It's gone a bit messy in the run_ml_server testing section. 

-- Sort that out and you'll be good.

-- run_ml_server is now clean and works with both vllm and llamacpp
