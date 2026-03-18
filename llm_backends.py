import gc
from vllm import SamplingParams
import yaml
import logging
from llama_cpp import Llama
import os

class BaseLLMBackend:
    def load_model(self, model_path, model_config):
        raise NotImplementedError
    def unload_model(self):
        raise NotImplementedError
    def generate(self, prompts, sampling_params):
        raise NotImplementedError
    def get_backend_name(self):
        return self.__class__.__name__

class VLLMBackend(BaseLLMBackend):
    def __init__(self):
        self.llm = None
        try:
            from vllm import LLM
            self.LLM = LLM
        except ImportError:
            raise RuntimeError("vLLM is not installed.")

    def load_model(self, model_path, model_config):
        self.llm = self.LLM(
            model=model_path,
            max_model_len=model_config.get('n_ctx', 4096),
            gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.90),
            tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
            trust_remote_code=model_config.get('trust_remote_code', False),
        )
        return self.llm

    def unload_model(self):
        if self.llm is not None:
            del self.llm
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            self.llm = None

    def generate(self, prompts, sampling_params):
        # Convert dict to SamplingParams if needed
        if isinstance(sampling_params, dict):
            sampling_params = SamplingParams(**{k: v for k, v in sampling_params.items() if v is not None})
        return self.llm.generate(prompts, sampling_params)

def _get_lib_path() -> str | None:
    try:
        with open('config.yaml') as f:
            c = yaml.safe_load(f)
        path = c.get('llama_cpp', {}).get('lib_path') or None
        return path if path else None
    except Exception:
        return None

def _check_library_consistency(expected_path: str | None) -> None:
    if not expected_path:
        return  # Using default wheel — nothing to check.
    try:
        import llama_cpp.llama_cpp as _lib_module
        loaded = getattr(_lib_module, '_lib', None)
        actual_path = getattr(loaded, '_name', None) if loaded else None
        if actual_path and actual_path != expected_path:
            logging.warning(
                'llama.cpp library mismatch.\n'
                f'  Expected : {expected_path}\n'
                f'  Loaded   : {actual_path}\n'
                'The model may still run. Clear llama_cpp.lib_path in config.yaml '
                'to silence this warning and use the default build.'
            )
        else:
            logging.info(f'llama.cpp library confirmed: {actual_path or "default"}')
    except Exception as e:
        logging.warning(f'Could not verify llama.cpp library path: {e}')

import json
import subprocess
import time
import urllib.request
import urllib.error

def _get_config_value(expr):
    with open('config.yaml') as f:
        c = yaml.safe_load(f)
    return eval(expr, {'c': c})

class LlamaCppBackend(BaseLLMBackend):
    def __init__(self):
        self.process     = None
        self.server_url  = None
        self._model_path = None

        # Path to llama-server binary, read from config.
        # Falls back to 'llama-server' if not set (i.e. it must be on PATH).
        self.server_bin = _get_config_value(
            "c.get('llama_cpp', {}).get('server_bin', 'llama-server')"
        )

    def load_model(self, model_path, model_config):
        # Check if model_path exists before launching llama-server
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.unload_model()  # Clean up any existing process.

        n_ctx        = model_config.get('n_ctx', 4096)
        n_gpu_layers = model_config.get('n_gpu_layers', -1)
        port         = _get_config_value(
            "c.get('llama_cpp', {}).get('server_port', 28000)"
        )
        lora_path    = model_config.get('lora_path')

        cmd = [
            self.server_bin,
            '--model',        model_path,
            '--ctx-size',     str(n_ctx),
            '--n-gpu-layers', str(n_gpu_layers),
            '--port',         str(port),
            '--host',         '127.0.0.1',
            '--jinja',                                          # enable chat template
            '--chat-template-kwargs', '{"enable_thinking":false}',
            '--reasoning-budget', '0',                          # belt and braces
            '--presence-penalty', '1.5',                        # stops repetition loops
            '--temp', '0.6', 
            '--top-k', '20', 
            '--top-p', '0.95',
        ]
        if lora_path:
            if not os.path.isfile(lora_path):
                raise FileNotFoundError(f"LoRA file not found: {lora_path}")
            cmd += ['--lora', lora_path]

        self.process     = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.server_url  = f'http://127.0.0.1:{port}'
        self._model_path = model_path

        self._wait_for_ready()
        return self

    def unload_model(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        self.server_url  = None
        self._model_path = None
        gc.collect()

    def generate(self, prompts, sampling_params):
        if not self.server_url:
            raise RuntimeError('No model loaded. Call load_model() first.')

        results = []
        for prompt in prompts:
            payload = json.dumps({
                'prompt':      prompt,
                'n_predict':   sampling_params.get('max_tokens', 256),
                'temperature': sampling_params.get('temperature', 0.3),
                'top_p':       sampling_params.get('top_p', 0.95),
                'stop':        sampling_params.get('stop', ["</s>", "\n\n"]),  # Both .encode()
            }).encode()

            req = urllib.request.Request(
                f'{self.server_url}/completion',
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                output = json.loads(resp.read().decode())

            results.append({
                'response':          output.get('content', '').strip(),
                'finish_reason':     output.get('stop_type', 'stop'),
                'prompt_tokens':     output.get('tokens_evaluated', None),
                'completion_tokens': output.get('tokens_predicted', None),
            })

        return results

    # ── Private ───────────────────────────────────────────────────────

    def _wait_for_ready(self, timeout=60):
        """Poll /health until llama-server reports ready."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                stderr = b''
                if self.process.stderr:
                    try:
                        stderr = self.process.stderr.read().decode(errors='replace')
                    except Exception:
                        stderr = b''
                raise RuntimeError(
                    f'llama-server exited unexpectedly during startup.\n{stderr}'
                )
            try:
                with urllib.request.urlopen(
                    f'{self.server_url}/health', timeout=2
                ) as resp:
                    status = json.loads(resp.read().decode()).get('status')
                    if status == 'ok':
                        return
            except (urllib.error.URLError, OSError):
                pass  # Not up yet.
            time.sleep(1)

        if self.process is not None:
            self.process.kill()
        raise RuntimeError(
            f'llama-server failed to become ready within {timeout}s. '
            f'Check the model path and available VRAM.'
        )
