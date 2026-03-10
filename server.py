from flask import Flask, request, jsonify
"""
LLMServer
Compact Flask server managing a single vLLM model (thread-safe via Lock).
Endpoints: /health, /load_model, /unload_model, /inference, /extract_names.
Uses a YAML prompts file with key "extract_names" for name extraction.

Quick curl examples:
1) Health:
    curl -s http://localhost:{PORT}/health

2) Load model:
    curl -X POST http://localhost:{PORT}/load_model \
      -H "Content-Type: application/json" \
      -d '{"model_path":"Qwen/Qwen2.5-7B-Instruct"}'

3) Inference:
    curl -X POST http://localhost:{PORT}/inference \
      -H "Content-Type: application/json" \
      -d '{"prompt":"Summarize this...","max_tokens":150}'

4) Batch inference:
    curl -X POST http://localhost:{PORT}/inference \
      -H "Content-Type: application/json" \
      -d '{"prompt":["Prompt one","Prompt two"],"max_tokens":150}'

5) Extract names:
    curl -X POST http://localhost:{PORT}/extract_names \
      -H "Content-Type: application/json" \
      -d '{"text":"Alice and Bob...","prompts_path":"textflow/prompts.yaml"}'

Notes on load/unload vs llama.cpp version:
- vLLM initialises the full model on /load_model and holds it in GPU memory.
- /unload_model deletes the LLM object and flushes CUDA cache, but vLLM does
  not have a first-class unload API — a process restart is the cleanest way
  to fully reclaim VRAM if that matters to you.
- model_path can be a HuggingFace repo ID (e.g. "Qwen/Qwen2.5-7B-Instruct")
  or an absolute path to a local HF model directory.
"""

from utils import extract_names

from vllm import LLM, SamplingParams

import gc
import torch
import yaml
import os
from threading import Lock

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

def load_config(path=CONFIG_PATH):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

from llm_backends import VLLMBackend, LlamaCppBackend

class LLMServer:
    def __init__(self):
        self.config = load_config()
        self.app = Flask(__name__)
        self.current_backend = None
        self.current_model_path = None
        self.current_backend_name = None
        self.model_lock = Lock()
        self.register_routes()
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_prompts_path = os.path.join(
            self.this_dir,
            self.config['defaults'].get('prompts_path', 'prompts.yaml')
        )
        self.model_config = self.config.get('model', {})

    def _try_load_backend(self, backend_cls, model_path):
        backend = backend_cls()
        backend.load_model(model_path, self.model_config)
        return backend

    def _build_sampling_params(self, data: dict):
        # Returns a dict for llama-cpp, or SamplingParams for vLLM
        return {
            'max_tokens': data.get('max_tokens', 256),
            'temperature': data.get('temperature', 0.3),
            'top_p': data.get('top_p', 0.95),
            'stop': data.get('stop', None),
        }

    def register_routes(self):
        app = self.app

        @app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'ok',
                'model_loaded': self.current_backend is not None,
                'model_path': self.current_model_path if self.current_backend is not None else None,
                'backend': self.current_backend_name,
            })

        @app.route('/checkcwd', methods=['POST', 'GET'])
        def cwd():
            cwd = os.getcwd()
            data = request.get_json()
            message = data.get('message') if data else None
            import sys
            print(f"VIRTUAL_ENV: {sys.prefix}")
            print(f"Current working directory: {cwd}")
            return jsonify({
                'status': 'ok',
                'message': message + sys.prefix + sys.base_prefix if message else '',
            })

        @app.route('/load_model', methods=['POST'])
        def load_model_endpoint():
            data = request.get_json()
            if not data or 'model_path' not in data:
                return jsonify({'error': 'model_path is required'}), 400
            model_path = data['model_path']
            print(f"Loading model: {model_path}")
            with self.model_lock:
                # Only unload current model if new one loads successfully
                backend = None
                backend_name = None
                error_msgs = []
                try:
                    backend = self._try_load_backend(VLLMBackend, model_path)
                    backend_name = 'vllm'
                except Exception as e:
                    error_msgs.append(f"vLLM failed: {e}")
                    try:
                        backend = self._try_load_backend(LlamaCppBackend, model_path)
                        backend_name = 'llama-cpp'
                    except Exception as e2:
                        error_msgs.append(f"llama-cpp failed: {e2}")
                        if not (os.path.exists(model_path) or '/' in model_path):
                            code = 404
                        else:
                            code = 500
                        return jsonify({'error': 'Both backends failed', 'details': error_msgs}), code
                # Only here if new model loaded successfully
                if self.current_backend is not None:
                    self.current_backend.unload_model()
                self.current_backend = backend
                self.current_backend_name = backend_name
                self.current_model_path = model_path
                msg = f"Model loaded with backend: {backend_name}"
                if error_msgs:
                    msg += f" (fallback used, errors: {error_msgs})"
                return jsonify({
                    'status': 'success',
                    'message': msg,
                    'backend': backend_name,
                })

        @app.route('/unload_model', methods=['POST'])
        def unload_model_endpoint():
            try:
                with self.model_lock:
                    if self.current_backend is None:
                        return jsonify({'status': 'success', 'message': 'No model loaded'}), 200
                    self.current_backend.unload_model()
                    self.current_backend = None
                    self.current_model_path = None
                    self.current_backend_name = None
                return jsonify({
                    'status': 'success',
                    'message': 'Model unloaded',
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @app.route('/extract_names', methods=['POST'])
        def extract_names_endpoint():
            if self.current_backend is None:
                return jsonify({'error': 'No model loaded. Please load a model first.'}), 400
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({'error': 'text is required'}), 400
            text = data['text']
            prompts_path = data.get('prompts_path', self.default_prompts_path)
            if not os.path.exists(prompts_path):
                return jsonify({'error': f'Prompts file not found: {prompts_path}'}), 404
            try:
                with self.model_lock:
                    names = extract_names(text, self.current_backend.llm, prompts_path)
                return jsonify({'status': 'success', 'names': names, 'backend': self.current_backend_name})
            except Exception as e:
                return jsonify({'error': str(e), 'backend': self.current_backend_name}), 500

        @app.route('/inference', methods=['POST'])
        def inference_endpoint():
            if self.current_backend is None:
                return jsonify({'error': 'No model loaded. Please load a model first.'}), 400

            data = request.get_json()
            if not data or 'prompt' not in data:
                return jsonify({'error': 'prompt is required'}), 400

            # Accept a single string or a list — list triggers batch mode
            prompt_input = data['prompt']
            prompts = prompt_input if isinstance(prompt_input, list) else [prompt_input]
            sampling_params = self._build_sampling_params(data)

            try:
                with self.model_lock:
                    results = self.current_backend.generate(prompts, sampling_params)
                # vLLM returns list of objects with .outputs, llama-cpp returns list of dicts
                if self.current_backend_name == 'vllm':
                    results = [
                        {
                            'index': i,
                            'response': output.outputs[0].text.strip(),
                            'finish_reason': output.outputs[0].finish_reason,
                            'prompt_tokens': len(output.prompt_token_ids),
                            'completion_tokens': len(output.outputs[0].token_ids),
                        }
                        for i, output in enumerate(results)
                    ]
                else:
                    # llama-cpp already returns a list of dicts
                    for i, r in enumerate(results):
                        r['index'] = i
                return jsonify({
                    'status': 'success',
                    'backend': self.current_backend_name,
                    'results': results if isinstance(prompt_input, list) else results[0],
                })
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[inference error] {e}\n{tb}")
                return jsonify({'error': str(e), 'traceback': tb, 'backend': self.current_backend_name}), 500

    def load_prompts(self, yaml_path: str):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def run(self):
        self.app.run(
            host=self.config['server'].get('host', '0.0.0.0'),
            port=self.config['server'].get('port', 27776),
            debug=self.config['server'].get('debug', False),
            threaded=True
        )

if __name__ == '__main__':
    server = LLMServer()
    server.run()