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


class LLMServer:
    def __init__(self):
        self.config = load_config()
        self.app = Flask(__name__)
        self.current_model = None
        self.current_model_path = None
        self.model_lock = Lock()
        self.register_routes()
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_prompts_path = os.path.join(
            self.this_dir,
            self.config['defaults'].get('prompts_path', 'prompts.yaml')
        )
        self.model_config = self.config.get('model', {})

    def load_model(self, model_path: str) -> LLM:
        """Load a model using vLLM. model_path can be a HF repo ID or local directory."""
        return LLM(
            model=model_path,
            max_model_len=self.model_config.get('n_ctx', 4096),
            gpu_memory_utilization=self.model_config.get('gpu_memory_utilization', 0.90),
            tensor_parallel_size=self.model_config.get('tensor_parallel_size', 1),
            trust_remote_code=self.model_config.get('trust_remote_code', False),
        )

    def unload_model(self, llm: LLM):
        """Best-effort unload — frees the Python object and flushes CUDA cache."""
        if llm is not None:
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_prompts(self, yaml_path: str):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def _build_sampling_params(self, data: dict) -> SamplingParams:
        return SamplingParams(
            max_tokens=data.get('max_tokens', 256),
            temperature=data.get('temperature', 0.3),
            top_p=data.get('top_p', 0.95),
            stop=data.get('stop', None),
        )

    def register_routes(self):
        app = self.app

        @app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'ok',
                'model_loaded': self.current_model is not None,
                'model_path': self.current_model_path,
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

            # Allow HF repo IDs (contain '/') as well as local paths
            if not os.path.exists(model_path) and '/' not in model_path:
                return jsonify({'error': f'Model not found: {model_path}'}), 404

            try:
                with self.model_lock:
                    if self.current_model is not None:
                        self.unload_model(self.current_model)
                        self.current_model = None
                    self.current_model = self.load_model(model_path)
                    self.current_model_path = model_path

                return jsonify({
                    'status': 'success',
                    'message': f'Model loaded: {model_path}',
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @app.route('/unload_model', methods=['POST'])
        def unload_model_endpoint():
            try:
                with self.model_lock:
                    if self.current_model is None:
                        return jsonify({'message': 'No model loaded'}), 200
                    self.unload_model(self.current_model)
                    self.current_model = None
                    self.current_model_path = None
                return jsonify({
                    'status': 'success',
                    'message': 'Model unloaded (restart process for full VRAM reclaim)'
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @app.route('/extract_names', methods=['POST'])
        def extract_names_endpoint():
            if self.current_model is None:
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
                    names = extract_names(text, self.current_model, prompts_path)
                return jsonify({'status': 'success', 'names': names})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @app.route('/inference', methods=['POST'])
        def inference_endpoint():
            if self.current_model is None:
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
                    outputs = self.current_model.generate(prompts, sampling_params)

                results = [
                    {
                        'index': i,
                        'response': output.outputs[0].text.strip(),
                        'finish_reason': output.outputs[0].finish_reason,
                        'prompt_tokens': len(output.prompt_token_ids),
                        'completion_tokens': len(output.outputs[0].token_ids),
                    }
                    for i, output in enumerate(outputs)
                ]

                return jsonify({
                    'status': 'success',
                    # Single prompt → single object; list → list
                    'results': results if isinstance(prompt_input, list) else results[0],
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

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