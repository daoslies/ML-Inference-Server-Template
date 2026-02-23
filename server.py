from flask import Flask, request, jsonify
"""
LLMServer
Compact Flask server managing a single llama_cpp.Llama model (thread-safe via Lock).
Endpoints: /health, /load_model, /unload_model, /inference, /extract_names.
Uses a YAML prompts file with key "extract_names" for name extraction.

Quick curl examples:
1) Health:
    curl -s http://localhost:{PORT}/health

2) Load model:
    curl -X POST http://localhost:{PORT}/load_model \
      -H "Content-Type: application/json" \
      -d '{"model_path":"/path/to/model.gguf"}'

3) Inference:
    curl -X POST http://localhost:{PORT}/inference \
      -H "Content-Type: application/json" \
      -d '{"prompt":"Summarize this...","max_tokens":150}'

4) Extract names:
    curl -X POST http://localhost:{PORT}/extract_names \
      -H "Content-Type: application/json" \
      -d '{"text":"Alice and Bob...","prompts_path":"textflow/prompts.yaml"}'
"""

from llama_cpp import Llama
from transformers import pipeline

import json
import yaml
import os
from threading import Lock
import ast

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

def load_config(path=CONFIG_PATH):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class LLMServer:
    def __init__(self):
        self.config = load_config()
        self.app = Flask(__name__)
        self.current_model = None
        self.model_lock = Lock()
        self.register_routes()
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_prompts_path = os.path.join(self.this_dir, self.config['defaults'].get('prompts_path', 'prompts.yaml'))
        self.model_config = self.config.get('model', {})

    def load_model(self, model_path: str):
        """Load a GGUF model using llama.cpp"""
        return Llama(
            model_path=model_path,
            n_ctx=self.model_config.get('n_ctx', 4096),
            n_threads=self.model_config.get('n_threads', 16),
            n_gpu_layers=self.model_config.get('n_gpu_layers', -1),
            verbose=self.model_config.get('verbose', False),
        )

    def unload_model(self, llm):
        """Unload model and free memory"""
        if llm is not None:
            del llm

    def load_prompts(self, yaml_path: str):
        """Load prompts from a YAML file"""
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def extract_names(self, text: str, llm: Llama, prompts_path: str):
        """Extract person names from text using LLM"""
        prompts = self.load_prompts(prompts_path)
        prompt = prompts['extract_names'].format(text=text)
        response = llm(
            prompt,
            max_tokens=256,
            temperature=0.3,
            stop=["</s>", "\n\n"],
        )

        llm.reset() #### This is required for the small falcon model. 
                    #### Not really sure why, something about resetting context, 
                    #### but it breaks trying to do more than one call without it.


        print('\n text: \n', response)
        if isinstance(response, dict) and 'choices' in response:
            response_text = response['choices'][0]['text'].strip()
        elif isinstance(response, str):
            response_text = response.strip()
        else:
            response_text = str(response).strip()
        try:
            # Extract the first list-like substring from the response_text
            start = response_text.find('[')
            end = response_text.rfind(']')
            if start != -1 and end != -1 and end > start:
                list_str = response_text[start:end+1]
                try:
                    names = ast.literal_eval(list_str)
                except Exception as e:
                    print(f"Failed to parse list with ast.literal_eval: {e}")
                    names = list_str  # fallback: return as string
            else:
                names = response_text  # fallback: return as string
            return names
        except Exception as e:
            print(f"Failed to extract names (possibly due to looking for a list brackets): {e}")
            return []

    def register_routes(self):
        app = self.app
        
        @app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'ok',
                'model_loaded': self.current_model is not None
            })

        @app.route('/checkcwd', methods=['POST', 'GET'])
        def cwd():
            cwd = os.getcwd()
            data = request.get_json()
            message = data.get('message') if data else None
            import sys
            print("Health check called.")
            print(f"VIRTUAL_ENV: {sys.prefix}")
            print(f"BASE_PREFIX: {sys.base_prefix}")
            print(f"Current working directory: {cwd}")
            return jsonify({
                'status': 'ok',
                'message': message + sys.prefix + 'yes' + sys.base_prefix if message else '',
            })

        @app.route('/load_model', methods=['POST'])
        def load_model_endpoint():
            data = request.get_json()
            if not data or 'model_path' not in data:
                return jsonify({'error': 'model_path is required'}), 400
            model_path = data['model_path']
            print(f"Loading model from path: {model_path}")
            if not os.path.exists(model_path):
                return jsonify({'error': f'Model file not found: {model_path}'}), 404
            try:
                with self.model_lock:
                    if self.current_model is not None:
                        self.unload_model(self.current_model)
                    self.current_model = self.load_model(model_path)
                    print('HERE')
                    print(self.current_model)
                return jsonify({
                    'status': 'success',
                    'message': f'Model loaded from {model_path} ',
                })
            except Exception as e:
                return jsonify({'error': str(e) + f"{model_path} \n \n Current dir: {os.getcwd()}"}), 500

        @app.route('/unload_model', methods=['POST'])
        def unload_model_endpoint():
            try:
                with self.model_lock:
                    if self.current_model is None:
                        return jsonify({'message': 'No model loaded'}), 200
                    self.unload_model(self.current_model)
                    self.current_model = None
                return jsonify({
                    'status': 'success',
                    'message': 'Model unloaded'
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
            print('here')
            print(prompts_path)
            if not os.path.exists(prompts_path):
                return jsonify({'error': f'Prompts file not found: {prompts_path}'}), 404
            try:
                with self.model_lock:
                    names = self.extract_names(text, self.current_model, prompts_path)
                return jsonify({
                    'status': 'success',
                    'names': names
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @app.route('/inference', methods=['POST'])
        def inference_endpoint():
            if self.current_model is None:
                return jsonify({'error': 'No model loaded. Please load a model first.'}), 400
            data = request.get_json()
            if not data or 'prompt' not in data:
                return jsonify({'error': 'prompt is required'}), 400
            prompt = data['prompt']
            max_tokens = data.get('max_tokens', 256)
            temperature = data.get('temperature', 0.3)
            stop = data.get('stop', ["</s>", "\n\n"])
            try:
                with self.model_lock:
                    response = self.current_model(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop,
                    )
                if isinstance(response, dict) and 'choices' in response:
                    response_text = response['choices'][0]['text'].strip()
                elif isinstance(response, str):
                    response_text = response.strip()
                else:
                    response_text = str(response).strip()
                return jsonify({
                    'status': 'success',
                    'response': response_text,
                    'full_response': response
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