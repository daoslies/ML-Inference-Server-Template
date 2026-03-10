import gc
from vllm import SamplingParams

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

class LlamaCppBackend(BaseLLMBackend):
    def __init__(self):
        self.llm = None
        try:
            from llama_cpp import Llama
            self.Llama = Llama
        except ImportError:
            raise RuntimeError("llama-cpp-python is not installed.")

    def load_model(self, model_path, model_config):
        self.llm = self.Llama(
            model_path=model_path,
            n_ctx=model_config.get('n_ctx', 4096),
            n_gpu_layers=model_config.get('n_gpu_layers', -1),
            # Add more config as needed
        )
        return self.llm

    def unload_model(self):
        if self.llm is not None:
            del self.llm
            gc.collect()
            self.llm = None

    def generate(self, prompts, sampling_params):
        # llama-cpp-python expects a single prompt at a time
        results = []
        for prompt in prompts:
            output = self.llm(
                prompt,
                max_tokens=sampling_params.get('max_tokens', 256),
                temperature=sampling_params.get('temperature', 0.3),
                top_p=sampling_params.get('top_p', 0.95),
                stop=sampling_params.get('stop', None),
            )
            # Wrap to match vLLM output structure as much as possible
            results.append({
                'response': output['choices'][0]['text'].strip(),
                'finish_reason': output['choices'][0].get('finish_reason', 'stop'),
                'prompt_tokens': output.get('usage', {}).get('prompt_tokens', None),
                'completion_tokens': output.get('usage', {}).get('completion_tokens', None),
            })
        return results
