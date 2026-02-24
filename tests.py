import os
import pytest
import requests
import yaml

# --- Config ---

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="session")
def base_url():
    config = load_config()
    host = config['server'].get('host', 'localhost')
    # 0.0.0.0 means "bind to all interfaces" — for requests we want localhost
    if host == '0.0.0.0':
        host = 'localhost'
    port = config['server'].get('port', 27776)
    return f"http://{host}:{port}"

@pytest.fixture(scope="session", autouse=True)
def check_server(base_url):
    """Fail fast if the server isn't running before any tests execute."""
    try:
        requests.get(f"{base_url}/health", timeout=3)
    except requests.ConnectionError:
        pytest.exit(f"Server not reachable at {base_url} — is it running?")

# --- Helpers ---

def get_test_model():
    config = load_config()
    return config.get('tests', config.get('defaults', {})).get('test_model', 'Qwen/Qwen3-0.6B')

TEST_MODEL = get_test_model()

# -------------------------------------------------------
# 1. Health
# -------------------------------------------------------

class TestHealth:
    def test_status_ok(self, base_url):
        r = requests.get(f"{base_url}/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_has_model_loaded_field(self, base_url):
        r = requests.get(f"{base_url}/health")
        assert "model_loaded" in r.json()

    def test_has_model_path_field(self, base_url):
        r = requests.get(f"{base_url}/health")
        assert "model_path" in r.json()

# -------------------------------------------------------
# 2. Load model
# -------------------------------------------------------

class TestLoadModel:
    def test_load_valid_model(self, base_url):
        r = requests.post(f"{base_url}/load_model", json={"model_path": TEST_MODEL})
        assert r.status_code == 200
        assert r.json()["status"] == "success"

    def test_health_reflects_loaded_model(self, base_url):
        r = requests.get(f"{base_url}/health")
        body = r.json()
        assert body["model_loaded"] is True
        assert body["model_path"] == TEST_MODEL

    def test_load_missing_model_path(self, base_url):
        r = requests.post(f"{base_url}/load_model", json={})
        assert r.status_code == 400
        assert "error" in r.json()

    def test_load_invalid_path(self, base_url):
        r = requests.post(f"{base_url}/load_model", json={"model_path": "this_does_not_exist"})
        assert r.status_code == 404
        assert "error" in r.json()

# -------------------------------------------------------
# 3. Inference (requires model to be loaded — runs after TestLoadModel)
# -------------------------------------------------------

class TestInference:
    def test_single_prompt(self, base_url):
        r = requests.post(f"{base_url}/inference", json={
            "prompt": "Say the word hello and nothing else.",
            "max_tokens": 10,
            "temperature": 0.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert "results" in body
        assert "response" in body["results"]

    def test_single_prompt_has_token_counts(self, base_url):
        r = requests.post(f"{base_url}/inference", json={
            "prompt": "Say the word hello and nothing else.",
            "max_tokens": 10,
            "temperature": 0.0,
        })
        result = r.json()["results"]
        assert "prompt_tokens" in result
        assert "completion_tokens" in result
        assert result["prompt_tokens"] > 0
        assert result["completion_tokens"] > 0

    def test_batch_prompts(self, base_url):
        r = requests.post(f"{base_url}/inference", json={
            "prompt": [
                "Say the word yes and nothing else.",
                "Say the word no and nothing else.",
            ],
            "max_tokens": 10,
            "temperature": 0.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert isinstance(body["results"], list)
        assert len(body["results"]) == 2

    def test_batch_results_have_indices(self, base_url):
        r = requests.post(f"{base_url}/inference", json={
            "prompt": ["Prompt one.", "Prompt two."],
            "max_tokens": 10,
            "temperature": 0.0,
        })
        results = r.json()["results"]
        assert results[0]["index"] == 0
        assert results[1]["index"] == 1

    def test_missing_prompt_field(self, base_url):
        r = requests.post(f"{base_url}/inference", json={"max_tokens": 50})
        assert r.status_code == 400
        assert "error" in r.json()

# -------------------------------------------------------
# 4. Unload model
# -------------------------------------------------------

class TestUnloadModel:
    def test_unload(self, base_url):
        r = requests.post(f"{base_url}/unload_model")
        assert r.status_code == 200
        assert r.json()["status"] == "success"

    def test_health_reflects_unloaded(self, base_url):
        r = requests.get(f"{base_url}/health")
        body = r.json()
        assert body["model_loaded"] is False
        assert body["model_path"] is None

    def test_unload_when_no_model(self, base_url):
        """Unloading again should be graceful, not an error."""
        r = requests.post(f"{base_url}/unload_model")
        assert r.status_code == 200

    def test_inference_with_no_model(self, base_url):
        r = requests.post(f"{base_url}/inference", json={"prompt": "hello", "max_tokens": 10})
        assert r.status_code == 400
        assert "error" in r.json()