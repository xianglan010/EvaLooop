"""Offline tests for the Ollama backend."""

import json

from evaloop.models.factory import ModelFactory
from evaloop.models.ollama import OllamaLLM


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"response": "  generated text  "}).encode()


def test_ollama_generate_sends_greedy_options(monkeypatch):
    captured = {}

    def fake_urlopen(http_request, timeout):
        captured.update(json.loads(http_request.data.decode()))
        assert timeout == 600
        return FakeResponse()

    monkeypatch.setattr("evaloop.models.ollama.request.urlopen", fake_urlopen)
    model = OllamaLLM({
        "name": "test",
        "model_id": "qwen-test",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 128,
        "seed": 42,
    })
    assert model.generate("hello") == "generated text"
    assert captured["model"] == "qwen-test"
    assert captured["think"] is False
    assert captured["options"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": 128,
        "seed": 42,
    }


def test_factory_creates_ollama_without_torch():
    model = ModelFactory().create_model({
        "name": "test", "type": "ollama", "model_id": "qwen-test"
    })
    assert isinstance(model, OllamaLLM)
