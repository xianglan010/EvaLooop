"""Ollama backend for local, reproducible model inference."""

import json
from urllib import error, request
from typing import Any, Dict

from .base import BaseLLM


class OllamaLLM(BaseLLM):
    """Generate completions through Ollama's local HTTP API."""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model_id = model_config["model_id"]
        self.base_url = model_config.get(
            "base_url", "http://127.0.0.1:11434"
        ).rstrip("/")
        self.timeout = float(model_config.get("timeout", 600))
        self.temperature = float(model_config.get("temperature", 0.0))
        self.top_p = float(model_config.get("top_p", 1.0))
        self.max_new_tokens = int(model_config.get("max_new_tokens", 1024))
        self.seed = model_config.get("seed")
        self.keep_alive = model_config.get("keep_alive", "30m")

    def generate(self, prompt: str, **kwargs) -> str:
        """Return one non-streaming completion with explicit decoding options."""
        options = {
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "num_predict": kwargs.get("max_new_tokens", self.max_new_tokens),
        }
        seed = kwargs.get("seed", self.seed)
        if seed is not None:
            options["seed"] = int(seed)

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Ollama generation failed: {exc}") from exc

        if "error" in result:
            raise RuntimeError(f"Ollama generation failed: {result['error']}")
        return result.get("response", "").strip()

    def __str__(self) -> str:
        return f"OllamaLLM(model_id={self.model_id})"
