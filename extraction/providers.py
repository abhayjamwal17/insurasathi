"""LLM provider abstraction.

The rest of the application talks to `LLMProvider.complete_json()` only, so
swapping between Groq (cloud) and Ollama (local) — or a deterministic
in-memory provider for tests/demos — never touches extraction logic.

No API keys are hardcoded: GroqProvider reads GROQ_API_KEY from the
environment (via config.settings) unless one is passed explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import requests


class LLMProviderError(Exception):
    """Raised when a provider cannot produce a completion."""


class LLMProvider(ABC):
    """A chat-completion backend that returns raw text expected to be JSON."""

    @abstractmethod
    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        """Send a system+user prompt and return the raw model output text."""


class GroqProvider(LLMProvider):
    """Groq's OpenAI-compatible chat completions API."""

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise LLMProviderError(
                "GROQ_API_KEY is not set. Set it in .env or pass api_key explicitly."
            )
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = requests.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise LLMProviderError(f"Groq request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMProviderError(f"Groq API error {response.status_code}: {response.text}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMProviderError(f"Unexpected Groq response shape: {data}") from exc


class OllamaProvider(LLMProvider):
    """A local Ollama server's /api/chat endpoint."""

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3.1"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise LLMProviderError(
                f"Ollama request failed (is `ollama serve` running at {self._base_url}?): {exc}"
            ) from exc

        if response.status_code != 200:
            raise LLMProviderError(f"Ollama API error {response.status_code}: {response.text}")

        data = response.json()
        try:
            return data["message"]["content"]
        except KeyError as exc:
            raise LLMProviderError(f"Unexpected Ollama response shape: {data}") from exc


class StaticProvider(LLMProvider):
    """Returns a fixed, pre-set response. For tests and offline demos only.

    Lets extraction logic (prompt building, JSON parsing, provenance
    attachment, unknown/conflict handling) be exercised deterministically
    without a network call or a running LLM.
    """

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        return self._response_text


def get_provider(
    provider_name: str,
    *,
    groq_api_key: str | None = None,
    groq_model: str = GroqProvider.DEFAULT_MODEL,
    ollama_base_url: str = OllamaProvider.DEFAULT_BASE_URL,
    ollama_model: str = OllamaProvider.DEFAULT_MODEL,
) -> LLMProvider:
    """Factory: build the configured provider by name ("groq" or "ollama")."""
    name = provider_name.strip().lower()
    if name == "groq":
        return GroqProvider(api_key=groq_api_key, model=groq_model)
    if name == "ollama":
        return OllamaProvider(model=ollama_model, base_url=ollama_base_url)
    raise LLMProviderError(f"Unknown LLM provider: {provider_name!r} (expected 'groq' or 'ollama')")
