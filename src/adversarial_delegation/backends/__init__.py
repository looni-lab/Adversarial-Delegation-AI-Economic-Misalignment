"""Backend registry. Providers are imported lazily so optional SDKs
(``google.generativeai`` for Gemini) are only required when actually used.
"""

from __future__ import annotations

from .base import Backend, TrialResult

BACKENDS = ("openai", "vllm", "gemini")


def get_backend(name: str, **kwargs) -> Backend:
    """Instantiate a backend by name: ``openai``, ``vllm``, or ``gemini``."""
    key = name.lower()
    if key == "openai":
        from .openai_backend import OpenAIBackend
        return OpenAIBackend(**kwargs)
    if key == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(**kwargs)
    if key == "gemini":
        from .gemini_backend import GeminiBackend
        return GeminiBackend(**kwargs)
    raise ValueError(f"Unknown backend {name!r}; choose from {BACKENDS}.")


__all__ = ["Backend", "TrialResult", "get_backend", "BACKENDS"]
