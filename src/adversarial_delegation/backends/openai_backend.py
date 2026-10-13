"""OpenAI (and OpenAI-compatible) backend.

Targets the OpenAI API by default; pass ``base_url``/``api_key`` to point at any
OpenAI-compatible endpoint (Azure, a gateway, etc.). Structured output uses the
native ``response_format`` json_schema path.
"""

from __future__ import annotations

import os

from .base import Backend, TrialResult
from ._openai_wire import run_openai_wire


class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        from openai import OpenAI
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY or pass api_key.")
        self._client = OpenAI(api_key=key, **({"base_url": base_url} if base_url else {}))

    def run_trial(self, *, model, domain, system_prompt, user_message, tools, handler) -> TrialResult:
        return run_openai_wire(
            client=self._client, model=model, domain=domain,
            system_prompt=system_prompt, user_message=user_message, tools=tools,
            handle=handler.handle,
            searched=lambda: bool(handler.access_record["search_calls"]),
            structured_mode="response_format",
        )
