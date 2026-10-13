"""vLLM backend for local open-weight models (OpenAI-compatible server).

Same wire format as OpenAI, with two differences: structured output uses vLLM's
``guided_json`` guided decoding, and ``extra_body`` carries sampling controls
(thinking suppression and ``top_k``) that vLLM accepts but OpenAI does not.
"""

from __future__ import annotations

import os

from .base import Backend, TrialResult
from ._openai_wire import run_openai_wire

# Suppress chain-of-thought emission and pin sampling for reproducibility.
EXTRA_BODY = {
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": False},
}


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, base_url: str | None = None, api_key: str = "unused"):
        from openai import OpenAI
        url = base_url or os.environ.get("ADV_VLLM_BASE_URL", "http://localhost:8001/v1")
        self._client = OpenAI(api_key=api_key, base_url=url)

    def run_trial(self, *, model, domain, system_prompt, user_message, tools, handler) -> TrialResult:
        return run_openai_wire(
            client=self._client, model=model, domain=domain,
            system_prompt=system_prompt, user_message=user_message, tools=tools,
            handle=handler.handle,
            searched=lambda: bool(handler.access_record["search_calls"]),
            structured_mode="guided_json",
            extra_body=EXTRA_BODY,
        )
