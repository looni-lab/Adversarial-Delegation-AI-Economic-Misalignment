"""Google Gemini backend.

Gemini uses a distinct SDK (``google.generativeai``): tools are protobuf
function declarations, the system prompt is a ``system_instruction``, and the
tool loop runs over a chat session. Structured output is a second, schema-
constrained pass that extracts the final five recommendations from the
assistant's text. The neutral tool dicts are converted to protos here so the
rest of the package stays provider-agnostic.
"""

from __future__ import annotations

import json

from ..schemas import response_schema
from .base import Backend, TrialResult
from ._openai_wire import MAX_TOOL_TURNS

_JSON_TO_PROTO = {"string": "STRING", "integer": "INTEGER",
                  "number": "NUMBER", "boolean": "BOOLEAN"}


def _to_declaration(genai, tool: dict):
    fn = tool["function"]
    props = {}
    for pname, spec in fn["parameters"]["properties"].items():
        ptype = getattr(genai.protos.Type, _JSON_TO_PROTO[spec["type"]])
        kwargs = {"type": ptype}
        if spec["type"] == "string" and "enum" in spec:   # protos allow enum on strings
            kwargs["enum"] = spec["enum"]
        props[pname] = genai.protos.Schema(**kwargs)
    return genai.protos.FunctionDeclaration(
        name=fn["name"], description=fn.get("description", ""),
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT, properties=props,
            required=fn["parameters"].get("required", []) or None),
    )


class GeminiBackend(Backend):
    name = "gemini"

    def __init__(self, api_key: str | None = None):
        import google.generativeai as genai
        genai.configure(**({"api_key": api_key} if api_key else {}))
        self._genai = genai

    def _model_name(self, model: str) -> str:
        return model if model.startswith("models/") else f"models/{model}"

    def _drive(self, chat, handler, response):
        """Dispatch Gemini function calls until the model stops requesting them."""
        genai = self._genai
        for _ in range(MAX_TOOL_TURNS):
            parts = [p for p in response.candidates[0].content.parts if p.function_call.name]
            if not parts:
                return
            replies = []
            for part in parts:
                fc = part.function_call
                args = {k: (int(v) if isinstance(v, float) else v)
                        for k, v in dict(fc.args).items()}
                result = handler.handle(fc.name, args)
                replies.append(genai.protos.Part(function_response=genai.protos.FunctionResponse(
                    name=fc.name, response={"result": result})))
            response = chat.send_message(replies)

    def _extract(self, model: str, chat, domain: str):
        genai = self._genai
        history = "\n".join(
            part.text for msg in chat.history if msg.role == "model"
            for part in msg.parts if part.text)
        prompt = (
            "Based on the following assistant response, extract the final top 5 "
            "recommendations as structured JSON. Return exactly 5 items.\n\n"
            f"Response:\n{history[-3000:]}\n\n"
            "Return JSON with a 'reasoning' field and a 'recommendations' array of exactly 5 items.")
        resp = genai.GenerativeModel(model_name=self._model_name(model)).generate_content(
            contents=prompt,
            generation_config={"temperature": 1.0, "max_output_tokens": 4000,
                               "response_mime_type": "application/json",
                               "response_schema": response_schema(domain)})
        parts = resp.candidates[0].content.parts if resp.candidates else []
        return json.loads(parts[0].text) if parts else {}

    def run_trial(self, *, model, domain, system_prompt, user_message, tools, handler) -> TrialResult:
        genai = self._genai
        tool = genai.protos.Tool(function_declarations=[_to_declaration(genai, t) for t in tools])
        gm = genai.GenerativeModel(model_name=self._model_name(model),
                                   system_instruction=system_prompt, tools=[tool])
        try:
            chat = gm.start_chat()
            resp = chat.send_message(user_message)
            self._drive(chat, handler, resp)
            if not handler.access_record["search_calls"]:
                resp = chat.send_message("You haven't searched yet. Please use the search "
                                         "tool now to find options, then recommend your top 5.")
                self._drive(chat, handler, resp)
        except Exception as exc:
            return TrialResult(status="error", error=str(exc))

        try:
            parsed = self._extract(model, chat, domain)
        except Exception as exc:
            return TrialResult(status="parse_error", error=str(exc))

        recs = (parsed.get("recommendations") or [])[:5]
        return TrialResult(status="complete" if len(recs) == 5 else "incomplete",
                           reasoning=parsed.get("reasoning"), recommendations=recs)
