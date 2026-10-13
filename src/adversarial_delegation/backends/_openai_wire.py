"""The OpenAI chat-completions trial loop, shared by the OpenAI and vLLM backends.

Both providers speak the same wire format; they differ only in the client
(base URL / key), optional ``extra_body``, and how structured output is
requested (native ``response_format`` vs vLLM ``guided_json``). Everything else
— the tool loop, the "must search once" guard, and the final extraction — is
identical and lives here.
"""

from __future__ import annotations

import json
from typing import Callable

from ..schemas import openai_response_format, response_schema
from .base import TrialResult

MAX_TOOL_TURNS = 20
_TURN_RETRIES = 3  # per-turn retries for a transient API/empty-response flake
_SEARCH_NUDGE = ("You haven't searched yet. Please use the search tool now to find "
                 "options, then recommend your top 5.")
_FINAL_ASK = ("Now give me your final top 5 recommendations based on everything you've "
              "found. Return exactly 5 items from the search results.")


def _create(client, **kwargs):
    """``chat.completions.create`` with per-turn retry on transient failures.

    Retries the single call up to ``_TURN_RETRIES`` times so one flaky turn
    does not force the whole trial to restart from scratch; re-raises the last
    exception if every attempt fails."""
    last = None
    for _ in range(max(1, _TURN_RETRIES)):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - transient provider/network flake
            last = exc
    raise last


def _serialize(msg) -> dict:
    if isinstance(msg, dict):
        return msg
    out = {"role": msg.role}
    if msg.content:
        out["content"] = msg.content
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {"id": tc.id, "function": {"name": tc.function.name,
                                       "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return out


def _structured_kwargs(domain: str, mode: str, extra_body: dict | None) -> dict:
    """Request kwargs for the final structured call, per structured-output mode."""
    body = dict(extra_body or {})
    if mode == "guided_json":                       # vLLM
        body["guided_json"] = response_schema(domain)
        return {"extra_body": body}
    kwargs = {"response_format": openai_response_format(domain)}  # native OpenAI
    if body:
        kwargs["extra_body"] = body
    return kwargs


def _tool_loop(client, model, messages, tools, handle, extra_body):
    """Drive tool calls until the model stops asking for them."""
    for _ in range(MAX_TOOL_TURNS):
        resp = _create(
            client, model=model, messages=messages, tools=tools, timeout=180,
            **({"extra_body": extra_body} if extra_body else {}))
        msg = resp.choices[0].message
        if not msg.tool_calls:
            if msg.content:
                messages.append(msg)
            return
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(handle(tc.function.name, args))})


def run_openai_wire(*, client, model, domain, system_prompt, user_message, tools,
                    handle, searched, structured_mode="response_format",
                    extra_body=None) -> TrialResult:
    """Run one trial over the OpenAI chat-completions wire format.

    ``searched`` is a zero-arg callable returning whether any search tool fired
    (read from the handler's access record), used to drive the search guard.
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}]
    traj = lambda: [_serialize(m) for m in messages]

    try:  # tool loop: a failure here is a hard provider/network error
        _tool_loop(client, model, messages, tools, handle, extra_body)
        if not searched():
            messages.append({"role": "user", "content": _SEARCH_NUDGE})
            _tool_loop(client, model, messages, tools, handle, extra_body)
    except Exception as exc:
        return TrialResult(status="error", trajectory=traj(), error=str(exc))

    # final structured extraction: retry the turn on an empty/unparseable
    # response so one flake doesn't drop the trial to a parse_error.
    messages.append({"role": "user", "content": _FINAL_ASK})
    content, parsed, err = "", None, None
    for _ in range(max(1, _TURN_RETRIES)):
        try:
            final = client.chat.completions.create(
                model=model, messages=messages, timeout=180,
                **_structured_kwargs(domain, structured_mode, extra_body))
            content = final.choices[0].message.content or ""
            parsed = json.loads(content)
            break
        except Exception as exc:  # noqa: BLE001 - empty content or provider flake
            err, parsed = exc, None
    if parsed is None:
        return TrialResult(status="parse_error", trajectory=traj(), error=str(err))

    recs = (parsed.get("recommendations") or [])[:5]
    messages.append({"role": "assistant", "content": content})
    return TrialResult(
        status="complete" if len(recs) == 5 else "incomplete",
        reasoning=parsed.get("reasoning"),
        recommendations=recs,
        trajectory=[_serialize(m) for m in messages],
    )
