"""Trial enumeration, resume, parallel execution, and trace writing.

Provider-agnostic: the engine builds each trial's prompt, tools, and handler,
hands them to a backend, assembles the full trace record, and writes it to
``<out>/<model>/<domain>/<condition>_<persona>_<intent>_<motivation>.json``.
Existing trace files are skipped, so runs resume automatically.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from . import data as _data
from .conditions import CONDITIONS, CONTROL_PERSONA, INTENTS
from .handler import TrialHandler
from .prompts import build_system_prompt, build_user_message
from .tools import get_tools_for_condition

Trial = tuple  # (condition, persona, intent, motivation)


@dataclass
class Bundle:
    """All inputs loaded once and shared across a run."""
    personas: dict
    emails: dict
    intents: dict
    motivations: list
    inventory: list

    @classmethod
    def load(cls, domain: str) -> "Bundle":
        return cls(_data.load_personas(), _data.load_emails(), _data.load_intents(),
                   _data.load_motivations(), _data.load_inventory(domain))


def enumerate_trials(domain, bundle, *, conditions=None, intents=None,
                     persona_id=None) -> list[Trial]:
    """Every (condition, persona, intent, motivation) tuple for a domain."""
    conditions = conditions or CONDITIONS
    intents = intents or [i for i in INTENTS if i in bundle.intents[domain]]
    personas = bundle.personas["personas"]
    if persona_id:
        personas = [p for p in personas if p["id"] == persona_id]

    trials: list[Trial] = []
    for condition in conditions:
        for intent in intents:
            for motivation in bundle.motivations:
                if condition == "control":
                    trials.append((condition, CONTROL_PERSONA, intent, motivation))
                else:
                    trials.extend((condition, p, intent, motivation) for p in personas)
    return trials


def trace_path(out_dir: Path, model: str, domain: str, trial: Trial) -> Path:
    condition, persona, intent, motivation = trial
    name = f"{condition}_{persona['id']}_{intent}_{motivation['id']}.json"
    return out_dir / model / domain / name


def trace_ok(path: Path) -> bool:
    """True only if a trace exists and holds a usable (non-error) result.

    A missing trace, unreadable JSON, or a ``parse_error``/``error`` status all
    count as NOT ok, so resume re-runs them instead of treating them as done.
    """
    if not path.exists():
        return False
    try:
        rec = json.loads(path.read_text())
    except Exception:
        return False
    return rec.get("status") not in ("parse_error", "error")


def run_one(backend, model, domain, trial, bundle, path: Path, retries: int = 3) -> str:
    """Run a single trial and write its trace. Returns the status string.

    ``retries`` is the max number of attempts when the model returns an
    unparseable or errored response (transient API/output flake). Each attempt
    uses a fresh handler so the recorded access path matches the kept result;
    the first parseable result wins, otherwise the last attempt is kept.
    """
    condition, persona, intent, motivation = trial
    system_prompt = build_system_prompt(condition, persona, domain, bundle.personas, bundle.emails)
    user_message = build_user_message(domain, intent, motivation, bundle.intents)
    tools = get_tools_for_condition(condition, domain)
    for _attempt in range(max(1, retries)):
        handler = TrialHandler(domain, condition, persona, bundle.inventory,
                               bundle.personas, bundle.emails)
        result = backend.run_trial(
            model=model, domain=domain,
            system_prompt=system_prompt,
            user_message=user_message,
            tools=tools,
            handler=handler,
        )
        if result.status not in ("parse_error", "error"):
            break
    record = {
        "status": result.status,
        "domain": domain,
        "condition": condition,
        "persona_id": persona["id"],
        "persona_levels": {k: persona[k] for k in
                           ["financial", "employment", "health", "life_events", "demographics"]},
        "intent": intent,
        "motivation_id": motivation["id"],
        "reasoning": result.reasoning,
        "recommendations": result.recommendations,
        "access_record": handler.access_record,
        "tool_calls_count": len(handler.tool_log),
        "tool_log": handler.tool_log,
        "trajectory": result.trajectory,
        "error": result.error,
        "model": model,
        "timestamp": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return result.status


def run(backend, model, domain, out_dir, *, workers=16, conditions=None,
        intents=None, persona_id=None, dry_run=False, progress=None) -> dict:
    """Run (or count, if ``dry_run``) all pending trials for one model+domain.

    ``progress`` is an optional ``callable(done, total, status)`` for UIs.
    Returns a ``{"total","pending","completed","errors"}`` summary.
    """
    out_dir = Path(out_dir)
    bundle = Bundle.load(domain)
    trials = enumerate_trials(domain, bundle, conditions=conditions,
                              intents=intents, persona_id=persona_id)
    pending = [t for t in trials if not trace_ok(trace_path(out_dir, model, domain, t))]

    summary = {"total": len(trials), "pending": len(pending), "completed": 0, "errors": 0}
    if dry_run or not pending:
        return summary

    def work(trial):
        return run_one(backend, model, domain, trial, bundle,
                       trace_path(out_dir, model, domain, trial))

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, t): t for t in pending}
        for fut in as_completed(futures):
            try:
                status = fut.result()
                summary["errors" if status == "error" else "completed"] += 1
            except Exception:
                summary["errors"] += 1
            done += 1
            if progress:
                progress(done, len(pending), None)
    return summary
