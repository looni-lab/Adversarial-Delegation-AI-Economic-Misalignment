"""Live smoke test: one rich + one poor persona through each of the 14 conditions.

That is 14 x 2 = 28 live trials by default (``ADV_SMOKE_SAMPLES=2``).

For every condition the test runs a handful of real trials against a live
OpenAI-compatible endpoint and asserts the pipeline works and the condition's
mechanics actually fire — search happens, blocked axes stay withheld, and email
caps hold. Each trial's full trace is written under ``data/trajectories/``
(gitignored). It is opt-in: with no API key configured the module is skipped.

Configure via environment:
    ADV_SMOKE_API_KEY   (or OPENAI_API_KEY / META_API_KEY)   required
    ADV_SMOKE_BASE_URL  default https://api.meta.ai/v1
    ADV_SMOKE_MODEL     default muse-spark-1.2
    ADV_SMOKE_DOMAIN    default flights
    ADV_SMOKE_INTENT    default cheap  (same intent for rich & poor; only persona varies)
    ADV_SMOKE_SAMPLES   default 2  (one rich persona + one poor)

Run:  pytest tests/ -v    (or:  python tests/test_access_conditions.py)
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adversarial_delegation.backends import get_backend
from adversarial_delegation.conditions import (
    CONDITIONS, CONTROL_PERSONA, INTENTS, blocked_axis, email_cap)
from adversarial_delegation.engine import Bundle, run_one, trace_ok, trace_path

# Smoke trajectories are written here (gitignored).
TRAJ_DIR = Path(__file__).resolve().parent.parent / "data" / "trajectories"

API_KEY = (os.environ.get("ADV_SMOKE_API_KEY") or os.environ.get("OPENAI_API_KEY")
           or os.environ.get("META_API_KEY"))
BASE_URL = os.environ.get("ADV_SMOKE_BASE_URL", "https://api.meta.ai/v1")
MODEL = os.environ.get("ADV_SMOKE_MODEL", "muse-spark-1.2")
DOMAIN = os.environ.get("ADV_SMOKE_DOMAIN", "flights")
INTENT = os.environ.get("ADV_SMOKE_INTENT", "cheap")  # same request for rich & poor
SAMPLES = int(os.environ.get("ADV_SMOKE_SAMPLES", "2"))  # sample 0 = rich (P01), 1 = poor (P17)

pytestmark = pytest.mark.skipif(
    not API_KEY, reason="set ADV_SMOKE_API_KEY (or OPENAI_API_KEY/META_API_KEY) to run")


def _bundle():
    if not hasattr(_bundle, "_cached"):
        _bundle._cached = Bundle.load(DOMAIN)
    return _bundle._cached


def _persona(bundle, pid):
    return next(p for p in bundle.personas["personas"] if p["id"] == pid)


def _samples(condition, bundle, n=SAMPLES):
    """n (persona, intent, motivation) triples; intent and motivation are held
    fixed across samples so the only thing that varies is the persona (rich vs
    poor), making the rich-vs-poor gap directly comparable."""
    intent = INTENT if INTENT in bundle.intents[DOMAIN] else INTENTS[0]
    mot = bundle.motivations[0]
    triples = []
    for i in range(n):
        if condition == "control":
            persona = CONTROL_PERSONA
        else:  # alternate a rich and a poor persona, same request
            pid = f"P{i // 2 + 1:02d}" if i % 2 == 0 else f"P{17 + i // 2:02d}"
            persona = _persona(bundle, pid)
        triples.append((persona, intent, mot))
    return triples


def _run(backend, condition, bundle, persona, intent, motivation) -> dict:
    """Run one trial via the engine (writes a trace) and return the trace record.

    Resumes: a trace that already exists and is usable is reused as-is; a
    missing one — or one left in parse_error/error from a prior run — is
    re-run (``run_one`` itself retries parse/error flakes up to 3x)."""
    trial = (condition, persona, intent, motivation)
    path = trace_path(TRAJ_DIR, MODEL, DOMAIN, trial)
    if not trace_ok(path):
        run_one(backend, MODEL, DOMAIN, trial, bundle, path)
    return json.loads(path.read_text())


def _check(condition, record):
    ar = record["access_record"]
    assert record["status"] != "error", f"{condition}: backend error: {record.get('error')}"
    assert record["recommendations"], f"{condition}: no recommendations ({record['status']})"
    assert ar["search_calls"], f"{condition}: never searched"
    axis = blocked_axis(condition)
    if axis:  # a blocked axis must never be disclosed
        assert axis not in ar["axes_accessed"], f"{condition}: leaked blocked axis {axis}"
    cap = email_cap(condition)
    if cap is not None:  # the email cap must hold
        assert len(ar["emails_read"]) <= cap, f"{condition}: read {len(ar['emails_read'])} > cap {cap}"


@pytest.fixture(scope="session")
def backend():
    return get_backend("openai", api_key=API_KEY, base_url=BASE_URL)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_condition(backend, condition):
    """Run SAMPLES trials for one condition and assert every one is healthy."""
    bundle = _bundle()
    triples = _samples(condition, bundle)
    with ThreadPoolExecutor(max_workers=min(SAMPLES, 8)) as pool:
        records = list(pool.map(lambda t: _run(backend, condition, bundle, *t), triples))
    for record in records:
        _check(condition, record)


def _main(workers=16):
    """Script entry: every condition x sample runs in ONE flat thread pool."""
    if not API_KEY:
        print("skip: set ADV_SMOKE_API_KEY (or OPENAI_API_KEY/META_API_KEY)")
        return 0
    be = get_backend("openai", api_key=API_KEY, base_url=BASE_URL)
    bundle = _bundle()
    jobs = [(c, *t) for c in CONDITIONS for t in _samples(c, bundle)]  # 14 x SAMPLES

    def one(job):
        condition = job[0]
        record = _run(be, condition, bundle, *job[1:])
        try:
            _check(condition, record)
            return condition, True, ""
        except AssertionError as exc:
            return condition, False, str(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(one, jobs))

    passed = {c: 0 for c in CONDITIONS}
    for condition, ok, msg in outcomes:
        passed[condition] += int(ok)
        if not ok:
            print(f"  FAIL {condition}: {msg}")
    for condition in CONDITIONS:
        n = passed[condition]
        print(f"[{'PASS' if n == SAMPLES else 'FAIL'}] {condition:24s} {n}/{SAMPLES}")
    failures = sum(SAMPLES - n for n in passed.values())
    print(f"\n{'OK' if failures == 0 else f'{failures} failed'} "
          f"({len(CONDITIONS)} conditions x {SAMPLES} samples, {workers} workers)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
