"""Loaders for the synthetic experiment inputs and persona-derived views.

The inputs live in ``data/raw/`` at the repository root (bundled with the repo).
Resolution order for that directory:

1. the ``ADV_DATA_DIR`` environment variable, if set;
2. a ``data/raw`` folder found by walking up from this file;
3. a ``data/raw`` folder found by walking up from the current directory.

All inputs are synthetic. No real user data is involved.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

AXES = ["financial", "employment", "health", "life_events", "demographics"]


def _search_up(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        candidate = parent / "data" / "raw"
        if candidate.is_dir():
            return candidate
    return None


@lru_cache(maxsize=1)
def data_dir() -> Path:
    """Absolute path to the bundled ``data/raw`` directory."""
    env = os.environ.get("ADV_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        found = _search_up(start)
        if found:
            return found
    raise FileNotFoundError(
        "Could not locate data/raw. Set ADV_DATA_DIR to the inputs directory."
    )


def _load(name: str):
    with open(data_dir() / name) as fh:
        return json.load(fh)


def load_personas() -> dict:
    """The 32-persona panel plus the per-axis high/low description text."""
    return _load("personas.json")


def load_emails() -> dict:
    """Email subjects (shared across levels) and per-level bodies."""
    return _load("emails.json")


def load_intents() -> dict:
    """Opening request line per domain and intent."""
    return _load("domain-intents.json")


def load_motivations() -> list[dict]:
    """The five wealth-neutral motivation tails."""
    return _load("motivations.json")["motivations"]


_INVENTORY_SPEC = {
    "flights": ("flights.json", "flights"),
    "grad_schools": ("grad_programs.json", "programs"),
    "insurance": ("insurance_plans.json", "plans"),
}


def load_inventory(domain: str) -> list[dict]:
    """The 200-item search catalog for one domain."""
    filename, key = _INVENTORY_SPEC[domain]
    return _load(filename)[key]


def persona_attributes(persona: dict, personas_data: dict) -> dict:
    """Map each axis level of ``persona`` to its description text."""
    axes = personas_data["axes"]
    return {
        axis: {"level": persona[axis], "description": axes[axis][persona[axis]]}
        for axis in AXES
    }


def persona_emails(persona: dict, emails_data: dict) -> list[dict]:
    """The 10-email inbox for ``persona`` (two per axis), bodies level-matched."""
    subjects, bodies = emails_data["subjects"], emails_data["bodies"]
    inbox = []
    for axis in AXES:
        level = persona[axis]
        for suffix in ("1", "2"):
            key = f"{axis}_{suffix}"
            inbox.append({
                "id": key,
                "subject": subjects[key],
                "from": "notifications@service.com",
                "to": "alex@email.com",
                "date": "2025-12-15",
                "body": bodies[key][level],
                "axis": axis,
            })
    return inbox
