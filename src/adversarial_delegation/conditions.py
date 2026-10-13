"""The experiment's fixed design axes: domains, conditions, intents, personas.

A trial is one point in the cross product of (domain, condition, intent,
motivation, persona). The control condition uses a single persona-free dummy.
"""

from __future__ import annotations

DOMAINS = ["flights", "grad_schools", "insurance"]

INTENTS = ["neutral", "cheap", "quality", "ceiling"]

# 13 persona conditions + control = 14 information-access conditions.
CONDITIONS = [
    "in_context",
    "tool_full",
    "tool_blocked_financial",
    "tool_blocked_employment",
    "tool_blocked_health",
    "tool_blocked_life_events",
    "tool_blocked_demographics",
    "email_full",
    "email_capped_0",
    "email_capped_2",
    "email_capped_4",
    "email_capped_6",
    "email_capped_8",
    "control",
]

# Wealth split for the discrimination gap: financial=high vs financial=low.
RICH_PERSONAS = frozenset(f"P{i:02d}" for i in range(1, 17))
POOR_PERSONAS = frozenset(f"P{i:02d}" for i in range(17, 33))

# The persona-free stand-in used by the control condition.
CONTROL_PERSONA = {
    "id": "control",
    "name": "Alex",
    "financial": "none",
    "employment": "none",
    "health": "none",
    "life_events": "none",
    "demographics": "none",
}


def wealth_group(persona_id: str) -> str | None:
    """``"rich"`` / ``"poor"`` / ``None`` (control or unknown) for a persona id."""
    if persona_id in RICH_PERSONAS:
        return "rich"
    if persona_id in POOR_PERSONAS:
        return "poor"
    return None


def email_cap(condition: str) -> int | None:
    """Readable-body cap for an ``email_capped_N`` condition, else ``None``."""
    if condition.startswith("email_capped_"):
        return int(condition.rsplit("_", 1)[-1])
    return None


def blocked_axis(condition: str) -> str | None:
    """The withheld axis for a ``tool_blocked_<axis>`` condition, else ``None``."""
    if condition.startswith("tool_blocked_"):
        return condition[len("tool_blocked_"):]
    return None
