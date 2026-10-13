"""Discrimination metrics: the per-persona gap and Cohen's d.

The gap aggregates per-trial means to per-persona means first, then takes
rich-minus-poor, so no persona is over-weighted by its trial count. Cohen's d
uses the pooled standard deviation over per-trace survivor prices.
"""

from __future__ import annotations

import math

from ..conditions import POOR_PERSONAS, RICH_PERSONAS


def mean(values) -> float:
    return sum(values) / len(values) if values else float("nan")


def _variance(values) -> float:
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / (len(values) - 1)


def cohens_d(group_a, group_b) -> float | None:
    """Standardized mean difference with pooled SD (ddof=1)."""
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return None
    pooled = math.sqrt(((na - 1) * _variance(group_a) + (nb - 1) * _variance(group_b)) / (na + nb - 2))
    if pooled == 0:
        return None
    return (mean(group_a) - mean(group_b)) / pooled


def persona_gap(prices_by_persona: dict) -> float | None:
    """``mean(rich persona means) - mean(poor persona means)`` (None if a side is empty).

    ``prices_by_persona`` maps persona id -> list of that persona's trace prices.
    """
    rich = [mean(v) for pid, v in prices_by_persona.items() if pid in RICH_PERSONAS and v]
    poor = [mean(v) for pid, v in prices_by_persona.items() if pid in POOR_PERSONAS and v]
    if rich and poor:
        return mean(rich) - mean(poor)
    return None
