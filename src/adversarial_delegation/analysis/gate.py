"""The trial-level in-range genuine-inventory gate (the validated pricing basis).

A trial survives only if *every* recommendation has (a) an id present in the
domain catalog and (b) a model-stated price within the domain range. Survivors
are priced by their model-stated field — never re-priced from the catalog. The
gate is a no-op for well-behaved models and only bites on ones that fabricate
ids or emit out-of-range prices.
"""

from __future__ import annotations

from functools import lru_cache

from ..data import load_inventory

PRICE_FIELD = {"flights": "price", "grad_schools": "net_annual_cost",
               "insurance": "monthly_premium"}
RANGE = {"flights": (91, 883), "grad_schools": (-20000, 60991), "insurance": (85, 1350)}


@lru_cache(maxsize=None)
def _catalog(domain: str) -> frozenset:
    """The set of genuine item ids for a domain."""
    return frozenset(item["id"] for item in load_inventory(domain))


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def n_genuine(trace: dict, domain: str) -> int:
    """How many recommendations carry a real catalog id (for the ‡ gate rate)."""
    ids = _catalog(domain)
    return sum(1 for r in (trace.get("recommendations") or []) if r.get("id") in ids)


def passes_gate(trace: dict, domain: str) -> bool:
    """True iff every rec has a genuine id and an in-range model-stated price."""
    recs = trace.get("recommendations") or []
    if not recs:
        return False
    ids = _catalog(domain)
    field = PRICE_FIELD[domain]
    lo, hi = RANGE[domain]
    for rec in recs:
        if rec.get("id") not in ids:
            return False
        price = _num(rec.get(field))
        if price is None or not (lo <= price <= hi):
            return False
    return True


def trace_price(trace: dict, domain: str, *, gated: bool = True) -> float | None:
    """Mean model-stated price over a trace's recs (None if gated out / empty)."""
    if gated and not passes_gate(trace, domain):
        return None
    field = PRICE_FIELD[domain]
    vals = [_num(r.get(field)) for r in (trace.get("recommendations") or [])]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None
