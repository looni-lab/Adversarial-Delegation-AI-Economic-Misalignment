"""JSON schema for the final structured output, one per domain.

A single schema source feeds all three backends: OpenAI (``response_format``
json_schema), Gemini (``response_schema``), and vLLM (``guided_json``). Every
trial returns a ``reasoning`` string and exactly five ``recommendations``.
"""

from __future__ import annotations

_REC_FIELDS = {
    "flights": {
        "id": {"type": "string"},
        "price": {"type": "number"},
        "stops": {"type": "integer"},
        "duration_hrs": {"type": "number"},
        "cabin": {"type": "string"},
        "airline": {"type": "string"},
        "departure": {"type": "string"},
        "flexibility": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "grad_schools": {
        "id": {"type": "string"},
        "university": {"type": "string"},
        "ranking": {"type": "integer"},
        "funding": {"type": "string"},
        "net_annual_cost": {"type": "number"},
        "acceptance_rate": {"type": "number"},
        "research_fit": {"type": "number"},
        "location_col": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "insurance": {
        "id": {"type": "string"},
        "plan_name": {"type": "string"},
        "monthly_premium": {"type": "number"},
        "deductible": {"type": "number"},
        "oop_max": {"type": "number"},
        "network": {"type": "string"},
        "tier": {"type": "string"},
        "coverage_score": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
}

_REQUIRED = {
    "flights": ["id", "price", "stops", "cabin", "airline", "reasoning"],
    "grad_schools": ["id", "university", "ranking", "funding", "net_annual_cost", "reasoning"],
    "insurance": ["id", "plan_name", "monthly_premium", "deductible", "tier", "reasoning"],
}


def response_schema(domain: str) -> dict:
    """The bare JSON schema (``reasoning`` + ``recommendations``) for a domain."""
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": _REC_FIELDS[domain],
                    "required": _REQUIRED[domain],
                },
            },
        },
        "required": ["reasoning", "recommendations"],
    }


def openai_response_format(domain: str) -> dict:
    """Wrap the schema for the OpenAI ``response_format`` structured-output param."""
    return {
        "type": "json_schema",
        "json_schema": {"name": f"{domain}_recommendations",
                        "schema": response_schema(domain)},
    }
