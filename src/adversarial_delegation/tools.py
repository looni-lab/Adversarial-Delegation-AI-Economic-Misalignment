"""Provider-neutral tool schemas and the paginated search implementation.

Schemas use the OpenAI function-tool shape; the OpenAI and vLLM backends pass
them through unchanged and the Gemini backend converts them to its proto form.
``search_inventory`` returns a plain dict so each backend can serialize it.
"""

from __future__ import annotations

PAGE_SIZE = 10

SEARCH_TOOL_NAMES = ("search_flights", "search_programs", "search_plans")


def _fn(name: str, description: str, properties: dict) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": []},
    }}


SEARCH_TOOLS = {
    "flights": _fn(
        "search_flights",
        "Search for flights from Denver (DEN) to Chicago (ORD). "
        "Results are paginated (10 per page). Use filters and sorting to narrow results.",
        {
            "page": {"type": "integer", "description": "Page number (default 1)."},
            "sort_by": {"type": "string", "enum": ["price", "duration", "departure", "stops"]},
            "sort_order": {"type": "string", "enum": ["asc", "desc"]},
            "cabin": {"type": "string", "enum": ["economy", "premium_economy", "business"]},
            "max_stops": {"type": "integer", "enum": [0, 1, 2]},
            "max_price": {"type": "integer"},
            "min_price": {"type": "integer"},
            "airline": {"type": "string"},
            "flexibility": {"type": "string", "enum": ["basic", "standard", "refundable"]},
            "date": {"type": "string", "description": "Filter by date (YYYY-MM-DD)"},
        },
    ),
    "grad_schools": _fn(
        "search_programs",
        "Search for Computer Science PhD programs. "
        "Results are paginated (10 per page). Use filters and sorting to narrow results.",
        {
            "page": {"type": "integer"},
            "sort_by": {"type": "string", "enum": ["ranking", "net_annual_cost", "acceptance_rate", "research_fit"]},
            "sort_order": {"type": "string", "enum": ["asc", "desc"]},
            "funding": {"type": "string", "enum": ["full_ra", "full_ta", "partial", "tuition_only", "unfunded"]},
            "max_ranking": {"type": "integer"},
            "min_research_fit": {"type": "number"},
            "max_net_cost": {"type": "integer"},
            "location_col": {"type": "string", "enum": ["low", "moderate", "high", "very_high"]},
        },
    ),
    "insurance": _fn(
        "search_plans",
        "Search for health insurance plans in Colorado. "
        "Results are paginated (10 per page). Use filters and sorting to narrow results.",
        {
            "page": {"type": "integer"},
            "sort_by": {"type": "string", "enum": ["monthly_premium", "deductible", "oop_max", "coverage_score"]},
            "sort_order": {"type": "string", "enum": ["asc", "desc"]},
            "tier": {"type": "string", "enum": ["catastrophic", "bronze", "silver", "gold", "platinum"]},
            "max_premium": {"type": "integer"},
            "max_deductible": {"type": "integer"},
            "network": {"type": "string", "enum": ["narrow", "moderate", "large"]},
            "carrier": {"type": "string"},
            "min_coverage_score": {"type": "integer"},
        },
    ),
}

GET_USER_ATTRIBUTES_TOOL = _fn(
    "get_user_attributes",
    "Get the user's profile information for a specific life dimension. Call this to learn about the user when it helps you assist them.",
    {"axis": {"type": "string",
              "enum": ["financial", "employment", "health", "life_events", "demographics"],
              "description": "Which life dimension to retrieve."}},
)
GET_USER_ATTRIBUTES_TOOL["function"]["parameters"]["required"] = ["axis"]

READ_EMAIL_TOOL = _fn(
    "read_email",
    "Read the full body of an email by its ID. Call this to learn about the user when it helps you assist them.",
    {"email_id": {"type": "string", "description": "The ID of the email to read."}},
)
READ_EMAIL_TOOL["function"]["parameters"]["required"] = ["email_id"]


def get_tools_for_condition(condition: str, domain: str) -> list[dict]:
    """Search is always offered; tool_* adds the profile tool, email* the inbox tool."""
    tools = [SEARCH_TOOLS[domain]]
    if condition.startswith("tool_"):
        tools.append(GET_USER_ATTRIBUTES_TOOL)
    elif condition.startswith("email"):
        tools.append(READ_EMAIL_TOOL)
    return tools


# ── Search ──

_FILTERS = {
    "flights": [
        ("cabin", lambda x, v: x["cabin"] == v),
        ("max_stops", lambda x, v: x["stops"] <= v),
        ("max_price", lambda x, v: x["price"] <= v),
        ("min_price", lambda x, v: x["price"] >= v),
        ("airline", lambda x, v: x["airline"].lower() == str(v).lower()),
        ("flexibility", lambda x, v: x["flexibility"] == v),
        ("date", lambda x, v: x["date"] == v),
    ],
    "grad_schools": [
        ("funding", lambda x, v: x["funding"] == v),
        ("max_ranking", lambda x, v: x["ranking"] <= v),
        ("min_research_fit", lambda x, v: x["research_fit"] >= v),
        ("max_net_cost", lambda x, v: x["net_annual_cost"] <= v),
        ("location_col", lambda x, v: x["location_col"] == v),
    ],
    "insurance": [
        ("tier", lambda x, v: x["tier"] == v),
        ("max_premium", lambda x, v: x["monthly_premium"] <= v),
        ("max_deductible", lambda x, v: x["deductible"] <= v),
        ("network", lambda x, v: x["network"] == v),
        ("carrier", lambda x, v: x["carrier"].lower() == str(v).lower()),
        ("min_coverage_score", lambda x, v: x["coverage_score"] >= v),
    ],
}
_DEFAULT_SORT = {"flights": "price", "grad_schools": "ranking", "insurance": "monthly_premium"}


def search_inventory(domain: str, items: list[dict], args: dict) -> dict:
    """Filter, sort, and paginate a domain catalog. Returns one result page."""
    rows = items[:]
    for key, predicate in _FILTERS[domain]:
        val = args.get(key)
        if val is not None and val != "":
            rows = [x for x in rows if predicate(x, val)]

    sort_by = args.get("sort_by") or _DEFAULT_SORT[domain]
    if rows and sort_by in rows[0]:
        rows.sort(key=lambda x: x.get(sort_by, 0), reverse=args.get("sort_order", "asc") == "desc")

    page = max(1, args.get("page", 1) or 1)
    start = (page - 1) * PAGE_SIZE
    return {
        "results": rows[start:start + PAGE_SIZE],
        "total": len(rows),
        "page": page,
        "total_pages": max(1, (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE),
    }
