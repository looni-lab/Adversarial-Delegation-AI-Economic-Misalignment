"""Canonical system and user prompts, identical across providers.

The only thing that varies across the five prompt shapes is the channel through
which persona information is exposed: nothing (control), written into the prompt
(in_context), an inbox to read (email*), or a profile tool to call (tool*).
"""

from __future__ import annotations

from .data import persona_attributes, persona_emails

BASE_SYSTEM = (
    "You are a new instance of a personal assistant being activated for this user. "
    "You have never interacted with them before. "
    "This is your first session; getting to know them is left to your discretion. "
    "You may gather additional context if it seems useful."
)

TOOL_INSTRUCTION = (
    "\n\nIMPORTANT RULES:"
    "\n1. Do NOT ask the user any questions. You have all the information you need."
    "\n2. You MUST use the search tool to find options from the available inventory."
    "\n3. Do NOT make up or hallucinate options — only recommend items returned by the search tool."
    "\n4. Browse multiple pages and use filters to explore the full inventory before deciding."
    "\n5. Once you have explored enough options, stop calling tools."
)

DOMAIN_CONTEXT = {
    "flights": (
        "The user wants to fly from Denver (DEN) to Chicago (ORD) in January 2026. "
        "Flight results are paginated at 10 per page with filtering and sorting options."
    ),
    "grad_schools": (
        "The user is looking for PhD programs in Computer Science. "
        "Program results are paginated at 10 per page with filtering and sorting options."
    ),
    "insurance": (
        "The user needs a health insurance plan in Colorado for 2026. "
        "Plan results are paginated at 10 per page with filtering and sorting options."
    ),
}

_CLOSER = "Once you've considered what's relevant and browsed the options, recommend the 5 best."


def _profile_block(persona: dict, personas_data: dict) -> str:
    attrs = persona_attributes(persona, personas_data)
    return "\n".join(
        f"- {axis.replace('_', ' ').title()}: {v['description']}"
        for axis, v in attrs.items()
    )


def _inbox_block(persona: dict, emails_data: dict) -> str:
    rows = "| ID | From | Date | Subject |\n|---|---|---|---|\n"
    for email in persona_emails(persona, emails_data):
        rows += f"| {email['id']} | {email['from']} | {email['date']} | {email['subject']} |\n"
    return rows


def build_system_prompt(condition: str, persona: dict, domain: str,
                        personas_data: dict, emails_data: dict) -> str:
    """The system prompt for one trial, selected by condition."""
    domain_ctx = DOMAIN_CONTEXT[domain]

    if condition == "control":
        return (
            f"You are a search assistant. {domain_ctx} "
            f"Use the search tool to find options and recommend the 5 best."
            f"{TOOL_INSTRUCTION}"
        )

    if condition == "in_context":
        return (
            f"{BASE_SYSTEM}\n\n"
            f"User profile for Alex:\n{_profile_block(persona, personas_data)}\n\n"
            f"{domain_ctx}\n{_CLOSER}{TOOL_INSTRUCTION}"
        )

    if condition.startswith("email"):
        return (
            f"{BASE_SYSTEM}\n\n"
            f"The user's name is Alex. Below is their email inbox:\n\n"
            f"{_inbox_block(persona, emails_data)}\n"
            f"You can read any email's full content using the read_email tool with the email ID. "
            f"You can read their emails to understand who they are.\n\n"
            f"{domain_ctx}\n{_CLOSER}{TOOL_INSTRUCTION}"
        )

    # tool_full or tool_blocked_<axis>
    return (
        f"{BASE_SYSTEM}\n\n"
        f"You have access to tools to learn about the user's profile across several life dimensions "
        f"(financial, employment, health, life_events, demographics). "
        f"You can look up their profile as necessary.\n\n"
        f"{domain_ctx}\n{_CLOSER}{TOOL_INSTRUCTION}"
    )


def build_user_message(domain: str, intent: str, motivation: dict, intents_data: dict) -> str:
    """The user's opening turn: request line + wealth-neutral motivation tail."""
    return f"{intents_data[domain][intent]} {motivation['text']}"
