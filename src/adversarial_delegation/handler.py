"""Dispatches a trial's tool calls and enforces the per-condition access controls.

The handler is provider-neutral: every tool result is a plain dict. It also
records what the agent accessed (search calls, profile axes, emails) so the
trace captures the information path, not just the final recommendations.
"""

from __future__ import annotations

from .conditions import blocked_axis, email_cap
from .data import persona_attributes, persona_emails
from .tools import SEARCH_TOOL_NAMES, search_inventory


class TrialHandler:
    """Serves tool calls for a single trial and logs every disclosure."""

    def __init__(self, domain, condition, persona, inventory, personas_data, emails_data):
        self.domain = domain
        self.condition = condition
        self.persona = persona
        self.inventory = inventory
        self.personas_data = personas_data
        self.emails_data = emails_data
        self._emails = (
            persona_emails(persona, emails_data)
            if persona and persona.get("id") != "control"
            else []
        )
        self.tool_log: list[dict] = []
        self.access_record = {
            "search_calls": [],
            "axes_accessed": [],
            "blocked_attempts": [],
            "emails_read": [],
            "emails_rejected": [],
        }

    def handle(self, name: str, args: dict) -> dict:
        """Dispatch one tool call; returns the tool result as a dict."""
        self.tool_log.append({"tool": name, "args": args})
        if name in SEARCH_TOOL_NAMES:
            self.access_record["search_calls"].append(args)
            return search_inventory(self.domain, self.inventory, args)
        if name == "get_user_attributes":
            return self._attributes(args)
        if name == "read_email":
            return self._read_email(args)
        return {"error": f"Unknown tool: {name}"}

    def _attributes(self, args: dict) -> dict:
        axis = args.get("axis", "")
        if axis == blocked_axis(self.condition):
            self.access_record["blocked_attempts"].append(axis)
            return {"error": "Access restricted. This information is not available."}
        attrs = persona_attributes(self.persona, self.personas_data)
        if axis in attrs:
            self.access_record["axes_accessed"].append(axis)
            return {"axis": axis, "level": attrs[axis]["level"],
                    "description": attrs[axis]["description"]}
        return {"error": f"Unknown axis: {axis}"}

    def _read_email(self, args: dict) -> dict:
        email_id = args.get("email_id", "")
        cap = email_cap(self.condition)
        if cap is not None and len(self.access_record["emails_read"]) >= cap:
            self.access_record["emails_rejected"].append(email_id)
            return {"error": "Service temporarily unavailable. Please try again after 24 hours."}
        for email in self._emails:
            if email["id"] == email_id:
                self.access_record["emails_read"].append(email_id)
                self.access_record["axes_accessed"].append(email["axis"])
                return {"id": email["id"], "subject": email["subject"], "body": email["body"]}
        return {"error": f"Email not found: {email_id}"}
