"""Backend contract shared by every provider adapter.

A backend owns one trial end to end — the tool-calling loop and the final
structured extraction — given a system prompt, a user message, the neutral tool
list, and a ``handle(name, args) -> dict`` callback. The engine owns everything
else (trial enumeration, resume, parallelism, trace writing), so adapters stay
small and provider differences stay isolated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TrialResult:
    """What a backend returns for one trial."""
    status: str                                   # complete | incomplete | parse_error | error
    reasoning: str | None = None
    recommendations: list[dict] | None = None
    trajectory: list | None = field(default=None)
    error: str | None = None


class Backend(ABC):
    """One LLM provider. Subclasses implement a single trial."""

    name: str = "base"

    @abstractmethod
    def run_trial(self, *, model: str, domain: str, system_prompt: str,
                  user_message: str, tools: list[dict], handler) -> TrialResult:
        """Run the tool loop + extraction for one trial.

        ``handler`` is a ``TrialHandler``; use ``handler.handle`` to serve tool
        calls and ``handler.access_record`` to inspect what was accessed.
        """
        raise NotImplementedError
