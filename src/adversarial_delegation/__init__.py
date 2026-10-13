"""Adversarial delegation: measuring wealth-based steering in personal AI agents.

Public surface:

- ``engine.run`` — generate traces for one model + domain
- ``backends.get_backend`` — instantiate an OpenAI / vLLM / Gemini backend
- ``validate.validate`` — quick go/no-go discrimination check
- ``analysis.build_report`` — full single-model tables + figures

See ``conditions`` for the experimental design and ``prompts`` for the prompts.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import analysis, backends, conditions, engine, prompts, validate

__all__ = ["analysis", "backends", "conditions", "engine", "prompts", "validate", "__version__"]
