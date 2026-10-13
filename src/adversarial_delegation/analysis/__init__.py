"""Single-model analytics: the validated gate, metrics, and report rendering."""

from .compute import compute_model
from .report import build_report

__all__ = ["compute_model", "build_report"]
