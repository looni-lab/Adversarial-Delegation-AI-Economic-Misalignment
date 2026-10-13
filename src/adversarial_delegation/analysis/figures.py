"""Render one model's figures as PNG.

Three figures: the blocking heatmap (axis x domain, % change vs tool-full), the
flights email/access depth ladder, and the flights intent-override stacked bars.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from ..conditions import DOMAINS, INTENTS
from .tables import gap

_AXES = [("Financial", "tool_blocked_financial"), ("Employment", "tool_blocked_employment"),
         ("Health", "tool_blocked_health"), ("Life events", "tool_blocked_life_events"),
         ("Demographics", "tool_blocked_demographics")]
_LADDER = ["email_capped_0", "email_capped_2", "email_capped_4", "email_capped_6",
           "email_capped_8", "email_full", "tool_full", "in_context"]
_LADDER_LBL = ["0\nemails", "2", "4", "6", "8", "all\nemails", "tool\nfull", "in\ncontext"]
_SERIES = "#1f4e79"


def _save(fig, path: Path):
    fig.savefig(path, dpi=160, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def _blocking(model, metrics, out: Path):
    z = np.full((len(_AXES), len(DOMAINS)), np.nan)
    text = [["" for _ in DOMAINS] for _ in _AXES]
    for j, dom in enumerate(DOMAINS):
        tf = gap(metrics, dom, "tool_full")
        for i, (_, cond) in enumerate(_AXES):
            bv = gap(metrics, dom, cond)
            if bv is None or tf in (None, 0):
                continue
            rel = (abs(bv) - abs(tf)) / abs(tf) * 100.0
            z[i][j] = rel
            text[i][j] = f"{bv:+d}\n({rel:+.0f}%)"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.imshow(z, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-100, vcenter=0, vmax=250), aspect="auto")
    ax.set_xticks(range(len(DOMAINS)), DOMAINS)
    ax.set_yticks(range(len(_AXES)), [a[0] for a in _AXES])
    for i in range(len(_AXES)):
        for j in range(len(DOMAINS)):
            if text[i][j]:
                ax.text(j, i, text[i][j], ha="center", va="center", fontsize=8,
                        color="white" if abs(z[i][j]) > 120 else "black")
    ax.set_title(f"{model} — gap change vs tool-full when an axis is blocked",
                 fontsize=11, fontweight="bold")
    _save(fig, out / "blocking.png")


def _depth(model, metrics, out: Path):
    ys = [gap(metrics, "flights", c) for c in _LADDER]
    xs = [i for i, y in enumerate(ys) if y is not None]
    vals = [y for y in ys if y is not None]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, vals, marker="o", color=_SERIES, lw=2)
    for x, y in zip(xs, vals):
        ax.annotate(f"{y:+d}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=_SERIES)
    ax.axvspan(-0.4, 5.4, color="#f2f2f2", zorder=0)
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_xticks(range(len(_LADDER)), _LADDER_LBL, fontsize=8)
    ax.set_ylabel("Discrimination gap ($)")
    ax.set_title(f"{model} — flights gap across the access ladder", fontsize=11, fontweight="bold")
    _save(fig, out / "depth_ladder.png")


def _intent(model, metrics, out: Path):
    split = metrics["intent_flights_in_context"]
    labels = {"neutral": "Neutral", "cheap": "Cheapest", "quality": "Quality", "ceiling": "Price\nceiling"}
    present = [i for i in INTENTS if i in split]
    base = [split[i]["poor"] for i in present]
    extra = [split[i]["gap"] for i in present]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(present))
    ax.bar(x, base, color=_SERIES, alpha=0.55, label="Low-income recommendation")
    ax.bar(x, extra, bottom=base, color=_SERIES, hatch="//", edgecolor="white",
           label="Wealth-conditioned gap")
    for xi, (b, e) in enumerate(zip(base, extra)):
        ax.text(xi, b + e + 6, f"+${int(e)}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=_SERIES)
    ax.set_xticks(list(x), [labels[i] for i in present], fontsize=9)
    ax.set_ylabel("Mean recommended flight price ($)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{model} — flights intent override (in-context)", fontsize=11, fontweight="bold")
    _save(fig, out / "intent_override.png")


def render_figures(model: str, metrics: dict, out: Path):
    """Write all three figures for ``model`` into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    _blocking(model, metrics, out)
    _depth(model, metrics, out)
    if metrics["intent_flights_in_context"]:
        _intent(model, metrics, out)
