"""Render one model's metric tables as PNG + Markdown + LaTeX.

Four tables: the capability summary (gap per domain + Cohen's d), the gap by
information condition, the flights gap by blocked axis, and the financial
retrieval vs effect summary. ‡ marks cells where under half the completed
trials pass the inventory gate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..conditions import CONDITIONS, DOMAINS

UNIT = {"flights": "$", "insurance": "$/mo", "grad_schools": "$/yr"}
_COND_ROWS = [c for c in CONDITIONS if c != "control"]


def gap(metrics: dict, domain: str, cond: str):
    """Gap for a cell, or None if ‡ (gate < 0.5) or missing."""
    rate = metrics["gate_pass_rate"][domain].get(cond)
    if rate is not None and rate < 0.5:
        return None
    return metrics["gaps"][domain].get(cond)


def fmt(value) -> str:
    return "‡" if value is None else f"{'+' if value >= 0 else '-'}{abs(int(round(value))):,}"


def _save_png(path: Path, header, rows, title, caption):
    char_w = [max([len(str(header[c]))] + [len(str(r[c])) for r in rows]) for c in range(len(header))]
    total = sum(char_w) or 1
    fig, ax = plt.subplots(figsize=(max(7, 0.16 * total + 2), max(2.2, 0.4 * (len(rows) + 2) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center",
                   colWidths=[w / total for w in char_w])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#2b2b2b")
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=12)
    fig.text(0.01, 0.01, caption, fontsize=7, color="#333333", wrap=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def _write_md(path: Path, header, rows, title, caption):
    lines = [f"# {title}", "", f"*{caption}*", "",
             "| " + " | ".join(header) + " |",
             "|" + "|".join(["--"] + ["--:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    path.write_text("\n".join(lines) + "\n")


def _write_tex(path: Path, header, rows, align):
    body = [r"\begin{tabular}{" + align + "}", r"\toprule",
            " & ".join(header) + r" \\", r"\midrule"]
    body += [" & ".join(str(c) for c in row) + r" \\" for row in rows]
    body += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(body) + "\n")


def _emit(out: Path, stem, header, rows, title, caption, align):
    _save_png(out / f"{stem}.png", header, rows, title, caption)
    _write_md(out / f"{stem}.md", header, rows, title, caption)
    _write_tex(out / f"{stem}.tex", header, rows, align)


def render_tables(model: str, metrics: dict, out: Path):
    """Write all four tables for ``model`` into ``out``."""
    out.mkdir(parents=True, exist_ok=True)

    # 1. capability
    header = ["Domain", "Gap", "Cohen's d"]
    rows = [[f"{d} ({UNIT[d]})", fmt(gap(metrics, d, "tool_full")),
             f"{metrics['d_by_domain'].get(d):.2f}" if d in metrics["d_by_domain"] else "–"]
            for d in DOMAINS]
    rows.append(["mean", "", f"{metrics['mean_d']:.2f}" if metrics["mean_d"] is not None else "–"])
    _emit(out, "capability", header, rows,
          f"{model} — discrimination gap by domain (tool-full)",
          "Rich (P01–P16) minus poor (P17–P32) mean recommended price. ‡ = inventory-gate exclusion.",
          "lcc")

    # 2. gap by information condition
    header = ["Condition"] + [f"{d} ({UNIT[d]})" for d in DOMAINS]
    rows = [[c] + [fmt(gap(metrics, d, c)) for d in DOMAINS] for c in _COND_ROWS]
    _emit(out, "by_condition", header, rows,
          f"{model} — gap by information condition",
          "Discrimination gap per domain and condition, pooled over intents and motivations.",
          "l" + "c" * len(DOMAINS))

    # 3. flights gap by blocked axis
    axes = [("None (tool-full)", "tool_full"), ("Financial", "tool_blocked_financial"),
            ("Employment", "tool_blocked_employment"), ("Health", "tool_blocked_health"),
            ("Life events", "tool_blocked_life_events"), ("Demographics", "tool_blocked_demographics")]
    rows = [[label, fmt(gap(metrics, "flights", cond))] for label, cond in axes]
    _emit(out, "blocking", ["Blocked axis", "Flights gap ($)"], rows,
          f"{model} — flights gap by blocked axis",
          "Blocking financial typically collapses the gap; other axes preserve or amplify it.",
          "lc")

    # 4. financial retrieval vs effect
    fr = metrics["financial_retrieval_pct"]
    rows = [["Financial retrieval, flights (%)",
             f"{fr['flights']:.1f}" if fr["flights"] is not None else "–"],
            ["Financial retrieval, all domains (%)",
             f"{fr['all_domains']:.1f}" if fr["all_domains"] is not None else "–"],
            ["Flights Cohen's d", f"{metrics['d_by_domain'].get('flights'):.2f}"
             if "flights" in metrics["d_by_domain"] else "–"],
            ["Flights gap ($)", fmt(gap(metrics, "flights", "tool_full"))]]
    _emit(out, "retrieval", ["Metric", "Value"], rows,
          f"{model} — financial retrieval vs effect (tool-full)",
          "High retrieval need not imply strong steering, and vice versa.", "lc")
