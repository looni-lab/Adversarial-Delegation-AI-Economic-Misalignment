# CLAUDE.md — guide for agents working in this repo

## What this is
A harness that measures wealth-based steering in LLM recommendation agents.
A trial = (domain, condition, intent, motivation, persona). The agent browses a
fixed catalog with tools and returns five recommendations; we compare mean
recommended price for rich vs poor personas making the same request.

## Where things live
- `src/adversarial_delegation/`
  - `conditions.py` — the design: domains, the 14 conditions, intents, the
    rich/poor persona split, control persona. Start here.
  - `prompts.py` — the five system-prompt shapes + user message. **Single source
    of truth for prompt wording**; do not fork it per backend.
  - `tools.py` — neutral (OpenAI-shape) tool schemas + the paginated search.
  - `handler.py` — `TrialHandler`: dispatches tools, enforces `tool_blocked_*`
    and `email_capped_*`, and records the access path.
  - `schemas.py` — one JSON output schema per domain, consumed by all backends.
  - `engine.py` — enumerate trials, resume (skip existing traces), parallelise,
    write the trace record. Provider-agnostic.
  - `backends/` — `base.Backend` contract; `_openai_wire.py` is the shared
    OpenAI-wire trial loop used by `openai_backend` and `vllm_backend`;
    `gemini_backend` converts the neutral tools to protos.
  - `analysis/` — `gate.py` (the validated gate), `metrics.py`, `compute.py`
    (one streaming pass → metrics), `tables.py`/`figures.py`/`report.py`.
  - `validate.py` — quick go/no-go check. `cli.py` — `adv run|validate|analyze`.
- `data/raw/` — bundled synthetic inputs (personas, inventories, emails,
  intents, motivations). Inputs, not generations.

## Invariants (keep these true)
- **No file exceeds 300 lines.** Split before you cross it.
- **Prompts are defined once** in `prompts.py`; backends never re-word them.
- **One analysis method:** the trial-level in-range genuine-inventory gate
  (`analysis/gate.py`). It is a no-op for clean models; do not add a second
  pricing basis.
- **Never commit generations.** Traces/reports go to ignored dirs
  (`traces/`, `reports/`, `dump/`).
- Adding a provider = one adapter in `backends/` + one line in
  `backends/__init__.py`. Adding an OpenAI-compatible endpoint needs no new
  code — pass `--base-url`.

## Run / verify
```bash
pip install -e .
# dry-run prints the trial grid and exits — no API calls
adv run --backend openai --model <m> --domain flights --dry-run
# real run (traces land in --out, default ./data/trajectories/<model>/<domain>/)
adv run --backend openai --model <m> --domain flights
adv validate --model-dir data/trajectories/<m>
adv analyze  --model <m> --model-dir data/trajectories/<m> --out reports/<m>
```
`run` enumerates every (condition, intent, motivation, persona) trial, resumes
by skipping traces already on disk, and fans out across `--workers` (default 16).
Backends talk to live APIs; there is no offline fixture — keys come from the
environment (`OPENAI_API_KEY`, `GEMINI`/`GOOGLE_API_KEY`) or `--api-key`. Any
OpenAI-compatible endpoint (vLLM, Meta/Muse, etc.) works through the `openai`
backend with `--base-url`, no new code:
```bash
adv run --backend openai --base-url https://api.meta.ai/v1 \
        --model muse-spark-1.2 --domain flights --api-key "$META_API_KEY"
```
Smoke-test with a single `--condition`/`--persona`/`--intent` slice first.
