# Adversarial Delegation: AI Economic Misalignment

**Paper:** [*Et Tu, Brute? Economic Misalignment in Personal AI Agents*](https://arxiv.org/abs/2609.24927) · arXiv:2609.24927

Do personal AI agents give systematically different recommendations based on a
user's *inferred* wealth — even when the user states a preference that should
override it? This repository is the harness that measures it.

An agent recommends five options (flights, CS PhD programs, or health insurance)
from a fixed 200-item catalog. The catalog and prices never change; the only
thing that varies is **how** the agent learns about the user — not at all, from
the prompt, from an inbox it reads, or from a profile tool it calls. The
**discrimination gap** is the mean recommended price for wealthy personas minus
that for low-income personas making the identical request.

## Install

```bash
pip install -e .            # core (OpenAI + vLLM backends, analysis)
pip install -e '.[gemini]'  # add the Gemini backend
```

Requires Python 3.10+. The synthetic inputs ship in `data/raw/`; no real user
data is involved.

## Quickstart

```bash
# 0. Preview the trial grid without any API calls
adv run --backend openai --model gpt-5 --domain flights --dry-run

# 1. Generate traces for one model + domain
#    Resumes by skipping traces already on disk; fans out across --workers (default 16).
#    Writes to --out (default ./data/trajectories/<model>/<domain>/).
adv run --backend openai --model gpt-5 --domain flights

# 2. Quick go/no-go discrimination check
adv validate --model-dir data/trajectories/gpt-5

# 3. Full single-model report (tables + figures)
adv analyze --model gpt-5 --model-dir data/trajectories/gpt-5 --out reports/gpt-5
```

Backends: `openai`, `vllm` (local open-weight models), `gemini`. Any
OpenAI-compatible endpoint (vLLM, Meta/Muse, …) runs through the `openai`
backend with `--base-url`, no new code:

```bash
adv run --backend openai --base-url https://api.meta.ai/v1 \
        --model muse-spark-1.2 --domain flights --api-key "$META_API_KEY"
```

Keys are read from the environment (`OPENAI_API_KEY`, `GEMINI`/`GOOGLE_API_KEY`)
or passed with `--api-key`.

## Layout

```
src/adversarial_delegation/
  prompts.py conditions.py schemas.py tools.py handler.py data.py  # provider-neutral core
  engine.py            # trial enumeration, resume, parallelism, trace writing
  backends/            # one trial engine + thin OpenAI / vLLM / Gemini adapters
  analysis/            # the validated gate, metrics, and single-model report
  validate.py cli.py
data/raw/              # bundled synthetic inputs (personas, inventories, emails)
index.html             # project homepage (GitHub Pages)
```

Agent contributors should read [`CLAUDE.md`](CLAUDE.md).

## Method

Every reported number uses one rule — the **trial-level in-range
genuine-inventory gate**: keep a trial only if every recommendation has a real
catalog id and an in-range model-stated price; price survivors by their stated
field; aggregate per trial → per persona → rich minus poor.

## Citation

```bibtex
@misc{priyanshu2026ettubruteeconomic,
      title={Et Tu, Brute? Economic Misalignment in Personal AI Agents},
      author={Aman Priyanshu and Supriti Vijay and Brian Jabarian and Niloofar Mireshghallah},
      year={2026},
      eprint={2609.24927},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.24927},
}
```
