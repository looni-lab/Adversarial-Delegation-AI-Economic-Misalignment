# tests

A single **live** smoke test: one rich (P01) and one poor (P17) persona through
each of the 14 information-access conditions — 14 × 2 = **28 trials** — asserting
the pipeline runs and each condition's mechanics fire (search happens, blocked
axes stay withheld, email caps hold).

It talks to a real OpenAI-compatible endpoint, so it is **opt-in** — with no API
key configured the module is skipped. Each trial's full trace is saved under
`data/trajectories/<model>/<domain>/` (gitignored).

```bash
export ADV_SMOKE_API_KEY=sk-...          # or OPENAI_API_KEY / META_API_KEY
# optional overrides:
export ADV_SMOKE_BASE_URL=https://api.meta.ai/v1
export ADV_SMOKE_MODEL=muse-spark-1.2
export ADV_SMOKE_DOMAIN=flights
export ADV_SMOKE_SAMPLES=5

pytest tests/ -v                 # or: python tests/test_access_conditions.py
```

That is 14 × `ADV_SMOKE_SAMPLES` live trials (default 2 = one rich + one poor);
set `ADV_SMOKE_SAMPLES=1` for a quick check.
