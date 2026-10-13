"""Command-line interface: ``adv run`` / ``adv validate`` / ``adv analyze``.

    adv run      --backend openai --model gpt-5 --domain flights --out traces
    adv validate --model-dir traces/gpt-5
    adv analyze  --model gpt-5 --model-dir traces/gpt-5 --out reports/gpt-5
"""

from __future__ import annotations

import argparse
import json
import sys

from .backends import BACKENDS, get_backend
from .conditions import DOMAINS


def _add_run(sub):
    p = sub.add_parser("run", help="generate traces for one model + domain")
    p.add_argument("--backend", required=True, choices=BACKENDS)
    p.add_argument("--model", required=True)
    p.add_argument("--domain", required=True, choices=DOMAINS)
    p.add_argument("--out", default="data/trajectories",
                   help="output root (default: ./data/trajectories)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--condition", help="run only this condition")
    p.add_argument("--intent", help="run only this intent")
    p.add_argument("--persona", help="run only this persona id")
    p.add_argument("--base-url", help="OpenAI-compatible / vLLM endpoint override")
    p.add_argument("--api-key", help="API key override (else read from environment)")
    p.add_argument("--dry-run", action="store_true", help="print trial counts and exit")


def _backend_kwargs(args) -> dict:
    kwargs = {}
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if args.base_url and args.backend in ("openai", "vllm"):
        kwargs["base_url"] = args.base_url
    return kwargs


def _cmd_run(args) -> int:
    from .engine import run
    if args.dry_run:
        backend = None
    else:
        backend = get_backend(args.backend, **_backend_kwargs(args))

    def progress(done, total, _status):
        print(f"\r  {args.domain}/{args.model}: {done}/{total}", end="", file=sys.stderr)

    summary = run(backend, args.model, args.domain, args.out, workers=args.workers,
                  conditions=[args.condition] if args.condition else None,
                  intents=[args.intent] if args.intent else None,
                  persona_id=args.persona, dry_run=args.dry_run,
                  progress=None if args.dry_run else progress)
    print(("\n" if not args.dry_run else "") + json.dumps(summary))
    return 0


def _cmd_validate(args) -> int:
    from .validate import validate
    passed, report = validate(args.model_dir)
    print(json.dumps(report, indent=2))
    print("RESULT:", "DISCRIMINATION CONFIRMED" if passed else "INSUFFICIENT SIGNAL")
    return 0 if passed else 1


def _cmd_analyze(args) -> int:
    from .analysis import build_report
    metrics = build_report(args.model, args.model_dir, args.out)
    print(f"wrote report for {args.model} -> {args.out}")
    print(json.dumps({"mean_d": metrics["mean_d"],
                      "tool_full_gaps": {d: metrics["gaps"][d].get("tool_full") for d in DOMAINS}},
                     indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adv", description="Adversarial delegation experiment.")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run(sub)
    v = sub.add_parser("validate", help="quick discrimination check on a trace dir")
    v.add_argument("--model-dir", required=True)
    a = sub.add_parser("analyze", help="full single-model tables + figures")
    a.add_argument("--model", required=True, help="label used in titles/output")
    a.add_argument("--model-dir", required=True, help="the model's trace tree")
    a.add_argument("--out", required=True, help="report output directory")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"run": _cmd_run, "validate": _cmd_validate, "analyze": _cmd_analyze}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
