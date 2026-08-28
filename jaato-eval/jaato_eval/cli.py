"""``python -m jaato_eval`` — run a sweep, or report on one.

Exit codes mirror the verdict semantics, so CI can branch on them:

===  ==========================================================
0    every arm passed
1    at least one arm failed — a real defect in what was tested
2    at least one arm was blocked, or nothing ran at all
===  ==========================================================

A CI job that treats 2 as success is the vacuous pass this engine exists
to refuse; the code is distinct precisely so that mistake has to be made
deliberately.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .arm import ArmResult
from .manifest import ManifestError, discover_tasks
from .report import render_markdown
from .results import ResultStore
from .sweep import DEFAULT_CONCURRENCY, build_matrix, pool_size_advice, run_sweep
from .verdict import BLOCKED, FAIL, PASS


def _progress(result: ArmResult) -> None:
    glyph = {PASS: "✓", FAIL: "✘", BLOCKED: "○"}[result.state]
    tail = ""
    if result.blocked_reason:
        tail = f"  ({result.blocked_reason})"
    cost = (result.usage or {}).get("cost_usd")
    if isinstance(cost, (int, float)):
        tail += f"  ${cost:.4f}"
    print(f"{glyph} {result.spec.arm_id}{tail}", file=sys.stderr, flush=True)


def _exit_code(results: Sequence[ArmResult]) -> int:
    if any(r.state == FAIL for r in results):
        return 1
    if any(r.state == BLOCKED for r in results) or not results:
        return 2
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        tasks = discover_tasks(Path(args.tasks))
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2
    if not tasks:
        print(f"no task.yaml found under {args.tasks}", file=sys.stderr)
        return 2

    profile_sets: List[Optional[str]] = (
        [s.strip() for s in args.profile_set.split(",") if s.strip()]
        if args.profile_set else []
    )
    arms = build_matrix(tasks, profile_sets)
    store = ResultStore(Path(args.out))

    print(f"{len(tasks)} task(s), {len(arms)} arm(s), concurrency {args.concurrency}",
          file=sys.stderr)
    print(pool_size_advice(args.concurrency), file=sys.stderr)

    results = asyncio.run(run_sweep(
        arms, store=store, workspace_root=Path(args.workspaces),
        concurrency=args.concurrency, socket_path=args.socket,
        keep_workspaces=args.keep_workspaces, resume=args.resume,
        arm_timeout_seconds=args.arm_timeout,
        on_result=_progress,
    ))

    print("", file=sys.stderr)
    print(render_markdown(store.read()))
    return _exit_code(results)


def cmd_report(args: argparse.Namespace) -> int:
    store = ResultStore(Path(args.results))
    records = store.read()
    if not records:
        print(f"no records in {args.results}", file=sys.stderr)
        return 2
    print(render_markdown(records))
    states = [r.get("state") for r in records]
    if FAIL in states:
        return 1
    if BLOCKED in states:
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jaato_eval", description="Run agent eval sweeps over jaato sessions.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a sweep")
    run.add_argument("tasks", help="directory containing task.yaml files")
    run.add_argument("--profile-set", default="",
                     help="comma-separated profile sets to sweep across "
                          "(default: whatever each task declares)")
    run.add_argument("--out", default="results.jsonl", help="results JSONL path")
    run.add_argument("--workspaces", default=".jaato-eval-workspaces",
                     help="parent directory for per-arm scratch workspaces")
    run.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    run.add_argument("--socket", default=None, help="daemon IPC socket path")
    run.add_argument("--keep-workspaces", action="store_true",
                     help="leave scratch workspaces on disk for inspection")
    run.add_argument("--arm-timeout", type=float, default=None,
                     help="wall-clock ceiling per arm in seconds "
                          "(default 900; 0 disables). An arm that exceeds it "
                          "is BLOCKED — a task pool's `seconds` cannot bound "
                          "this, because it is reconciled when a session ends "
                          "and a session that never ends never consumes it.")
    run.add_argument("--resume", action="store_true",
                     help="skip arms already present in --out")
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="pivot an existing results file")
    rep.add_argument("results", help="results JSONL path")
    rep.set_defaults(func=cmd_report)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
