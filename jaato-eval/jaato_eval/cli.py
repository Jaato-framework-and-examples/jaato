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
from .report_html import ReportDependencyError, write_html, write_pdf
from .results import ResultStore
from .sweep import DEFAULT_CONCURRENCY, build_matrix, pool_size_advice, run_sweep
from .verdict import BLOCKED, FAIL, PASS


def _progress(result: ArmResult) -> None:
    glyph = {PASS: "✓", FAIL: "✘", BLOCKED: "○"}[result.state]
    tail = ""
    if result.blocked_reason:
        tail = f"  ({result.blocked_reason})"
    elif result.error:
        # A VERDICT AND AN ERROR TERMINAL AT ONCE.  An arm graded through a
        # missing signal_completion (jaato #773) has a real state, so it
        # gets a ✓ or ✘ like any other — but it also ended badly, and the
        # live line is where an operator is actually looking.  Without this
        # such an arm is indistinguishable from a clean one until someone
        # opens the results file.
        tail = f"  (graded without a sign-off: {result.error})"
    cost = (result.usage or {}).get("cost_usd")
    if isinstance(cost, (int, float)):
        tail += f"  ${cost:.4f}"
    print(f"{glyph} {result.spec.arm_id}{tail}", file=sys.stderr, flush=True)


def _absolute(path: str) -> Path:
    """Resolve a CLI path against THIS process's cwd.

    A workspace path is sent to the daemon, which has its own cwd and a
    lifetime longer than any sweep.  Left relative, ``--workspaces
    .jaato-eval-workspaces`` meant one directory to the harness (which
    writes each arm's fixture) and another to the daemon (which runs the
    agent in it) whenever the two processes were started from different
    places — so the agent got a workspace with its worktree but no
    fixture, and the grader read a workspace with the fixture but no
    repository, with no error on either side (issue #742).  The daemon now
    refuses a relative path; resolving it here, where the cwd it is
    relative to actually lives, is the caller's half of that contract.
    """
    return Path(path).expanduser().resolve()


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
        arms, store=store, workspace_root=_absolute(args.workspaces),
        concurrency=args.concurrency, socket_path=args.socket,
        keep_workspaces=args.keep_workspaces, resume=args.resume,
        arm_timeout_seconds=args.arm_timeout,
        on_result=_progress,
    ))

    print("", file=sys.stderr)
    records = store.read()
    print(render_markdown(records))
    rendered = _render_documents(args, records)
    return rendered if rendered else _exit_code(results)


def _render_documents(args: argparse.Namespace,
                      records: Sequence[dict]) -> int:
    """Write the per-arm document(s) asked for.  ``0`` unless PDF failed.

    Returns an exit code ONLY for the one failure a caller must not miss:
    ``--pdf`` without the optional renderer.  A sweep run unattended asked
    for a PDF and did not get one, and silently exiting on the verdict
    codes would report that as an ordinary result.  Every other outcome
    returns 0 and lets the verdict decide the exit, because the verdict is
    what the exit codes mean.
    """
    if getattr(args, "html", None):
        path = write_html(records, _absolute(args.html))
        print(f"per-arm report: {path}", file=sys.stderr)
    if not getattr(args, "pdf", None):
        return 0
    try:
        path = write_pdf(records, _absolute(args.pdf))
    except ReportDependencyError as exc:
        print(f"--pdf: {exc}", file=sys.stderr)
        return 2
    print(f"per-arm report: {path}", file=sys.stderr)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store = ResultStore(Path(args.results))
    records = store.read()
    if not records:
        print(f"no records in {args.results}", file=sys.stderr)
        return 2
    print(render_markdown(records))
    rendered = _render_documents(args, records)
    if rendered:
        return rendered
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
                     help="parent directory for per-arm scratch workspaces "
                          "(resolved against this process's cwd before it "
                          "is sent to the daemon)")
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
    _add_document_arguments(run)
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="pivot an existing results file")
    rep.add_argument("results", help="results JSONL path")
    _add_document_arguments(rep)
    rep.set_defaults(func=cmd_report)
    return parser


def _add_document_arguments(parser: argparse.ArgumentParser) -> None:
    """The per-arm document flags, shared by ``run`` and ``report``.

    Both subcommands take them because the two are the same question at
    different times: a sweep wants the document written as it finishes,
    and an old ``results.jsonl`` wants it written now — including one
    produced before this feature existed, whose new columns simply render
    as unknown.
    """
    parser.add_argument("--html", default=None, metavar="PATH",
                        help="write the per-arm report as a self-contained "
                             "HTML document (no dependencies; carries print "
                             "CSS, so any browser prints it to PDF)")
    parser.add_argument("--pdf", default=None, metavar="PATH",
                        help="also render that document to PDF. Needs the "
                             "optional renderer: pip install "
                             "'jaato-eval[report]'. Missing it is an error, "
                             "not a silent HTML-only fallback.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
