#!/usr/bin/env python3
"""Daemon-level measurements for the references catalog (phase 0, part 2).

Companion to ``bench_references_scale.py``, which drives the plugin
in-process. This one goes through a REAL daemon and its runners, and adds
the embedding index:

1. ``index_*``: give every bundle a vector index in-process
   (``bundle index`` with the installed embedding provider, which is
   jaato-premium's sentence-transformers model), timed per catalog;
2. ``invoke_*``: ``listReferences`` / ``selectReferences`` through
   ``IPCClient.invoke_plugin_tool`` -- a runner per call, plugin initialised
   with the given config, the call, the runner closed. Measured with
   ``lookup_strategy: tags_only`` and ``hybrid`` (the second loads the
   embedding model in the runner and ranks a capped walk by vectors);
3. ``session_*``: ``create_session`` with a profile enabling ``references``
   (hybrid), and the RSS of every runner process of this daemon after it;
4. ``promote_*``: a claim written in-process, promoted through the daemon
   into an indexed bundle (``reference.promote``), whose index is then
   reconciled with vectors from the session's runner.

The daemon is ``jaato_sdk.conformance.daemon.ConformanceDaemon``: a
foreground child on a private temp socket with its own pid file, killed by
process group at the end. It never touches a daemon on the default socket.
HOME is a private directory (so ``~/.jaato`` is the benchmark's), HF_HOME
stays the operator's so the cached model is used and nothing is downloaded.

Not covered: AppArmor confinement. The private daemon runs as the invoking
user, and loading a profile needs root or passwordless ``apparmor_parser``.

Usage::

    .venv/bin/python scripts/bench_references_daemon.py \\
        --root /path/to/scratch --size 10000 --out daemon-10k.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_references_scale import AGE_SECONDS, generate  # noqa: E402

PROFILE = """\
name: bench
description: references scaling benchmark session
model: echo
provider: echo
plugins: [references]
plugin_configs:
  echo:
    usage: {prompt_tokens: 10, output_tokens: 5, total_tokens: 15}
  references:
    lookup_strategy: hybrid
    max_transitive_references: 25
"""


def _age(ws: Path) -> None:
    past = time.time() - AGE_SECONDS
    for dirpath, dirnames, filenames in os.walk(ws):
        for name in filenames + dirnames:
            os.utime(os.path.join(dirpath, name), (past, past))
        os.utime(dirpath, (past, past))


def _rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def _runner_rss(home: str) -> List[Dict[str, Any]]:
    """Every runner process whose HOME is this benchmark's private one."""
    rows = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{entry}/cmdline").read_bytes().split(b"\0")
            env = Path(f"/proc/{entry}/environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"jaato_server.server.runner" not in cmd:
            continue
        if f"HOME={home}".encode() not in env:
            continue
        ppid = 0
        for line in Path(f"/proc/{entry}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
        rows.append({"pid": int(entry), "ppid": ppid,
                     "rss_mb": round(_rss_mb(int(entry)), 1)})
    # A pool slot is forked from the template and keeps its command line, so
    # the role is read from the parent: a runner whose parent is another
    # runner is a slot; a cold-spawned runner or the template is a child of
    # the daemon.
    pids = {r["pid"] for r in rows}
    for r in rows:
        r["role"] = "slot" if r["ppid"] in pids else "template/cold"
    return rows


def index_bundles(ws: Path) -> Dict[str, Any]:
    """Give every workspace bundle a vector index, in-process."""
    from jaato_server.shared.plugins.references.plugin import create_plugin

    plugin = create_plugin()
    t = time.perf_counter()
    plugin.initialize({"workspace_path": str(ws), "lookup_strategy": "hybrid"})
    plugin.set_workspace_path(str(ws))
    out: Dict[str, Any] = {"index_plugin_init_s": round(time.perf_counter() - t, 2)}
    names = sorted(b.qualified_ref for b in plugin._bundles if not b.has_index)
    t = time.perf_counter()
    errors = []
    for name in names:
        res = plugin._cmd_bundle_index(name)
        if isinstance(res, dict) and res.get("error"):
            errors.append(f"{name}: {res['error']}")
    out["index_bundles"] = len(names)
    out["index_total_s"] = round(time.perf_counter() - t, 1)
    out["index_errors"] = errors[:3]
    return out


def write_claim(ws: Path) -> str:
    """Propose one reference as an agent would; return the claim id."""
    from jaato_server.shared.plugins.references.plugin import create_plugin
    from jaato_server.shared.session_context import isolated_current_session

    plugin = create_plugin()
    plugin.initialize({"workspace_path": str(ws), "lookup_strategy": "tags_only"})
    plugin.set_workspace_path(str(ws))
    with isolated_current_session():
        claim = plugin._execute_propose({
            "id": f"bench-promoted-{int(time.time())}", "name": "Benchmark page",
            "description": "promoted by the scaling benchmark", "tags": ["bench"],
            "content": "# Benchmark page\n\nWritten by bench_references_daemon.py.\n"})
    if "claim_id" not in claim:
        raise RuntimeError(f"proposal refused: {claim}")
    return claim["claim_id"]


async def through_daemon(ws: Path, home: str) -> Dict[str, Any]:
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.conformance.daemon import ConformanceDaemon
    from jaato_sdk.events import ClientType

    out: Dict[str, Any] = {}
    with ConformanceDaemon(ws) as daemon:
        client = IPCClient(daemon.socket_path, client_type=ClientType.API, auto_start=False,
                           workspace_path=str(ws), env_file=str(ws / ".env"))
        if not await client.connect(timeout=60):
            raise RuntimeError("could not connect to the private daemon")
        try:
            calls = (
                ("list_tags", "listReferences", {}, {"lookup_strategy": "tags_only"}),
                ("select_cap25_tags", "selectReferences", {"ids": ["ref-000000"]},
                 {"lookup_strategy": "tags_only", "max_transitive_references": 25}),
                ("select_cap25_hybrid", "selectReferences", {"ids": ["ref-000000"]},
                 {"lookup_strategy": "hybrid", "max_transitive_references": 25}),
            )
            for label, tool, args, cfg in calls:
                t = time.perf_counter()
                res = await client.invoke_plugin_tool(
                    "references", tool, args, plugin_configs={"references": cfg}, timeout=1800)
                out[f"invoke_{label}_s"] = round(time.perf_counter() - t, 2)
                out[f"invoke_{label}_ok"] = bool(res.ok and res.success)
                out[f"invoke_{label}_bytes"] = len(json.dumps(res.result, default=str))
                if not (res.ok and res.success):
                    out[f"invoke_{label}_error"] = (res.error or str(res.result))[:300]
                if label == "select_cap25_hybrid" and isinstance(res.result, dict):
                    trunc = res.result.get("truncated") or {}
                    out["select_cap25_hybrid_ranked_by"] = trunc.get("ranked_by")

            out["runners_before_session"] = _runner_rss(home)
            t = time.perf_counter()
            await client.create_session(profile="bench", timeout=600)
            out["session_new_s"] = round(time.perf_counter() - t, 2)
            await asyncio.sleep(2)
            out["runners_after_session"] = _runner_rss(home)

            claim_id = write_claim(ws)
            t = time.perf_counter()
            res = await client.promote_reference_claim(claim_id, bundle="bundle-0000", timeout=1800)
            out["promote_s"] = round(time.perf_counter() - t, 2)
            out["promote_ok"] = res.ok
            out["promote_reconcile"] = getattr(res, "reconcile", None)
            out["promote_reconcile_detail"] = (getattr(res, "reconcile_detail", "") or "")[:300]
            if not res.ok:
                out["promote_error"] = f"{res.category}: {res.error}"[:300]
            out["runners_after_promote"] = _runner_rss(home)
        finally:
            await client.disconnect()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--size", type=int, default=10000)
    ap.add_argument("--per-bundle", type=int, default=1000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    ws = args.root / f"dws-{args.size}"
    home = args.root / f"dhome-{args.size}"
    home.mkdir(parents=True, exist_ok=True)
    real_home = os.environ.get("HOME", "")
    os.environ.setdefault("HF_HOME", os.path.join(real_home, ".cache", "huggingface"))
    os.environ["HOME"] = str(home)
    os.environ.pop("REFERENCES_CONFIG_PATH", None)
    os.environ.setdefault("JAATO_RUNNER_POOL_SIZE", "1")

    row: Dict[str, Any] = {"n": args.size}
    if not (ws / ".jaato" / "references").exists():
        row.update(generate(ws, args.size, per_bundle=args.per_bundle, mentions=3,
                            max_links=3, body_kb=3, seed=1145))
    (ws / ".env").touch()
    profiles = ws / ".jaato" / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / "bench.yaml").write_text(PROFILE, encoding="utf-8")
    row.update(index_bundles(ws))
    _age(ws)
    row.update(asyncio.run(through_daemon(ws, str(home))))
    print(json.dumps(row, indent=2))
    if args.out:
        args.out.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
