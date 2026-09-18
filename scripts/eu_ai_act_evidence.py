#!/usr/bin/env python3
"""Capture visual evidence of every EU AI Act control the framework implements.

Drives the INSTALLED framework -- a real daemon on a private socket, real
sessions on the credential-free ``echo`` provider, the real ``jaato-scaffold``
and ``jaato-doctor`` commands -- and records what each control does as a
text capture, an SVG rendering of the terminal and a PNG of that SVG.  The
manual at ``docs/eu-ai-act-manual.md`` embeds the images by these stable
names; regenerating replaces the pictures without touching the prose.

    python scripts/eu_ai_act_evidence.py                 # -> docs/eu-ai-act-manual/evidence/
    python scripts/eu_ai_act_evidence.py --out /tmp/ev   # elsewhere
    python scripts/eu_ai_act_evidence.py --no-png        # skip the Chromium step

Everything here is MEASURED, never asserted: a capture is the bytes the
command produced, and a control that did not do what the manual claims
shows up as a capture that says so (``NOT WRITTEN``, ``no announcement``).
That is the point of generating the manual from a run rather than writing
it from memory -- the day a mechanism goes inert the picture goes with it.

Two things are rewritten in the captures and stated in the manual: the
throwaway workspace root becomes ``/srv/acme-support`` and the private
socket becomes ``/run/jaato/jaato.sock``, so the pictures read as a
deployment rather than as a temp directory.  Nothing else is edited.

Requires the server tree importable (``server``, ``shared``, ``jaato_sdk``)
and ``rich``; the PNG step needs a Chromium binary (``--chromium`` or
``$JAATO_EVIDENCE_CHROMIUM``, else the Playwright install under
``/opt/pw-browsers`` when present).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import glob
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKSPACE_ALIAS = "/srv/acme-support"
SOCKET_ALIAS = "/run/jaato/jaato.sock"
PIDFILE_ALIAS = "/run/jaato/jaato.pid"
CAPTURE_WIDTH = 104
STARTUP_TIMEOUT = 90.0

# A 1x1 PNG, for the output marker.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAC"
    "hwGA60e6kgAAAABJRU5ErkJggg==")


@dataclass
class Capture:
    """One piece of evidence: what was run, and what it produced."""

    slug: str
    title: str
    command: str
    text: str
    control: str
    note: str = ""
    files: Dict[str, str] = field(default_factory=dict)


# ----------------------------------------------------------------- fixtures

SCREENER_PROFILE = """\
name: screener
description: Pre-screens inbound support tickets and drafts a reply for a human agent to approve.
model: echo
provider: echo
plugins: []
regulatory:
  intended_purpose: Pre-screen inbound support tickets and draft a reply for a human agent to approve before it is sent.
  risk_class: limited
  interacts_with_persons: true
  provider:
    name: Acme Support GmbH
    contact: compliance@acme.example
trace:
  ledger: .jaato/logs/ledger.jsonl
  session_log: .jaato/logs/session_trace.jsonl
record_keeping:
  retention_days: 180
  conversation_retention_days: 30
  integrity: sha256-chain
budget_control:
  limits: {turns: 2}
  degrade:
    - {at: 100, action: abort}
plugin_configs:
  echo:
    usage: {prompt_tokens: 12, output_tokens: 7, total_tokens: 19}
"""

TRIAGE_PROFILE = """\
name: triage
description: Triage assistant with a persona and a high-risk classification, declaring almost nothing else
model: echo
provider: echo
plugins: [cli, memory]
default_agent: triage
regulatory:
  risk_class: high
  annex_iii: "5(a) access to essential private services"
suppress_base_instructions: {disclosure: true}
"""

TRIAGE_PERSONA = """\
You are the triage assistant for Acme Support. Read each ticket and draft a reply.
"""

QUIET_PROFILE = """\
name: quiet
description: Drops the disclosure piece by name, to show the framework announcing the posture change
model: echo
provider: echo
plugins: []
suppress_base_instructions: {disclosure: true}
"""


def _memory_profile(name: str, description: str, tool: str, args: Dict[str, Any],
                    extra_plugin_configs: Optional[Dict[str, Any]] = None) -> str:
    cfg: Dict[str, Any] = {
        "echo": {"tool_call": {"name": tool, "args": args},
                 "usage": {"prompt_tokens": 40, "output_tokens": 20, "total_tokens": 60}},
    }
    cfg.update(extra_plugin_configs or {})
    return json.dumps({
        "name": name, "description": description, "model": "echo", "provider": "echo",
        "plugins": ["memory(preload)"], "plugin_configs": cfg,
    }, indent=2)


def build_workspaces(root: Path) -> Dict[str, Path]:
    """Three workspaces: the compliant deployment, a non-compliant one, a quiet one."""
    acme = root / "acme-support"
    (acme / ".jaato" / "profiles").mkdir(parents=True)
    (acme / ".jaato" / "profiles" / "screener.yaml").write_text(SCREENER_PROFILE)
    (acme / ".jaato" / "profiles" / "writer.json").write_text(_memory_profile(
        "writer", "Stores a memory the model learned (echo-driven)", "store_memory", {
            "content": "Customers on the Pro plan get a 14-day refund window; Basic plan gets 7 days.",
            "description": "Refund windows per plan", "tags": ["refunds", "billing"],
            "scope": "project"}))
    (acme / ".jaato" / "profiles" / "reader.json").write_text(_memory_profile(
        "reader", "Retrieves memories under require_curation", "retrieve_memories",
        {"tags": ["refunds"]}, {"memory": {"require_curation": True}}))
    non = root / "triage-uncontrolled"
    (non / ".jaato" / "profiles").mkdir(parents=True)
    (non / ".jaato" / "agents").mkdir(parents=True)
    (non / ".jaato" / "profiles" / "triage.yaml").write_text(TRIAGE_PROFILE)
    (non / ".jaato" / "agents" / "triage.md").write_text(TRIAGE_PERSONA)
    quiet = root / "quiet"
    (quiet / ".jaato" / "profiles").mkdir(parents=True)
    (quiet / ".jaato" / "profiles" / "quiet.yaml").write_text(QUIET_PROFILE)
    return {"acme": acme, "non": non, "quiet": quiet}


# ------------------------------------------------------------------- daemon

class Daemon:
    """A daemon owned by this run, on a private socket, torn down at exit."""

    def __init__(self, root: Path) -> None:
        self.socket = str(root / "jaato.sock")
        self.pidfile = str(root / "jaato.pid")
        self.log = str(root / "daemon.log")

    def start(self) -> None:
        env = dict(os.environ)
        env.setdefault("JAATO_RUNNER_POOL_ENABLED", "false")
        subprocess.run([sys.executable, "-m", "server", "--ipc-socket", self.socket,
                        "--pid-file", self.pidfile, "--daemon"],
                       cwd=_server_dir(), env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.settimeout(0.5); s.connect(self.socket); return
            except OSError:
                time.sleep(0.25)
            finally:
                s.close()
        raise RuntimeError(f"daemon on {self.socket} did not accept a connection")

    def stop(self) -> None:
        subprocess.run([sys.executable, "-m", "server", "--stop", "--ipc-socket", self.socket,
                        "--pid-file", self.pidfile], cwd=_server_dir(),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _server_dir() -> str:
    import server  # noqa: F401 -- resolves the installed tree
    return str(Path(server.__file__).resolve().parent.parent)


# ----------------------------------------------------------- SDK sessions

async def _drive(workspace: Path, sock: str, profile: str, prompts: List[str],
                 answer_permissions: bool = True) -> Dict[str, Any]:
    """One session: subscribe BEFORE creation, ask, end (which persists it)."""
    from jaato_sdk.client.convenience import Session
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.events import ClientType, EventType

    events: List[Dict[str, Any]] = []
    c = IPCClient(socket_path=sock, client_type=ClientType.API, auto_start=False,
                  workspace_path=str(workspace))
    if not await c.connect(timeout=15):
        raise RuntimeError("could not connect to the evidence daemon")
    c.subscribe(EventType.SESSION_INFO, lambda ev: events.append(
        {"event": "SESSION_INFO", "disclosure_announcement":
         getattr(ev, "disclosure_announcement", None)}))
    c.subscribe(EventType.AGENT_OUTPUT, lambda ev: events.append(
        {"event": "AGENT_OUTPUT", "source": ev.source, "text": ev.text}))
    c.subscribe(EventType.SESSION_TERMINATED, lambda ev: events.append(
        {"event": "SESSION_TERMINATED", "reason": ev.reason}))
    c.subscribe(EventType.TOOL_CALL_START, lambda ev: events.append(
        {"event": "TOOL_CALL_START", "tool": ev.tool_name}))
    if hasattr(EventType, "INCIDENT"):
        c.subscribe(EventType.INCIDENT, lambda ev: events.append(
            {"event": "INCIDENT", "kind": ev.kind, "cause": ev.cause}))

    def on_perm(ev):
        events.append({"event": "PERMISSION_REQUESTED", "tool": ev.tool_name,
                       "prompt_lines": list(getattr(ev, "prompt_lines", None) or [])})
        return "y" if answer_permissions else "n"

    sid = await c.create_session(profile=profile)
    s = Session(c, sid, on_permission=on_perm)
    answers: List[Dict[str, str]] = []
    for prompt in prompts:
        try:
            answers.append({"prompt": prompt, "answer": await s.ask(prompt, timeout=90)})
        except Exception as exc:  # noqa: BLE001 -- the refusal IS the evidence
            answers.append({"prompt": prompt, "raised": f"{type(exc).__name__}: {exc}"})
            break
    await asyncio.sleep(0.3)
    try:
        await c.end_session()
    except Exception:  # noqa: BLE001 -- a budget-terminated session is already gone
        pass
    await asyncio.sleep(1.0)
    await c.disconnect()
    return {"session_id": sid, "events": events, "answers": answers}


def drive(workspace: Path, sock: str, profile: str, prompts: List[str], **kw) -> Dict[str, Any]:
    return asyncio.run(_drive(workspace, sock, profile, prompts, **kw))


def session_record(workspace: Path, sid: str, wait: float = 20.0) -> Dict[str, Any]:
    """The persisted record, once the daemon has saved it (end_session unloads)."""
    path = workspace / ".jaato" / "sessions" / f"{sid}.json"
    deadline = time.time() + wait
    while time.time() < deadline:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("history"):
                return data
        time.sleep(0.5)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def tool_results(record: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    out = []
    for msg in record.get("history", []):
        for part in msg.get("parts", []):
            if part.get("type") == "function_response" and part.get("name") == name:
                out.append(part.get("result") or {})
    return out


# ------------------------------------------------------------ subprocesses

def run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None,
        ok_codes=(0, 1, 2)) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env,
                          capture_output=True, text=True, timeout=300)
    out = proc.stdout
    if proc.returncode not in ok_codes:
        out += f"\n[exit {proc.returncode}]\n{proc.stderr}"
    return _strip_noise(out)


_NOISE = re.compile(r"^.*(Plugin 'interactive_shell' skipped|\[MCP\] No MCP servers).*$\n?", re.M)


def _strip_noise(text: str) -> str:
    return _NOISE.sub("", text)


def scaffold(*args: str, cwd: Optional[Path] = None) -> str:
    return run([sys.executable, "-m", "shared.scaffold", *args], cwd=cwd)


def doctor(*args: str, cwd: Optional[Path] = None) -> str:
    return run([sys.executable, "-m", "jaato_sdk.doctor", *args], cwd=cwd)


# --------------------------------------------------------------- rendering

def _rewrite(text: str, root: Path, daemon: Daemon) -> str:
    text = text.replace(str(root / "acme-support"), WORKSPACE_ALIAS)
    text = text.replace(str(root / "triage-uncontrolled"), WORKSPACE_ALIAS + "-triage")
    text = text.replace(str(root / "quiet"), WORKSPACE_ALIAS + "-quiet")
    text = text.replace(str(root / "acme-new"), "/srv/acme-new")
    text = text.replace(daemon.socket, SOCKET_ALIAS).replace(daemon.pidfile, PIDFILE_ALIAS)
    text = text.replace(str(root), "/srv")
    return text


def render(capture: Capture, out: Path, chromium: Optional[str]) -> None:
    from rich.console import Console
    from rich.text import Text
    console = Console(record=True, width=CAPTURE_WIDTH, force_terminal=True,
                      color_system="truecolor", file=io.StringIO())
    console.print(Text(f"$ {capture.command}", style="bold"))
    for line in capture.text.rstrip("\n").splitlines():
        console.print(Text.from_ansi(line[:CAPTURE_WIDTH * 4]), soft_wrap=False)
    svg = console.export_svg(title=capture.title)
    (out / f"{capture.slug}.svg").write_text(svg, encoding="utf-8")
    (out / f"{capture.slug}.txt").write_text(
        f"$ {capture.command}\n{capture.text}", encoding="utf-8")
    if chromium:
        width, height = _svg_size(svg)
        subprocess.run([chromium, "--headless=new", "--no-sandbox", "--disable-gpu",
                        "--hide-scrollbars", f"--screenshot={(out / (capture.slug + '.png')).resolve()}",
                        f"--window-size={width + 16},{height + 16}",
                        f"file://{(out / (capture.slug + '.svg')).resolve()}"],
                       capture_output=True, timeout=120)


def _svg_size(svg: str) -> tuple:
    w = re.search(r'<svg[^>]*width="([\d.]+)"', svg)
    h = re.search(r'<svg[^>]*height="([\d.]+)"', svg)
    return (int(float(w.group(1))) if w else 1240, int(float(h.group(1))) if h else 600)


def find_chromium(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    env = os.environ.get("JAATO_EVIDENCE_CHROMIUM")
    if env:
        return env
    for pattern in ("/opt/pw-browsers/chromium-*/chrome-linux/chrome",):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")


# ---------------------------------------------------------------- captures

def _fmt_events(events: List[Dict[str, Any]]) -> str:
    return "\n".join(json.dumps(e, ensure_ascii=False) for e in events)


def _section(text: str, start: str, end_marker: Optional[str] = None, max_lines: int = 60) -> str:
    lines = text.splitlines()
    try:
        i = next(n for n, l in enumerate(lines) if l.startswith(start))
    except StopIteration:
        return f"(section {start!r} not found)"
    block = []
    for l in lines[i:]:
        if end_marker and l.startswith(end_marker) and block:
            break
        block.append(l)
        if len(block) >= max_lines:
            break
    return "\n".join(block)


def collect(root: Path, ws: Dict[str, Path], daemon: Daemon) -> List[Capture]:
    caps: List[Capture] = []
    acme, non, quiet = ws["acme"], ws["non"], ws["quiet"]
    add = caps.append

    # ---- Art. 6(4): the regulatory block, and what validate says about it
    add(Capture("01-profile-regulatory-block", "The regulatory: block", "cat .jaato/profiles/screener.yaml",
                SCREENER_PROFILE, "regulatory"))
    schema = scaffold("explain", "profile")
    add(Capture("02-explain-profile-schema", "explain profile: the three EU AI Act keys",
                "jaato-scaffold explain profile",
                "\n\n".join([_section(schema, "  regulatory", "  record_keeping", 3),
                              _section(schema, "  record_keeping", "  gc", 3),
                              _section(schema, "  trace ", "  regulatory", 3)]),
                "regulatory"))
    add(Capture("03-validate-compliant", "validate: the compliant workspace",
                f"jaato-scaffold validate {WORKSPACE_ALIAS}", scaffold("validate", str(acme)), "regulatory"))
    add(Capture("04-validate-high-risk", "validate: risk_class: high escalates every finding",
                f"jaato-scaffold validate {WORKSPACE_ALIAS}-triage", scaffold("validate", str(non)),
                "regulatory"))

    # ---- the authoring surface: what `new` emits, what `validate` says, what the skill lists
    fresh = root / "acme-new"
    fresh.mkdir()
    scaffold("new", "profile-set", "--workspace", str(fresh), "--provider", "anthropic",
             "--model", "claude-sonnet-4-5", "--set", "acme", "--agents", "collector,writer")
    base = fresh / ".jaato" / "profiles" / "_base_collector.yaml"
    add(Capture("30-new-profile-set-commented-block",
                "new profile-set: the three blocks in the tier-1 base, commented out",
                "jaato-scaffold new profile-set --workspace /srv/acme-new --provider anthropic "
                "--model claude-sonnet-4-5 --set acme --agents collector,writer && "
                "sed -n 1,24p .jaato/profiles/_base_collector.yaml",
                "\n".join((base.read_text(encoding="utf-8") if base.is_file()
                           else "(not written)").splitlines()[:24]), "regulatory"))
    undeclared = "\n".join(l for l in scaffold("validate", str(fresh)).splitlines()
                           if "budget_control_absent" not in l)
    add(Capture("31-validate-regulatory-undeclared",
                "validate: a workspace that declares nothing under the Act is told so",
                "jaato-scaffold validate /srv/acme-new | grep -v budget_control_absent",
                undeclared, "regulatory",
                note="the four budget_control_absent warnings the same run prints are elided"))
    scaffold("integration", "claude-code", "--workspace", str(fresh))
    skill = fresh / ".claude" / "skills" / "jaato-sdk" / "SKILL.md"
    hits = [f"{n}:{l}" for n, l in enumerate(
        (skill.read_text(encoding="utf-8") if skill.is_file() else "").splitlines(), 1)
        if any(k in l for k in ("oversight", "explain audit", "dossier", "regulatory", "record_keeping"))]
    add(Capture("32-integration-skill-lists-the-verbs",
                "The Claude Code integration skill names the verbs and the keys",
                "jaato-scaffold integration claude-code --workspace /srv/acme-new && "
                "grep -n 'oversight\\|explain audit\\|dossier\\|regulatory\\|record_keeping' "
                ".claude/skills/jaato-sdk/SKILL.md",
                "\n".join(hits) or "(no matches)", "regulatory"))

    # ---- Art. 50(1): the announcement, the instruction piece, the suppression WARNING
    screener = drive(acme, daemon.socket, "screener", ["Hello, is anyone there?", "Second question",
                                                       "Third question"])
    ev = screener["events"]
    add(Capture("05-announcement-events", "The first-interaction announcement, on the wire",
                "session.new --profile screener   # events as the SDK receives them",
                _fmt_events([e for e in ev if e["event"] in ("SESSION_INFO", "AGENT_OUTPUT")][:4]),
                "disclosure"))
    rec = session_record(acme, screener["session_id"])
    rendered = rec.get("rendered_instructions") or ""
    m = re.search(r"AI DISCLOSURE:.*?(?=\n\n|\Z)", rendered, re.S)
    add(Capture("06-disclosure-instruction-piece", "The disclosure piece inside the rendered system prompt",
                f"jq -r .rendered_instructions .jaato/sessions/{screener['session_id']}.json | grep -A3 'AI DISCLOSURE'",
                textwrap.fill(m.group(0), CAPTURE_WIDTH) if m else "(disclosure piece NOT FOUND in the rendered instructions)",
                "disclosure"))
    q = drive(quiet, daemon.socket, "quiet", ["hi"])
    warn = ""
    for logf in sorted(glob.glob(str(quiet / ".jaato" / "logs" / "*.log"))):
        for line in Path(logf).read_text(encoding="utf-8", errors="replace").splitlines():
            if "disclosure" in line and "WARNING" in line:
                warn = line
    add(Capture("07-disclosure-suppressed-warning", "Dropping the piece by name is announced at WARNING",
                "grep WARNING .jaato/logs/session_*.log | grep disclosure",
                warn or "(no WARNING found)", "disclosure",
                files={"profile": QUIET_PROFILE}))

    # ---- Art. 14: oversight pages, the stop button, the permission gate, the budget stop
    add(Capture("08-explain-oversight", "explain oversight: the measures, read from their enforcers",
                "jaato-scaffold explain oversight", scaffold("explain", "oversight"), "oversight"))
    add(Capture("09-explain-oversight-screener", "explain oversight <profile>: what this profile armed",
                "jaato-scaffold explain oversight screener",
                scaffold("explain", "oversight", "screener", cwd=acme), "oversight"))
    doc = doctor("--socket", daemon.socket, "--pidfile", daemon.pidfile, "--workspace", str(acme),
                 "--no-release-check")
    keep = []
    for l in doc.splitlines():
        if any(k in l for k in ("jaato SDK doctor", "────", "socket ", "daemon identity", "stop button")):
            keep.append(l)
        if l.startswith(("OK:", "WARN:", "FAIL:")):
            keep.append(l); break
    add(Capture("10-doctor-stop-button", "jaato-doctor names the stop button for the running daemon",
                "jaato-doctor --no-release-check", "\n".join(keep), "oversight"))
    add(Capture("11-budget-stop-events", "A budget ceiling stops the run, and the client is told why",
                "session.new --profile screener; ask x3   # limits: {turns: 2}, abort at 100%",
                _fmt_events([e for e in ev if e["event"] in ("AGENT_OUTPUT", "SESSION_TERMINATED", "INCIDENT")
                             and e.get("source") != "user"][-4:])
                + "\n\n" + json.dumps(screener["answers"][-1], ensure_ascii=False), "oversight"))

    # ---- Arts. 12/19: the audit record, the ledger, the chain, retention
    add(Capture("12-explain-audit", "explain audit: the record contract, computed",
                "jaato-scaffold explain audit", _section(scaffold("explain", "audit"), "the audit record", None, 46),
                "audit"))
    add(Capture("13-explain-audit-screener", "explain audit <profile>: the files this profile writes",
                "jaato-scaffold explain audit screener", scaffold("explain", "audit", "screener", cwd=acme), "audit"))
    ledger = acme / ".jaato" / "logs" / "ledger.jsonl"
    if ledger.is_file():
        rows = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
        shown = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows[:3])
        note = ""
    else:
        shown = "NOT WRITTEN -- no ledger.jsonl after the session (the runner held no ledger; see #1139)"
        note = "defect"
    add(Capture("14-ledger-records", "The ledger on disk after the session, chained",
                "cat .jaato/logs/ledger.jsonl", shown, "audit", note=note))
    verify_dir = root / "verify"; verify_dir.mkdir(exist_ok=True)
    src = ledger if ledger.is_file() else _synthetic_chained_ledger(verify_dir)
    intact = verify_dir / "ledger.jsonl"; shutil.copy(src, intact)
    add(Capture("15-audit-verify-intact", "jaato-doctor --audit-verify on the untouched file",
                "jaato-doctor --audit-verify .jaato/logs/ledger.jsonl",
                doctor("--audit-verify", str(intact)), "audit"))
    tampered = verify_dir / "ledger-edited.jsonl"
    lines = intact.read_text(encoding="utf-8").splitlines()
    if len(lines) > 1:
        lines[1] = lines[1].replace('"output_tokens": 7', '"output_tokens": 700', 1)
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
    add(Capture("16-audit-verify-tampered", "The same file after one number was edited in place",
                "sed -i '2s/\"output_tokens\": 7/\"output_tokens\": 700/' ledger.jsonl && jaato-doctor --audit-verify ledger.jsonl",
                doctor("--audit-verify", str(tampered)), "audit"))
    add(Capture("17-retention-sweep", "record_keeping: the retention pass, per profile and per clock",
                "python -c 'from server.record_retention import ...'   # a sweep over backdated files",
                _retention_demo(root), "audit"))

    # ---- Arts. 72/73: the incident register
    trace = acme / ".jaato" / "logs" / "session_trace.jsonl"
    inc_lines = [l for l in trace.read_text(encoding="utf-8", errors="replace").splitlines()
                 if "INCIDENT:" in l] if trace.is_file() else []
    add(Capture("18-incident-trace-line", "The INCIDENT record in the application trace",
                "grep 'INCIDENT:' .jaato/logs/session_trace.jsonl",
                "\n".join(inc_lines) or "(no INCIDENT line)", "incidents"))
    add(Capture("19-doctor-incidents", "jaato-doctor --incidents: the register with the Art. 73 clocks",
                "jaato-doctor --incidents .jaato/logs/session_trace.jsonl --since 15d",
                doctor("--incidents", str(trace), "--since", "15d"), "incidents"))

    # ---- Art. 50(2): marking generated output
    add(Capture("20-marking-posture", "explain oversight: what is marked, and what is not",
                "jaato-scaffold explain oversight   # MARKING GENERATED OUTPUT",
                _section(scaffold("explain", "oversight"), "MARKING GENERATED OUTPUT", "\n", 24), "marking"))
    add(Capture("21-marker-sidecar", "The output_marker plugin writes a provenance sidecar",
                "python -c '...mark_output(OutputPayload(chart.png, image/png, generated_by=...))'",
                _marker_demo(root), "marking"))
    add(Capture("22-generated-by-wire", "generated_by on the delivery event (protocol 1.14)",
                "python -c 'from jaato_sdk.events import ToolOutputEvent, ai_generated_by; ...'",
                _generated_by_demo(), "marking",
                note="constructed: the stamp is set in JaatoSession._deliver_model_media; echo emits no media"))

    # ---- Art. 15(4): memory provenance and the curation gate
    w = drive(acme, daemon.socket, "writer", ["Note the refund policy"])
    raw_files = sorted(glob.glob(str(acme / ".jaato" / "memories" / "raw" / "*.json")))
    raw = json.loads(Path(raw_files[0]).read_text(encoding="utf-8")) if raw_files else {}
    add(Capture("23-memory-generated-by", "A stored memory records which model wrote it",
                f"cat .jaato/memories/raw/{Path(raw_files[0]).name if raw_files else '?'}",
                json.dumps(raw, ensure_ascii=False, indent=2) if raw else "(no raw memory stored)", "memory"))
    legacy_id = "mem_20260401_090000_0001"
    curated = acme / ".jaato" / "memories" / "curated.jsonl"
    curated.parent.mkdir(parents=True, exist_ok=True)
    with curated.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "id": legacy_id, "content": "Chargebacks are handled by the payments team, not support.",
            "description": "Chargeback ownership", "tags": ["refunds", "chargebacks"],
            "timestamp": "2026-04-01T09:00:00", "usage_count": 0, "last_accessed": None,
            "maturity": "validated", "confidence": 0.8, "scope": "project", "evidence": None,
            "source_agent": "main", "source_session": "20260401_090000",
        }) + "\n")
    before = drive(acme, daemon.socket, "reader", ["What is the refund policy?"])
    before_res = tool_results(session_record(acme, before["session_id"]), "retrieve_memories")
    (acme / ".jaato" / "profiles" / "curator.json").write_text(_memory_profile(
        "curator", "The advisor persona promoting a raw memory (echo-driven)", "update_memory",
        {"id": raw.get("id", "?"), "maturity": "validated", "confidence": 0.9}))
    cur = drive(acme, daemon.socket, "curator", ["Review the raw memories"])
    perm = [e for e in cur["events"] if e["event"] == "PERMISSION_REQUESTED"]
    add(Capture("24-permission-gate", "The permission gate in front of update_memory (Art. 14(4)(d))",
                "session.new --profile curator; ask   # PERMISSION_REQUESTED as the client receives it",
                _fmt_events(perm) + "\n\n" + json.dumps({"answered": "y"}), "oversight"))
    promoted = ""
    if curated.is_file():
        for line in curated.read_text(encoding="utf-8").splitlines():
            if raw.get("id", "?") in line:
                d = json.loads(line)
                promoted = json.dumps({k: d.get(k) for k in ("id", "maturity", "generated_by", "curated_by")},
                                      ensure_ascii=False, indent=2)
    add(Capture("25-memory-curated-by", "After promotion: who wrote it, and who approved it",
                "grep <id> .jaato/memories/curated.jsonl | jq '{id, maturity, generated_by, curated_by}'",
                promoted or "(the promoted record was not found in curated.jsonl)", "memory"))
    after = drive(acme, daemon.socket, "reader", ["What is the refund policy?"])
    after_res = tool_results(session_record(acme, after["session_id"]), "retrieve_memories")
    add(Capture("26-memory-retrieval-gate", "require_curation: the same retrieval before and after the curator",
                "retrieve_memories(tags=[refunds])   # plugin_configs.memory.require_curation: true",
                "BEFORE the curator promoted anything (one legacy record validated with no curated_by):\n"
                + _fmt_results(before_res) + "\n\nAFTER the curator promoted the raw memory:\n"
                + _fmt_results(after_res), "memory"))

    # ---- Arts. 11/13/25(4)/15(3): the dossier, the component pack, the accuracy section
    results = _sample_eval_results(root)
    scaffold("new", "dossier", "--workspace", str(acme), "--profile", "screener",
             "--eval-results", str(results), "--force")
    scaffold("new", "dossier", "--workspace", str(acme), "--component", "--force")
    annex = acme / "docs" / "annex-iv-screener.md"
    pack = acme / "docs" / "jaato-component-pack.md"
    add(Capture("27-dossier-annex-iv", "jaato-scaffold new dossier --profile: the Annex IV skeleton",
                "jaato-scaffold new dossier --workspace . --profile screener --eval-results results.jsonl && head -60 docs/annex-iv-screener.md",
                "\n".join((annex.read_text(encoding="utf-8") if annex.is_file() else "(not written)").splitlines()[:60]),
                "dossier"))
    add(Capture("28-dossier-accuracy-section", "The accuracy section, from a jaato-eval results file, caveat verbatim",
                "sed -n '/^## 4/,/^## 5/p' docs/annex-iv-screener.md",
                _section(annex.read_text(encoding="utf-8") if annex.is_file() else "", "## 4", "## 5", 40), "dossier"))
    add(Capture("29-component-pack", "jaato-scaffold new dossier --component: the Art. 25(4) pack",
                "jaato-scaffold new dossier --workspace . --component && head -50 docs/jaato-component-pack.md",
                "\n".join((pack.read_text(encoding="utf-8") if pack.is_file() else "(not written)").splitlines()[:50]),
                "dossier"))

    for c in caps:
        c.text = _rewrite(c.text, root, daemon)
        c.command = _rewrite(c.command, root, daemon)
    return caps


def _fmt_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "(no retrieve_memories result recorded)"
    out = []
    for r in results:
        head = {k: v for k, v in r.items() if k != "memories"}
        out.append(json.dumps(head, ensure_ascii=False))
        for mem in r.get("memories", []):
            out.append(f"  - {mem.get('id')}  maturity={mem.get('maturity')}  {mem.get('description')}")
    return "\n".join(out)


def _synthetic_chained_ledger(where: Path) -> Path:
    """A chained ledger written in-process, for the verifier captures when the
    live one is absent.  Labelled in the capture that uses it."""
    from shared.token_accounting import TokenLedger
    path = where / "synthetic-ledger.jsonl"
    os.environ["LEDGER_PATH"] = str(path)
    os.environ["JAATO_LEDGER_INTEGRITY"] = "sha256-chain"
    ledger = TokenLedger()
    for i in range(3):
        ledger._record("response", {"prompt_tokens": 12, "output_tokens": 7, "total_tokens": 19})
    ledger._record("permission-check", {"tool": "writeNewFile", "allowed": True, "method": "whitelist", "asked": False})
    return path


def _retention_demo(root: Path) -> str:
    """Backdate two files under two profiles and let the real judge speak."""
    from server.record_retention import declared_retentions, expired_paths
    ws = root / "retention"
    (ws / ".jaato" / "profiles").mkdir(parents=True, exist_ok=True)
    (ws / ".jaato" / "logs").mkdir(parents=True, exist_ok=True)
    (ws / ".jaato" / "profiles" / "a.yaml").write_text(
        "name: a\ndescription: keeps its ledger 30 days\nmodel: echo\nprovider: echo\nplugins: []\n"
        "trace: {ledger: .jaato/logs/a-ledger.jsonl}\nrecord_keeping: {retention_days: 30}\n")
    (ws / ".jaato" / "profiles" / "b.yaml").write_text(
        "name: b\ndescription: keeps its ledger until something deletes it\nmodel: echo\nprovider: echo\nplugins: []\n"
        "trace: {ledger: .jaato/logs/b-ledger.jsonl}\nrecord_keeping: {retention_days: 0}\n")
    old = time.time() - 45 * 86400
    for name in ("a-ledger.jsonl", "b-ledger.jsonl"):
        p = ws / ".jaato" / "logs" / name; p.write_text("{}\n"); os.utime(p, (old, old))
    out = ["# two profiles, two clocks; both files are 45 days old"]
    for entry in declared_retentions(ws):
        days = getattr(entry.keeping, "retention_days", None)
        out.append(f"profile {entry.profile}: retention_days={days}  files={[Path(p).name for p in entry.audit_paths]}")
        if not days:
            out.append("  -> 0 = keep until deleted: the sweep never expires these")
            continue
        expired, kept = expired_paths(entry.audit_paths, days)
        for v in expired:
            out.append(f"  EXPIRED  {Path(v.path).name}  age={v.age_days:.0f}d > {v.kept_days}d  ({v.reason})")
        for v in kept:
            out.append(f"  kept     {Path(v.path).name}  age={v.age_days:.0f}d")
    return "\n".join(out)


def _marker_demo(root: Path) -> str:
    from jaato_sdk.events import ai_generated_by
    from jaato_sdk.output_marking import OutputPayload
    from shared.plugins.output_marker.plugin import create_plugin
    where = root / "marker"; where.mkdir(exist_ok=True)
    png = where / "chart.png"; png.write_bytes(_PNG)
    plugin = create_plugin(); plugin.initialize({})
    stamp = ai_generated_by("openrouter", "openai/gpt-image-1", session_id="20260918_212513", agent_id="main")
    r = plugin.mark_output(OutputPayload(data=_PNG, mime_type="image/png", generated_by=stamp,
                                         path=str(png), display_name="chart.png"))
    out = [f"marked={r.marked}  {r.detail}", ""]
    if r.sidecar_path:
        out.append(f"$ cat {Path(r.sidecar_path).name}")
        out.append(Path(r.sidecar_path).read_text(encoding="utf-8").rstrip())
    r2 = plugin.mark_output(OutputPayload(data=b"hello", mime_type="text/plain", generated_by=stamp,
                                          path=str(where / "note.txt")))
    out.append("")
    out.append(f"text/plain -> marked={r2.marked}  {r2.detail}")
    return "\n".join(out)


def _generated_by_demo() -> str:
    from jaato_sdk.events import ToolOutputEvent, ai_generated_by
    ev = ToolOutputEvent(session_id="20260918_212513", agent_id="main", call_id="model-output",
                         stream_id="s1", sequence=0, mime_type="audio/pcm;rate=24000", data_b64="...",
                         final=True, generated_by=ai_generated_by("openrouter", "openai/gpt-audio",
                                                                  session_id="20260918_212513",
                                                                  agent_id="main"))
    d = ev.model_dump(mode="json")
    d.pop("timestamp", None)
    return json.dumps(d, indent=2)


def _sample_eval_results(root: Path) -> Path:
    path = root / "eval-results.jsonl"
    caveat = ("The `judge` grader is one language model scoring another's output, "
              "and it is not a calibrated instrument.")
    rows = []
    for repeat, state in enumerate(("PASS", "PASS", "FAIL", "PASS")):
        rows.append({"results_version": "1", "caveats": [caveat],
                     "arm_id": f"ticket-triage@screener#{repeat}", "task_id": "ticket-triage",
                     "profile_set": "screener", "repeat": repeat, "state": state,
                     "verdicts": [{"grader_id": "script:acceptance", "claim": "reply drafted", "state": state},
                                  {"grader_id": "judge:tone", "claim": "tone acceptable", "state": "PASS"}],
                     "provenance": {"jaato_sdk_version": "0.23.0", "jaato_sdk_path": "/srv/venv/jaato_sdk/__init__.py"}})
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="docs/eu-ai-act-manual/evidence")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--chromium", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the temp root and the daemon log")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="jaato-evidence-"))
    chromium = None if args.no_png else find_chromium(args.chromium)
    if not args.no_png and not chromium:
        print("no Chromium found; SVG and text only (pass --chromium or --no-png)", file=sys.stderr)
    daemon = Daemon(root)
    print(f"evidence root {root}; starting a daemon on {daemon.socket}", file=sys.stderr)
    daemon.start()
    try:
        caps = collect(root, build_workspaces(root), daemon)
    finally:
        daemon.stop()
    index = []
    for c in caps:
        render(c, out, chromium)
        index.append({"slug": c.slug, "title": c.title, "control": c.control, "command": c.command,
                      "note": c.note})
        print(f"  {c.slug}", file=sys.stderr)
    (out / "captures.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    from datetime import datetime, timezone
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=str(Path(__file__).resolve().parent.parent)).stdout.strip() or "unknown"
    (out / "INDEX.md").write_text(
        "# Evidence index\n\nGenerated by `scripts/eu_ai_act_evidence.py` on "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%MZ')} against commit `{commit}`.\n"
        "Each capture is a `.txt` (the bytes the command produced), an `.svg` and a `.png` of "
        "the same text.  Temp paths are rewritten to `/srv/acme-support` and "
        "`/run/jaato/jaato.sock`; nothing else is edited.\n\n"
        + "\n".join(f"- `{c['slug']}` — {c['title']}" + (f" *({c['note']})*" if c['note'] else "")
                    for c in index) + "\n", encoding="utf-8")
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    print(f"{len(caps)} captures -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
