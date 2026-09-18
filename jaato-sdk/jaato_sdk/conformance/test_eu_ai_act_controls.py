"""The EU AI Act controls, asserted against a running daemon.

Every one of these mechanisms has a unit guard with a reversion the
meta-guard proves live.  None of those could see jaato #1139: the ledger
class was tested thoroughly, the runner's bootstrap was pinned, and both
passed while the default path wrote nothing -- because no test asked the
question end to end.  The evidence harness asked it and the capture came
back ``NOT WRITTEN``.  This module asks it in CI, on every PR, at zero token
cost: the ``echo`` provider is deterministic, credential-free and
network-free, and reports the spend its profile declares.

Each test is a statement of what the manual (``docs/eu-ai-act-manual.md``)
shows, so a failure reads as "the control no longer does what the manual
says" rather than "a test broke".  The profiles are the manual's own: the
``screener`` that declares everything the controls read, the ``quiet`` one
that declares nothing, and the three echo-driven memory profiles.

Written against the SDK alone (``jaato_sdk``), like the rest of this
directory: the daemon is the thing under test, not a module to import.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from jaato_sdk import IPCClient
from jaato_sdk.client.convenience import Session, SessionEnded
from jaato_sdk.conformance.daemon import ConformanceDaemon
from jaato_sdk.events import ClientType, EventType, HistoryEvent

pytestmark = pytest.mark.conformance

SCREENER = """\
name: screener
description: Pre-screens inbound support tickets and drafts a reply for a human agent to approve.
model: echo
provider: echo
plugins: []
regulatory:
  intended_purpose: Pre-screen inbound support tickets and draft a reply for a human agent to approve.
  risk_class: limited
  interacts_with_persons: true
  provider: {name: Acme Support GmbH, contact: compliance@acme.example}
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

QUIET = """\
name: quiet
description: Declares nothing under the Act.
model: echo
provider: echo
plugins: []
plugin_configs:
  echo:
    usage: {prompt_tokens: 12, output_tokens: 7, total_tokens: 19}
"""


def _memory_profile(name: str, tool: str, args: Dict[str, Any],
                    extra: Dict[str, Any] = None) -> str:
    cfg: Dict[str, Any] = {"echo": {
        "tool_call": {"name": tool, "args": args},
        "usage": {"prompt_tokens": 40, "output_tokens": 20, "total_tokens": 60}}}
    cfg.update(extra or {})
    return json.dumps({"name": name, "description": f"{name} (echo-driven)",
                       "model": "echo", "provider": "echo",
                       "plugins": ["memory(preload)"], "plugin_configs": cfg})


@pytest.fixture(scope="module")
def act_daemon():
    root = Path(tempfile.mkdtemp(prefix="jaato-act-conf-"))
    profiles = root / ".jaato" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "screener.yaml").write_text(SCREENER, encoding="utf-8")
    (profiles / "quiet.yaml").write_text(QUIET, encoding="utf-8")
    (profiles / "writer.json").write_text(_memory_profile("writer", "store_memory", {
        "content": "Customers on the Pro plan get a 14-day refund window.",
        "description": "Refund window", "tags": ["refunds"], "scope": "project"}))
    (profiles / "reader.json").write_text(_memory_profile(
        "reader", "retrieve_memories", {"tags": ["refunds"]},
        {"memory": {"require_curation": True}}))
    d = ConformanceDaemon(root).start()
    try:
        yield d
    finally:
        d.stop()


# ------------------------------------------------------------- driving

async def _run(daemon, profile: str, prompts: List[str], answer: str = "y",
               await_tool: Optional[str] = None) -> Dict[str, Any]:
    """One session: subscribe BEFORE creation, ask, end (which persists it).

    ``await_tool`` names a tool whose RESULTS are fetched over the wire
    (``request_history`` -> ``HistoryEvent``) from the live session before
    it is ended, and returned under ``results``.  Not from the persisted
    record: the daemon also persists on ``ToolCallStartEvent`` (crash
    recovery), that snapshot carries the call without its result, and on a
    CI runner the after-turn save was observed never to land at all in the
    30 s the session stayed loaded -- so a disk reader was handed the
    mid-turn snapshot however long it waited.  The wire read asks the
    runner for its history as it stands, which is what the record is
    supposed to be a copy of.
    """
    c = IPCClient(socket_path=daemon.socket_path, client_type=ClientType.API,
                  workspace_path=str(daemon.workspace), auto_start=False)
    assert await c.connect(timeout=60), "could not connect to the test daemon"
    events: List[Dict[str, Any]] = []
    c.subscribe(EventType.SESSION_INFO, lambda ev: events.append(
        {"event": "SESSION_INFO",
         "disclosure_announcement": getattr(ev, "disclosure_announcement", None)}))
    c.subscribe(EventType.AGENT_OUTPUT, lambda ev: events.append(
        {"event": "AGENT_OUTPUT", "source": ev.source, "text": ev.text}))
    c.subscribe(EventType.SESSION_TERMINATED, lambda ev: events.append(
        {"event": "SESSION_TERMINATED", "reason": ev.reason}))
    perms: List[str] = []

    def on_perm(ev):
        perms.append(ev.tool_name)
        return answer

    try:
        sid = await c.create_session(profile=profile)
        s = Session(c, sid, on_permission=on_perm)
        outcome: Dict[str, Any] = {"answers": [], "ended": None}
        for prompt in prompts:
            try:
                outcome["answers"].append(await s.ask(prompt, timeout=90))
            except SessionEnded as exc:
                outcome["ended"] = exc.reason
                break
        if await_tool is not None:
            outcome["results"] = await _tool_results_over_the_wire(c, await_tool)
        await asyncio.sleep(0.3)
        try:
            await c.end_session()
        except Exception:  # noqa: BLE001 -- a budget-terminated session is gone already
            pass
        await asyncio.sleep(1.0)
    finally:
        await c.disconnect()
    return {"session_id": sid, "events": events, "permissions": perms, **outcome}


async def _tool_results_over_the_wire(c: IPCClient, tool: str,
                                      wait: float = 30.0) -> List[Dict[str, Any]]:
    """The ``response`` of every ``function_response`` part named ``tool`` in
    the live session's history, asked for over the wire.  Re-asks until one
    is there or ``wait`` runs out, and on the deadline names the parts the
    last answer held, so a wrong tool name is not reported as silence."""
    deadline = time.time() + wait
    last: Optional[List[Dict[str, Any]]] = None
    while True:
        got = asyncio.Event()
        answer: List[Any] = []

        def on_any(ev):
            if isinstance(ev, HistoryEvent) or type(ev).__name__ == "ErrorEvent":
                answer.append(ev)
                got.set()

        unsub = c.subscribe_all(on_any)
        try:
            await c.request_history()
            await asyncio.wait_for(got.wait(), timeout=15)
        finally:
            unsub()
        ev = answer[0] if answer else None
        if isinstance(ev, HistoryEvent):
            last = ev.history
            found = [p.get("response") or {} for m in last for p in m.get("parts", [])
                     if p.get("type") == "function_response" and p.get("name") == tool]
            if found:
                return found
        if time.time() >= deadline:
            break
        await asyncio.sleep(1.0)
    seen = ("no HistoryEvent answered" if last is None else "last answer held parts: "
            + repr([(p.get("type"), p.get("name")) for m in last for p in m.get("parts", [])]))
    raise AssertionError(f"no {tool} result reached the session's history; {seen}")


def run(daemon, profile, prompts, **kw):
    return asyncio.run(_run(daemon, profile, prompts, **kw))


def record(daemon, sid: str, wait: float = 30.0, need_history: bool = True,
           until: Optional[Callable[[Dict[str, Any]], bool]] = None) -> Dict[str, Any]:
    """The persisted session record, once ``until`` holds on it.

    The daemon writes the record at creation (rendered instructions, no
    history), on every ``ToolCallStartEvent`` (the history up to the CALL,
    no result yet) and after each turn (the whole history).  So "has a
    history" is not "has the tool result": a reader of tool results passes
    an ``until`` that looks for the result, a reader of the rendered prompt
    takes the first write.  On the deadline the error names what the last
    snapshot held, so a wrong predicate is not reported as a missing file.
    """
    path = daemon.workspace / ".jaato" / "sessions" / f"{sid}.json"
    deadline = time.time() + wait
    last: Optional[Dict[str, Any]] = None
    while time.time() < deadline:
        if path.is_file():
            try:
                last = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:      # a write in progress
                last = None
            if last is not None and (last.get("history") or not need_history) \
                    and (until is None or until(last)):
                return last
        time.sleep(0.5)
    seen = "no record on disk" if last is None else "last snapshot held parts: " + repr(
        [(p.get("type"), p.get("name")) for m in last.get("history", [])
         for p in m.get("parts", [])])
    raise AssertionError(f"session record {path} did not reach the awaited state"
                         + (" (a history was required)" if need_history else "")
                         + f"; {seen}")


def tool_results(rec: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    return [part.get("result") or {}
            for msg in rec.get("history", []) for part in msg.get("parts", [])
            if part.get("type") == "function_response" and part.get("name") == name]


# ---------------------------------------------------------- Art. 50(1)

def test_a_person_is_told_before_the_first_turn(act_daemon):
    """A profile declaring interacts_with_persons announces once, first."""
    out = run(act_daemon, "screener", ["Hello, is anyone there?"])
    info = [e for e in out["events"] if e["event"] == "SESSION_INFO"]
    assert info and info[0]["disclosure_announcement"], (
        "SessionInfo carries no announcement for a profile that declared it talks to people")
    assert "Acme Support GmbH" in info[0]["disclosure_announcement"]
    outputs = [e for e in out["events"] if e["event"] == "AGENT_OUTPUT"]
    assert outputs and outputs[0]["source"] == "system", (
        "the announcement must be the FIRST output, before any user or model text")
    assert "AI system" in outputs[0]["text"]


def test_a_profile_that_declares_nothing_announces_nothing(act_daemon):
    """Absent is undeclared: the framework never announces on a profile's behalf."""
    out = run(act_daemon, "quiet", ["hi"])
    assert all(not e["disclosure_announcement"]
               for e in out["events"] if e["event"] == "SESSION_INFO")
    assert not [e for e in out["events"]
                if e["event"] == "AGENT_OUTPUT" and e["source"] == "system"
                and "AI system" in e["text"]]


def test_the_disclosure_piece_is_in_the_rendered_prompt(act_daemon):
    out = run(act_daemon, "quiet", ["hi"])
    rendered = record(act_daemon, out["session_id"],
                      need_history=False).get("rendered_instructions") or ""
    assert "AI DISCLOSURE:" in rendered, (
        "the disclosure instruction piece is not in the system prompt the session persisted")


# --------------------------------------------------- Arts. 12, 19, 73(6)

def test_the_ledger_explain_audit_names_is_written_and_chained(act_daemon):
    """jaato #1139: the file `explain audit <profile>` names must exist after
    the session, with one response record per metered turn, chained."""
    from jaato_sdk.audit_chain import verify
    ledger = act_daemon.workspace / ".jaato" / "logs" / "ledger.jsonl"
    before = len(ledger.read_text(encoding="utf-8").splitlines()) if ledger.is_file() else 0
    run(act_daemon, "screener", ["First", "Second"])
    assert ledger.is_file(), "trace.ledger named a file the session never wrote (#1139)"
    lines = ledger.read_text(encoding="utf-8").splitlines()
    new = [json.loads(l) for l in lines[before:] if l.strip()]
    responses = [r for r in new if r.get("stage") == "response"]
    assert len(responses) == 2, f"two metered turns, {len(responses)} response records: {new}"
    assert all(r.get("prev_digest") and r.get("digest") for r in responses), (
        "record_keeping.integrity: sha256-chain declared and the records carry no chain")
    intact, breaks = verify(lines)
    assert intact, f"the untouched ledger does not verify: {breaks}"
    edited = list(lines)
    edited[-1] = edited[-1].replace('"output_tokens": 7', '"output_tokens": 700', 1)
    intact, breaks = verify(edited)
    assert not intact and breaks, "an in-place edit of the last record went undetected"


# --------------------------------------------------------- Arts. 72, 73

def test_a_budget_stop_is_recorded_in_the_incident_register(act_daemon):
    from jaato_sdk.incidents import parse_incident_trace
    out = run(act_daemon, "screener", ["one", "two", "three"])
    assert out["ended"] == "budget_exhausted", (
        f"limits: {{turns: 2}} with an abort rung should end the third turn; got {out!r}")
    assert any(e["event"] == "SESSION_TERMINATED" and e["reason"] == "budget_exhausted"
               for e in out["events"])
    trace = act_daemon.workspace / ".jaato" / "logs" / "session_trace.jsonl"
    assert trace.is_file(), "trace.session_log named a file the session never wrote"
    incidents = [parse_incident_trace(line)
                 for line in trace.read_text(encoding="utf-8", errors="replace").splitlines()
                 if "INCIDENT: " in line]
    incidents = [i for i in incidents if i is not None]
    mine = [i for i in incidents if i.session_id == out["session_id"]]
    assert mine, f"no INCIDENT line names session {out['session_id']}: {incidents}"
    assert mine[0].kind == "budget_exhausted"


# ------------------------------------------------------------ Art. 15(4)

def test_the_learning_loop_records_provenance_and_the_gate_holds(act_daemon):
    """A stored memory names its author; retrieval under require_curation
    withholds it until a curator promotes it, and the promotion names the
    curator.  The permission gate stands in front of the promotion."""
    ws = act_daemon.workspace
    run(act_daemon, "writer", ["note the refund policy"])
    raw_files = sorted((ws / ".jaato" / "memories" / "raw").glob("*.json"))
    assert raw_files, "store_memory stored nothing"
    raw = json.loads(raw_files[-1].read_text(encoding="utf-8"))
    assert raw.get("generated_by", {}).get("provider") == "echo", raw.get("generated_by")
    assert raw.get("curated_by") is None

    # A validated record predating the stamp: what require_curation must withhold.
    curated = ws / ".jaato" / "memories" / "curated.jsonl"
    curated.parent.mkdir(parents=True, exist_ok=True)
    legacy = {**raw, "id": "mem_legacy_0001", "maturity": "validated",
              "description": "Chargeback ownership", "tags": ["refunds"]}
    legacy.pop("generated_by", None); legacy.pop("curated_by", None)
    with curated.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(legacy) + "\n")

    before = run(act_daemon, "reader", ["what is the refund policy?"],
                 await_tool="retrieve_memories")
    res = before["results"]
    assert res and res[0].get("withheld_uncurated", 0) >= 1, res
    assert not [m for m in res[0].get("memories", []) if m["id"] == "mem_legacy_0001"]

    (ws / ".jaato" / "profiles" / "curator.json").write_text(_memory_profile(
        "curator", "update_memory", {"id": raw["id"], "maturity": "validated"}))
    cur = run(act_daemon, "curator", ["review the raw memories"])
    assert "update_memory" in cur["permissions"], (
        "the permission gate (Art. 14(4)(d)) did not stand in front of the promotion")
    promoted = [json.loads(l) for l in curated.read_text(encoding="utf-8").splitlines()
                if raw["id"] in l]
    assert promoted and promoted[-1].get("curated_by", {}).get("session_id") == cur["session_id"], (
        "the promotion did not stamp the CURATOR's session as the approver")

    after = run(act_daemon, "reader", ["what is the refund policy?"],
                await_tool="retrieve_memories")
    res = after["results"]
    assert res and any(m["id"] == raw["id"] for m in res[0].get("memories", [])), (
        "the curated memory is still withheld after the curator promoted it")
