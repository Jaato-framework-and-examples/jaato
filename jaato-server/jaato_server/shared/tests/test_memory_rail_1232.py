"""The memory rail (#1232): which copy answers, who may change it, who approved it.

The web client's side rail lists a session's memories and lets the
workspace owner approve, dismiss, edit and remove them.  Three properties
make that honest, and each has been wrong somewhere in this tree before:

**The answer comes from the copy that HOLDS the store.**  ``memory`` is
``PLUGIN_TIER = "runner"``, so on a runner-served session -- the default --
the plugin the model writes through lives in the runner.  The daemon's
registry discovers a copy of its own that nothing on that path writes to,
and the ``memory`` command's ``MemoryListEvent`` used to be filled from
THAT copy after the command itself had run on the runner (#1179's defect
class).  ``JaatoServer.memory_op`` asks the runner (``session.memory``,
control lane), and a failed ask answers ``ok=False,
category="runner_unreachable"`` -- never an empty list, which reads as
"nothing remembered", and never the daemon's copy.  Only with NO runner at
all (embedded, standalone WS) does the daemon's plugin answer, because
there it IS the store.

**Only the workspace owner may change memories.**  ``may_curate`` is the
one predicate: the owner may, anyone may on an unowned workspace, and an
identity-less connection on an OWNED workspace may look and not change.
The identity is the transport's, never a field of the request, and a
refused mutation never reaches the runner.

**An approval records WHO approved it.**  Approve and dismiss move a
memory's maturity through ``_stamp_curation``, the one writer of
``curated_by`` (an AST guard in ``test_memory_provenance`` requires every
maturity writer to stamp).  A rail action has no model in context, so the
daemon hands the plugin the person the transport authenticated; demotion
clears the stamp, because a withdrawn approval must not keep reading as one.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from jaato_sdk.events import (
    ClientConfigRequest,
    MemoryDeleteRequest,
    MemoryDeleteResultEvent,
    MemoryGetRequest,
    MemoryListEvent,
    MemoryListRequest,
    MemoryUpdateRequest,
    MemoryUpdateResultEvent,
)
from jaato_server.server.core import JaatoServer
from jaato_server.server.memory_verbs import (
    answer_memory_request,
    curator_stamp,
    may_curate,
)
from jaato_server.server.runner.rpc import RunnerRPC
from jaato_server.server.session_manager import SessionManager
from jaato_server.shared.plugins.memory.indexer import MemoryIndexer
from jaato_server.shared.plugins.memory.models import (
    MATURITY_DISMISSED,
    MATURITY_RAW,
    MATURITY_VALIDATED,
    Memory,
)
from jaato_server.shared.plugins.memory.plugin import MemoryPlugin
from jaato_server.shared.plugins.memory.storage import MemoryStore
from jaato_server.shared.plugins.memory.verbs import serve_memory_op
from jaato_server.shared.tests.reversion import Reversion

_CORE = "jaato-server/jaato_server/server/core.py"
_VERBS = "jaato-server/jaato_server/server/memory_verbs.py"
_PLUGIN = "jaato-server/jaato_server/shared/plugins/memory/plugin.py"
_SM = "jaato-server/jaato_server/server/session_manager.py"


# ---------------------------------------------------------------- doubles


def _mem(mid: str, *, maturity: str = MATURITY_RAW,
         session: Optional[str] = None) -> Memory:
    return Memory(
        id=mid, content=f"content of {mid}", description=f"about {mid}",
        tags=["alpha", "beta"], timestamp="2026-09-01T00:00:00",
        maturity=maturity, source_session=session,
    )


def _plugin(tmp_path: Path, *, workspace=(), global_=()) -> MemoryPlugin:
    plugin = MemoryPlugin()
    plugin._storage = MemoryStore(str(tmp_path / "ws"))
    plugin._indexer = MemoryIndexer()
    plugin._global_storage = MemoryStore(str(tmp_path / "global"))
    plugin._global_indexer = MemoryIndexer()
    for m in workspace:
        plugin._storage.save(m)
    for m in global_:
        plugin._global_storage.save(m)
    return plugin


class _Registry:
    """A registry holding (and exposing) one memory plugin."""

    def __init__(self, plugin: Any, exposed: bool = True):
        self.plugin, self.exposed = plugin, exposed

    def is_exposed(self, name: str) -> bool:
        return name == "memory" and self.exposed

    def get_plugin(self, name: str) -> Any:
        return self.plugin if name == "memory" else None


class _Runtime:
    def __init__(self, registry: Any):
        self.registry = registry


class _RPCClient:
    """A runner whose memory plugin is the one the model writes through."""

    def __init__(self, registry: Any = None, raises: bool = False):
        self.registry, self.raises, self.calls = registry, raises, []

    def session_memory_threadsafe(self, op, args, *, timeout=None):
        self.calls.append((op, args))
        if self.raises:
            raise RuntimeError("runner is gone")
        return serve_memory_op(self.registry, op, args)


def _server(*, daemon_registry=None, rpc=None) -> JaatoServer:
    s = JaatoServer.__new__(JaatoServer)
    s._runtime = _Runtime(daemon_registry) if daemon_registry is not None else None
    s._runner_rpc = rpc
    return s


# --------------------------------------------- which copy answers the list


def test_the_runner_is_asked_not_the_daemons_own_copy(tmp_path):
    daemon = _plugin(tmp_path / "daemon", workspace=[_mem("stale")])
    runner = _plugin(tmp_path / "runner", workspace=[_mem("real")])
    rpc = _RPCClient(_Registry(runner))
    answer = _server(daemon_registry=_Registry(daemon), rpc=rpc).memory_op(
        "list", {"session_id": "s1"})
    assert rpc.calls, "the runner holding the store was never asked"
    assert answer["ok"] is True and answer["source"] == "runner"
    assert [r["id"] for r in answer["memories"]] == ["real"]


def test_a_failed_ask_is_a_failure_not_an_empty_store(tmp_path):
    daemon = _plugin(tmp_path, workspace=[_mem("stale")])
    answer = _server(daemon_registry=_Registry(daemon),
                     rpc=_RPCClient(raises=True)).memory_op("list", {})
    assert answer["ok"] is False
    assert answer["category"] == "runner_unreachable"
    assert "memories" not in answer, (
        "a failed ask must not carry a list -- neither [] nor the daemon's copy")


def test_with_no_runner_the_daemons_plugin_IS_the_store(tmp_path):
    daemon = _plugin(tmp_path, workspace=[_mem("here")])
    answer = _server(daemon_registry=_Registry(daemon)).memory_op("list", {})
    assert answer["ok"] is True and answer["source"] == "daemon"
    assert [r["id"] for r in answer["memories"]] == ["here"]


def test_the_command_push_is_withheld_when_the_store_could_not_be_read(tmp_path):
    """The unsolicited push has no request_id; an empty list there would
    tell the completion cache and the rail "nothing remembered"."""
    s = _server(daemon_registry=_Registry(_plugin(tmp_path)),
                rpc=_RPCClient(raises=True))
    assert s.memory_list_event() is None


def test_the_command_push_carries_the_runners_rows(tmp_path):
    runner = _plugin(tmp_path, workspace=[_mem("real")])
    event = _server(rpc=_RPCClient(_Registry(runner))).memory_list_event()
    assert isinstance(event, MemoryListEvent)
    assert [r["id"] for r in event.memories] == ["real"]


def test_an_unexposed_plugin_is_no_plugin_not_an_empty_store(tmp_path):
    """A discovered-but-unexposed plugin was never initialized: its storage
    is None, and reading it would answer as an empty store."""
    answer = serve_memory_op(_Registry(_plugin(tmp_path), exposed=False),
                             "list", {})
    assert answer == {"ok": False, "category": "no_plugin",
                      "error": answer["error"]}


# ------------------------------------------------------------- the rows


def test_rows_cover_both_tiers_and_the_raw_queue(tmp_path):
    plugin = _plugin(
        tmp_path,
        workspace=[_mem("w-raw"), _mem("w-cur", maturity=MATURITY_VALIDATED)],
        global_=[_mem("g-raw")],
    )
    rows = {r["id"]: r for r in plugin.memory_rows("s1")}
    assert set(rows) == {"w-raw", "w-cur", "g-raw"}
    assert rows["w-raw"]["tier"] == "workspace"
    assert rows["g-raw"]["tier"] == "global"
    assert rows["w-raw"]["maturity"] == MATURITY_RAW
    assert "content" not in rows["w-raw"], "a list row carries no content"
    for key in ("timestamp", "last_accessed", "usage_count", "generated_by",
                "curated_by", "source_agent", "source_session", "scope"):
        assert key in rows["w-raw"], key


def test_this_session_flags_distinguish_written_from_retrieved(tmp_path):
    plugin = _plugin(tmp_path, workspace=[
        _mem("mine", session="s1"), _mem("theirs", session="s0"),
        _mem("recalled", session="s0", maturity=MATURITY_VALIDATED)])
    plugin._retrieved_by_session["s1"] = {"recalled"}
    rows = {r["id"]: r for r in plugin.memory_rows("s1")}
    assert rows["mine"]["written_this_session"] is True
    assert rows["theirs"]["written_this_session"] is False
    assert rows["recalled"]["retrieved_this_session"] is True
    assert rows["mine"]["retrieved_this_session"] is False


def test_get_carries_the_content(tmp_path):
    plugin = _plugin(tmp_path, global_=[_mem("g1")])
    answer = serve_memory_op(_Registry(plugin), "get", {"memory_id": "g1"})
    assert answer["ok"] is True
    assert answer["memory"]["content"] == "content of g1"
    assert answer["memory"]["tier"] == "global"
    missing = serve_memory_op(_Registry(plugin), "get", {"memory_id": "nope"})
    assert (missing["ok"], missing["category"]) == (False, "not_found")


# ------------------------------------------------- approve / dismiss / edit


_HUMAN = {"kind": "human", "via": "memory.update", "user": "app:alice"}


def test_approving_records_the_person_who_approved(tmp_path):
    plugin = _plugin(tmp_path, workspace=[_mem("m1")])
    answer = plugin.edit_memory_structured(
        "m1", maturity=MATURITY_VALIDATED, curator=_HUMAN)
    assert answer["ok"] is True
    stored = plugin._storage.get_by_id("m1")
    assert stored.maturity == MATURITY_VALIDATED
    assert stored.curated_by is not None, "an approval nobody recorded"
    assert stored.curated_by["user"] == "app:alice"
    assert stored.curated_by["kind"] == "human"
    assert "at" in stored.curated_by


def test_a_withdrawn_approval_is_withdrawn(tmp_path):
    plugin = _plugin(tmp_path, workspace=[_mem("m1")])
    plugin.edit_memory_structured("m1", maturity=MATURITY_VALIDATED, curator=_HUMAN)
    assert plugin._storage.get_by_id("m1").curated_by is not None
    plugin.edit_memory_structured("m1", maturity=MATURITY_RAW, curator=_HUMAN)
    assert plugin._storage.get_by_id("m1").curated_by is None


def test_dismissing_a_raw_memory_removes_it_from_the_next_list(tmp_path):
    plugin = _plugin(tmp_path, workspace=[_mem("m1"), _mem("m2")])
    assert plugin.edit_memory_structured(
        "m1", maturity=MATURITY_DISMISSED, curator=_HUMAN)["ok"] is True
    assert [r["id"] for r in plugin.memory_rows()] == ["m2"]


def test_an_invalid_edit_changes_nothing(tmp_path):
    plugin = _plugin(tmp_path, workspace=[_mem("m1")])
    answer = plugin.edit_memory_structured("m1", description="   ")
    assert (answer["ok"], answer["category"]) == (False, "invalid")
    assert plugin._storage.get_by_id("m1").description == "about m1"


def test_a_structured_edit_rewrites_the_fields(tmp_path):
    plugin = _plugin(tmp_path, workspace=[_mem("m1")])
    answer = serve_memory_op(_Registry(plugin), "update", {
        "memory_id": "m1",
        "fields": {"description": "new", "content": "body", "tags": ["gamma"]},
    })
    assert answer["ok"] is True
    stored = plugin._storage.get_by_id("m1")
    assert (stored.description, stored.content, stored.tags) == (
        "new", "body", ["gamma"])
    assert stored.maturity == MATURITY_RAW and stored.curated_by is None


def test_remove_goes_through_the_plugins_delete_path(tmp_path):
    plugin = _plugin(tmp_path, global_=[_mem("g1", maturity=MATURITY_VALIDATED)])
    answer = serve_memory_op(_Registry(plugin), "delete", {"memory_id": "g1"})
    assert answer == {"ok": True, "memory_id": "g1"}
    assert plugin._global_storage.get_by_id("g1") is None
    again = serve_memory_op(_Registry(plugin), "delete", {"memory_id": "g1"})
    assert (again["ok"], again["category"]) == (False, "not_found")


# ------------------------------------------------------------ the owner gate


@pytest.mark.parametrize("owner,user,allowed", [
    (None, None, True),              # unowned: anyone who can see it
    (None, "app:bob", True),
    ("app:alice", "app:alice", True),
    ("app:alice", "app:bob", False),
    ("app:alice", None, False),      # identity-less on an owned workspace
])
def test_only_the_owner_may_change_memories(owner, user, allowed):
    assert may_curate(owner, user) is allowed


class _GateServer:
    def __init__(self):
        self.calls: List[Any] = []

    def memory_op(self, op, args=None, *, timeout=5.0):
        self.calls.append((op, args))
        if op == "list":
            return {"ok": True, "memories": [], "source": "runner"}
        return {"ok": True, "memory_id": (args or {}).get("memory_id"),
                "memory": None, "source": "runner"}


@pytest.mark.parametrize("request_", [
    MemoryUpdateRequest(request_id="r1", memory_id="m1", maturity="validated"),
    MemoryDeleteRequest(request_id="r1", memory_id="m1"),
])
def test_a_refused_mutation_never_reaches_the_runner(request_):
    server = _GateServer()
    answer = answer_memory_request(server, request_, session_id="s1",
                                   user_id="app:bob", owner="app:alice")
    assert server.calls == []
    assert answer.ok is False and answer.category == "not_owner"
    assert answer.request_id == "r1"


def test_a_non_owner_may_still_look():
    server = _GateServer()
    answer = answer_memory_request(server, MemoryListRequest(request_id="r2"),
                                   session_id="s1", user_id="app:bob",
                                   owner="app:alice")
    assert answer.ok is True and answer.may_curate is False
    assert server.calls == [("list", {"session_id": "s1"})]


def test_the_approval_stamp_is_the_transports_identity():
    server = _GateServer()
    answer_memory_request(
        server,
        MemoryUpdateRequest(request_id="r3", memory_id="m1", maturity="validated"),
        session_id="s1", user_id="app:alice", owner="app:alice")
    op, args = server.calls[0]
    assert op == "update"
    assert args["curator"] == curator_stamp("app:alice")
    assert args["curator"]["user"] == "app:alice"
    assert args["fields"] == {"maturity": "validated"}


def test_no_session_is_answered_under_the_callers_request_id():
    answer = answer_memory_request(None, MemoryGetRequest(request_id="r4",
                                                          memory_id="m1"),
                                   session_id="", user_id=None, owner=None)
    assert (answer.ok, answer.category, answer.request_id) == (
        False, "no_session", "r4")


# ------------------------------------------------ the daemon's one arm


class _Session:
    def __init__(self, server, workspace_path="/ws/a"):
        self.server, self.workspace_path = server, workspace_path


class _Resolver:
    def __init__(self, owner):
        self.owner = owner

    def owner_for(self, path):
        return self.owner


def _manager(session, owner) -> "tuple[SessionManager, list]":
    sm = SessionManager.__new__(SessionManager)
    sent: List[Any] = []
    sm.get_session = lambda sid: session if sid == "s1" else None  # type: ignore[method-assign]
    sm._emit_to_client = lambda cid, ev: sent.append((cid, ev))  # type: ignore[method-assign]
    sm._app_secret_resolver = _Resolver(owner)
    return sm, sent


def test_handle_request_answers_the_memory_verbs_with_the_workspace_owner():
    server = _GateServer()
    sm, sent = _manager(_Session(server), owner="app:alice")
    sm.handle_request("c1", "s1",
                      MemoryDeleteRequest(request_id="r5", memory_id="m1"),
                      user_id="app:bob")
    assert server.calls == [], "a non-owner's delete reached the runner"
    [(cid, event)] = sent
    assert cid == "c1" and isinstance(event, MemoryDeleteResultEvent)
    assert (event.ok, event.category) == (False, "not_owner")


def test_handle_request_lets_the_owner_through():
    server = _GateServer()
    sm, sent = _manager(_Session(server), owner="app:alice")
    sm.handle_request("c1", "s1",
                      MemoryUpdateRequest(request_id="r6", memory_id="m1",
                                          description="d"),
                      user_id="app:alice")
    assert server.calls and server.calls[0][0] == "update"
    [(_cid, event)] = sent
    assert isinstance(event, MemoryUpdateResultEvent) and event.ok is True


def test_client_config_still_answers_before_the_memory_arm():
    """The memory arm is placed after ClientConfigRequest, before the
    session lookup: pin that it did not swallow the config path."""
    sm, _sent = _manager(None, owner=None)
    seen: List[Any] = []
    sm._apply_client_config = lambda cid, ev, peer=None: seen.append(ev)  # type: ignore[method-assign]
    ev = ClientConfigRequest()
    sm.handle_request("c1", "s1", ev)
    assert seen == [ev]


# ------------------------------------------------------- the runner side


class _Host:
    def __init__(self, registry):
        self.session = type("S", (), {"_runtime": _Runtime(registry)})()


def test_the_runner_handler_serves_its_own_registry(tmp_path):
    rpc = RunnerRPC.__new__(RunnerRPC)
    session = _Host(_Registry(_plugin(tmp_path, workspace=[_mem("r1")]))).session
    rpc._require_ready_session = lambda: (True, None, session)  # type: ignore[method-assign]
    ok, answer = rpc._handle_session_memory({"op": "list", "args": {}})
    assert ok is True and answer["ok"] is True
    assert [r["id"] for r in answer["memories"]] == ["r1"]


def test_session_memory_is_a_named_runner_method():
    from jaato_server.server.runner.rpc import NAMED_METHOD_HANDLERS
    assert NAMED_METHOD_HANDLERS.get("session.memory") == "_handle_session_memory"


# ---------------------------------------------------------------- guard


def test_the_push_after_the_memory_command_asks_the_holding_copy():
    """``execute_command``'s memory push must go through memory_list_event
    (the runner) and never through the daemon plugin's own metadata."""
    tree = ast.parse(Path(__file__).resolve().parents[2]
                     .joinpath("server", "core.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "execute_command")
    calls = {n.func.attr for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "memory_list_event" in calls
    assert "get_memory_metadata" not in calls


REVERSIONS = [
    Reversion(
        target=_CORE,
        find="""        rpc = getattr(self, "_runner_rpc", None)
        if rpc is None:
            registry = self._runtime.registry""",
        replace="""        rpc = None
        if rpc is None:
            registry = self._runtime.registry""",
        test="test_the_runner_is_asked_not_the_daemons_own_copy",
        because=(
            "the memory list answered from the daemon's own copy of a "
            "runner-tier plugin -- a store no runner-served session writes to"
        ),
    ),
    Reversion(
        target=_VERBS,
        find="    return owner is None or (user_id is not None and owner == user_id)",
        replace="    return True",
        test="test_only_the_owner_may_change_memories",
        because="anyone who can see a workspace could approve or delete its memories",
    ),
    Reversion(
        target=_VERBS,
        find="    elif isinstance(event, MUTATING_REQUEST_TYPES) and not allowed:",
        replace="    elif False:",
        test="test_a_refused_mutation_never_reaches_the_runner",
        because="the owner gate computed and never applied",
    ),
    Reversion(
        target=_PLUGIN,
        find="""            memory.maturity = draft["maturity"]
            self._stamp_curation(memory, draft["maturity"], curator=curator)""",
        replace="""            memory.maturity = draft["maturity"]""",
        test="test_approving_records_the_person_who_approved",
        because=(
            "a rail approval that records no approver -- withheld by "
            "require_curation with no error anywhere"
        ),
    ),
    Reversion(
        target=_PLUGIN,
        find="""        if curator:
            stamp.update(curator)
            memory.curated_by = stamp
            return
""",
        replace="",
        test="test_approving_records_the_person_who_approved",
        because="the person who clicked Approve replaced by whatever model was in context",
    ),
    Reversion(
        target=_SM,
        find="""        if isinstance(event, MEMORY_REQUEST_TYPES):
            self._handle_memory_request(client_id, session_id, event, user_id=user_id)
            return
""",
        replace="",
        test="test_handle_request_answers_the_memory_verbs_with_the_workspace_owner",
        because="the daemon answering the memory verbs 'Unknown request type'",
    ),
]
