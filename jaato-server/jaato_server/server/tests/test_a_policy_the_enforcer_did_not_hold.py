"""The permission status a client reads is the one the enforcer holds.

Reported from the web client as *"the permission was to become a button,
and now I do not even see it"* -- the status bar's ``permissions ask``
segment, absent entirely.  It is gated on having been TOLD the policy
(``{permStatus && ...}``), and the daemon had never told that client.
Driving a real daemon over IPC found three facts, of which the report is
only the first::

    A. session.new                          -> PERMISSION_STATUS ('ask', None)
    B. session.attach                       -> nothing at all
    C. permissions default deny, then:
         the command's own answer           -> "deny (session override, was: ask)"
         the PermissionStatusEvent beside it-> ('ask', None)

**B is the reported symptom.**  ``emit_current_state`` is the one door
for "tell a client arriving mid-session what the state is" -- it replays
the agents, the conversation, the statuses, the instruction budget, the
subagents and the tool-id registry -- and the policy was not among them.
So a client that ATTACHED rather than created (a reconnect, a session
switch, a resume from the picker) never learned it, and the web client
resets that field on attach, so the segment vanished and did not come
back.

**C is worse, and it is why this is not a one-line emit.**  There are
two ``PermissionPlugin`` objects on a runner-served session -- the
default -- and only one of them decides anything.  The daemon builds its
own at ``initialize()`` and seeds it from the profile; the RUNNER's is
the plugin ``check_permission`` consults and the one a ``permissions``
command mutates.  ``emit_permission_status`` read the daemon's, so the
value was true until somebody changed the policy and wrong from then on.
Emitting that on attach would have made a stale fact arrive more
reliably -- the defect wearing the fix as a disguise.

The segment is a CONTROL since the plate landed: it marks which default
is in force and offers Suspend or Resume from this value.  A control
whose readout disagrees with the thing it controls is worse than one
that shows nothing, which is the argument ``test_envelope_carries_gc``
makes about a GC strategy displayed and never run.

So the runner is asked (``session.get_permission_status``), and **a
failed ask reports nothing**: falling back to the daemon's copy is
reading the stale value this exists to stop reading, and the client then
keeps what it last knew rather than being handed a new claim.  The
fallback applies only where there is no runner at all -- the embedded
client, standalone WS, the legacy daemon-local path -- and there the
daemon's plugin IS the enforcer, so it is the right answer rather than a
tolerated one.

Measured on the same live daemon after the fix::

    A. create                        -> [('ask', None)]
    B. attach (was: nothing)         -> [('ask', None)]
    C. after 'default deny' (was ask) -> [('deny', None)]
    D. attach again                  -> [('deny', None)]
    E. after 'suspend --turn'        -> [('deny', 'turn')]

NOT closed here, and stated rather than implied: an ``a`` / ``t`` / ``i``
answer to a prompt re-emits only on the daemon-local path
(``on_permission_resolved``), which is dead on a runner-served session.
The runner applies the suspension asynchronously after
``resolve_response`` hands the answer over, so a re-emit at that seam
would race the change it is reporting -- and reporting the pre-change
value is this file's own defect in a third place.  The honest fix is for
the runner-side plugin to announce its own policy change, which is a
notification frame rather than a pull, and its own change.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jaato_sdk.events import PermissionStatusEvent
from jaato_server.server.core import JaatoServer
from jaato_server.server.runner.rpc import RunnerRPC
from jaato_server.shared.tests.reversion import Reversion

_CORE = "jaato-server/jaato_server/server/core.py"
_RPC = "jaato-server/jaato_server/server/runner/rpc.py"


# ---------------------------------------------------------------- doubles


class _Plugin:
    """The daemon's own permission plugin -- deliberately the STALE one."""

    def __init__(self, default: str = "ask", scope: Optional[str] = None):
        self.default, self.scope = default, scope

    def get_permission_status(self) -> Dict[str, Any]:
        return {
            "effective_default": self.default,
            "suspension_scope": self.scope,
            "is_suspended": self.scope is not None,
        }


class _RPCClient:
    """A runner whose plugin holds the policy that is actually enforced."""

    def __init__(self, status: Any = None, raises: bool = False):
        self.status, self.raises, self.calls = status, raises, 0

    def session_get_permission_status_threadsafe(self, *, timeout=None):
        self.calls += 1
        if self.raises:
            raise RuntimeError("runner is gone")
        return self.status


def _server(plugin=None, rpc=None) -> JaatoServer:
    s = JaatoServer.__new__(JaatoServer)
    s.permission_plugin = plugin
    if rpc is not None:
        s._runner_rpc = rpc
    return s


# ------------------------------------------------------- C: whose policy


def test_the_runner_is_asked_not_the_daemons_own_copy():
    """The enforcer's answer wins over the daemon's seeded copy."""
    rpc = _RPCClient({"effective_default": "deny", "suspension_scope": None})
    event = _server(plugin=_Plugin("ask"), rpc=rpc).permission_status_event()
    assert rpc.calls == 1
    assert event is not None and event.effective_default == "deny"


def test_a_suspension_the_runner_applied_reaches_the_client():
    rpc = _RPCClient({"effective_default": "deny", "suspension_scope": "turn"})
    event = _server(plugin=_Plugin("ask"), rpc=rpc).permission_status_event()
    assert event is not None
    assert (event.effective_default, event.suspension_scope) == ("deny", "turn")


def test_a_failed_ask_reports_nothing_rather_than_the_stale_copy():
    """Absence is not a claim; the daemon's copy would be a wrong one."""
    rpc = _RPCClient(raises=True)
    assert _server(plugin=_Plugin("ask"), rpc=rpc).permission_status_event() is None


def test_with_no_runner_the_daemons_plugin_IS_the_enforcer():
    """Embedded / standalone-WS / daemon-local: unchanged behaviour."""
    event = _server(plugin=_Plugin("allow", "idle")).permission_status_event()
    assert event is not None
    assert (event.effective_default, event.suspension_scope) == ("allow", "idle")


def test_no_plugin_and_no_runner_reports_nothing():
    assert _server().permission_status_event() is None


def test_emit_permission_status_publishes_nothing_when_there_is_nothing():
    sent: List[Any] = []
    s = _server(plugin=_Plugin("ask"), rpc=_RPCClient(raises=True))
    s.emit = sent.append  # type: ignore[method-assign]
    s.emit_permission_status()
    assert sent == []


def test_emit_permission_status_publishes_the_runners_answer():
    sent: List[Any] = []
    s = _server(plugin=_Plugin("ask"),
                rpc=_RPCClient({"effective_default": "deny",
                                "suspension_scope": None}))
    s.emit = sent.append  # type: ignore[method-assign]
    s.emit_permission_status()
    assert len(sent) == 1 and isinstance(sent[0], PermissionStatusEvent)
    assert sent[0].effective_default == "deny"


# ------------------------------- B: an arriving client is told the policy


def _is_resolver_call(node: ast.AST) -> bool:
    """``<anything>.permission_status_event(...)``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "permission_status_event"
    )


def _resolver_result_names(fn: ast.FunctionDef) -> set:
    """The local names the resolver's answer was assigned to."""
    names = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or not _is_resolver_call(node.value):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _emits_one_of(node: ast.AST, names: set) -> bool:
    """``emit(<name>)`` for one of those names."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return False
    if node.func.id != "emit" or len(node.args) != 1:
        return False
    return isinstance(node.args[0], ast.Name) and node.args[0].id in names


def _emit_current_state_calls_the_resolver() -> Tuple[bool, bool]:
    """Does ``emit_current_state`` resolve the policy, and EMIT it?

    Asserted by AST over the real source rather than by driving the
    method: ``emit_current_state`` reaches a dozen subsystems, and a test
    that stubbed enough of a server to run it would be asserting the
    stubs.  What the reported defect was is a missing CALL SITE, which is
    exactly what a walk of the call sites can answer.

    The three predicates above are separate because this file is subject
    to the complexity ratchet like any other: one comprehension answering
    all of it scored 20 against a ceiling of 15.
    """
    tree = ast.parse(Path(_CORE).read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "emit_current_state"
    )
    nodes = list(ast.walk(fn))
    names = _resolver_result_names(fn)
    return (
        any(_is_resolver_call(n) for n in nodes),
        any(_emits_one_of(n, names) for n in nodes),
    )


def test_emit_current_state_tells_an_attaching_client_the_policy():
    resolves, emits = _emit_current_state_calls_the_resolver()
    assert resolves, (
        "emit_current_state does not resolve the permission status, so a "
        "client that ATTACHED never learns the policy and the web status "
        "bar's segment stays hidden"
    )
    assert emits, (
        "emit_current_state resolves the permission status and does not "
        "emit it to the arriving client"
    )


# ------------------------------------------------ the runner-side handler


def _handler_registered() -> bool:
    return "session.get_permission_status" in Path(_RPC).read_text()


def test_the_runner_answers_the_verb():
    assert _handler_registered()
    rpc = RunnerRPC.__new__(RunnerRPC)

    class _Sess:
        _runtime = type("_RT", (), {"permission_plugin": _Plugin("deny", "idle")})()

    rpc._require_ready_session = lambda: (True, None, _Sess())  # type: ignore[assignment]
    ok, payload = rpc._handle_session_get_permission_status()
    assert ok and payload["status"]["effective_default"] == "deny"
    assert payload["status"]["suspension_scope"] == "idle"


def test_a_session_with_no_permission_plugin_is_refused_not_defaulted():
    """``ask`` invented here is the same lie one process over."""
    rpc = RunnerRPC.__new__(RunnerRPC)

    class _Sess:
        _runtime = type("_RT", (), {"permission_plugin": None})()

    rpc._require_ready_session = lambda: (True, None, _Sess())  # type: ignore[assignment]
    ok, payload = rpc._handle_session_get_permission_status()
    assert not ok and payload["stage"] == "no_plugin"


REVERSIONS = [
    Reversion(
        target=_CORE,
        find="""        permission_status = self.permission_status_event()
        if permission_status is not None:
            emit(permission_status)
""",
        replace="",
        test="test_emit_current_state_tells_an_attaching_client_the_policy",
        because=(
            "a client that attaches -- a reconnect, a session switch, a "
            "resume from the picker -- never being told the permission "
            "policy, so the web status bar's segment vanishes and does "
            "not come back"
        ),
    ),
    Reversion(
        target=_CORE,
        find="""        rpc = getattr(self, "_runner_rpc", None)
        if rpc is not None:
            forwarder = getattr(
                rpc, "session_get_permission_status_threadsafe", None,
            )""",
        replace="""        rpc = None
        if rpc is not None:
            forwarder = getattr(
                rpc, "session_get_permission_status_threadsafe", None,
            )""",
        test="test_the_runner_is_asked_not_the_daemons_own_copy",
        because=(
            "the daemon reporting its own profile-seeded copy of the "
            "policy, which is true until somebody changes the policy and "
            "wrong from then on"
        ),
    ),
    Reversion(
        target=_CORE,
        find="""                        "RPC failed (%s) — reporting no status rather "
                        "than the daemon's stale copy", exc,
                    )
                    return None""",
        replace="""                        "RPC failed (%s) — reporting no status rather "
                        "than the daemon's stale copy", exc,
                    )""",
        test="test_a_failed_ask_reports_nothing_rather_than_the_stale_copy",
        because=(
            "a runner that could not answer falling through to the "
            "daemon's stale copy -- the wrong value this exists to stop "
            "reading, now reported as though it had been measured"
        ),
    ),
    Reversion(
        target=_RPC,
        find="""            return False, {
                "error": (
                    "session.get_permission_status: the runner session's "
                    "runtime carries no permission plugin"
                ),
                "stage": "no_plugin",
            }""",
        replace="""            return True, {"status": {"effective_default": "ask",
                                     "suspension_scope": None,
                                     "is_suspended": False}}""",
        test="test_a_session_with_no_permission_plugin_is_refused_not_defaulted",
        because=(
            "a session with no permission plugin reporting the framework "
            "default as though it were its policy"
        ),
    ),
]
