"""``PermissionResolvedEvent`` names who decided (issue #859).

The daemon-side resolved hook (``JaatoServer._setup_permission_hooks``)
is where the permission plugin's resolution becomes the event clients
and the event bus see.  It now forwards two identities:

- ``user_id`` — passed by the plugin for a runner-relayed ASK (it came
  back on the ``PromptResponse``), or, on the legacy daemon-side queue
  path where the channel cannot carry it, read back from the value
  ``respond_to_permission`` parked for that request_id.
- ``approver`` — passed by the plugin from a webhook / file response.

A policy decision fires the hook with neither, and the event says so.
"""

from __future__ import annotations

import queue
from typing import Any, Dict, List, Optional

from jaato_sdk.events import PermissionResolvedEvent

from server.core import JaatoServer


class _HookCapturingPlugin:
    """The only surface the hook setup touches on the permission plugin."""

    def __init__(self) -> None:
        self.on_requested = None
        self.on_resolved = None

    def set_permission_hooks(self, on_requested=None, on_resolved=None):
        self.on_requested = on_requested
        self.on_resolved = on_resolved

    def get_permission_status(self) -> Dict[str, Any]:
        return {"effective_default": "ask", "suspension_scope": None}


def _server_with_hooks() -> Any:
    srv = JaatoServer.__new__(JaatoServer)
    srv._emitted: List[Any] = []
    srv.emit = lambda e: srv._emitted.append(e)  # type: ignore[method-assign]
    srv.permission_plugin = _HookCapturingPlugin()
    srv._current_tool_agent_id = "main"
    srv._main_agent_id = "main"
    srv._channel_input_queue = queue.Queue()
    srv._waiting_for_channel_input = False
    srv._pending_permission_request_id: Optional[str] = None
    srv._pending_edited_arguments: Optional[Dict[str, Any]] = None
    srv._pending_permission_user_id: Optional[str] = None
    srv._prompt_operator_handler = None
    srv._setup_permission_hooks()
    return srv


def _resolved(srv: Any) -> List[PermissionResolvedEvent]:
    return [e for e in srv._emitted if isinstance(e, PermissionResolvedEvent)]


def test_runner_relayed_resolution_carries_both_identities() -> None:
    srv = _server_with_hooks()
    srv.permission_plugin.on_resolved(
        "deploy", "req-1", True, "user_approved",
        comment="", user_id="sso|alice", approver="Alice",
    )
    (event,) = _resolved(srv)
    assert event.granted is True
    assert event.method == "user_approved"
    assert event.user_id == "sso|alice"
    assert event.approver == "Alice"


def test_policy_decision_names_nobody() -> None:
    """The plugin fires the hook positionally for whitelist / blacklist
    hits; both identity fields must default to None, not fail."""
    srv = _server_with_hooks()
    srv.permission_plugin.on_resolved("read_file", "", True, "whitelist")
    (event,) = _resolved(srv)
    assert event.method == "whitelist"
    assert event.user_id is None
    assert event.approver is None


def test_legacy_queue_path_reads_back_the_parked_user() -> None:
    """Daemon-side QueueChannel: the queue carries only the response key,
    so ``respond_to_permission`` parked the responder and the hook
    reads it back for the matching request_id -- then clears it."""
    srv = _server_with_hooks()
    srv._pending_permission_request_id = "req-legacy"
    srv.respond_to_permission("req-legacy", "y", user_id="sso|bob")
    assert srv._pending_permission_user_id == "sso|bob"

    srv.permission_plugin.on_resolved("deploy", "req-legacy", True, "user_approved")
    (event,) = _resolved(srv)
    assert event.user_id == "sso|bob"
    assert srv._pending_permission_user_id is None
    assert srv._pending_permission_request_id is None


def test_parked_user_does_not_leak_onto_another_request() -> None:
    """A whitelist auto-decision racing the visible prompt fires with an
    empty request_id; it must not consume or carry the parked identity."""
    srv = _server_with_hooks()
    srv._pending_permission_request_id = "req-visible"
    srv._pending_permission_user_id = "sso|bob"

    srv.permission_plugin.on_resolved("read_file", "", True, "whitelist")
    (auto,) = _resolved(srv)
    assert auto.user_id is None
    assert srv._pending_permission_user_id == "sso|bob"
    assert srv._pending_permission_request_id == "req-visible"


def test_explicit_user_wins_over_a_parked_one() -> None:
    srv = _server_with_hooks()
    srv._pending_permission_request_id = "req-x"
    srv._pending_permission_user_id = "sso|parked"
    srv.permission_plugin.on_resolved(
        "deploy", "req-x", False, "user_denied", user_id="sso|explicit",
    )
    (event,) = _resolved(srv)
    assert event.user_id == "sso|explicit"
    assert srv._pending_permission_user_id is None


def test_event_serializes_the_identity_fields() -> None:
    event = PermissionResolvedEvent(
        request_id="r", tool_name="t", granted=True, method="user_approved",
        user_id="sso|alice", approver="Alice",
    )
    data = event.model_dump()
    assert data["user_id"] == "sso|alice"
    assert data["approver"] == "Alice"
    assert PermissionResolvedEvent().user_id is None
