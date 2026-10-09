"""One plugin tool call with no session: the daemon half (#1606).

``PluginToolDescribeRequest`` / ``PluginToolInvokeRequest`` (protocol 1.37)
let a client reach one plugin tool directly -- jaato-mcp's ``kind: plugin``
tool, with kbwiki offering the ``template`` plugin to its MCP callers.  The
alternative the issue names, importing ``jaato_server.shared.plugins.<x>`` and
calling its executor, needs jaato-server in the client's environment, treats
internals as an API, and runs the call outside the profile's confinement.

WHAT THE CALL RUNS IN.  A runner, and no session.  A tool needs its plugin
initialized with the profile's configuration, the profile's permission
policy and a process inside the workspace's boundary; it does not need a
conversation, a provider or a model.  :meth:`SessionManager.spawn_plugin_host`
resolves what the caller's own ``session.new`` would (its client config,
the workspace, the profile), runs the spawn half of session bootstrap --
the same confinement provisioning, runner uid and pool slot -- and the runner
builds a ``PluginToolHost`` (``shared/plugin_tool_call.py``) where it would
have built a ``JaatoSession``.  Nothing is recorded, listed or persisted;
:meth:`SessionManager.release_plugin_host` closes the runner when the answer
is sent.

A permission ASK the call raises is emitted by the host's daemon-side server
to :class:`_CallSink`, which FORWARDS it to the caller tagged with the
request's id (``PermissionRequestedEvent.origin_request_id``) and keeps any
``ErrorEvent`` so a failed setup is answered with its cause.  The caller
answers with the ordinary ``PermissionResponseRequest``;
:meth:`PluginToolCalls.route_permission_response` finds the call whose
server holds that ``request_id`` and resolves it there.

Setting up a runner and the call take seconds, and an IPC connection reads
its next message only when the previous one was handled -- so the work runs
on its own thread and the answer is sent when it is ready.  Otherwise a
caller could not send the permission answer the call is waiting on.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Seconds a call may take, a pending permission ASK included, when the
#: request names no ``timeout``.
DEFAULT_TIMEOUT = 300.0

#: The longest ``timeout`` a request may ask for.
MAX_TIMEOUT = 3600.0

#: Seconds a ``describe`` may take once its runner is up.
DESCRIBE_TIMEOUT = 60.0

#: Events forwarded from the call's runner to the caller: what a client
#: needs to show a permission ASK and to clear it.
FORWARDED_EVENTS = frozenset({
    "PermissionRequestedEvent",
    "PermissionInputModeEvent",
    "PermissionResolvedEvent",
})


def effective_timeout(requested: float) -> float:
    """``requested`` clamped to ``(0, MAX_TIMEOUT]``; non-positive is the default."""
    try:
        value = float(requested)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0:
        return DEFAULT_TIMEOUT
    return min(value, MAX_TIMEOUT)


def request_problem(event: Any) -> str:
    """Why a plugin-tool request cannot be served as written, or ``""``."""
    if not str(getattr(event, "plugin", "") or "").strip():
        return "plugin is required"
    if not str(getattr(event, "tool", "") or "").strip():
        return "tool is required"
    configs = getattr(event, "plugin_configs", None)
    if getattr(event, "profile", "") and configs is not None:
        return "profile and plugin_configs are mutually exclusive"
    if configs is not None and not isinstance(configs, dict):
        return "plugin_configs must be an object"
    return ""


@dataclass
class _CallSink:
    """Where the events of one call's runner go (see the module docstring)."""

    caller: str
    send: Callable[[str, Any], None]
    origin_request_id: str = ""
    errors: List[str] = field(default_factory=list)

    def __call__(self, event: Any) -> None:
        name = type(event).__name__
        if name == "ErrorEvent":
            self.errors.append(str(getattr(event, "error", "") or ""))
        if name == "PermissionRequestedEvent":
            # A copy: the relay keeps the original for replay to its own
            # attached clients, which must not see another caller's tag.
            event = event.model_copy(
                update={"origin_request_id": self.origin_request_id or None})
        if name in FORWARDED_EVENTS:
            try:
                self.send(self.caller, event)
            except Exception:  # noqa: BLE001 -- a dead caller fails the ASK by timeout
                logger.warning("plugin tool call: could not forward %s to %s",
                               name, self.caller, exc_info=True)


@dataclass
class _InFlight:
    caller: str
    host_id: str
    server: Any = None


class PluginToolCalls:
    """The two plugin-tool verbs, owned by the :class:`CommandRouter`.

    Args:
        session_manager: Spawns and releases the call's plugin host.
        send: ``(client_id, event)`` to the caller's transport.
        get_user: The identity the transport authenticated for a client.
    """

    def __init__(
        self,
        session_manager: Any,
        send: Callable[[str, Any], None],
        get_user: Callable[[str], Optional[str]],
    ) -> None:
        self._session_manager = session_manager
        self._send = send
        self._get_user = get_user
        self._lock = threading.Lock()
        self._inflight: Dict[str, _InFlight] = {}

    # ------------------------------------------------------------------
    # Entry points (called by the router on the transport's thread)
    # ------------------------------------------------------------------

    def start(self, client_id: str, event: Any, workspace: Optional[str],
              no_workspace_detail: str = "") -> None:
        """Validate ``event`` and run it on its own thread.

        Answers at once (on this thread) for a malformed request or a caller
        with no workspace; anything that needs a runner is answered from
        the worker thread when it is ready.
        """
        op = "invoke" if type(event).__name__ == "PluginToolInvokeRequest" else "describe"
        problem = request_problem(event)
        if problem:
            self._answer(client_id, op, event, category="invalid_request",
                         error=f"plugin tool {op}: {problem}")
            return
        if not workspace:
            self._answer(client_id, op, event, category="no_workspace",
                         error=f"plugin tool {op}: the caller has no workspace"
                               + (f" ({no_workspace_detail})" if no_workspace_detail else ""))
            return
        threading.Thread(
            target=self._run, args=(client_id, op, event, workspace),
            name=f"plugin-tool-{op}", daemon=True,
        ).start()

    def route_permission_response(
        self, client_id: str, session_id: str, event: Any,
    ) -> bool:
        """Resolve a permission answer that belongs to one of ``client_id``'s calls.

        Returns True when the request was handled here: resolved on the
        call's runner, or -- when the client is attached to no session of
        its own -- refused as ``no_session`` the way the IPC transport
        refuses any request that needs a session.  False hands it on to the
        ordinary routing.
        """
        if type(event).__name__ != "PermissionResponseRequest":
            return False
        request_id = str(getattr(event, "request_id", "") or "")
        for call in self._calls_of(client_id):
            if call.server is not None and request_id in _pending_prompt_ids(call.server):
                # The identity is the transport's, never the request's (#859).
                call.server.respond_to_permission(
                    request_id, event.response,
                    edited_arguments=getattr(event, "edited_arguments", None),
                    user_id=self._get_user(client_id),
                )
                return True
        if session_id:
            return False
        from jaato_sdk.events import ErrorEvent
        self._send(client_id, ErrorEvent(
            error="PermissionResponseRequest: no session selected and no "
                  "plugin tool call is waiting on this request.",
            error_type="RequestError",
            request_id=request_id or None,
            details={"category": "no_session",
                     "request_type": "PermissionResponseRequest"},
        ))
        return True

    # ------------------------------------------------------------------
    # The worker
    # ------------------------------------------------------------------

    def _calls_of(self, client_id: str) -> List[_InFlight]:
        with self._lock:
            return [c for c in self._inflight.values() if c.caller == client_id]

    def _run(self, client_id: str, op: str, event: Any, workspace: str) -> None:
        sink = _CallSink(caller=client_id, send=self._send,
                         origin_request_id=str(getattr(event, "request_id", "") or ""))
        server, host_id, key = None, "", uuid.uuid4().hex
        try:
            server, host_id, error = self._session_manager.spawn_plugin_host(
                client_id, workspace_path=workspace,
                profile_name=str(getattr(event, "profile", "") or "") or None,
                plugin=event.plugin,
                plugin_configs=getattr(event, "plugin_configs", None),
                created_by=self._get_user(client_id),
                on_event=sink,
            )
            if server is None:
                cause = "; ".join([error] + [e for e in sink.errors if e])
                self._answer(client_id, op, event, category="host_failed",
                             error=f"plugin tool {op}: no runner could be set up "
                                   f"for the call: {cause}")
                return
            with self._lock:
                self._inflight[key] = _InFlight(client_id, host_id, server)
            answer = server.plugin_tool_op(
                op, {"plugin": event.plugin, "tool": event.tool,
                     "args": dict(getattr(event, "args", None) or {}),
                     "call_id": f"plugin-tool-{uuid.uuid4().hex[:12]}"},
                timeout=(effective_timeout(event.timeout) if op == "invoke"
                         else DESCRIBE_TIMEOUT),
            )
            self._answer_from(client_id, op, event, answer)
        except Exception as exc:  # noqa: BLE001 -- the caller is always answered
            logger.exception("plugin tool %s failed", op)
            self._answer(client_id, op, event, category="host_failed",
                         error=f"plugin tool {op}: {type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            if server is not None:
                self._session_manager.release_plugin_host(server, host_id)

    # ------------------------------------------------------------------
    # Answers
    # ------------------------------------------------------------------

    def _answer_from(self, client_id: str, op: str, event: Any,
                     answer: Dict[str, Any]) -> None:
        category = str(answer.get("category") or "")
        if category:
            self._answer(client_id, op, event, category=category,
                         error=f"plugin tool {op}: {answer.get('error') or category}")
            return
        if op == "describe":
            self._answer(client_id, op, event, exists=bool(answer.get("exists")),
                         reason=str(answer.get("reason") or ""),
                         detail=str(answer.get("detail") or ""),
                         tool_schema=answer.get("tool_schema"))
            return
        if not answer.get("ran"):
            self._answer(client_id, op, event, category="not_in_surface",
                         error=f"plugin tool invoke: {answer.get('detail') or answer.get('reason')}")
            return
        self._answer(client_id, op, event, success=bool(answer.get("success")),
                     result=answer.get("result"))

    def _answer(self, client_id: str, op: str, event: Any, *,
                category: str = "", error: str = "", **fields: Any) -> None:
        from jaato_sdk.events import PluginToolDescribeEvent, PluginToolInvokeResultEvent

        logger.info("plugin tool %s: client=%s plugin=%s tool=%s ok=%s category=%s",
                    op, client_id, getattr(event, "plugin", ""),
                    getattr(event, "tool", ""), not category, category or "-")
        if not client_id:
            return
        if op == "describe":
            answer: Any = PluginToolDescribeEvent(
                request_id=event.request_id, ok=not category, category=category,
                error=error, **fields)
        else:
            answer = PluginToolInvokeResultEvent(
                request_id=event.request_id, ok=not category, category=category,
                error=error, **fields)
        self._send(client_id, answer)


def _pending_prompt_ids(server: Any) -> List[str]:
    """The ``request_id`` of every permission ASK pending on ``server``."""
    handler = getattr(server, "_prompt_operator_handler", None)
    if handler is None:
        return []
    try:
        return [str(getattr(e, "request_id", "") or "")
                for e in handler.pending_events()]
    except Exception:  # noqa: BLE001
        return []
