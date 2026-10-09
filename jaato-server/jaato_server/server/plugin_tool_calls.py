"""One plugin tool call with no model turn: the daemon half (#1606).

``PluginToolDescribeRequest`` / ``PluginToolInvokeRequest`` (protocol 1.37)
let a client reach one plugin tool directly -- jaato-mcp's ``kind: plugin``
tool, with kbwiki offering the ``template`` plugin to its MCP callers.  The
alternative the issue names, importing ``jaato_server.shared.plugins.<x>`` and
calling its executor, needs jaato-server in the client's environment, treats
internals as an API, and runs the call outside the profile's confinement.

WHAT THE CALL RUNS IN.  A short-lived session, because then everything a
session already does applies and nothing is re-implemented here:

* it is created by :meth:`SessionManager.create_session`, from a copy of the
  CALLER's client configuration, in the caller's workspace
  (``resolve_caller_workspace``, the rule every daemon-level verb uses) --
  so the config search path, the profile resolution, the runner, its
  AppArmor / SELinux boundary and its uid are the ones the caller's own
  ``session.new`` would get;
* the runner answers against that session
  (``shared/plugin_tool_call.py``): the schema after ``narrow_tool_schema``
  and the session's surface, and the call through the session's own
  ``ToolExecutor`` (permission policy, coercion, redaction, the failure
  contract);
* it is deleted when the answer is sent, record and all.

The session is created under a SYNTHETIC client id
(``_plugin_tool:<hex>``) registered with
:meth:`SessionManager.register_ephemeral_client`, so the caller's own
attachment (``_client_to_session``) is untouched and every event addressed
to the session reaches :class:`_CallSink` instead of a transport.  That sink
keeps the creation error, so ``session_failed`` names its cause rather than
the empty session id a headless create returns, and FORWARDS a permission
ASK to the caller.  The caller answers with the ordinary
``PermissionResponseRequest``; :meth:`PluginToolCalls.route_permission_response`
finds the call whose session holds that ``request_id`` and resolves it there.

The verbs block for as long as session creation and the call take, and an
IPC connection reads its next message only when the previous one was
handled -- so the work runs on its own thread and the answer is sent when it
is ready.  Otherwise a caller could not send the permission answer the call
is waiting on.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Seconds a call may take, a pending permission ASK included, when the
#: request names no ``timeout``.
DEFAULT_TIMEOUT = 300.0

#: The longest ``timeout`` a request may ask for.
MAX_TIMEOUT = 3600.0

#: Seconds a ``describe`` may take once its session exists.
DESCRIBE_TIMEOUT = 60.0

#: Prefix of the synthetic client ids the calls run under.
CLIENT_ID_PREFIX = "_plugin_tool:"

#: Events forwarded from the call's session to the caller: what a client
#: needs to show a permission ASK and to clear it.
FORWARDED_EVENTS = frozenset({
    "PermissionRequestedEvent",
    "PermissionInputModeEvent",
    "PermissionResolvedEvent",
})

#: The provider the profile-less (``plugin_configs``) form binds.  A session
#: must name a model, and a tool call needs none: ``echo`` is in-tree, needs
#: no credential and makes no network call, and no turn ever reaches it.
INLINE_PROVIDER = "echo"


def inline_spec(plugin: str, plugin_configs: Dict[str, Any]) -> Dict[str, Any]:
    """The inline profile the ``plugin_configs`` form runs under.

    Enables ``plugin`` and nothing else (plus what every session has), with
    the caller's block as its ``plugin_configs``.
    """
    return {
        "name": f"plugin-tool-{plugin}",
        "model": INLINE_PROVIDER,
        "provider": INLINE_PROVIDER,
        "plugins": [plugin],
        "plugin_configs": dict(plugin_configs),
    }


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
    """Where the events of one call's session go (see the module docstring)."""

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
    session_id: str
    server: Any = None


class PluginToolCalls:
    """The two plugin-tool verbs, owned by the :class:`CommandRouter`.

    Args:
        session_manager: Creates, finds and deletes the call's session.
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
        with no workspace; anything that needs a session is answered from
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
        call's session, or -- when the client is attached to no session of
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
        synthetic = f"{CLIENT_ID_PREFIX}{uuid.uuid4().hex[:12]}"
        sink = _CallSink(caller=client_id, send=self._send,
                         origin_request_id=str(getattr(event, "request_id", "") or ""))
        session_id = ""
        try:
            self._session_manager.register_ephemeral_client(
                synthetic, sink, config_from=client_id)
            session_id, server = self._create(synthetic, client_id, event, workspace)
            if not session_id:
                cause = "; ".join(e for e in sink.errors if e) or "see the daemon log"
                self._answer(client_id, op, event, category="session_failed",
                             error=f"plugin tool {op}: the session could not be "
                                   f"created: {cause}")
                return
            with self._lock:
                self._inflight[synthetic] = _InFlight(client_id, session_id, server)
            answer = server.plugin_tool_op(
                op, {"plugin": event.plugin, "tool": event.tool,
                     "args": dict(getattr(event, "args", None) or {}),
                     "call_id": f"plugin-tool-{uuid.uuid4().hex[:12]}"},
                timeout=(effective_timeout(event.timeout) if op == "invoke"
                         else DESCRIBE_TIMEOUT),
            )
            self._answer_from(client_id, op, event, answer, session_id)
        except Exception as exc:  # noqa: BLE001 -- the caller is always answered
            logger.exception("plugin tool %s failed", op)
            self._answer(client_id, op, event, category="session_failed",
                         error=f"plugin tool {op}: {type(exc).__name__}: {exc}",
                         session_id=session_id)
        finally:
            with self._lock:
                self._inflight.pop(synthetic, None)
            if session_id:
                try:
                    self._session_manager.delete_session(session_id)
                except Exception:  # noqa: BLE001
                    logger.warning("plugin tool call: could not delete session %s",
                                   session_id, exc_info=True)
            self._session_manager.unregister_ephemeral_client(synthetic)

    def _create(self, synthetic: str, client_id: str, event: Any,
                workspace: str) -> Tuple[str, Any]:
        profile = str(getattr(event, "profile", "") or "")
        spec = None if profile else inline_spec(
            event.plugin, getattr(event, "plugin_configs", None) or {})
        session_id = self._session_manager.create_session(
            synthetic, f"plugin-tool:{event.plugin}.{event.tool}",
            workspace_path=workspace,
            profile_name=profile or None,
            inline_profile_data=spec,
            created_by=self._get_user(client_id),
        )
        if not session_id:
            return "", None
        session = self._session_manager.get_session(session_id)
        server = getattr(session, "server", None) if session is not None else None
        if server is None:
            return "", None
        return session_id, server

    # ------------------------------------------------------------------
    # Answers
    # ------------------------------------------------------------------

    def _answer_from(self, client_id: str, op: str, event: Any,
                     answer: Dict[str, Any], session_id: str) -> None:
        category = str(answer.get("category") or "")
        if category:
            self._answer(client_id, op, event, category=category,
                         error=f"plugin tool {op}: {answer.get('error') or category}",
                         session_id=session_id)
            return
        if op == "describe":
            self._answer(client_id, op, event, exists=bool(answer.get("exists")),
                         reason=str(answer.get("reason") or ""),
                         detail=str(answer.get("detail") or ""),
                         tool_schema=answer.get("tool_schema"))
            return
        if not answer.get("ran"):
            self._answer(client_id, op, event, category="not_in_surface",
                         error=f"plugin tool invoke: {answer.get('detail') or answer.get('reason')}",
                         session_id=session_id)
            return
        self._answer(client_id, op, event, success=bool(answer.get("success")),
                     result=answer.get("result"), session_id=session_id)

    def _answer(self, client_id: str, op: str, event: Any, *,
                category: str = "", error: str = "", **fields: Any) -> None:
        from jaato_sdk.events import PluginToolDescribeEvent, PluginToolInvokeResultEvent

        logger.info("plugin tool %s: client=%s plugin=%s tool=%s ok=%s category=%s",
                    op, client_id, getattr(event, "plugin", ""),
                    getattr(event, "tool", ""), not category, category or "-")
        if not client_id:
            return
        if op == "describe":
            fields.pop("session_id", None)
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
