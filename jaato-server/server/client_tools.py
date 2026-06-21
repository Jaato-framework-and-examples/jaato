"""Shared client-provided ("host") tool registration (transport-agnostic).

Both transports register client tools the same way — a proxy ``ToolSchema`` on
the session registry whose executor sends a ``ToolExecuteRequestEvent`` to the
session's attached client(s) and blocks for the matching
``ToolExecuteResultEvent``.  Only the transport-specific *send* and the *waiters*
dict differ, so they are injected.

The recorded schema is also stashed on ``JaatoServer.client_tool_schemas`` so
``spawn_session_runner`` ferries it to the RUNNER-tier model via
``envelope.client_tools`` — the model runs runner-side, and registering only on
the daemon registry left it blind (see ``project_client_tools_runner_split_gap``
/ PR #349).

Used by the IPC transport (``server/ipc.py``).  The WebSocket transport
(``websocket._register_client_tools``) still inlines an equivalent today; folding
it onto this helper is a no-behavior-change cleanup follow-up.
"""

import threading
import time
import uuid
from typing import Any, Callable, Dict, List

from jaato_sdk.plugins.model_provider.types import ToolSchema
from jaato_sdk.events import ToolExecuteRequestEvent

# (threading.Event, {"result": ..., "error": ...}) keyed by call_id.
WaiterEntry = "tuple[threading.Event, Dict[str, Any]]"


def make_client_tool_executor(
    tool_name: str,
    timeout_s: float,
    send_request: Callable[[ToolExecuteRequestEvent], bool],
    waiters: Dict[str, Any],
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Build the proxy executor for one client tool.

    Args:
        tool_name: The model-facing tool name.
        timeout_s: Seconds to wait for the client's result.
        send_request: Transport send — dispatches the event to the session's
            attached client(s); returns True iff at least one received it.  May
            be retried while no client is connected (mirrors permission requests
            during disconnection).
        waiters: ``call_id -> (Event, holder)`` dict the transport's incoming
            ``ToolExecuteResultEvent`` handler fills (``holder["result"]`` /
            ``holder["error"]``) and ``.set()``s.
    """
    def _executor(args: Dict[str, Any]) -> Dict[str, Any]:
        call_id = str(uuid.uuid4())[:8]
        waiter = threading.Event()
        holder: Dict[str, Any] = {"result": None, "error": None}
        waiters[call_id] = (waiter, holder)

        event = ToolExecuteRequestEvent(
            call_id=call_id, agent_id="", tool_name=tool_name, tool_args=args)

        deadline = time.time() + timeout_s
        sent = send_request(event)
        while not sent and time.time() < deadline:
            time.sleep(1.0)
            sent = send_request(event)
        if not sent:
            waiters.pop(call_id, None)
            return {"error": f"No client connected to receive tool call {tool_name}"}

        if waiter.wait(timeout=max(0.0, deadline - time.time())):
            return ({"error": holder["error"]} if holder["error"]
                    else {"result": holder["result"]})
        waiters.pop(call_id, None)
        return {"error": f"Client tool {tool_name} timed out after {timeout_s}s"}

    return _executor


def register_client_tools(
    registry: Any,
    server: Any,
    tools: List[Dict[str, Any]],
    *,
    send_request: Callable[[ToolExecuteRequestEvent], bool],
    waiters: Dict[str, Any],
) -> List[str]:
    """Register each client tool as a core tool (model-visible) with a proxy
    executor, and record its schema on ``server.client_tool_schemas`` so the
    runner-tier model receives it.  Returns the registered tool names.
    """
    registered: List[str] = []
    for tool_def in tools or []:
        name = tool_def.get("name")
        if not name:
            continue
        description = tool_def.get("description", "")
        parameters = tool_def.get("parameters", {})
        category = tool_def.get("category", "")
        timeout_s = tool_def.get("timeout", 30000) / 1000.0  # ms -> s
        auto_approve = tool_def.get("auto_approve", True)

        schema = ToolSchema(
            name=name,
            description=description + " [client-provided]",
            parameters=parameters,
            category=category or None,
        )
        registry.register_core_tool(
            schema,
            make_client_tool_executor(name, timeout_s, send_request, waiters),
            auto_approved=auto_approve,
        )
        server.client_tool_schemas[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "category": category or "",
        }
        registered.append(name)
    return registered
