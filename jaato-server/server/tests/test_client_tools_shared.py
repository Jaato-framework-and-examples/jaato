"""Shared client-tool registration helper (server/client_tools.py).

The proxy executor's send -> wait -> result flow (transport-agnostic), and the
registration that records schemas onto ``JaatoServer.client_tool_schemas``.
"""

from server.client_tools import make_client_tool_executor, register_client_tools


def test_proxy_executor_send_wait_result():
    waiters = {}

    def send_request(event):                      # simulate the client replying
        w, holder = waiters[event.call_id]
        holder["result"] = '{"ok": true}'
        w.set()
        return True

    ex = make_client_tool_executor("send_to_telegram", 5.0, send_request, waiters)
    assert ex({"text": "hi"}) == {"result": '{"ok": true}'}


def test_proxy_executor_no_client_errors():
    ex = make_client_tool_executor("t", 0.1, lambda e: False, {})
    assert "error" in ex({})


def test_proxy_executor_client_error_propagates():
    waiters = {}

    def send_request(event):
        w, holder = waiters[event.call_id]
        holder["error"] = "boom"
        w.set()
        return True

    ex = make_client_tool_executor("t", 5.0, send_request, waiters)
    assert ex({}) == {"error": "boom"}


class _Reg:
    def __init__(self):
        self.core = {}

    def register_core_tool(self, schema, executor, auto_approved=False):
        self.core[schema.name] = (schema, executor, auto_approved)


class _Srv:
    def __init__(self):
        self.client_tool_schemas = {}


def test_register_records_schema_and_registers_proxy():
    reg, srv = _Reg(), _Srv()
    names = register_client_tools(
        reg, srv,
        [{"name": "t1", "description": "d", "parameters": {}, "auto_approve": True},
         {"description": "no name"}],            # skipped (no name)
        send_request=lambda e: True, waiters={})
    assert names == ["t1"]
    schema, executor, auto = reg.core["t1"]
    assert auto is True and callable(executor)
    assert srv.client_tool_schemas["t1"]["name"] == "t1"
    assert "[client-provided]" in schema.description
