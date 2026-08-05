"""Shared client-tool registration helper (server/client_tools.py).

The proxy executor's send -> wait -> result flow (transport-agnostic), and the
registration that records schemas onto ``JaatoServer.client_tool_schemas``.
"""

from server.client_tools import (
    make_client_tool_executor, register_client_tools, decode_client_tool_result,
)
from shared.completion_processors import build_tool_call_ledger
from types import SimpleNamespace


def test_decode_client_tool_result():
    # JSON-encoded string -> native value (the symmetric decode).
    assert decode_client_tool_result('{"invoice_total": 250}') == {"invoice_total": 250}
    assert decode_client_tool_result('[1, 2]') == [1, 2]
    assert decode_client_tool_result('"hi"') == "hi"
    assert decode_client_tool_result('42') == 42
    # Empty (the error case, SDK sends result="" when error set) -> None.
    assert decode_client_tool_result("") is None
    assert decode_client_tool_result(None) is None
    # Malformed (client contract violation) -> raw passthrough, no crash.
    assert decode_client_tool_result("{not json") == "{not json"


def test_proxy_executor_returns_decoded_result_flat():
    # In production the transport decodes the JSON string into the holder
    # (decode_client_tool_result) before waking the executor; the executor
    # returns it VERBATIM (flat), symmetric with an in-process tool — NOT
    # wrapped under a "result" envelope.
    waiters = {}

    def send_request(event):                      # simulate the decoded reply
        w, holder = waiters[event.call_id]
        holder["result"] = {"invoice_total": 250, "status": "issued"}
        w.set()
        return True

    ex = make_client_tool_executor("issue_refund", 5.0, send_request, waiters)
    assert ex({"invoice_id": "INV-1007"}) == {"invoice_total": 250, "status": "issued"}


def test_host_tool_result_is_flat_in_the_ledger():
    # End-to-end shape: a decoded+flattened host-tool result lands in the
    # tool-call ledger as the client's flat dict — so a provenance check that
    # reads result.get('invoice_total') at the top level succeeds (the bug was
    # {"result": "<json-string>"} → .get() always None).
    flat = decode_client_tool_result('{"invoice_total": 250, "status": "issued"}')
    fc = SimpleNamespace(id="c1", name="issue_refund", args={"invoice_id": "INV-1007"})
    fr = SimpleNamespace(call_id="c1", result=flat, enrichment_metadata=None)
    history = [
        SimpleNamespace(parts=[SimpleNamespace(function_call=fc, function_response=None)]),
        SimpleNamespace(parts=[SimpleNamespace(function_call=None, function_response=fr)]),
    ]
    ledger = build_tool_call_ledger(history)
    entry = next(e for e in ledger if e["name"] == "issue_refund")
    assert entry["result"].get("invoice_total") == 250
    assert entry["result"].get("status") == "issued"


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
