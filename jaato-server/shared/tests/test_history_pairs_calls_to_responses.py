"""The call side of ``request_history`` carried no identifier at all.

``CommandRouter._serialize_part`` emitted::

    function_call      -> {"type", "name", "args"}
    function_response  -> {"type", "name", "call_id", "response", ...}

``build_tool_call_ledger`` pairs the two BY IDENTIFIER, so the field the
pairing depends on survived on only one of the two Parts once it crossed the
wire.  A client reading ``request_history`` could not rebuild the ledger that
completion processors receive as ``context.tool_calls``.

WHY PAIRING BY NAME-IN-ORDER IS NOT A WORKAROUND.  It fails in exactly the
case that matters: an agent calls a tool, it errors, it retries -- two calls
and two responses sharing a name.  Positional pairing credits the retry's
success to the call that failed, so a grader reports a fabricated artefact as
verified.  That silently INVERTS a correctness check rather than degrading it.

AND THE OBVIOUS ONE-LINE FIX IS A SILENT NO-OP.  The identifier is ``fc.id``
on the call and ``fr.call_id`` on the response::

    FunctionCall fields: ['id', 'name', 'args']
    ToolResult   fields: ['call_id', 'name', 'result', 'is_error', ...]

Mirroring the response branch -- ``getattr(fc, "call_id", "")`` -- emits the
empty string forever.  Every test below would still pass on `` == ""`` if it
only checked that the KEY exists, which is why they check the VALUE and check
it against the response's.

This is the third defect of the same shape in this one function: tool results
sent as reprs (#600-#610), and now the call identifier dropped.  Each time a
``getattr``/``hasattr`` fallback made a wrong or missing attribute produce
something that looked like a value.
"""

from jaato_sdk.plugins.model_provider.types import FunctionCall, Part, ToolResult
from server.command_router import CommandRouter


def _call(fc):
    return CommandRouter._serialize_part(Part(function_call=fc))


def _resp(tr):
    return CommandRouter._serialize_part(Part(function_response=tr))


def test_the_call_carries_its_identifier():
    out = _call(FunctionCall(id="call_abc123", name="write_file",
                             args={"path": "a.py"}))

    assert out["call_id"] == "call_abc123", (
        "the call side must carry the identifier the pairing depends on; "
        f"got {out.get('call_id')!r}"
    )


def test_the_identifier_is_read_from_id_not_from_call_id():
    """The no-op fix, pinned as the thing that must not ship.

    ``FunctionCall`` has ``id``.  A fix written as ``getattr(fc, 'call_id')``
    would add the key and leave it empty -- indistinguishable from "this
    provider did not supply an id" at every consumer.
    """
    fc = FunctionCall(id="call_xyz", name="t", args={})
    assert not hasattr(fc, "call_id"), (
        "FunctionCall grew a call_id field; re-check which one _serialize_part "
        "should read, and whether they can disagree"
    )
    assert _call(fc)["call_id"] == "call_xyz"


def test_a_call_and_its_response_agree_on_the_identifier():
    """The property that makes pairing possible, asserted across BOTH branches.

    Checking the call side alone would pass with any non-empty string.
    """
    cid = "call_7f3a"
    call = _call(FunctionCall(id=cid, name="write_file", args={"path": "a.py"}))
    resp = _resp(ToolResult(call_id=cid, name="write_file", result={"ok": True}))

    assert call["call_id"] == resp["call_id"] != ""


def test_a_retry_is_distinguishable_from_the_call_it_retried():
    """The failure that name-order pairing inverts.

    Two calls, same name, one failed then retried.  Pairing by identifier
    attributes each result to the call that produced it; pairing by position
    credits the retry's success to the failed call.
    """
    failed = _call(FunctionCall(id="call_1", name="write_file",
                                args={"path": "a.py"}))
    retry = _call(FunctionCall(id="call_2", name="write_file",
                               args={"path": "a.py"}))
    r_failed = _resp(ToolResult(call_id="call_1", name="write_file",
                                result={"error": "EACCES"}, is_error=True))
    r_retry = _resp(ToolResult(call_id="call_2", name="write_file",
                               result={"ok": True}))

    assert failed["call_id"] != retry["call_id"], (
        "two calls of the same tool are indistinguishable on the wire; a "
        "grader cannot tell a retry from the call it retried"
    )

    paired = {c["call_id"]: r for c, r in
              ((failed, r_failed), (retry, r_retry))}
    assert paired["call_id" and failed["call_id"]]["is_error"] is True
    assert paired[retry["call_id"]]["is_error"] is False


def test_a_provider_without_ids_yields_empty_not_a_crash():
    """Absent stays absent.

    Not every provider supplies tool-call ids (the prose-emulated
    ``prose_tool_calls`` path hashes its own).  An empty string here means
    "this provider gave no id", and a consumer must be able to tell that from
    a populated one -- which is the whole reason the value is asserted above
    rather than the key's presence.
    """
    out = _call(FunctionCall(id="", name="t", args={}))
    assert out["call_id"] == ""
    assert out["name"] == "t"


def test_the_wire_shape_is_symmetric_on_the_pairing_field():
    """Both branches expose the identifier under the SAME key.

    They read different attributes (``id`` / ``call_id``) and that asymmetry
    is real, but it must not reach the client -- a consumer pairing two dicts
    should not need to know which side it is holding.
    """
    call = _call(FunctionCall(id="c", name="t", args={}))
    resp = _resp(ToolResult(call_id="c", name="t", result={}))

    assert "call_id" in call and "call_id" in resp
    assert set(call) == {"type", "name", "args", "call_id"}
