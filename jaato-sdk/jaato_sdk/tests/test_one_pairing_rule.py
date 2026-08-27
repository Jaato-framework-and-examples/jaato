"""The pairing rule exists ONCE, and both carriers agree on its answer.

The SDK published the ledger entry TYPE and no way to obtain a ledger, so a
consumer wanting one wrote its own pairing — and every such copy has to
re-derive that pairing is by IDENTIFIER, not by name-in-order.  Copies of a
rule rot independently unless something executes the comparison.

``build_ledger`` is that one rule.  ``shared.completion_processors.
build_tool_call_ledger`` is now a thin alias of it, so the server and every
SDK consumer answer the question the same way by construction rather than by
agreement.

THE TWO CARRIERS DIFFER ONLY IN READING:

    in-process   FunctionCall.id / ToolResult.call_id  (ASYMMETRIC — and
                 reading ``call_id`` off a FunctionCall silently yields
                 nothing, which is how the wire came to carry no identifier
                 at all)
    wire         both sides expose ``call_id``; the response body is under
                 ``response``; enrichment is not transported

The equivalence test below is the one that makes "one rule" checkable: build
the same conversation both ways and demand the same ledger.
"""

from __future__ import annotations

import pytest

from jaato_sdk.completion_processors import ToolCallEntry, build_ledger
from jaato_sdk.plugins.model_provider.types import (
    FunctionCall, Message, Part, Role, ToolResult,
)


# --------------------------------------------------------------- builders

def _obj_history():
    """The retry conversation, as in-process objects."""
    return [
        Message(role=Role.MODEL, parts=[
            Part(function_call=FunctionCall(
                id="call_1", name="write_file", args={"path": "a.py"})),
        ]),
        Message(role=Role.USER, parts=[
            Part(function_response=ToolResult(
                call_id="call_1", name="write_file",
                result={"error": "EACCES"}, is_error=True)),
        ]),
        Message(role=Role.MODEL, parts=[
            Part(function_call=FunctionCall(
                id="call_2", name="write_file", args={"path": "a.py"})),
        ]),
        Message(role=Role.USER, parts=[
            Part(function_response=ToolResult(
                call_id="call_2", name="write_file", result={"ok": True})),
        ]),
    ]


def _wire_history():
    """The same conversation, as ``CommandRouter._serialize_part`` emits it."""
    return [
        {"role": "model", "parts": [
            {"type": "function_call", "name": "write_file",
             "args": {"path": "a.py"}, "call_id": "call_1"},
        ]},
        {"role": "user", "parts": [
            {"type": "function_response", "name": "write_file",
             "call_id": "call_1", "response": {"error": "EACCES"},
             "is_error": True, "untrusted": False, "untrusted_source": None},
        ]},
        {"role": "model", "parts": [
            {"type": "function_call", "name": "write_file",
             "args": {"path": "a.py"}, "call_id": "call_2"},
        ]},
        {"role": "user", "parts": [
            {"type": "function_response", "name": "write_file",
             "call_id": "call_2", "response": {"ok": True},
             "is_error": False, "untrusted": False, "untrusted_source": None},
        ]},
    ]


# ------------------------------------------------------------ equivalence

def test_both_carriers_produce_the_same_ledger():
    """The assertion that makes "one rule" a fact rather than an intention.

    If the two ever diverge, a grader reading serialized history and a
    completion processor reading live history disagree about the same
    conversation — which is the bug this consolidation exists to prevent, and
    it would be invisible from either side alone.
    """
    assert build_ledger(_obj_history()) == build_ledger(_wire_history())


def test_the_server_builder_is_the_same_rule():
    """``shared.build_tool_call_ledger`` must not drift back into a copy."""
    from shared.completion_processors import build_tool_call_ledger

    assert build_tool_call_ledger(_obj_history()) == build_ledger(_obj_history())


# ----------------------------------------------------- the case that matters

@pytest.mark.parametrize("history,label", [
    (_obj_history(), "in-process"),
    (_wire_history(), "wire"),
])
def test_a_retry_does_not_inherit_the_failure_or_lend_its_success(history, label):
    """Two calls, one name, opposite outcomes.

    Name-in-order pairing credits the retry's success to the call that
    failed.  A grader built on that reports a fabricated artefact as
    verified — the verdict is inverted, not merely weakened.
    """
    ledger = build_ledger(history)

    assert len(ledger) == 2, label
    first, second = ledger
    assert first["call_id"] == "call_1" and first["success"] is False, label
    assert second["call_id"] == "call_2" and second["success"] is True, label
    assert first["result"] == {"error": "EACCES"}, label


# -------------------------------------------------------------- edge cases

@pytest.mark.parametrize("label,history", [
    ("in-process", [Message(role=Role.MODEL, parts=[
        Part(function_call=FunctionCall(id="c9", name="pending", args={}))])]),
    ("wire", [{"role": "model", "parts": [
        {"type": "function_call", "name": "pending", "args": {},
         "call_id": "c9"}]}]),
])
def test_an_unanswered_call_is_emitted_not_dropped(label, history):
    """A terminal turn's pending calls must still appear.

    Dropping them would make a validator's "claimed in the payload but never
    successfully called" check find nothing to look at — passing by absence.
    """
    ledger = build_ledger(history)

    assert len(ledger) == 1, label
    assert ledger[0]["success"] is False
    assert ledger[0]["result"] == {"error": "no_response"}


def test_a_non_dict_result_is_wrapped_on_both_carriers():
    """So ``"error" not in result`` applies rather than raising on a string."""
    obj = [Message(role=Role.MODEL, parts=[
               Part(function_call=FunctionCall(id="c", name="t", args={}))]),
           Message(role=Role.USER, parts=[
               Part(function_response=ToolResult(
                   call_id="c", name="t", result="just a string"))])]
    wire = [{"role": "model", "parts": [
                {"type": "function_call", "name": "t", "args": {},
                 "call_id": "c"}]},
            {"role": "user", "parts": [
                {"type": "function_response", "name": "t", "call_id": "c",
                 "response": "just a string", "is_error": False}]}]

    assert build_ledger(obj) == build_ledger(wire)
    assert build_ledger(wire)[0]["result"] == {"result": "just a string"}
    assert build_ledger(wire)[0]["success"] is True


def test_non_dict_args_are_preserved_under_raw():
    wire = [{"role": "model", "parts": [
        {"type": "function_call", "name": "t", "args": "not-a-dict",
         "call_id": "c"}]}]

    assert build_ledger(wire)[0]["args"] == {"_raw": "not-a-dict"}


def test_success_ignores_the_wires_is_error_flag():
    """One success rule, not two that usually agree.

    The wire carries an explicit ``is_error``; the ledger derives success
    from ``"error" not in result`` on both carriers.  Consulting the flag
    would make the two carriers answerable differently for the same
    conversation the moment a provider set it inconsistently.
    """
    wire = [{"role": "model", "parts": [
                {"type": "function_call", "name": "t", "args": {},
                 "call_id": "c"}]},
            {"role": "user", "parts": [
                {"type": "function_response", "name": "t", "call_id": "c",
                 # flag says error; body says otherwise
                 "response": {"ok": True}, "is_error": True}]}]

    assert build_ledger(wire)[0]["success"] is True


def test_enrichment_is_carried_in_process_and_absent_on_the_wire():
    """The one place the carriers legitimately differ — stated, not hidden.

    Enrichment is in-memory only.  ``None`` on the wire means "not
    transported", which a consumer cannot distinguish from "the tool
    produced none" — worth knowing before building a check on it.
    """
    meta = {"lsp": {"total_errors": 2}}
    obj = [Message(role=Role.MODEL, parts=[
               Part(function_call=FunctionCall(id="c", name="t", args={}))]),
           Message(role=Role.USER, parts=[
               Part(function_response=ToolResult(
                   call_id="c", name="t", result={"ok": True},
                   enrichment_metadata=meta))])]

    assert build_ledger(obj)[0]["enrichment_metadata"] == meta


def test_the_published_type_matches_what_is_emitted():
    """The type must describe the dict consumers actually receive.

    ``enrichment_metadata`` was emitted by the builder and documented in its
    docstring from the start, and was missing from ``ToolCallEntry`` — so the
    published type described a NARROWER dict than the real one, and anyone
    typing against it saw no reason the key existed.  Same shape as an
    annotation promising a value the code cannot produce: the mismatch is
    invisible until someone reads the field.
    """
    entry = build_ledger(_wire_history())[0]

    assert set(entry) == set(ToolCallEntry.__annotations__), (
        f"emitted {sorted(set(entry) - set(ToolCallEntry.__annotations__))} "
        f"not in the type; type declares "
        f"{sorted(set(ToolCallEntry.__annotations__) - set(entry))} not emitted"
    )


def test_mixed_carriers_in_one_history_do_not_confuse_the_reader():
    """Each message is read independently, so a mixed list is well-defined.

    Not a shape production produces — but a consumer concatenating live and
    restored history would produce it, and silently mispairing there would be
    very hard to find.
    """
    mixed = [_obj_history()[0], _wire_history()[1]]

    ledger = build_ledger(mixed)

    assert len(ledger) == 1
    assert ledger[0]["call_id"] == "call_1"
    assert ledger[0]["success"] is False
