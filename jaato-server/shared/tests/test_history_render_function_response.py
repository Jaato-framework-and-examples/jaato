"""``request_history`` must send tool results as DATA, not as a repr.

``CommandRouter._serialize_part`` rendered a function-response part as::

    "response": fr.response if hasattr(fr, 'response') else str(fr)

``ToolResult`` has ``result``; it has never had ``response``.  So the
``hasattr`` was always False and EVERY tool response in ``request_history``
was sent as ``str(fr)`` — the dataclass repr.  A client could not read a tool
result structurally at all: to recover ``is_error``, or the result dict, or
the untrusted mark, it had to parse a Python repr.  A large result was
stringified whole into the history payload.

The ``hasattr`` guard is what hid it. A WRONG attribute name and an ABSENT
one are indistinguishable to ``hasattr``, and the fallback produced something
that looked like a value — so nothing ever failed.

Found while tracing the cascade-coordination probe's Finding 3: the probe was
reading `untrusted` out of a repr string, which is what made it visible.
"""

from jaato_sdk.plugins.model_provider.types import Part, ToolResult
from server.command_router import CommandRouter


ROSTER = {"status": "ok", "siblings": [{"sibling_name": "hostile"}]}


def _render(tr):
    return CommandRouter._serialize_part(Part(function_response=tr))


def test_the_result_is_sent_as_data():
    out = _render(ToolResult(call_id="c1", name="list_siblings", result=ROSTER))
    assert out["response"] == ROSTER, "must be the dict, not its repr"
    assert not isinstance(out["response"], str)


def test_a_repr_never_reaches_the_client():
    """The specific corruption, pinned by its fingerprint."""
    out = _render(ToolResult(call_id="c1", name="list_siblings", result=ROSTER))
    assert "ToolResult(" not in str(out["response"])


def test_error_state_is_readable_without_parsing():
    out = _render(ToolResult(
        call_id="c1", name="list_siblings", result={"error": "nope"},
        is_error=True,
    ))
    assert out["is_error"] is True
    assert out["call_id"] == "c1"


def test_the_untrusted_boundary_is_readable_without_parsing():
    """A client deciding how to display or re-feed a result needs this.

    Recovering it from a repr means a client that renders sibling- or
    web-authored text has no structured way to know it is attacker-authored.
    """
    out = _render(ToolResult(
        call_id="c1", name="list_siblings", result=ROSTER,
        untrusted=True, untrusted_source="list_siblings",
    ))
    assert out["untrusted"] is True
    assert out["untrusted_source"] == "list_siblings"


def test_unmarked_results_report_trusted_explicitly():
    out = _render(ToolResult(call_id="c1", name="cli", result={"ok": True}))
    assert out["untrusted"] is False
    assert out["untrusted_source"] is None


def test_a_string_result_still_passes_through():
    """Tools that return bare strings must not regress into a repr."""
    out = _render(ToolResult(call_id="c1", name="cli", result="hello"))
    assert out["response"] == "hello"
