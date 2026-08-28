"""``tool_result_status`` and the ``ToolCallEndEvent.result_status`` field.

A consumer watching the event stream could tell that a tool call failed, and
could tell that a "successful" call carried an error body
(``is_error_result``).  It could not tell WHICH failure it was, because the
only place the distinction survived was ``error_message`` — prose.

That distinction is load-bearing for a cascade driver.  ``send_to_sibling``
returns ``accepted`` / ``queued`` / ``refused`` / ``sibling_cold`` /
``no_such_sibling``; ``refused`` is backpressure (the peer is busy — let it
work) while ``sibling_cold`` means the peer will never wake and the loop is
over.  Both arrive as ``success=False``.  A driver branching between "keep
going" and "the run has ended" was matching on a sentence.

The framework does not own the vocabulary — a tool may define any statuses it
likes — so the helper copies the string and interprets nothing.
"""

import pytest

from jaato_sdk.events import ToolCallEndEvent
from jaato_sdk.event_payloads import ToolCallCompletedPayload
from jaato_sdk.plugins.model_provider.types import (
    tool_result_is_error,
    tool_result_status,
)


@pytest.mark.parametrize("result,expected", [
    ({"status": "accepted"}, "accepted"),
    ({"status": "refused", "error": "cap reached"}, "refused"),
    ({"status": "sibling_cold"}, "sibling_cold"),
    ({"error": "boom"}, None),          # a failure that declares no status
    ({}, None),
    ({"status": 200}, None),            # not a string — not a status
    ({"status": None}, None),
    ("a string result", None),          # not a dict
    (None, None),
])
def test_tool_result_status(result, expected):
    assert tool_result_status(result) is expected or \
        tool_result_status(result) == expected


def test_status_and_is_error_answer_different_questions():
    """``refused`` is a failure WITH a status; a bare ``{"error"}`` is a
    failure without one.  Neither helper can stand in for the other."""
    refused = {"status": "refused", "error": "SIBLING_PENDING_CAP"}
    assert tool_result_is_error(refused) is True
    assert tool_result_status(refused) == "refused"

    accepted = {"status": "accepted", "sibling_name": "subconscient"}
    assert tool_result_is_error(accepted) is False
    assert tool_result_status(accepted) == "accepted"


def test_the_two_sibling_failures_are_distinguishable():
    """The case the field exists for.

    Pre-fix both of these produced ``success=False`` and nothing else a
    consumer could branch on.
    """
    backpressure = ToolCallEndEvent(
        tool_name="send_to_sibling", success=False,
        result_status=tool_result_status({"status": "refused"}),
        error_message="peer has 20 pending messages. Let it work.",
    )
    fatal = ToolCallEndEvent(
        tool_name="send_to_sibling", success=False,
        result_status=tool_result_status({"status": "sibling_cold"}),
        error_message="sibling is not loaded.",
    )

    assert backpressure.success == fatal.success       # indistinguishable here
    assert backpressure.result_status != fatal.result_status


def test_result_status_round_trips_on_the_wire():
    event = ToolCallEndEvent(tool_name="send_to_sibling",
                             result_status="sibling_cold")
    dumped = event.model_dump(mode="json")
    assert dumped["result_status"] == "sibling_cold"
    assert ToolCallEndEvent(**dumped).result_status == "sibling_cold"


def test_result_status_defaults_to_none():
    """Silence, not an outcome.  Most tools declare no status at all, and a
    consumer must not read ``None`` as a verdict."""
    assert ToolCallEndEvent().result_status is None


def test_payload_type_carries_result_status():
    assert "result_status" in ToolCallCompletedPayload.__annotations__
