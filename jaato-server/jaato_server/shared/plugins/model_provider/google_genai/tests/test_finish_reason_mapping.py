"""Google's abnormal finish reasons must not read as success or tool use.

Part of #687.  Google reports why generation stopped in a
``candidate.finish_reason`` enum, and the converter mapped it with four
substring tests::

    if 'STOP' in reason:            -> STOP
    elif 'MAX' in reason or 'LENGTH' -> MAX_TOKENS
    elif 'SAFETY' in reason:        -> SAFETY
    elif 'TOOL' in reason or 'FUNCTION' in reason: -> TOOL_USE
    else:                           -> UNKNOWN

Two of those were wrong, in opposite directions.

**The TOOL/FUNCTION branch never matched a tool-use turn.**  Gemini
reports ``STOP`` for a turn that emits function calls -- it has no
tool-use finish reason at all.  What the branch did match was every
*error* whose name mentions tools: ``MALFORMED_FUNCTION_CALL`` (the
model emitted a call its own serialiser rejected),
``UNEXPECTED_TOOL_CALL``, ``TOO_MANY_TOOL_CALLS``.  Each became "the
model wants a tool run", which is a failed turn presenting as a
request -- the shape #745 closed for output-cap truncation, arriving
by a different route.

**The fall-through hid the rest.**  ``RECITATION``, ``BLOCKLIST``,
``PROHIBITED_CONTENT``, ``SPII`` and ``OTHER`` matched nothing and
resolved to ``UNKNOWN``, which is a SUCCESS value downstream.  A turn
the safety filter stopped read as a clean answer.

The fix maps by member NAME, and keeps ``UNKNOWN`` for a name the table
does not carry -- a reason we do not know is not a reason to guess.
"""

import pytest

from jaato_sdk.plugins.model_provider.types import FinishReason

from ..converters import (
    extract_finish_reason_from_response,
    finish_reason_from_sdk,
)


class _Candidate:
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, *reasons):
        self.candidates = [_Candidate(r) for r in reasons]


# ==================== The regression itself ====================


@pytest.mark.parametrize("name", [
    "MALFORMED_FUNCTION_CALL",
    "UNEXPECTED_TOOL_CALL",
    "TOO_MANY_TOOL_CALLS",
])
def test_a_tool_shaped_error_is_not_a_tool_use_turn(name):
    """The one assertion the substring mapping fails."""
    resolved = finish_reason_from_sdk(name)
    assert resolved is not FinishReason.TOOL_USE, (
        f"{name} is a generation FAILURE. Reading it as TOOL_USE makes "
        f"the session execute or nudge on a turn that did not produce a "
        f"call it can run."
    )
    assert resolved is FinishReason.ERROR


@pytest.mark.parametrize("name", [
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
])
def test_a_filtered_turn_is_not_a_clean_stop(name):
    """These all fell through to UNKNOWN, which is a success value."""
    assert finish_reason_from_sdk(name) is FinishReason.SAFETY


@pytest.mark.parametrize("name", ["OTHER", "LANGUAGE", "UNSUPPORTED_LANGUAGE"])
def test_a_named_failure_is_not_a_clean_stop(name):
    assert finish_reason_from_sdk(name) is FinishReason.ERROR


# ==================== The cases that must stay working ====================


def test_a_normal_stop_is_still_a_stop():
    assert finish_reason_from_sdk("STOP") is FinishReason.STOP


def test_the_output_cap_is_still_max_tokens():
    """#745 keys rewind-with-hint on this; it must not have moved."""
    assert finish_reason_from_sdk("MAX_TOKENS") is FinishReason.MAX_TOKENS


def test_an_unrecognised_name_is_still_unknown():
    """Google adds members faster than this table can.

    Guessing at an unfamiliar reason is how the substring mapping got
    ``MALFORMED_FUNCTION_CALL`` wrong in the first place.
    """
    assert finish_reason_from_sdk("SOME_FUTURE_REASON") is FinishReason.UNKNOWN


def test_nothing_reported_is_unknown():
    assert finish_reason_from_sdk(None) is FinishReason.UNKNOWN
    assert finish_reason_from_sdk("") is FinishReason.UNKNOWN


# ==================== The three shapes a reason arrives in ====================


def test_a_dotted_enum_repr_reduces_to_its_member_name():
    """``str()`` of an SDK enum is ``"FinishReason.STOP"``."""
    assert finish_reason_from_sdk("FinishReason.STOP") is FinishReason.STOP
    assert finish_reason_from_sdk(
        "FinishReason.MALFORMED_FUNCTION_CALL",
    ) is FinishReason.ERROR


def test_an_object_with_a_name_attribute_is_read_by_name():
    class _Enum:
        name = "RECITATION"

        def __str__(self):
            return "FinishReason.RECITATION"

    assert finish_reason_from_sdk(_Enum()) is FinishReason.SAFETY


def test_lowercase_is_folded():
    assert finish_reason_from_sdk("max_tokens") is FinishReason.MAX_TOKENS


# ==================== Batch and streaming agree ====================


@pytest.mark.parametrize("name", [
    "STOP", "MAX_TOKENS", "SAFETY", "RECITATION",
    "MALFORMED_FUNCTION_CALL", "OTHER", "SOME_FUTURE_REASON",
])
def test_the_batch_path_maps_identically(name):
    """The two paths carried two copies of the mapping, so two defects.

    ``extract_finish_reason_from_response`` now delegates, which is what
    keeps them from drifting apart again.
    """
    assert extract_finish_reason_from_response(_Response(name)) == \
        finish_reason_from_sdk(name)


def test_a_response_with_no_candidates_is_unknown():
    assert extract_finish_reason_from_response(_Response()) is \
        FinishReason.UNKNOWN
    assert extract_finish_reason_from_response(None) is FinishReason.UNKNOWN


def test_a_candidate_reporting_nothing_falls_to_the_next_one():
    """A ``None`` reason is not a reason; keep looking."""
    assert extract_finish_reason_from_response(None, ) is FinishReason.UNKNOWN
    assert extract_finish_reason_from_response(
        _Response(None, "MAX_TOKENS"),
    ) is FinishReason.MAX_TOKENS
