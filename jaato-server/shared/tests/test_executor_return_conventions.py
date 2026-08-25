"""``ToolExecutor`` must not confuse two 2-tuple conventions.

TWO conventions shared one representation:

    (ok_bool,     payload_dict)   split_executor_result — 19 executors
    (result_dict, metadata_dict)  ToolExecutor side-channel — 4 producers

``ToolExecutor`` unwrapped ANY 2-tuple whose second element was a dict, so
``(False, receipt)`` was read as result=``False`` / metadata=``receipt``.  The
merge was skipped because ``False`` is not a dict, and the call returned
``(True, False)`` — flag inverted, payload GONE.  Both halves were affected:
``(True, {...})`` became ``(True, True)``.

The model saw ``{"result": false}`` for a failed tool, and ``is_error`` was
False, so BOTH consumer-side error checks reported success.

Found by the cascade-coordination probe: nine ``send_to_sibling`` receipts from
one daemon, eight resolvable names returning proper receipts (bare dicts, not
affected) and the one non-resolving name returning ``{"result": false}``.

The fix names the metadata convention (``WithMetadata``) so the bare tuple is
unambiguous.  Discriminating on ``isinstance(x[0], bool)`` would have
ARBITRATED the ambiguity rather than removed it.
"""

import pytest

from jaato_sdk.plugins.model_provider.types import WithMetadata
from shared.ai_tool_runner import ToolExecutor


def _run(fn):
    ex = ToolExecutor()
    ex._map["probe"] = fn
    return ex.execute("probe", {})


# ----------------------------------------------------------------------
# The (ok, payload) contract
# ----------------------------------------------------------------------

def test_a_domain_failure_keeps_its_flag_and_its_payload():
    """The exact shape that became ``(True, False)``."""
    payload = {"status": "no_such_sibling", "error": "no sibling named 'gone'"}
    ok, result = _run(lambda a: (False, payload))
    assert ok is False, "the ok flag must survive"
    assert result == payload, "the payload must survive"


def test_the_success_half_is_equally_affected():
    """``(True, {...})`` became ``(True, True)`` — payload gone, silently."""
    payload = {"status": "accepted", "bytes": 26}
    ok, result = _run(lambda a: (True, payload))
    assert ok is True
    assert result == payload


def test_the_model_never_sees_a_bare_boolean():
    """``{"result": false}`` is the fingerprint of the collapse.

    ``normalize_result_dict`` wraps a non-dict payload, so a bare ``False``
    reaching it produces a result dict whose only key is ``result`` — which
    ``tool_result_is_error`` cannot read, and which says nothing to the model.
    """
    _ok, result = _run(lambda a: (False, {"error": "nope"}))
    assert result is not False
    assert isinstance(result, dict) and "error" in result


# ----------------------------------------------------------------------
# The metadata convention — must still work
# ----------------------------------------------------------------------

def test_metadata_is_merged_into_the_result():
    ok, result = _run(
        lambda a: WithMetadata({"answer": 1}, {"continuation_id": "c1"}))
    assert ok is True
    assert result == {"answer": 1, "continuation_id": "c1"}


def test_metadata_reaches_the_level_the_session_reads():
    """The session reads these at ``executor_result[1]``, beside
    ``auto_backgrounded`` — so the merge target matters, not just the value."""
    _ok, result = _run(lambda a: WithMetadata(
        {"out": "x"}, {"continuation_id": "s1", "show_output": False}))
    assert result["continuation_id"] == "s1"
    assert result["show_output"] is False


def test_metadata_on_a_non_dict_result_does_not_crash():
    ok, result = _run(lambda a: WithMetadata("plain text", {"continuation_id": "c"}))
    assert ok is True and result == "plain text"


# ----------------------------------------------------------------------
# Everything else
# ----------------------------------------------------------------------

def test_a_bare_dict_is_unaffected():
    ok, result = _run(lambda a: {"status": "ok"})
    assert ok is True and result == {"status": "ok"}


def test_a_raised_exception_is_still_a_failure():
    def boom(a):
        raise RuntimeError("kaboom")
    ok, result = _run(boom)
    assert ok is False and "kaboom" in result["error"]


def test_the_four_shapes_are_all_distinguishable():
    """The property, not the four instances.

    Each shape must round-trip to its own meaning — that is what makes the
    conventions non-colliding rather than merely currently-correct.
    """
    cases = [
        ((False, {"e": 1}), (False, {"e": 1})),
        ((True, {"v": 2}), (True, {"v": 2})),
        (WithMetadata({"v": 3}, {"m": 4}), (True, {"v": 3, "m": 4})),
        ({"v": 5}, (True, {"v": 5})),
    ]
    for returned, expected in cases:
        assert _run(lambda a, r=returned: r) == expected, f"{returned!r}"


def test_no_producer_returns_the_old_bare_metadata_tuple():
    """A structural guard over the tree, not over the four sites I changed.

    A plugin reintroducing ``return (result, {...})`` silently rejoins the
    collision, and nothing would fail — the tuple is now read as
    ``(ok, payload)``, so its result becomes the ok flag.
    """
    import pathlib
    import re
    root = pathlib.Path("jaato-server/shared/plugins")
    offenders = []
    pattern = re.compile(r"return\s*\(\s*[A-Za-z_][\w\.]*\s*,\s*\{")
    for f in root.rglob("*.py"):
        if "/tests/" in str(f):
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{f}:{i}: {line.strip()}")
    assert offenders == [], (
        "bare (result, {metadata}) tuples found — use WithMetadata:\n"
        + "\n".join(offenders)
    )
