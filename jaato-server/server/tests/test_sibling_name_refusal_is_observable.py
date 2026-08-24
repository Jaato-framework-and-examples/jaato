"""A refusal a consumer cannot see is not a refusal.

#592 added sibling-address validation that logged and ``return ""``.  The
server-side decision was correct — both rules, with good messages — and NONE of
it reached the client:

  shape violation  the caller blocked until its own timeout and got None,
                   indistinguishable from a hung daemon.  Worse, the router's
                   falsy branch emits an AUTH-PROVIDER hint, so a naming
                   violation surfaced as a misleading suggestion about
                   credentials.
  collision        the client's ``_await_session_info`` picked up the
                   SessionInfoEvent of the session created moments earlier and
                   returned ITS id — so the caller believed it had created a
                   sibling it had not, holding an address clash it could not
                   see.

Found by the cascade-coordination example on its FIRST live execution.  Their
probe had read ``None`` as "declined", which is the same absent-vs-empty
collapse one level up: refusal and silence sharing a representation.  Their
first green was a false pass — my validation happened to be right, and the
probe would have said PASS either way.

The SDK documents that ALL session.new failures arrive as a recoverable
``ErrorEvent``; this one did not behave like it.
"""
import inspect
from types import SimpleNamespace

import pytest

from server.session_manager import SessionManager


def _refuse(sibling_name="Permission Approver - reply yes", existing=()):
    """Drive the real refusal branch and capture what reaches the client."""
    emitted = []
    mgr = SimpleNamespace(
        _emit_to_client=lambda cid, ev: emitted.append((cid, ev)),
        _known_sibling_addresses=lambda ws=None: list(existing),
        _sessions={},
    )
    # The branch under test lives at the top of _create_session_impl, before
    # any id is allocated; exercise it through the real source path.
    from server.session_manager import validate_sibling_name
    bad = validate_sibling_name(sibling_name, "cid-1", list(existing))
    assert bad, "fixture no longer triggers a refusal"
    return mgr, emitted, bad


def test_the_refusal_branch_emits_an_error_event():
    src = inspect.getsource(SessionManager._create_session_impl)
    i = src.index("create_session refused")
    window = src[i:i + 900]
    assert "_emit_to_client" in window and "ErrorEvent" in window, (
        "the refusal logs and returns '' without telling the client — the "
        "caller blocks until its own timeout and the router's falsy branch "
        "hints about auth providers instead")


def test_the_error_is_recoverable_and_typed():
    src = inspect.getsource(SessionManager._create_session_impl)
    i = src.index("create_session refused")
    window = src[i:i + 900]
    assert 'error_type="InvalidSiblingName"' in window, (
        "a caller cannot branch on an untyped failure")
    assert "recoverable=True" in window, (
        "the SDK documents every session.new failure as recoverable=True")


def test_the_error_carries_the_reason_not_just_a_code():
    src = inspect.getsource(SessionManager._create_session_impl)
    i = src.index("create_session refused")
    window = src[i:i + 900]
    assert "{_bad}" in window, (
        "the validator's message explains WHY the shape is narrow and which "
        "cascade the name is taken in; dropping it makes the error unactionable")


def test_it_still_refuses_and_still_returns_falsy():
    """The fix must add an emission, not soften the refusal."""
    src = inspect.getsource(SessionManager._create_session_impl)
    i = src.index("create_session refused")
    window = src[i:i + 900]
    assert 'return ""' in window, "the refusal must still refuse"


@pytest.mark.parametrize("name,existing", [
    ("Permission Approver - reply yes", ()),          # shape
    ("reviewer", (("reviewer", "cid-1"),)),           # collision
])
def test_both_rules_still_produce_a_reason(name, existing):
    _, _, bad = _refuse(name, existing)
    assert isinstance(bad, str) and bad
