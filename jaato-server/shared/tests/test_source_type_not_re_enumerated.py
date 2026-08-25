"""No prose may hold a partial copy of ``SourceType``.

When ``SourceType.SIBLING`` shipped, the enum and its tier frozensets were
updated and THREE DOCSTRINGS were not:

    runner/rpc.py         'one of "parent", "child", "user", "system", "event"'
    session_manager.py    '(USER/PARENT/SYSTEM/EVENT/CHILD)'
    runner_rpc_client.py  'one of "parent", "child", "user", "system", "event"'

The runtime was never wrong — validation is ``SourceType(value)`` and the
rejection message enumerates from the enum — so ``sibling`` was accepted the
whole time.  Only the DOCUMENTATION said otherwise, which is its own hazard:
a caller reading "one of five" concludes the sixth is unsupported and works
around a restriction that does not exist.

Same shape as the rest of this arc — a second copy of a fact that has an owner
— but a rarer flavour: the copy that drifted was the one humans read, so
nothing failed and nothing could.

This guard allows either style (list them ALL, or reference the enum) and
forbids only the stale-subset state in between.
"""

import inspect
import re

import pytest

from shared.message_queue import SourceType


def _surfaces():
    from server.runner.rpc import RunnerRPC
    from server.runner_rpc_client import RunnerRPCClient
    from server.session_manager import SessionManager
    return [
        ("runner RPC handler",
         RunnerRPC._handle_session_inject_prompt),
        ("SessionManager.inject_prompt_to_session",
         SessionManager.inject_prompt_to_session),
        ("RunnerRPCClient.session_inject_prompt_threadsafe",
         RunnerRPCClient.session_inject_prompt_threadsafe),
    ]


@pytest.mark.parametrize("label,fn", _surfaces(), ids=[s[0] for s in _surfaces()])
def test_no_docstring_holds_a_partial_copy_of_the_enum(label, fn):
    """List every member, or none — never a subset that can go stale."""
    doc = (inspect.getdoc(fn) or "").lower()
    values = {s.value for s in SourceType}

    # Count mentions in an ENUMERATING shape: quoted, or slash/paren
    # separated.  "the user's message" is prose, not a list.
    listed = {
        v for v in values
        if re.search(rf'["\']{v}["\']', doc) or re.search(rf'\b{v}[/)]', doc)
    }

    # AN ENUMERATION IS TWO OR MORE.  A single mention is a DEFAULT
    # (`defaults to "user"`) or an EXAMPLE (`e.g. "user"`) — both correct and
    # both must stay legal.  The first version of this guard treated any
    # mention as a list and failed docstrings that were already fixed: a
    # check whose signal could not distinguish an example from an
    # enumeration, which is precisely the defect it exists to catch.
    #
    # A stale copy is ≥2 by construction — the three that drifted listed five.
    if len(listed) < 2:
        return
    assert listed == values, (
        f"{label} enumerates {sorted(listed)} but the enum has "
        f"{sorted(values)} — missing {sorted(values - listed)}. "
        f"Either list them all or reference SourceType."
    )


def test_the_runtime_rejection_message_enumerates_from_the_enum():
    """The one place a list IS correct: built from the enum, not typed out."""
    from server.runner.rpc import RunnerRPC
    src = inspect.getsource(RunnerRPC._handle_session_inject_prompt)
    assert "sorted(s.value for s in SourceType)" in src, (
        "the rejection message must derive its list, not hardcode one"
    )


def test_every_enum_member_decodes_off_the_wire():
    """The behavioural claim the docstrings were wrong about."""
    for member in SourceType:
        assert SourceType(member.value) is member


def test_sibling_specifically_decodes():
    """The member whose absence from the prose prompted this."""
    assert SourceType("sibling") is SourceType.SIBLING
