"""IPCRecoveryClient must not fall behind IPCClient.

``IPCRecoveryClient`` wraps ``IPCClient`` by HAND -- it is not a subclass and
has no ``__getattr__`` delegation -- so every method added to IPCClient has to
be mirrored deliberately.  Nothing enforced that, and it drifted:

    commit e2ced167 (2026-08-07) "IPC verbs for cascade budgets" added BOTH
    ``payload=`` on ``IPCClient.execute_command`` AND
    ``cascade_budget_set/get/clear``, and never touched recovery.py.

Fifteen days later the first caller to need them -- an SDK-only cascade whose
driver must survive a daemon restart, so IPCRecoveryClient is load-bearing --
hit an AttributeError on the budget call and a TypeError on every
``session.wake``.  ``cascade_events`` had drifted the same way, unnoticed,
because nothing had needed it yet.

This test is the mechanism that prose about "keep the wrapper in sync" was
not.  When it fails, the fix is one of exactly two things:

  * forward the new method on IPCRecoveryClient, or
  * add it to ``INTENTIONALLY_ABSENT`` below WITH a reason.

That list is therefore the record of what is deliberate rather than forgotten.
"""
import inspect

import pytest

from jaato_sdk import IPCClient, IPCRecoveryClient


# Members of IPCClient that IPCRecoveryClient deliberately does NOT expose.
# Each entry needs a reason: an unexplained entry is indistinguishable from
# the drift this test exists to catch.
INTENTIONALLY_ABSENT = {
    "connection_state":
        "recovery owns a richer state machine and exposes it as `.state` "
        "(DISCONNECTED/CONNECTING/CONNECTED/RECONNECTING/CLOSED)",
    "supports_reconnection":
        "recovery IS the reconnection layer; the question is meaningless on "
        "it, and `.state` / `.is_reconnecting` answer what callers want",
    "MIN_PROTOCOL_VERSION":
        "protocol floor is negotiated by the inner IPCClient; recovery takes "
        "`min_protocol_version` as a constructor argument instead",
}


def _public(cls):
    return {name for name in dir(cls) if not name.startswith("_")}


def test_recovery_client_exposes_everything_ipcclient_does():
    """Every public IPCClient member is forwarded or explicitly excused."""
    missing = _public(IPCClient) - _public(IPCRecoveryClient)
    undocumented = sorted(missing - set(INTENTIONALLY_ABSENT))

    assert not undocumented, (
        "IPCRecoveryClient has fallen behind IPCClient: "
        f"{undocumented}.\n"
        "Forward each on IPCRecoveryClient, or add it to "
        "INTENTIONALLY_ABSENT with a reason."
    )


def test_exclusion_list_has_no_stale_entries():
    """An excuse for something that IS forwarded is itself drift."""
    stale = sorted(set(INTENTIONALLY_ABSENT) & _public(IPCRecoveryClient))
    assert not stale, (
        f"INTENTIONALLY_ABSENT excuses members that recovery now exposes: "
        f"{stale}. Remove them so the list keeps meaning something."
    )


def test_exclusion_list_only_names_real_ipcclient_members():
    """Guard against typos and against excuses outliving their method."""
    unknown = sorted(set(INTENTIONALLY_ABSENT) - _public(IPCClient))
    assert not unknown, (
        f"INTENTIONALLY_ABSENT names members IPCClient does not have: "
        f"{unknown}"
    )


@pytest.mark.parametrize("name", sorted(
    _public(IPCClient) & _public(IPCRecoveryClient)
))
def test_forwarded_methods_keep_a_compatible_signature(name):
    """Same name is not enough -- the parameters have to match too.

    The `session.wake` half of the drift was exactly this: recovery HAD
    execute_command, just without `payload=`, so callers got a TypeError
    rather than an AttributeError.
    """
    original = getattr(IPCClient, name)
    forwarded = getattr(IPCRecoveryClient, name)
    if not (inspect.isfunction(original) and inspect.isfunction(forwarded)):
        return  # properties / constants carry no signature to compare

    want = inspect.signature(original).parameters
    got = inspect.signature(forwarded).parameters

    dropped = sorted(set(want) - set(got))
    assert not dropped, (
        f"IPCRecoveryClient.{name} drops parameter(s) {dropped} that "
        f"IPCClient.{name} accepts -- callers get a TypeError, not a clear "
        f"'not supported'."
    )
