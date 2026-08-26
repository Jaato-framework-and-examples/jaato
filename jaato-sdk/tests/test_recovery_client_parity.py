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
from jaato_sdk.events import ClientType


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
    "MIN_CORRELATION_PROTOCOL":
        "implementation detail of the INNER IPCClient's session.new "
        "correlation; recovery delegates create_session to it and never "
        "consults the constant itself",
    "MIN_PROTOCOL_VERSION":
        "protocol floor is negotiated by the inner IPCClient; recovery takes "
        "`min_protocol_version` as a constructor argument instead",
    "MIN_INJECT_RESULT_PROTOCOL":
        "implementation detail of the INNER IPCClient's inject_prompt "
        "delivery reporting; recovery delegates inject_prompt to it and "
        "never consults the constant itself -- same reasoning as "
        "MIN_CORRELATION_PROTOCOL above.  The METHOD is forwarded with a "
        "matching signature (asserted in "
        "jaato_sdk/tests/test_sdk_parity_methods.py), which is the part a "
        "caller can observe",
}


# Constructor arguments of IPCClient that IPCRecoveryClient deliberately does
# NOT accept.  Separate list from INTENTIONALLY_ABSENT because ctor args are a
# DIFFERENT drift axis -- and one this test was blind to until it bit someone.
INTENTIONALLY_ABSENT_CTOR_ARGS = {
    "self": "not an argument",
}


def _public(cls):
    return {name for name in dir(cls) if not name.startswith("_")}


def _ctor_args(cls):
    return set(inspect.signature(cls.__init__).parameters)


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


# ---------------------------------------------------------------------------
# Constructor-argument parity -- the axis this test used to miss entirely.
#
# ``_public`` compares dir(), i.e. METHODS and attributes.  Constructor kwargs
# are neither, so ``config_root`` / ``apparmor`` / ``autostart_timeout`` were
# invisible to every check above.  They stayed IPCClient-only for months and
# were found the same way the payload= gap was: by a user hitting it.
#
# ``config_root`` was the one that mattered.  A recovery-driven session got
# ``config_root=None`` with NO route to set it (unlike ``apparmor``, which a
# profile can request via ``apparmor: true``).  AppArmor composition then
# silently dropped every plugin rule gated on config_root and file_edit lost
# its backup subtree -- while the profile still loaded and still logged
# "runner confined (enforce)".  A confinement profile missing part of its
# intended policy is indistinguishable from a complete one in the log, which
# makes this strictly worse than an AttributeError.


def test_recovery_client_accepts_every_ipcclient_ctor_arg():
    """Every IPCClient ctor arg is accepted by recovery or explicitly excused."""
    missing = _ctor_args(IPCClient) - _ctor_args(IPCRecoveryClient)
    undocumented = sorted(missing - set(INTENTIONALLY_ABSENT_CTOR_ARGS))

    assert not undocumented, (
        "IPCRecoveryClient cannot be constructed with: "
        f"{undocumented}.\n"
        "Recovery clients are load-bearing for anything that must survive a "
        "daemon restart, so an IPCClient-only ctor arg is unreachable for "
        "those callers -- there is no escape hatch unless the arg happens to "
        "have a profile-level equivalent.\n"
        "Forward each on IPCRecoveryClient, or add it to "
        "INTENTIONALLY_ABSENT_CTOR_ARGS with a reason."
    )


def test_ctor_exclusion_list_has_no_stale_entries():
    """An excused ctor arg that recovery now accepts must leave the list."""
    stale = sorted(
        (set(INTENTIONALLY_ABSENT_CTOR_ARGS) & _ctor_args(IPCRecoveryClient))
        - {"self"}
    )
    assert not stale, (
        f"INTENTIONALLY_ABSENT_CTOR_ARGS names args recovery DOES accept: "
        f"{stale}. Remove them -- a stale excuse hides the next real gap."
    )


def test_ctor_exclusion_list_only_names_real_ipcclient_args():
    bogus = sorted(set(INTENTIONALLY_ABSENT_CTOR_ARGS) - _ctor_args(IPCClient))
    assert not bogus, (
        f"INTENTIONALLY_ABSENT_CTOR_ARGS names non-existent IPCClient ctor "
        f"args: {bogus}. A typo'd entry silently excuses nothing."
    )


def test_forwarded_ctor_args_reach_the_inner_client():
    """Accepting the arg is not enough -- it must reach the inner IPCClient.

    A ctor that stores an arg and never forwards it passes the signature
    check above while changing nothing, which is the same
    observable-reports-success shape the arg was added to fix.
    """
    rc = IPCRecoveryClient(
        client_type=ClientType.API,
        config_root="/tmp/cfg-root",
        apparmor=True,
        autostart_timeout=7.5,
    )
    inner = rc._make_client(auto_start=False)
    assert inner.config_root == "/tmp/cfg-root", (
        "config_root was accepted and dropped -- the session still boots with "
        "config_root=None and confinement is still silently incomplete")
    assert inner.apparmor is True
    assert inner.autostart_timeout == 7.5
