"""Live conformance guard — the SDK's contracts, asserted against a real daemon.

WHY THIS EXISTS.  Around 3600 tests pass in this repository, and a consumer
building an eval harness against a live daemon found five real defects within
an hour: an exhausted cascade handing back a session id for a session that can
never run, a history request answered with silence, a cascade id suppressing
event delivery, ``cost_usd`` arriving None beside a tracker holding the real
figure for the same turn, and a zero token limit refusing nothing.

None of them were exotic.  All of them were invisible here, because every test
that touches these paths supplies its own fixture — and **a fixture cannot be
wrong in the way production is wrong, because the person writing it is the
person who believes the contract.**  Three separate times in one day a test in
this repository was greener than reality:

* a ``SessionRefused`` test echoed back a ``request_id`` the daemon never
  stamps, so the contract it pinned could not fire on the one real path;
* a fake ``_send_event`` returned ``None``, so every test in its file asserted
  the not-sent branch instead of what it was named for;
* a type-conformance guard held a hardcoded copy of a field list that had
  already diverged from the thing it claimed to check.

A conformance test has no fixture for the behaviour under test.  It starts a
daemon, drives it the way a consumer does, and asserts what the SDK's own
docstrings promise.  It cannot be greener than reality because reality is what
it runs against.

WHY IT COSTS NOTHING.  The ``echo`` provider is deterministic, creds-free and
network-free, and it reports a CONFIGURED spend
(``plugin_configs.echo.usage``), so ceilings, refusals and cost propagation
are assertable without a provider that bills.  Every invariant here runs in
CI on every push.

WHY IT IS IN THE SDK.  Two audiences.  Ours is CI, where these are guards and
delegating them to a consumer — who finds our defects on their own tokens — is
backwards.  The other is a consumer asking a question our CI structurally
cannot answer for them: *is MY daemon, with MY provider and MY config,
behaving the way the SDK says?*  Same assertions, different daemon.

RUNNING THEM.  Deselected by default via the ``conformance`` marker, because
they start a subprocess and are therefore slower and likelier to flake than a
unit test::

    pytest -m conformance jaato-sdk/jaato_sdk/conformance/

Against your own daemon, point the fixture at it::

    JAATO_CONFORMANCE_SOCKET=/tmp/mine.sock pytest -m conformance ...
"""

from jaato_sdk.conformance.daemon import ConformanceDaemon, echo_workspace

__all__ = ["ConformanceDaemon", "echo_workspace"]
