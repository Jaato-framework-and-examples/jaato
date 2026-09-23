"""A WS session is confined before its runner serves work (#1253).

On the WebSocket path a pre-warm pool slot was dispatched to serve a new
session BEFORE the session's AppArmor profile was provisioned (or, on the
core path, when provisioning FAILED), so the runner spawned with
``confined=False`` and ran unconfined for its whole life — even though the
session was configured ``sandbox_mode: apparmor`` and the record claimed it.
A silent confinement bypass: model-driven ``cli`` / subprocess code ran with
no kernel boundary, and the only trace was one ``confined=False`` spawn line
that reads as normal.  #1100 documented and deferred the race; #1253 filed it
with live daemon-log evidence.

The invariant this pins: **a session whose configuration requests AppArmor
confinement must never serve model-driven work with ``confined=False``** — it
is confined, or it does not run.  Two layers hold it, and they fail CLOSED:

1. Daemon (WS pre-init hook, ``websocket.py``): confinement is REQUIRED
   whenever the daemon holds an available ``AppArmorManager``.  If the
   profile fails to provision, or the runner spawn raises, the hook records
   a bootstrap outcome and RETURNS without spawning — so
   ``initialize_or_refuse`` refuses the session by name rather than falling
   back to an unconfined runner or unconfined in-process execution.  (Those
   behavioural cases live in ``test_ws_always_spawn_runner.py``.)

2. Runner (``_maybe_self_confine``, ``runner/session.py``): the envelope
   carries ``confinement_required``, and an empty ``profile_name`` with it
   set RAISES ``BootstrapError`` rather than running unconfined — the
   defence-in-depth backstop for a pre-warm slot or a peer daemon that
   reaches the runner with the profile missing.

``confinement_required`` is the discriminator the runner lacked: an empty
``profile_name`` alone cannot tell a genuinely-unconfined session (operator
opt-out, or a host with no AppArmor) from one that WANTED confinement and
did not get it.  ``False`` (the default, and every unconfined session) keeps
the gate inert, so nothing about the pre-existing unconfined path changes.

Ordering: this fix is safe only WITH #1252 (the ``//child`` exec-grant fix)
on ``main`` — before it, a confined-from-spawn WS runner had a ``cli`` dead
from turn 1.  With #1252 merged, making the first runner confined is safe.

**No kernel here.** This container carries no AppArmor LSM, as #1023 / #1033
/ #1100 all record.  These tests exercise the FRAMEWORK's decision — that it
refuses to serve an apparmor-configured session unconfined — from fabricated
envelope / manager state, never the kernel's behaviour.  What is
kernel-unverified: that a runner which self-confines then actually enforces.
"""

from __future__ import annotations

import pytest

from jaato_server.server.runner.session import (
    BootstrapError,
    _maybe_self_confine,
)
from jaato_server.shared.session_envelope import SessionInitEnvelope


# ----------------------------------------------------------------------
# Reversions — read by
# ``shared/tests/test_every_guard_detects_its_own_reversion.py``, which
# puts each defect back in a disposable copy of the checkout and fails if
# the named test still passes.  Anchors are tight (the gate itself), so an
# insertion nearby does not make one go stale.
# ----------------------------------------------------------------------
try:  # pragma: no cover - import shape differs per invocation
    from jaato_server.shared.tests.reversion import Reversion
except Exception:  # pragma: no cover
    Reversion = None  # type: ignore[assignment]

_SESSION = "jaato-server/jaato_server/server/runner/session.py"
_ENVELOPE = "jaato-server/jaato_server/shared/session_envelope.py"
_WS = "jaato-server/jaato_server/server/websocket.py"

REVERSIONS = [] if Reversion is None else [
    # Layer 2 — the runner-side gate.  Disabling the raise restores the
    # pre-#1253 behaviour: an empty profile with confinement required falls
    # through to the "unconfined session" skip and serves work with no
    # boundary.
    Reversion(
        target=_SESSION,
        find='        if getattr(envelope, "confinement_required", False):',
        replace="        if False:  # #1253 reversion: gate disabled",
        test="test_maybe_self_confine_refuses_confined_session_with_no_profile",
        because=(
            "the runner would again bootstrap unconfined when confinement "
            "was required but no profile reached it (the pre-warm-slot race)"
        ),
    ),
    # The invariant only reaches the runner gate if the envelope carries it.
    # Reading a fixed False on ingest drops it, so the gate is inert on every
    # wire.
    Reversion(
        target=_ENVELOPE,
        find=(
            "            confinement_required=bool("
            'd.get("confinement_required", False)),'
        ),
        replace="            confinement_required=False,",
        test="test_envelope_round_trips_confinement_required",
        because=(
            "confinement_required no longer survives the daemon->runner wire, "
            "so the runner gate can never see that confinement was required"
        ),
    ),
    # Layer 1 — the daemon refuses a provisioning failure.  Falling through
    # (``return`` -> ``pass``) spawns an unconfined runner for a session that
    # required confinement: exactly the #1253 bypass.
    Reversion(
        target=_WS,
        find=(
            '                        "unconfined runner (#1253)",\n'
            "                    )\n"
            "                    return"
        ),
        replace=(
            '                        "unconfined runner (#1253)",\n'
            "                    )\n"
            "                    pass  # #1253 reversion: fall through to spawn"
        ),
        test="test_ws_hook_refuses_when_provisioning_fails",
        because=(
            "the WS hook again spawns an unconfined runner when profile "
            "provisioning fails instead of refusing the session"
        ),
    ),
    # Layer 1 — a confined session's spawn failure must refuse, not fall back
    # to unconfined in-process execution.  Neutering the gate in
    # ``_report_confined_spawn_failure`` restores the in-process fallback for
    # a session that required confinement.
    Reversion(
        target=_WS,
        find=(
            "    if confinement_required:\n"
            "        logger.warning(\n"
            '            "AppArmor pre-init: runner spawn failed for session %s "'
        ),
        replace=(
            "    if False:  # #1253 reversion\n"
            "        logger.warning(\n"
            '            "AppArmor pre-init: runner spawn failed for session %s "'
        ),
        test="test_ws_hook_spawn_failure_refuses_a_confined_session",
        because=(
            "a confined session whose runner spawn fails again falls back to "
            "unconfined in-process tool execution instead of being refused"
        ),
    ),
]


def _envelope(*, profile_name: str, confinement_required: bool) -> SessionInitEnvelope:
    """A minimal envelope for the ``_maybe_self_confine`` gate.

    ``workspace_path=None`` avoids the absolute-path ``__post_init__`` guard;
    the gate reads only ``profile_name`` and ``confinement_required``, and
    both no-profile branches (raise, and the unconfined skip) return before
    any AppArmor import, so no kernel or ``/proc`` access is involved.
    """
    return SessionInitEnvelope(
        session_id="s-1253",
        workspace_path=None,
        profile_name=profile_name,
        provider_name="echo",
        model_name="echo-1",
        confinement_required=confinement_required,
    )


def test_maybe_self_confine_refuses_confined_session_with_no_profile() -> None:
    """The load-bearing invariant: confinement required + no profile RAISES.

    This is the runner-side backstop that turns the #1253 silent bypass into
    a refused bootstrap.  The daemon turns the ``BootstrapError`` into a
    ``RunnerBootstrapFailed`` refusal (#1033); the session serves no
    model-driven work rather than serving it unconfined.
    """
    envelope = _envelope(profile_name="", confinement_required=True)
    with pytest.raises(BootstrapError) as exc_info:
        _maybe_self_confine(envelope, recycle_pools=None)
    # The stage names the confinement step, and the message names the issue.
    assert exc_info.value.stage == "confine"
    assert "1253" in exc_info.value.message


def test_maybe_self_confine_allows_genuine_unconfined_session() -> None:
    """A genuinely-unconfined session (confinement NOT required) is a no-op.

    An operator opt-out, or a host with no AppArmor, reaches the gate with an
    empty profile and ``confinement_required=False``; it must skip silently,
    exactly as before #1253 — the fix must not refuse sessions nobody asked
    to confine.
    """
    envelope = _envelope(profile_name="", confinement_required=False)
    # Returns (no raise); nothing to assert beyond "does not raise".
    assert _maybe_self_confine(envelope, recycle_pools=None) is None


def test_envelope_round_trips_confinement_required() -> None:
    """``confinement_required`` survives the daemon -> runner wire.

    The runner gate reads it off the deserialized envelope, so it must
    round-trip through ``to_dict`` / ``from_dict``.  The default is False, so
    an older daemon's envelope (no such key) deserializes to the inert value.
    """
    env = _envelope(profile_name="", confinement_required=True)
    revived = SessionInitEnvelope.from_dict(env.to_dict())
    assert revived.confinement_required is True

    # Absent on the wire (an older daemon) => the inert default, not a raise.
    wire = env.to_dict()
    del wire["confinement_required"]
    assert SessionInitEnvelope.from_dict(wire).confinement_required is False
