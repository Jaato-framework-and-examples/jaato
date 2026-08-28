"""Phase 3 cascade-sharing — per-session apparmor with cross-session transitions.

Pin tests for the three changes:

  1. Per-session apparmor template carries the
     ``change_profile -> jaato-ws-*,`` rule (template v28+).
  2. ``PoolSlot.last_session_id`` is set on slot return + reused on next
     acquire for the apparmor-teardown lookup.
  3. SessionManager._teardown_prior_apparmor_profile_after_transition
     calls apparmor.teardown_profile only when the slot's
     last_session_id is set + differs from the current session.

Design reference: ``docs/design/runner-cascade-sharing.md`` §4.4
(amendment 2026-05-20).

These are unit tests; the kernel-level apparmor_parser interaction
is mocked via the ``_apparmor_manager`` injection point on
SessionManager.  Integration tests against a real AppArmor host
live in ``jaato-server/tests/integration/``.
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from server.apparmor import AppArmorManager
from server.runner_pool import PoolSlot


# ======================================================================
# 1. Template v28 carries the change_profile glob rule
# ======================================================================


class TestTemplateChangeProfileRule:
    def test_template_version_bumped_to_28(self) -> None:
        assert AppArmorManager._TEMPLATE_VERSION == 28

    def test_template_contains_jaato_ws_glob_change_profile_rule(self) -> None:
        """The per-session template's main scope must allow transitions
        to any jaato-ws-* profile (Phase 3 cascade-sharing slot reuse)."""
        template = AppArmorManager.PROFILE_TEMPLATE
        assert "change_profile -> jaato-ws-*," in template

    def test_template_still_has_unconfined_transition(self) -> None:
        """Phase 3 must NOT remove the existing change_profile rules
        (unconfined for __exit__ restore; //child for cli subprocess
        transitions)."""
        template = AppArmorManager.PROFILE_TEMPLATE
        assert "change_profile -> unconfined," in template
        assert (
            "change_profile -> jaato-ws-{session_id}//child," in template
        )


# ======================================================================
# 2. PoolSlot.last_session_id round-trips
# ======================================================================


class TestPoolSlotLastSessionId:
    def test_default_is_none(self) -> None:
        slot = PoolSlot(pid=1, sock=MagicMock())
        assert slot.last_session_id is None

    def test_can_be_stamped_after_session_end(self) -> None:
        """Caller pattern (JaatoServer.shutdown cascade-return path):
        set last_session_id on the slot before return_slot_after_session."""
        slot = PoolSlot(pid=1, sock=MagicMock())
        slot.last_session_id = "20260520_120000"
        assert slot.last_session_id == "20260520_120000"

    def test_field_survives_acquire_after_return(self) -> None:
        """The whole point: a slot returned with last_session_id=S1 keeps
        that ID accessible to the NEXT session acquiring the slot."""
        from server.runner_pool import PoolManager

        # target_size=8 — see the note in test_cascade_teardown_e2e: 0 is
        # "pool disabled", not an inert value for a return-path test.
        pool = PoolManager(template_manager=MagicMock(), target_size=8)
        slot = PoolSlot(
            pid=1, sock=MagicMock(),
            cascade_id="cascade-X",
            last_session_id="session-1",
        )
        pool.return_slot_after_session(slot)
        out = pool.acquire_slot(cascade_driver_id="cascade-X")
        assert out is not None
        assert out.last_session_id == "session-1"


# ======================================================================
# 3. _teardown_prior_apparmor_profile_after_transition
# ======================================================================


class TestTeardownPriorAfterTransition:
    """Pins the conditional under which SessionManager unloads the
    prior session's apparmor profile.

    Constructed without going through SessionManager.__init__ (which
    has heavy dependencies) by instantiating the bound method against
    a MagicMock self.
    """

    def _call(
        self,
        *,
        apparmor_mgr,
        spawned,
        current_session_id: str,
        current_profile_name: str,
    ):
        """Invoke _teardown_prior_apparmor_profile_after_transition with
        synthesized self + server."""
        from server.session_manager import SessionManager

        # Build a tiny self-shaped object exposing only the attrs the
        # method reads.  Bound method dispatch via __func__.
        self_stub = MagicMock()
        self_stub._apparmor_manager = apparmor_mgr

        server_stub = MagicMock()
        server_stub._spawned_runner = spawned

        return SessionManager._teardown_prior_apparmor_profile_after_transition(
            self_stub,
            server=server_stub,
            current_session_id=current_session_id,
            current_profile_name=current_profile_name,
        )

    def test_unloads_prior_profile_on_cascade_transition(self) -> None:
        """Happy path: slot has last_session_id; apparmor available;
        teardown_profile called once with the prior session_id."""
        apparmor = MagicMock(spec=AppArmorManager)
        apparmor.is_available.return_value = True
        apparmor.teardown_profile.return_value = True

        slot = PoolSlot(
            pid=1, sock=MagicMock(),
            cascade_id="cascade-X",
            last_session_id="session-1",
        )
        spawned = MagicMock()
        spawned.pool_slot = slot

        self._call(
            apparmor_mgr=apparmor,
            spawned=spawned,
            current_session_id="session-2",
            current_profile_name="jaato-ws-session-2",
        )

        apparmor.teardown_profile.assert_called_once_with("session-1")

    def test_skip_when_unconfined_session(self) -> None:
        """Empty profile_name means operator opted out — no transition
        happened, no teardown needed."""
        apparmor = MagicMock(spec=AppArmorManager)
        apparmor.is_available.return_value = True

        slot = PoolSlot(
            pid=1, sock=MagicMock(),
            cascade_id="cascade-X",
            last_session_id="session-1",
        )
        spawned = MagicMock()
        spawned.pool_slot = slot

        self._call(
            apparmor_mgr=apparmor,
            spawned=spawned,
            current_session_id="session-2",
            current_profile_name="",
        )

        apparmor.teardown_profile.assert_not_called()

    def test_skip_when_no_pool_slot(self) -> None:
        """Cold-spawned session (no pool_slot on SpawnedRunner) has no
        prior profile to unload."""
        apparmor = MagicMock(spec=AppArmorManager)
        apparmor.is_available.return_value = True

        spawned = MagicMock()
        spawned.pool_slot = None  # cold spawn

        self._call(
            apparmor_mgr=apparmor,
            spawned=spawned,
            current_session_id="session-1",
            current_profile_name="jaato-ws-session-1",
        )

        apparmor.teardown_profile.assert_not_called()

    def test_skip_when_first_cascade_session(self) -> None:
        """First session of a cascade — slot's last_session_id is None.
        No prior profile to unload."""
        apparmor = MagicMock(spec=AppArmorManager)
        apparmor.is_available.return_value = True

        slot = PoolSlot(
            pid=1, sock=MagicMock(),
            cascade_id="cascade-X",
            last_session_id=None,
        )
        spawned = MagicMock()
        spawned.pool_slot = slot

        self._call(
            apparmor_mgr=apparmor,
            spawned=spawned,
            current_session_id="session-1",
            current_profile_name="jaato-ws-session-1",
        )

        apparmor.teardown_profile.assert_not_called()

    def test_skip_when_apparmor_unavailable(self) -> None:
        """No apparmor support → nothing to do (defensive)."""
        apparmor = MagicMock(spec=AppArmorManager)
        apparmor.is_available.return_value = False

        slot = PoolSlot(
            pid=1, sock=MagicMock(),
            cascade_id="cascade-X",
            last_session_id="session-1",
        )
        spawned = MagicMock()
        spawned.pool_slot = slot

        self._call(
            apparmor_mgr=apparmor,
            spawned=spawned,
            current_session_id="session-2",
            current_profile_name="jaato-ws-session-2",
        )

        apparmor.teardown_profile.assert_not_called()

    def test_skip_when_no_apparmor_manager(self) -> None:
        """SessionManager._apparmor_manager is None (lazy-init never
        fired because no session opted into apparmor)."""
        slot = PoolSlot(
            pid=1, sock=MagicMock(),
            cascade_id="cascade-X",
            last_session_id="session-1",
        )
        spawned = MagicMock()
        spawned.pool_slot = slot

        # _apparmor_manager = None on self_stub
        self._call(
            apparmor_mgr=None,
            spawned=spawned,
            current_session_id="session-2",
            current_profile_name="jaato-ws-session-2",
        )
        # No teardown to assert; test passes if no AttributeError raised.

    def test_skip_when_current_equals_prior(self) -> None:
        """Defensive self-check: if somehow last_session_id == current,
        don't unload our own profile."""
        apparmor = MagicMock(spec=AppArmorManager)
        apparmor.is_available.return_value = True

        slot = PoolSlot(
            pid=1, sock=MagicMock(),
            cascade_id="cascade-X",
            last_session_id="session-1",
        )
        spawned = MagicMock()
        spawned.pool_slot = slot

        self._call(
            apparmor_mgr=apparmor,
            spawned=spawned,
            current_session_id="session-1",  # same as last_session_id
            current_profile_name="jaato-ws-session-1",
        )

        apparmor.teardown_profile.assert_not_called()

    def test_teardown_exception_does_not_propagate(self) -> None:
        """Best-effort: apparmor.teardown_profile raising must NOT
        propagate out (would abort the spawn chain)."""
        apparmor = MagicMock(spec=AppArmorManager)
        apparmor.is_available.return_value = True
        apparmor.teardown_profile.side_effect = RuntimeError("kernel busy")

        slot = PoolSlot(
            pid=1, sock=MagicMock(),
            cascade_id="cascade-X",
            last_session_id="session-1",
        )
        spawned = MagicMock()
        spawned.pool_slot = slot

        # Should NOT raise.
        self._call(
            apparmor_mgr=apparmor,
            spawned=spawned,
            current_session_id="session-2",
            current_profile_name="jaato-ws-session-2",
        )

        apparmor.teardown_profile.assert_called_once_with("session-1")
