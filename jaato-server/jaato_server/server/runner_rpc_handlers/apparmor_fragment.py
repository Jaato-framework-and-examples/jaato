"""Daemon-side ``apparmor.add_reference_fragment`` handler (Phase 3 §3.2.2).

Receives a fragment-load request from the runner-side references
plugin (when ``selectReferences`` admits a new external path
outside the workspace), validates the path against the per-session
allow-list, and loads the fragment into the session's AppArmor
profile.

Why daemon-side and not directly in the runner?  Per parent design
§4.7: the runner is AppArmor-confined under
``jaato-ws-{session_id}`` whose ``/usr/bin/** ix,`` rule strips
setuid on the exec'd ``apparmor_parser``.  Even though the daemon's
sudoers entry covers ``apparmor_parser`` for the runner's UID, the
runner cannot effectively elevate.  Switching the cli rule to
``Ux`` would let sudo run unconfined — exactly the escape vector
the per-session profile exists to close.  Independently, the
existing sudoers entry permits ``apparmor_parser`` with arbitrary
args; an LLM-driven runner with sudo access could load attacker-
controlled profiles.  Validation (``_validate_path_for_fragment``)
must run in a process the model doesn't drive directly.

Wire flow:
1. Runner-side references plugin's ``selectReferences`` admits a
   new external path.
2. Runner-side ``RunnerRPCClient.add_reference_fragment(args)``
   sends the RPC.
3. **THIS handler** validates the path via the daemon's
   ``AppArmorManager._validate_path_for_fragment``, calls
   ``add_reference_fragment``, returns the result.
4. Runner picks up the new rule on its next file open via the
   profile's ``include if exists`` directive — no runner restart
   needed.

The handler is stateless beyond a reference to the
:class:`AppArmorManager` instance; one per daemon, but the manager
internally serializes mutations on the loop thread (via
``_run_unconfined``) so concurrent fragment loads from multiple
sessions are safe.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class AppArmorFragmentHandler:
    """Stateful handler for the ``apparmor.add_reference_fragment`` RPC.

    Phase 3 §3.2.2.

    Constructed by the daemon's session-bootstrap path with a
    reference to the active :class:`AppArmorManager`.  Bound to the
    runner-RPC server's ``apparmor.add_reference_fragment`` method.
    """

    def __init__(
        self,
        apparmor_manager: Any,  # server.apparmor.AppArmorManager
        session_id: str,
    ) -> None:
        """Construct the handler.

        Args:
            apparmor_manager: The daemon's :class:`AppArmorManager`.
                Lazy-init in production; tests pass a stub.  We hold
                a reference rather than re-resolving per call so a
                future swap of the manager (Phase 6 narrow daemon
                profile work) updates one slot.
            session_id: The session this handler is bound to.  Used
                for `add_reference_fragment(session_id, ...)` — the
                runner's RPC payload SHOULD echo the session id, but
                we treat the handler-bound id as authoritative to
                prevent a confused-deputy attack where a runner
                under session A asks the daemon to load a fragment
                into session B's profile.
        """
        self._apparmor = apparmor_manager
        self._session_id = session_id

    async def handle(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """RPC handler entry point.

        Expected args:
            ref_id: Stable identifier for the reference (used as the
                fragment filename's safe-encoded suffix).
            path: Absolute path to grant readonly access to.

        Returns:
            ``{"ok": True}`` on success, or
            ``{"ok": False, "error": "<reason>"}`` on validation /
            kernel failure.  ``ok=False`` flows back as an envelope
            ``ok=True`` with the dict result — this is a domain-
            failure shape, not a transport error.

        Raises:
            ValueError: when ``args`` is malformed (missing required
                keys).  The dispatcher catches this and translates
                to the typed error envelope.
        """
        ref_id = args.get("ref_id")
        path = args.get("path")
        if not isinstance(ref_id, str) or not ref_id:
            raise ValueError("apparmor.add_reference_fragment: missing ref_id")
        if not isinstance(path, str) or not path:
            raise ValueError("apparmor.add_reference_fragment: missing path")

        # Echo-check: if the runner included a session_id in the
        # payload, it must match the handler's bound id.  Prevents a
        # bug where one session's runner accidentally references
        # another's profile.  Not a security mechanism (the runner
        # is trusted at this layer); a sanity check.
        payload_session_id = args.get("session_id")
        if payload_session_id and payload_session_id != self._session_id:
            raise ValueError(
                f"apparmor.add_reference_fragment: payload session_id "
                f"{payload_session_id!r} != handler-bound session_id "
                f"{self._session_id!r}"
            )

        # Validation lives daemon-side per §4.7: the LLM-driven
        # runner cannot be trusted to validate paths it asks to
        # load.  Re-run the same validator the in-process flow uses
        # so the rules are identical.
        validate = getattr(self._apparmor, "_validate_path_for_fragment", None)
        if validate is not None:
            err = validate(path)
            if err:
                logger.warning(
                    "apparmor_fragment: validation rejected path "
                    "for session %s ref_id=%s: %s",
                    self._session_id, ref_id, err,
                )
                return {"ok": False, "error": err}

        # Hand off to AppArmorManager.  Internally dispatches to
        # _run_unconfined so the apparmor_parser reload happens on
        # the daemon's main loop thread.
        try:
            ok = self._apparmor.add_reference_fragment(
                self._session_id, ref_id, path,
            )
        except Exception as exc:  # noqa: BLE001 — boundary surface
            logger.exception(
                "apparmor_fragment: add_reference_fragment crashed "
                "for session %s ref_id=%s",
                self._session_id, ref_id,
            )
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        if not ok:
            return {
                "ok": False,
                "error": (
                    f"add_reference_fragment returned False for "
                    f"session {self._session_id!r} ref_id {ref_id!r}; "
                    f"see daemon log"
                ),
            }
        return {"ok": True}


def register(
    rpc_server: Any,  # server.runner_rpc_server.RunnerRPCServer
    handler: AppArmorFragmentHandler,
) -> None:
    """Register *handler*'s ``handle`` method as the
    ``apparmor.add_reference_fragment`` RPC method.

    Convenience indirection so the bootstrap site doesn't repeat
    the method-name string literal.
    """
    rpc_server.register("apparmor.add_reference_fragment", handler.handle)
