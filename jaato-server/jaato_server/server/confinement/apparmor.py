"""The AppArmor backend: the tree's existing :class:`AppArmorManager`,
seen through :class:`ConfinementBackend`.

A thin adapter by design (design §3, phase 1): every rule, every template
version and every lifetime decision stays in ``server/apparmor.py``, and
this class only translates a :class:`Boundary` into that manager's
arguments and its results into a :class:`ConfinementHandle`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from jaato_server.server.confinement.base import Boundary, ConfinementHandle
from jaato_server.shared.lsm_label import BACKEND_APPARMOR


class AppArmorBackend:
    """Adapts an ``AppArmorManager`` (or anything shaped like one)."""

    name = BACKEND_APPARMOR

    def __init__(self, manager: Any) -> None:
        self._manager = manager

    @property
    def manager(self) -> Any:
        """The wrapped manager, for call sites not yet on the seam."""
        return self._manager

    def is_available(self) -> bool:
        return bool(self._manager.is_available())

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._manager.unavailable_reason

    def _boundary_kwargs(self, boundary: Boundary) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "config_root": boundary.config_root,
            "env_file": boundary.env_file,
            "requested_fragments": list(boundary.requested_fragments) or None,
            "plugin_rules": list(boundary.plugin_rules) or None,
        }
        # Passed only when set: the manager's callers follow the same rule
        # (``private_tmp_kwargs``), so a manager predating #1381 still works.
        if boundary.private_tmp_dir:
            kwargs["private_tmp_dir"] = boundary.private_tmp_dir
        return kwargs

    def confinement_id_for_boundary(self, boundary: Boundary) -> str:
        return self._manager.confinement_id_for_boundary(
            boundary.workspace_path, **self._boundary_kwargs(boundary),
        )

    def provision(
        self, session_id: str, boundary: Boundary,
    ) -> Optional[ConfinementHandle]:
        kwargs = self._boundary_kwargs(boundary)
        confinement_id = self._manager.confinement_id_for_boundary(
            boundary.workspace_path, **kwargs,
        )
        if not self._manager.provision_profile(
            session_id, boundary.workspace_path,
            confinement_id=confinement_id, **kwargs,
        ):
            return None
        profile = self._manager.get_profile_name(session_id)
        return ConfinementHandle(
            backend=BACKEND_APPARMOR,
            label=profile,
            confinement_id=confinement_id,
            child_label=f"{profile}//child",
            grants=_recorded_grants(confinement_id),
        )

    def release(self, handle: ConfinementHandle) -> None:
        self._manager.teardown_profile_by_confinement_id(handle.confinement_id)


def _recorded_grants(confinement_id: str) -> Dict[str, Any]:
    from jaato_server.server.apparmor import recorded_grants

    try:
        return dict(recorded_grants(confinement_id) or {})
    except Exception:  # a diagnostics record must never fail a provision
        return {}
