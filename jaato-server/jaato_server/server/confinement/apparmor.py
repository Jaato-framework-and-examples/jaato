"""The AppArmor backend: the tree's existing :class:`AppArmorManager`,
seen through :class:`ConfinementBackend`.

A thin adapter by design (design §3, phase 1): every rule, every template
version and every lifetime decision stays in ``server/apparmor.py``, and
this class only translates a :class:`Boundary` into that manager's
arguments and its results into a :class:`ConfinementHandle`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
            # ``None`` and ``[]`` are different profiles (unscoped vs
            # scoped ``//child``), so neither is folded into the other.
            "requested_fragments": (
                None if boundary.requested_fragments is None
                else list(boundary.requested_fragments)
            ),
            "plugin_rules": _plugin_rules(boundary),
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
            child_label=child_label_for(profile),
            # The record is keyed by the PROFILE NAME (``_record_grants``),
            # not the bare id: looking it up by the id always found nothing.
            grants=_recorded_grants(profile),
            complain=_complain(self._manager, session_id),
        )

    def release(self, handle: ConfinementHandle) -> None:
        self._manager.teardown_profile_by_confinement_id(handle.confinement_id)


def child_label_for(profile_name: str) -> str:
    """What a subprocess of a runner in *profile_name* execs into.

    ``<profile>//child`` for a main runner.  A profile that is already a
    sub-profile (the isolated sub-runner, name containing ``//``) has no
    child of its own: its subprocesses inherit it (v15 design intent).
    """
    return profile_name if "//" in profile_name else f"{profile_name}//child"


def envelope_descriptor(profile_name: Optional[str]) -> Optional[Dict[str, str]]:
    """``SessionInitEnvelope.confinement`` for an AppArmor *profile_name*.

    ``None`` for an unconfined session (empty name).  The one place the
    daemon writes the descriptor, so the runner's
    ``lsm_confine.resolve`` has one shape to read.
    """
    if not profile_name:
        return None
    return {
        "backend": BACKEND_APPARMOR,
        "label": profile_name,
        "child_label": child_label_for(profile_name),
    }


def _plugin_rules(boundary: Boundary) -> Optional[List[str]]:
    """The manager's ``plugin_rules`` argument, attribution restored (#1326)."""
    if boundary.plugin_rules is None:
        return None
    if not boundary.plugin_rule_owners:
        # A plain list, as the caller had: the record then reads it as
        # "(unattributed)" exactly as before.
        return list(boundary.plugin_rules)
    from jaato_server.server.apparmor import PluginRules

    return PluginRules(
        list(boundary.plugin_rules),
        {name: list(rules) for name, rules in boundary.plugin_rule_owners},
    )


def _complain(manager: Any, session_id: str) -> bool:
    """Whether the manager RENDERED the profile in complain mode (#1014).

    Read from the manager's record rather than the environment, which a
    profile's ``env:`` overlay may have changed since.  ``is True`` so a
    test double answering with a mock reads as enforcing.
    """
    probe = getattr(manager, "profile_is_complain_mode", None)
    if probe is None:
        return False
    return probe(session_id) is True


def _recorded_grants(profile_name: str) -> Dict[str, Any]:
    from jaato_server.server.apparmor import recorded_grants

    try:
        return dict(recorded_grants(profile_name) or {})
    except Exception:  # a diagnostics record must never fail a provision
        return {}
