"""The contract every confinement backend implements (design §3.1).

A backend turns a :class:`Boundary` (what a session may touch) into a
:class:`ConfinementHandle` (what the runner transitions to).  AppArmor does
it by rendering and loading a profile; SELinux by labelling the workspace
at a per-workspace MCS level and naming a fixed domain.  Callers see only
the handle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any, Dict, Optional, Protocol, Sequence, Tuple, runtime_checkable,
)


@dataclass(frozen=True)
class Boundary:
    """What one session may reach, independent of how a backend enforces it.

    Attributes:
        workspace_path: The workspace, already realpath-resolved.
        config_root: The config tree the session reads, if any.
        env_file: The ``.env`` the daemon resolved for the session.
        managed: The workspace sits under the daemon's ``workspace_root``
            (executables built in it may run, #1273/#1274).
        private_tmp_dir: ``<ws>/.tmp`` when the runner binds it over
            ``/tmp`` (#1381).
        requested_fragments: AppArmor fragment names the profile asked
            for.  ``None`` and ``()`` are different boundaries: ``None``
            composes every fragment on disk (an unscoped ``//child``),
            ``()`` composes none (the most locked-down stage).  Collapsing
            one into the other widens or narrows a session silently.
        plugin_rules: AppArmor rule lines plugins contributed, or ``None``.
        plugin_rule_owners: Which plugin contributed which of
            ``plugin_rules`` (#1326), as ``((plugin, (rule, ...)), ...)``.
            Read only by the grant record the diagnostics panel shows.
    """

    workspace_path: str
    config_root: Optional[str] = None
    env_file: Optional[str] = None
    managed: bool = False
    private_tmp_dir: Optional[str] = None
    requested_fragments: Optional[Tuple[str, ...]] = None
    plugin_rules: Optional[Tuple[str, ...]] = None
    plugin_rule_owners: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()


def plugin_rule_fields(plugin_rules: Optional[Sequence[str]]) -> Dict[str, Any]:
    """``Boundary`` keyword arguments for what ``resolve_plugin_apparmor_rules``
    returned: the rules, and who contributed each when the value carries a
    ``by_plugin`` map (``PluginRules``, #1326).  ``None`` stays ``None``."""
    if plugin_rules is None:
        return {"plugin_rules": None, "plugin_rule_owners": ()}
    by_plugin = getattr(plugin_rules, "by_plugin", None) or {}
    return {
        "plugin_rules": tuple(plugin_rules),
        "plugin_rule_owners": tuple(
            (name, tuple(rules)) for name, rules in sorted(by_plugin.items())
        ),
    }


def fragment_field(requested_fragments: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    """``requested_fragments`` as a ``Boundary`` holds it, ``None`` kept ``None``."""
    return None if requested_fragments is None else tuple(requested_fragments)


@dataclass(frozen=True)
class ConfinementHandle:
    """What a provisioned boundary gives the runner.

    Attributes:
        backend: ``"apparmor"`` or ``"selinux"``.
        label: What the runner transitions to: an AppArmor profile name,
            or an SELinux context (``system_u:system_r:jaato_runner_t:s0:c1,c2``).
        confinement_id: The slot-key component (#1033); equal handles run
            under equal rules.
        child_label: What model-driven subprocesses exec into.
        grants: The diagnostics record (#1326), backend-shaped.
        complain: The kernel loaded the label and does not enforce it
            (AppArmor complain mode, SELinux permissive domain, #1014).
            A session record must then not claim a boundary.
    """

    backend: str
    label: str
    confinement_id: str
    child_label: str
    grants: Dict[str, Any] = field(default_factory=dict)
    complain: bool = False


@runtime_checkable
class ConfinementBackend(Protocol):
    """One kernel LSM, as the daemon uses it."""

    name: str

    def is_available(self) -> bool:
        """Can this daemon provision boundaries on this host?"""

    @property
    def unavailable_reason(self) -> Optional[str]:
        """The first failing precondition, or ``None``."""

    def confinement_id_for_boundary(self, boundary: Boundary) -> str:
        """The id :meth:`provision` would give *boundary*, without provisioning."""

    def provision(
        self, session_id: str, boundary: Boundary,
    ) -> Optional[ConfinementHandle]:
        """Make *boundary* enforceable and describe it, or ``None`` on failure."""

    def release(self, handle: ConfinementHandle) -> None:
        """Drop what :meth:`provision` created once no runner wears it."""


def is_selinux(handle: Optional["ConfinementHandle"]) -> bool:
    """Does *handle* name an SELinux boundary?  ``None`` is no boundary."""
    return handle is not None and handle.backend == "selinux"


def selinux_descriptor(handle: "ConfinementHandle") -> Dict[str, str]:
    """``SessionInitEnvelope.confinement`` for an SELinux *handle*.

    Carries ``confinement_id`` beside the two domains: SELinux has no
    profile name to read the id back out of, and the runner keys its
    session tmpdir on it (#1171).
    """
    return {
        "backend": handle.backend,
        "label": handle.label,
        "child_label": handle.child_label,
        "confinement_id": handle.confinement_id,
    }


def no_boundary(profile_name: Optional[str], handle: Optional["ConfinementHandle"]) -> bool:
    """No kernel boundary was provisioned: no AppArmor profile, no SELinux handle."""
    return not profile_name and handle is None


def split_handle(handle: "ConfinementHandle") -> Tuple[str, Optional["ConfinementHandle"]]:
    """``(profile_name, selinux_handle)`` as the spawn and envelope take them.

    AppArmor's label IS the profile name and travels as before; an SELinux
    handle travels as itself, with an empty profile name.
    """
    if is_selinux(handle):
        return "", handle
    return handle.label, None
