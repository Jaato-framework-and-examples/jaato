"""The contract every confinement backend implements (design §3.1).

A backend turns a :class:`Boundary` (what a session may touch) into a
:class:`ConfinementHandle` (what the runner transitions to).  AppArmor does
it by rendering and loading a profile; SELinux by labelling the workspace
at a per-workspace MCS level and naming a fixed domain.  Callers see only
the handle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable


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
        requested_fragments: AppArmor fragment names the profile asked for.
        plugin_rules: AppArmor rule lines plugins contributed.
    """

    workspace_path: str
    config_root: Optional[str] = None
    env_file: Optional[str] = None
    managed: bool = False
    private_tmp_dir: Optional[str] = None
    requested_fragments: Tuple[str, ...] = ()
    plugin_rules: Tuple[str, ...] = ()


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
    """

    backend: str
    label: str
    confinement_id: str
    child_label: str
    grants: Dict[str, Any] = field(default_factory=dict)


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
