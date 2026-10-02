"""Kernel confinement backends behind one seam.

``docs/design/selinux-backend.md`` §3.  AppArmor is the backend the tree
has always used; SELinux is selected when AppArmor is not available.
:func:`select_backend` decides which one a daemon uses.

Phase 1: the daemon provisions through the seam (the WS pre-init hook and
its post-init re-run, and the IPC path), the envelope names the backend,
and the runner confines through ``server/runner/lsm_confine.py``.  Only
the AppArmor backend provisions; the SELinux backend reports whether a
host could run it, and a runner refuses an envelope naming it.
"""

from jaato_server.server.confinement.base import (
    Boundary,
    ConfinementBackend,
    ConfinementHandle,
    fragment_field,
    plugin_rule_fields,
)
from jaato_server.server.confinement.selection import (
    CONFINEMENT_ENV_VAR,
    REQUIRE_CONFINEMENT_ENV_VAR,
    BackendChoice,
    select_backend,
    select_daemon_backend,
)

__all__ = [
    "Boundary",
    "ConfinementBackend",
    "ConfinementHandle",
    "fragment_field",
    "plugin_rule_fields",
    "BackendChoice",
    "CONFINEMENT_ENV_VAR",
    "REQUIRE_CONFINEMENT_ENV_VAR",
    "select_backend",
    "select_daemon_backend",
]
