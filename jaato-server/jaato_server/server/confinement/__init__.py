"""Kernel confinement backends behind one seam.

``docs/design/selinux-backend.md`` §3.  AppArmor is the backend the tree
has always used; SELinux is selected when AppArmor is not available.
:func:`select_backend` decides which one a daemon uses.

Phase 1 of that design: the seam exists and is exercised, and no call site
has moved onto it yet, so no session's behaviour changes. The SELinux
backend reports whether a host could run it and provisions nothing.
"""

from jaato_server.server.confinement.base import (
    Boundary,
    ConfinementBackend,
    ConfinementHandle,
)
from jaato_server.server.confinement.selection import (
    CONFINEMENT_ENV_VAR,
    REQUIRE_CONFINEMENT_ENV_VAR,
    BackendChoice,
    select_backend,
)

__all__ = [
    "Boundary",
    "ConfinementBackend",
    "ConfinementHandle",
    "BackendChoice",
    "CONFINEMENT_ENV_VAR",
    "REQUIRE_CONFINEMENT_ENV_VAR",
    "select_backend",
]
