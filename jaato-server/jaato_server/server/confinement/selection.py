"""Which confinement backend a daemon uses (design §3.1).

``JAATO_CONFINEMENT`` names one (``apparmor`` / ``selinux`` / ``none``);
unset or ``auto`` means AppArmor when it is available, else SELinux when it
is available, else none.  The two cannot both be the active LSM on one
kernel, so on a real host at most one is ever available and the order only
decides which reason is reported when neither is.

``JAATO_REQUIRE_CONFINEMENT`` asks for SOME kernel backend: a caller that
sees :attr:`BackendChoice.refuse` true must refuse to start, the posture
``JAATO_REQUIRE_APPARMOR`` already takes for AppArmor specifically.

An unrecognised ``JAATO_CONFINEMENT`` is refused rather than read as
``auto``: a typo that silently picked a different backend (or none) would
change what every session is confined by with nothing saying so.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional

from jaato_server.server.confinement.base import ConfinementBackend
from jaato_server.shared.lsm_label import (
    BACKEND_APPARMOR,
    BACKEND_NONE,
    BACKEND_SELINUX,
)

CONFINEMENT_ENV_VAR = "JAATO_CONFINEMENT"
REQUIRE_CONFINEMENT_ENV_VAR = "JAATO_REQUIRE_CONFINEMENT"

_AUTO = "auto"
_CHOICES = (_AUTO, BACKEND_APPARMOR, BACKEND_SELINUX, BACKEND_NONE)
_TRUTHY = ("1", "true", "yes", "on")

BackendFactory = Callable[[], Optional[ConfinementBackend]]


@dataclass
class BackendChoice:
    """The outcome of :func:`select_backend`.

    Attributes:
        name: The chosen backend, or ``"none"``.
        backend: The chosen backend object, or ``None``.
        requested: What ``JAATO_CONFINEMENT`` asked for (``auto`` when unset).
        required: ``JAATO_REQUIRE_CONFINEMENT`` was set.
        reasons: Why each backend that was asked was not chosen.
        error: A configuration error (an unknown value); set means refuse.
    """

    name: str
    backend: Optional[ConfinementBackend]
    requested: str
    required: bool
    reasons: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def refuse(self) -> bool:
        """Must the daemon refuse to start?"""
        if self.error:
            return True
        return self.required and self.backend is None

    def describe(self) -> str:
        """One line for the startup log."""
        if self.error:
            return self.error
        if self.backend is not None:
            return f"kernel confinement: {self.name}"
        unavailable = "; ".join(f"{k}: {v}" for k, v in self.reasons.items())
        return (
            "kernel confinement: none"
            + (f" ({unavailable})" if unavailable else "")
            + " — workspace isolation is the directory-sandbox heuristic only"
        )


def _probe(
    name: str, factory: BackendFactory, reasons: Dict[str, str],
) -> Optional[ConfinementBackend]:
    try:
        backend = factory()
    except Exception as exc:  # a broken probe is an unavailable backend
        reasons[name] = f"probe failed: {exc}"
        return None
    if backend is None:
        reasons[name] = "not installed"
        return None
    if backend.is_available():
        return backend
    reasons[name] = backend.unavailable_reason or "unavailable"
    return None


def select_backend(
    *,
    apparmor: BackendFactory,
    selinux: BackendFactory,
    environ: Optional[Mapping[str, str]] = None,
) -> BackendChoice:
    """Pick the backend.  The factories are called only when asked for."""
    env = os.environ if environ is None else environ
    # env: which kernel confinement backend the daemon uses; host-scoped,
    # it names the LSM the kernel runs.
    requested = (env.get(CONFINEMENT_ENV_VAR, "") or _AUTO).strip().lower()
    # env: refuse to start without a kernel confinement backend; host-scoped.
    required = env.get(REQUIRE_CONFINEMENT_ENV_VAR, "").strip().lower() in _TRUTHY
    choice = BackendChoice(BACKEND_NONE, None, requested, required)
    if requested not in _CHOICES:
        choice.error = (
            f"{CONFINEMENT_ENV_VAR}={requested!r} is not one of "
            f"{', '.join(_CHOICES)}"
        )
        return choice
    order = {
        _AUTO: (BACKEND_APPARMOR, BACKEND_SELINUX),
        BACKEND_APPARMOR: (BACKEND_APPARMOR,),
        BACKEND_SELINUX: (BACKEND_SELINUX,),
        BACKEND_NONE: (),
    }[requested]
    factories = {BACKEND_APPARMOR: apparmor, BACKEND_SELINUX: selinux}
    for name in order:
        backend = _probe(name, factories[name], choice.reasons)
        if backend is not None:
            choice.name, choice.backend = name, backend
            break
    return choice
