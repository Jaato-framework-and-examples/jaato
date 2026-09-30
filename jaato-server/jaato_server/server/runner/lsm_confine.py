"""The runner's half of the confinement seam (selinux-backend.md §3.3).

The daemon provisions a boundary through a :class:`ConfinementBackend` and
names what it produced on the envelope (``SessionInitEnvelope.confinement``:
``{backend, label, child_label}``).  This module is where the runner reads
that and does the three per-LSM things a runner does:

- enter the label itself (:func:`self_confine`, bootstrap step 1c);
- move each model-driven subprocess into the child label
  (:func:`child_transition_callback`);
- check that every thread of the process wears the label
  (:func:`verify_threads`, #1023).

Only AppArmor is implemented, and each AppArmor branch delegates to the
functions the runner has always called, looked up on their modules at call
time so a test patching ``bootstrap.confine_to_profile`` still reaches
them.  An envelope naming any other backend is REFUSED by :func:`resolve`,
before anything is entered: a runner that cannot enter the label it was
given must not run the session unconfined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from jaato_server.server.confinement.apparmor import child_label_for
from jaato_server.shared.lsm_label import BACKEND_APPARMOR

#: The backends this runner can enter.  A daemon naming any other one is
#: refused at bootstrap (``BootstrapError`` stage ``confine``).
SUPPORTED_BACKENDS = frozenset({BACKEND_APPARMOR})


@dataclass(frozen=True)
class RunnerConfinement:
    """What the envelope says this runner must wear.

    Attributes:
        backend: ``"apparmor"`` (the only one this runner enters).
        label: The runner's own label (an AppArmor profile name).
        child_label: What model-driven subprocesses exec into.
    """

    backend: str
    label: str
    child_label: str


def _refuse(message: str) -> Exception:
    from jaato_server.server.runner.session import BootstrapError

    return BootstrapError("confine", message)


def resolve(envelope: Any) -> Optional[RunnerConfinement]:
    """Read and check the envelope's confinement descriptor.

    ``None`` means the session is unconfined (no ``profile_name`` and no
    descriptor); whether that is allowed is ``_maybe_self_confine``'s
    question (``confinement_required``, #1253), not this one.

    An envelope with no descriptor but a ``profile_name`` is read as
    AppArmor, which is what every daemon before the field meant.

    Raises:
        BootstrapError: the descriptor names a backend this runner cannot
            enter, is malformed, or disagrees with ``profile_name``.
    """
    profile = (getattr(envelope, "profile_name", None) or "").strip()
    descriptor = getattr(envelope, "confinement", None)
    if descriptor is None:
        if not profile:
            return None
        return RunnerConfinement(
            backend=BACKEND_APPARMOR, label=profile,
            child_label=child_label_for(profile),
        )
    if not isinstance(descriptor, Mapping):
        raise _refuse(
            f"envelope.confinement is not a mapping ({type(descriptor).__name__}); "
            "refusing to guess which boundary to enter")
    backend = str(descriptor.get("backend") or "")
    if backend not in SUPPORTED_BACKENDS:
        raise _refuse(
            f"the daemon provisioned a {backend or '(unnamed)'!r} boundary and "
            f"this runner can enter only {sorted(SUPPORTED_BACKENDS)}; refusing "
            "to run the session unconfined (selinux-backend.md §3.2)")
    label = str(descriptor.get("label") or "")
    if backend == BACKEND_APPARMOR and label != profile:
        raise _refuse(
            f"envelope.confinement.label {label!r} disagrees with "
            f"envelope.profile_name {profile!r}; refusing to pick one")
    if not label:
        return None
    child = str(descriptor.get("child_label") or child_label_for(label))
    return RunnerConfinement(backend=backend, label=label, child_label=child)


def self_confine(backend: str, label: str) -> None:
    """Enter *label* on the calling thread (bootstrap step 1c)."""
    if backend == BACKEND_APPARMOR:
        from jaato_server.server.runner import bootstrap

        bootstrap.confine_to_profile(label)
        return
    raise _refuse(f"cannot self-confine under backend {backend!r}")


def child_transition_callback(backend: str, label: str) -> Callable[[], None]:
    """The ``preexec_fn`` that moves a subprocess into *label*'s child."""
    if backend == BACKEND_APPARMOR:
        from jaato_server.server import apparmor

        return apparmor.make_child_transition_callback(label)
    raise _refuse(f"no child transition for backend {backend!r}")


def verify_threads(backend: str, label: str) -> Any:
    """Check every thread wears *label* (#1023); returns the scan."""
    if backend == BACKEND_APPARMOR:
        from jaato_server.server.runner import bootstrap

        return bootstrap.verify_thread_confinement(label)
    raise _refuse(f"cannot verify threads under backend {backend!r}")
