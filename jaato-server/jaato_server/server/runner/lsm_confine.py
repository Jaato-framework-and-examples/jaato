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

Each AppArmor branch delegates to the functions the runner has always
called, looked up on their modules at call time so a test patching
``bootstrap.confine_to_profile`` still reaches them.

SELinux (phase 2b) differs in one way that decides the rest: the runner
does not enter its domain at bootstrap.  A cold-spawned runner entered it
by the daemon's exec (``setexeccon`` before ``execve``, design §7.1); a
pool slot entered it at fork, while it had one thread (phase 4, §7.2).
So :func:`self_confine` only CONFIRMS the process already wears the label,
and refuses when it does not: SELinux refuses ``setcon`` in a threaded
process (phase 0), so a runner not already in its domain cannot be moved
into it now.  An envelope naming any other backend is
REFUSED by :func:`resolve`, before anything is entered: a runner that
cannot enter the label it was given must not run the session unconfined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import os

from jaato_server.server.confinement.apparmor import child_label_for
from jaato_server.shared.lsm_label import BACKEND_APPARMOR, BACKEND_SELINUX

#: The backends this runner can enter.  A daemon naming any other one is
#: refused at bootstrap (``BootstrapError`` stage ``confine``).
SUPPORTED_BACKENDS = frozenset({BACKEND_APPARMOR, BACKEND_SELINUX})

#: Where a task reads its own label and sets its children's.
_ATTR_CURRENT = "/proc/self/attr/current"
_ATTR_EXEC = "/proc/self/attr/exec"


@dataclass(frozen=True)
class RunnerConfinement:
    """What the envelope says this runner must wear.

    Attributes:
        backend: ``"apparmor"`` or ``"selinux"``.
        label: The runner's own label: an AppArmor profile name, or an
            SELinux context (``user:role:jaato_runner_t:s0:cA,cB``).
        child_label: What model-driven subprocesses exec into.
        confinement_id: The boundary's id, carried for SELinux (which has
            no profile name to read it out of); ``""`` for AppArmor, whose
            id is read from the profile name.
        enforcing_attested: SELinux only: the daemon attested the kernel
            enforces both domains (``descriptor["enforcing"]``, #1519).
            Only a literal ``True`` counts; absent (an older daemon) is
            ``False``.
    """

    backend: str
    label: str
    child_label: str
    confinement_id: str = ""
    enforcing_attested: bool = False


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
    if backend == BACKEND_SELINUX:
        return _resolve_selinux(label, profile, descriptor)
    if not label:
        return None
    child = str(descriptor.get("child_label") or child_label_for(label))
    return RunnerConfinement(backend=backend, label=label, child_label=child)


def _resolve_selinux(label: str, profile: str,
                     descriptor: Mapping[str, Any]) -> RunnerConfinement:
    """An SELinux descriptor: both labels named, no AppArmor profile."""
    child = str(descriptor.get("child_label") or "")
    if profile:
        raise _refuse(
            f"envelope names an SELinux boundary and the AppArmor profile "
            f"{profile!r}; one kernel runs one LSM, refusing to pick one")
    if not label or not child:
        raise _refuse(
            "envelope.confinement names SELinux without both label and "
            "child_label; refusing to guess the domains")
    return RunnerConfinement(
        backend=BACKEND_SELINUX, label=label, child_label=child,
        confinement_id=str(descriptor.get("confinement_id") or ""),
        enforcing_attested=descriptor.get("enforcing") is True)


def _own_selinux_label() -> str:
    with open(_ATTR_CURRENT, encoding="utf-8") as fh:
        return fh.read().replace("\x00", "").strip()


def self_confine(backend: str, label: str) -> None:
    """Enter *label* on the calling thread (bootstrap step 1c).

    SELinux: confirm instead.  The domain was entered by the exec that
    started this process, or by a pool slot at fork (phase 4); a runner
    not already in it cannot be moved into one now.
    """
    if backend == BACKEND_APPARMOR:
        from jaato_server.server.runner import bootstrap

        bootstrap.confine_to_profile(label)
        return
    if backend == BACKEND_SELINUX:
        try:
            actual = _own_selinux_label()
        except OSError as exc:
            raise _refuse(f"cannot read {_ATTR_CURRENT} ({exc}) to confirm "
                          f"the SELinux domain {label!r}") from exc
        if actual != label:
            raise _refuse(
                f"this runner is {actual!r}, the session's boundary is "
                f"{label!r}.  An SELinux runner enters its domain by the "
                "exec transition the daemon sets before spawning it, or, as "
                "a pool slot, at fork (selinux-backend.md §7.1, §7.2); a "
                "threaded runner outside it cannot be confined.")
        return
    raise _refuse(f"cannot self-confine under backend {backend!r}")


def child_transition_callback(
    backend: str, label: str, child_label: str = "",
) -> Callable[[], None]:
    """The ``preexec_fn`` that moves a subprocess into *label*'s child."""
    if backend == BACKEND_APPARMOR:
        from jaato_server.server import apparmor

        return apparmor.make_child_transition_callback(label)
    if backend == BACKEND_SELINUX:
        if not child_label:
            raise _refuse("no SELinux child label to transition subprocesses into")
        return _selinux_exec_transition(child_label)
    raise _refuse(f"no child transition for backend {backend!r}")


def _selinux_exec_transition(child_label: str) -> Callable[[], None]:
    """``setexeccon(child_label)`` between fork and exec.

    A write that fails raises in the forked child, so the spawn fails and
    nothing runs in the runner's own domain (fail closed, as AppArmor's
    ``//child`` callback does).  Unbuffered: a buffered write reports its
    error only on close.
    """
    payload = child_label.encode("utf-8")

    def preexec() -> None:
        fd = os.open(_ATTR_EXEC, os.O_WRONLY)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)

    return preexec


def verify_threads(backend: str, label: str) -> Any:
    """Check every thread wears *label* (#1023); returns the scan."""
    from jaato_server.server.runner import bootstrap

    if backend == BACKEND_APPARMOR:
        return bootstrap.verify_thread_confinement(label)
    if backend == BACKEND_SELINUX:
        return bootstrap.verify_thread_confinement(
            label, matcher=_same_selinux_context)
    raise _refuse(f"cannot verify threads under backend {backend!r}")


def _same_selinux_context(label: str, expected: str) -> bool:
    """SELinux threads of a confined runner wear exactly its context."""
    return label.replace("\x00", "").strip() == expected
