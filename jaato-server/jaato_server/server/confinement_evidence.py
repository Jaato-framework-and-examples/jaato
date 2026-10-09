"""What a session's runner reported wearing, against what it should wear (#1253).

A session that asked for kernel confinement used to be believed on the
strength of the DAEMON's own bookkeeping: a profile had been provisioned,
so the record said ``sandbox_mode: apparmor``.  Whether the runner serving
the session actually wore that profile was never asked.  #1100 and #1253
measured the gap on a live daemon: the WS path spawned the session's first
runner, a pre-warm pool slot, ~0.4-3 s BEFORE the profile was provisioned,
so the slot bootstrapped with an empty ``profile_name``, skipped its
self-confinement, and served the whole session unconfined, while the
post-init hook provisioned the profile afterwards and recorded
``apparmor``.  Nothing said so.

The individual paths have since been ordered (provision, then spawn) and
each refuses the failures it can see.  This module is the end-to-end check
that does not depend on every path getting its own order right: the
runner reports the label it wears once ``session.bootstrap`` has confined
it (``RunnerRPC._handle_session_bootstrap``), and the daemon compares that
with the boundary it provisioned.  A session that wanted confinement is
initialized only on positive evidence that its runner wears the boundary.

Three outcomes, each a refusal reason or ``None``:

* confinement was required and no runner reported at all (the hook
  returned before spawning, raised, or its bootstrap dispatch never ran);
* the runner reported a label that is not the boundary's;
* the runner reported the boundary: ``None``, the session may run.

Stdlib only, plus the one label parser (:mod:`shared.apparmor_label`), so
it is importable wherever the bootstrap answer is read.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from jaato_server.shared.apparmor_label import (
    parse_label,
    profile_name_ignoring_mode,
    sandbox_mode_for_profile,
)

#: Backend names, as :class:`server.confinement.base.ConfinementHandle` spells them.
BACKEND_APPARMOR = "apparmor"
BACKEND_SELINUX = "selinux"

#: Key of the runner's ``session.bootstrap`` answer that carries the report.
REPORT_KEY = "confinement"


def expected_boundary(
    profile_name: Optional[str], confinement: Any,
) -> Optional[Tuple[str, str]]:
    """``(backend, label)`` the runner must wear, or ``None`` when none was provisioned.

    Args:
        profile_name: The AppArmor profile the spawn carries (empty under
            SELinux, where the label travels on the handle).
        confinement: The SELinux :class:`ConfinementHandle`, or ``None``.
    """
    label = str(getattr(confinement, "label", "") or "") if confinement is not None else ""
    if label:
        backend = str(getattr(confinement, "backend", "") or BACKEND_SELINUX)
        return backend, label
    if profile_name:
        return BACKEND_APPARMOR, profile_name
    return None


def reported_label(report: Any) -> Optional[str]:
    """The raw label a bootstrap report carries, or ``None`` when it carries none."""
    if not isinstance(report, Mapping):
        return None
    label = report.get("label")
    return label if isinstance(label, str) else None


def _wears(backend: str, expected: str, raw: str) -> bool:
    """Does the reported *raw* label name the *expected* boundary?

    AppArmor: by profile NAME, mode-tolerant (#1014) -- a complain-mode
    profile is still the profile the session asked for, and the record
    says ``apparmor-complain`` for it rather than refusing.  A stack with
    ``unconfined`` reads as its profile (#1509).  SELinux: the exact
    context, as ``lsm_confine.self_confine`` compares it.
    """
    if backend == BACKEND_SELINUX:
        return raw.replace("\x00", "").strip() == expected
    return profile_name_ignoring_mode(raw) == expected


def shortfall(
    *,
    required: bool,
    expected: Optional[Tuple[str, str]],
    report: Any,
    reported: bool,
) -> Optional[str]:
    """Why a confinement-required session may not run, or ``None``.

    Args:
        required: The session was configured for confinement on a host
            that supports it (the #1253 ``confinement_required``).
        expected: :func:`expected_boundary` of the bootstrap that was
            dispatched, or ``None`` when none was.
        report: The runner's ``confinement`` answer.
        reported: Whether a bootstrap answer was received at all.
    """
    if not required:
        return None
    if expected is None or not reported:
        return (
            "kernel confinement was required for this session but no runner "
            "was bootstrapped into a provisioned boundary, so nothing "
            "establishes that the session's work would run confined (#1253)"
        )
    backend, label = expected
    raw = reported_label(report)
    if raw is None:
        return (
            f"kernel confinement was required for this session but its "
            f"runner did not report the label it wears; the {backend} "
            f"boundary {label!r} is unconfirmed (#1253)"
        )
    if not _wears(backend, label, raw):
        return (
            f"kernel confinement was required for this session but its "
            f"runner wears {raw!r}, not the {backend} boundary {label!r} "
            f"provisioned for it; refusing to serve the session with a "
            f"boundary it does not have (#1253)"
        )
    return None


def reported_sandbox_mode(
    expected: Optional[Tuple[str, str]], report: Any,
) -> Optional[str]:
    """The AppArmor ``sandbox_mode`` the runner's own label supports, or ``None``.

    Only for an AppArmor boundary the runner was confirmed to wear: the
    mode is the one the KERNEL reported in the label, ``(complain)`` giving
    ``apparmor-complain`` (#1014).  ``None`` (SELinux, no report, a label
    that is not the boundary) leaves the caller's own derivation in charge.
    """
    if expected is None or expected[0] != BACKEND_APPARMOR:
        return None
    raw = reported_label(report)
    if raw is None or not _wears(BACKEND_APPARMOR, expected[1], raw):
        return None
    return sandbox_mode_for_profile(complain=parse_label(raw).complaining)


def recorded_sandbox_mode(server: Any, derived: Optional[str]) -> Optional[str]:
    """The ``sandbox_mode`` to record for a session: the runner's word over the daemon's.

    *derived* is what the daemon concluded from provisioning (an AppArmor
    profile loaded, in enforce or complain mode).  When the runner reported
    wearing that profile, the mode it reported wins (#1014: the mode the
    kernel applies, read from the task, not from the render).  Otherwise
    *derived* stands -- a session that wanted confinement and has no such
    report is refused before its record is written
    (``JaatoServer.runner_bootstrap_error``).

    Args:
        server: The session's ``JaatoServer`` (anything without
            ``runner_reported_sandbox_mode`` keeps *derived*).
        derived: The provisioning-derived mode, or ``None``.
    """
    from jaato_server.shared.apparmor_label import sandbox_mode_is_apparmor

    if not sandbox_mode_is_apparmor(derived):
        return derived
    ask = getattr(server, "runner_reported_sandbox_mode", None)
    reported = ask() if callable(ask) else None
    return reported if isinstance(reported, str) and reported else derived
