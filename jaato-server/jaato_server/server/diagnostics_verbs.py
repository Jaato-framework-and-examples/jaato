"""The daemon side of the self-diagnostics verb (#1294): the owner gate,
the cached-vs-live split, and the audit line.

``DiagnosticsRequest`` arrives at ``SessionManager.handle_request``, which
resolves the caller's OWN attached session (never one it names) and the
workspace owner, and hands both to :func:`answer_diagnostics_request`.
This module decides three things and nothing else:

* **who may look** -- :func:`may_view_diagnostics`, the #1283
  ``may_curate`` shape repeated rather than reused: confinement facts (a
  runner pid, which AppArmor profile, whether the kernel is genuinely
  enforcing it) are arguably MORE sensitive than memory content, so the
  stricter of the two owner rules in this tree applies -- the workspace
  owner may look; on an UNOWNED workspace anyone who can see it may; an
  identity-less connection on an OWNED workspace may not.  This is the
  #1294 open question answered with its own suggested default.
* **which facts are cached and which are live** -- the whole reason the
  view exists (a cached ``sandbox_mode`` reading "confined" is exactly
  what #1253 was filed about), so the two are never merged into one
  verdict here: cached facts (``runner_identity``, ``confinement_id``,
  ``sandbox_mode``) come from the daemon's own ``Session`` record; the
  live half (``probe``) is whatever the runner just measured, or ``None``
  when there was no runner to ask.
* **what gets logged about the asking** -- a live re-probe is a
  diagnostic action on a security boundary, and #1294 asks that its own
  use never be silent.  One line, at the single exit, for every outcome
  including a refusal: "who asked, for which session, what was found (or
  why nothing was)".

What a verb DOES is not decided here: the live probe comes from
``JaatoServer.diagnostics_probe``, which asks the runner holding the
threads -- there is no daemon-side fallback to guess from, because the
daemon does not have the runner's threads to look at.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from jaato_sdk.events import DiagnosticsRequest, DiagnosticsResultEvent

logger = logging.getLogger(__name__)

#: Every request this module answers, for ``handle_request``'s one arm --
#: the same shape ``memory_verbs.MEMORY_REQUEST_TYPES`` is read for.
DIAGNOSTICS_REQUEST_TYPES = (DiagnosticsRequest,)


def may_view_diagnostics(owner: Optional[str], user_id: Optional[str]) -> bool:
    """Whether *user_id* may see this session's confinement facts.

    Matches :func:`server.memory_verbs.may_curate` exactly, and is kept
    as its own definition rather than an import of that one: memory
    curation and diagnostics viewing are two different questions that
    happen to share an answer today, and tying this module to
    ``memory_verbs`` for the sake of one predicate would make it look
    like the two are the same policy rather than two policies that agree.
    """
    return owner is None or (user_id is not None and owner == user_id)


def _runner_identity_dict(session: Any) -> Optional[Dict[str, Any]]:
    """``Session.runner_identity`` as a plain dict, or ``None``.

    ``None`` covers both "this session has no runner" and "the identity
    could not be serialised" -- a diagnostics read must never raise for
    want of a field a client can simply not receive.
    """
    identity = getattr(session, "runner_identity", None)
    to_dict = getattr(identity, "to_dict", None)
    if not callable(to_dict):
        return None
    try:
        return dict(to_dict())
    except Exception:  # noqa: BLE001 -- a diagnostics read must not raise
        return None


def _apparmor_grants(
    confinement_id: str, server: Any,
) -> Optional[Dict[str, Any]]:
    """What the session's AppArmor profile grants, for the panel (#1326).

    ``None`` when the session names no AppArmor profile.  Otherwise the
    record :mod:`server.apparmor` kept when the profile was loaded, or
    ``{"recorded": False}`` when it has none (loaded before this daemon
    started, or on a path that does not record).  Either way it carries
    ``declared_by`` and ``requested_fragments`` from the session's own
    resolved profile, because which profile declared the fragments is a
    fact about the session, not about the boundary it shares.
    """
    from jaato_server.server.apparmor import recorded_grants

    if not confinement_id:
        return None
    grants = recorded_grants(confinement_id)
    result: Dict[str, Any] = dict(grants) if grants else {}
    result["recorded"] = grants is not None
    profile = getattr(server, "_profile", None)
    if profile is not None:
        result["declared_by"] = getattr(profile, "apparmor_fragments_source", None)
        if not grants:
            requested = getattr(profile, "apparmor_fragments", None)
            result["requested_fragments"] = (
                None if requested is None else list(requested))
    return result


def _compose_result(
    request_id: str, session: Any, probe_answer: Dict[str, Any],
    server: Any = None,
) -> DiagnosticsResultEvent:
    """Build the answer from the daemon's cached ``Session`` record plus
    whatever ``JaatoServer.diagnostics_probe`` returned.

    ``"probe" in probe_answer`` is the discriminator between "the runner
    answered" (every field populated) and "there was nothing to ask, or
    it did not answer" (``ok`` stays ``True`` for the CACHED facts, which
    are still a real answer; only ``probe`` is ``None``, and ``category``
    names why).
    """
    from jaato_server.server.command_router import _daemon_version

    identity = getattr(session, "runner_identity", None)
    confinement_id = str(getattr(identity, "apparmor_profile", "") or "")
    sandbox_mode = getattr(session, "sandbox_mode", None)
    runner_identity = _runner_identity_dict(session)
    apparmor_grants = _apparmor_grants(confinement_id, server)

    if "probe" in probe_answer:
        return DiagnosticsResultEvent(
            request_id=request_id,
            ok=True,
            runner_identity=runner_identity,
            confinement_id=confinement_id,
            sandbox_mode=sandbox_mode,
            consumption=probe_answer.get("consumption"),
            notebook_boundary_kind=probe_answer.get("notebook_boundary_kind"),
            protocol_version=str(probe_answer.get("protocol_version") or ""),
            server_version=_daemon_version(),
            probe=probe_answer.get("probe"),
            apparmor_grants=apparmor_grants,
        )

    # No runner to probe, or the runner did not answer.  The cached facts
    # are still an honest answer -- ``ok`` names whether ANYTHING could be
    # reported, and it can; only the LIVE half is missing, and ``category``
    # says why rather than the field silently reading ``None``.
    return DiagnosticsResultEvent(
        request_id=request_id,
        ok=True,
        runner_identity=runner_identity,
        confinement_id=confinement_id,
        sandbox_mode=sandbox_mode,
        server_version=_daemon_version(),
        probe=None,
        apparmor_grants=apparmor_grants,
        error=str(probe_answer.get("error") or ""),
        category=str(probe_answer.get("category") or "no_runner"),
    )


def _trace_probe(
    user_id: Optional[str],
    session_id: str,
    allowed: bool,
    result: DiagnosticsResultEvent,
) -> None:
    """The one audible record of a diagnostics probe, allowed or refused.

    A single exit, structural rather than per-branch, mirroring #951's
    ``[PERMISSION] ... DECISION`` line: a live re-probe of a security
    boundary is a diagnostic action worth an audit trail on its own, and
    a silent one is exactly the shape the #951 family exists to close.
    """
    probe = result.probe or {}
    logger.info(
        "[DIAGNOSTICS] probe: user=%s session=%s allowed=%s ok=%s "
        "enforced=%s confinement_id=%s sandbox_mode=%s category=%s",
        user_id or "-", session_id or "-", allowed, result.ok,
        probe.get("enforced") if result.probe is not None else "unknown",
        result.confinement_id or "-", result.sandbox_mode or "-",
        result.category or "-",
    )


def answer_diagnostics_request(
    server: Any,
    event: DiagnosticsRequest,
    *,
    session_id: str,
    user_id: Optional[str],
    owner: Optional[str],
    session: Any = None,
) -> DiagnosticsResultEvent:
    """Serve one ``DiagnosticsRequest`` end to end: gate, probe, compose.

    Args:
        server: The session's ``JaatoServer``, or ``None`` when the
            caller is attached to no loaded session (answered
            ``no_session``).
        event: The request.
        session_id: The session the request is served for -- the "this
            session" the caller's connection is attached to, resolved by
            the router before this function is reached.  Never a value
            read off the request: ``event`` declares no session-naming
            field of its own, so nothing here could resolve one even if
            it wanted to (#1294's no-cross-session-reach requirement).
        user_id: The identity the TRANSPORT authenticated for the caller,
            never a field of the request.
        owner: The session workspace's qualified owner, ``None`` when
            unowned.
        session: The daemon's own ``Session`` record, for the CACHED
            facts (``runner_identity``, ``sandbox_mode``) only it holds.
            ``None`` when there is no session (matches ``server is None``).

    A caller the owner gate refuses is answered ``not_owner`` WITHOUT
    asking the runner: no probe, no thread scan, for a request already
    decided not to be answered.

    Every outcome is traced once, allowed or refused, before returning.
    """
    allowed = may_view_diagnostics(owner, user_id)

    if server is None:
        result = DiagnosticsResultEvent(
            request_id=event.request_id,
            ok=False,
            error="no session is attached to this connection",
            category="no_session",
        )
    elif not allowed:
        result = DiagnosticsResultEvent(
            request_id=event.request_id,
            ok=False,
            error=(
                "only the owner of this session's workspace may view "
                "its diagnostics"
            ),
            category="not_owner",
        )
    else:
        probe_answer = server.diagnostics_probe()
        result = _compose_result(event.request_id, session, probe_answer, server)

    _trace_probe(user_id, session_id, allowed, result)
    return result


__all__ = [
    "DIAGNOSTICS_REQUEST_TYPES",
    "answer_diagnostics_request",
    "may_view_diagnostics",
]
