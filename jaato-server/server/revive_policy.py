"""Where a revived session gets its recipe and its prompt from (issue #787).

A session that is woken from disk — ``session.wake``, a reattach, anything
that reaches ``SessionManager._load_session`` — has two things it must come
back with: the **profile** it ran under (model, provider, plugins, GC,
budget) and the **persona**, meaning the fully rendered system instruction
the model was actually given on turn 1.

Historically both were RE-DERIVED at revive: the profile by re-resolving
``profile_name`` against ``.jaato/profiles/`` on disk, and the prompt by
re-reading the instruction layers, re-resolving the agent markdown, and
**re-running the persona's** ``{{!py:...}}`` **prefetch scripts**.  Three
things follow from that, and the first is what #787 reported:

1. ``agent_params`` were not persisted, so a mandatory prefetch that reads
   ``context.agent_params`` was handed an empty dict, raised, and aborted
   session-prep.  A session that was created fine, ran for eleven minutes
   and persisted fine could then not be woken by anything.
2. A prefetch is documented as running ONCE, at session-prep, before the
   model's first turn.  Re-deriving ran it again on every revive — and the
   reported case does not merely compute text, it materialises a git
   worktree.
3. The rebuilt prompt could differ from the original, because the files it
   is built from may have been edited since.  The session resumed with a
   history produced under one prompt and continued under another, silently.

The framework now PERSISTS both (``SessionState.profile_snapshot`` and
``SessionState.rendered_instructions``) and restores them, which removes
all three.  The operator decision recorded on #787 is explicit: *a revived
session keeps the instructions it was created under; a session that wants
new instructions is a new session.*

WHY THESE ARE ENV VARS AND NOT PROFILE KEYS.  Both knobs decide **whether
the profile is read at all**.  A key inside the profile could not express
that without being self-referential — the daemon would have to load the
file to learn whether it was allowed to load the file.  They are also
operator choices about a run rather than properties of an agent, which is
the other half of the argument.  See
``docs/design/env-vars-vs-profile-keys.md`` and the ``host``-scope entries
in ``shared/env_scope.py``.

THEY ARE PER-PROCESS, NOT PER-INVOCATION.  :func:`capture` resolves both
once, when the ``SessionManager`` is constructed, and the answer holds for
every session that process revives afterwards.  On a long-lived daemon
that means **changing the posture requires restarting the daemon**, and
until then it applies to every revive — not only the one the operator had
in mind.  For the motivating interrogation workflow the blast radius is
mild, because that combination needs only ``JAATO_REVIVE_PROFILE=disk``,
which is read-only.  It is ``JAATO_REVIVE_PERSONA=disk`` that has side
effects (below), so weigh that one against the fact that it stays on.

Freezing is also what makes the ``host`` scope true rather than merely
declared: a live ``os.environ`` read would be settable process-wide from
any single workspace's ``.env``.  :func:`capture` documents that vector in
full.

THE COMBINATION THAT MOTIVATED THE KNOBS.  Interrogating a finished session
— waking it to ask it to account for something — wants
``JAATO_REVIVE_PROFILE=disk`` with the persona left persisted: the profile
must re-resolve because interrogation selects a different contract via
``JAATO_PROFILE_SET`` (a profile set is resolved at ``discover_profiles``,
so a frozen profile would make the set selection silently inert), while the
persona must be the one the session actually saw, or the answers are about
a prompt it never had.  That combination is neither knob's default, which
is why both exist:

    ==========================================  =========  =========
    intent                                      profile    persona
    ==========================================  =========  =========
    resume the same work (default)              persisted  persisted
    interrogate under a different contract      disk       persisted
    test an alternative persona                 either     disk
    ==========================================  =========  =========

RE-RENDERING IS NOT A READ-ONLY OPTION.  ``JAATO_REVIVE_PERSONA=disk``
re-runs the persona's prefetch scripts (against the ORIGINAL, persisted
``agent_params`` — not against nothing, which was the bug).  Nothing in the
``{{!py:...}}`` contract requires a prefetch to be pure, and the case that
produced this issue is emphatically not: it runs ``git worktree add``.  So
that value may execute side effects, and a prefetch author who expects it
to be re-run should make theirs idempotent.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

#: Use what the session persisted at creation.  The default for both knobs.
PERSISTED = "persisted"

#: Re-read from disk at revive time — re-resolve the profile by name, or
#: re-render the persona (which RE-RUNS its prefetch scripts, see above).
#: Also the automatic fallback when nothing was persisted, which is what
#: makes this change backward compatible: every session already on disk
#: carries neither snapshot and revives exactly as it did before.
DISK = "disk"

#: Env var selecting where a revived session's PROFILE comes from.
ENV_REVIVE_PROFILE = "JAATO_REVIVE_PROFILE"

#: Env var selecting where a revived session's PERSONA (the rendered system
#: instruction) comes from.
ENV_REVIVE_PERSONA = "JAATO_REVIVE_PERSONA"

#: Accepted spellings, mapped to the canonical value.
#:
#: Deliberately minimal: the two documented values, plus the three ways an
#: operator unambiguously spells "go back to disk".  Anything else warns and
#: takes the default, which is the safe direction — the default never runs a
#: prefetch script.
#:
#: ``render`` is NOT here, and its absence is the point.  It read as ``disk``,
#: which is backwards: the PERSISTED value *is* the rendered prompt (the field
#: is ``rendered_instructions``).  An operator typing it plausibly means "give
#: me the rendered one" and would have silently got the opposite — and the
#: opposite is the side-effecting one, accepted without a warning because it
#: was a recognised value.  Now it warns and takes the default.
_ALIASES = {
    "persisted": PERSISTED,
    "disk": DISK,
    "reload": DISK,
    "rerender": DISK,
    "re-render": DISK,
}


def _resolve(var: str, raw: Optional[str]) -> str:
    """Map one env value to :data:`PERSISTED` / :data:`DISK`.

    Args:
        var: The env var's name, for the warning message.
        raw: Its raw value, or ``None`` / empty when unset.

    Returns:
        The canonical source.  Unset resolves to :data:`PERSISTED`; an
        unrecognised value ALSO resolves to :data:`PERSISTED`, but loudly —
        a silently-ignored knob is how an operator concludes the feature
        does not work.
    """
    if raw is None or not raw.strip():
        return PERSISTED
    resolved = _ALIASES.get(raw.strip().lower())
    if resolved is None:
        logger.warning(
            "%s=%r is not a recognised value (expected one of %s); "
            "using %r", var, raw, sorted(set(_ALIASES)), PERSISTED,
        )
        return PERSISTED
    return resolved


#: The posture, resolved ONCE and held for the process.  ``None`` until
#: :func:`capture` runs.  See that function for why this is not a per-call
#: read of ``os.environ``.
_CAPTURED: Optional[Dict[str, str]] = None


def capture() -> Dict[str, str]:
    """Resolve both knobs from ``os.environ`` once, and hold the answer.

    Called from ``SessionManager.__init__`` — the earliest point at which a
    daemon exists and the latest at which no session does, so the values
    read here are the ones the operator started the process with.

    WHY THIS IS NOT A PER-CALL ``os.environ`` READ.  ``JaatoServer.
    _with_session_env`` copies **every** key of a session's workspace
    ``.env`` into the daemon-global ``os.environ`` for the duration of that
    session's turn, with no scope filter::

        for key, value in self._session_env.items():
            saved[key] = os.environ.get(key)
            os.environ[key] = value

    A ``host``-scoped knob read live would therefore be settable, process-
    wide, from any one workspace's ``.env`` — and operators do put
    host-scoped vars there (the sweep harness's own ``.env`` carries
    ``JAATO_RUNNER_POOL_SIZE``, which the catalog classifies ``host``).  A
    concurrent revive reading during that window would see the other
    workspace's value; for ``JAATO_REVIVE_PERSONA`` that means re-running
    prefetch scripts, and the motivating one runs ``git worktree add``.

    Freezing removes the window rather than narrowing it, and makes the
    module's stated intent — a posture for the PROCESS — enforceable
    instead of merely documented.

    Returns:
        The captured mapping, for logging or assertion by the caller.
    """
    global _CAPTURED
    _CAPTURED = {
        ENV_REVIVE_PROFILE: _resolve(
            ENV_REVIVE_PROFILE, os.environ.get(ENV_REVIVE_PROFILE)),
        ENV_REVIVE_PERSONA: _resolve(
            ENV_REVIVE_PERSONA, os.environ.get(ENV_REVIVE_PERSONA)),
    }
    non_default = {k: v for k, v in _CAPTURED.items() if v != PERSISTED}
    if non_default:
        logger.info(
            "revive posture for this process: %s.  This is captured ONCE at "
            "startup and applies to every session revived until the daemon "
            "restarts; %s=disk re-runs the persona's prefetch scripts.",
            ", ".join(f"{k}={v}" for k, v in sorted(_CAPTURED.items())),
            ENV_REVIVE_PERSONA,
        )
    return dict(_CAPTURED)


def reset() -> None:
    """Drop the captured posture so the next read re-resolves.

    For tests, and for an embedded caller that deliberately wants to change
    the posture mid-process.  Production never calls this: the whole point
    of :func:`capture` is that the value cannot move under a running
    daemon.
    """
    global _CAPTURED
    _CAPTURED = None


def _source(var: str) -> str:
    """The captured value for ``var``, or a live read if nothing is captured.

    The live fallback keeps an embedded / test caller working without a
    ``SessionManager``.  It is NOT the production path — see
    :func:`capture`.
    """
    if _CAPTURED is not None:
        return _CAPTURED[var]
    return _resolve(var, os.environ.get(var))


def profile_source() -> str:
    """Where a revived session's profile comes from.

    Returns:
        :data:`PERSISTED` (default) or :data:`DISK`.
    """
    return _source(ENV_REVIVE_PROFILE)


def persona_source() -> str:
    """Where a revived session's rendered system instruction comes from.

    Returns:
        :data:`PERSISTED` (default) or :data:`DISK`.  ``DISK`` re-renders,
        which re-runs the persona's prefetch scripts — see the module
        docstring; that is not necessarily a read-only operation.
    """
    return _source(ENV_REVIVE_PERSONA)
