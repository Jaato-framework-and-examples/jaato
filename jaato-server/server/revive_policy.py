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
per-invocation operator choices (run the interrogation harness against a
finished session), not properties of an agent, which is the other half of
the argument.  See ``docs/design/env-vars-vs-profile-keys.md`` and the
``host``-scope entries in ``shared/env_scope.py``.

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
from typing import Optional

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

#: Accepted spellings, mapped to the canonical value.  Deliberately small:
#: an operator typing ``reload`` or ``rerender`` means ``disk``, and an
#: operator typing anything else has made a mistake worth saying out loud
#: rather than silently reading as the default.
_ALIASES = {
    "persisted": PERSISTED,
    "persist": PERSISTED,
    "saved": PERSISTED,
    "frozen": PERSISTED,
    "disk": DISK,
    "reload": DISK,
    "rerender": DISK,
    "re-render": DISK,
    "render": DISK,
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


def profile_source() -> str:
    """Where a revived session's profile comes from.

    Read from ``os.environ`` rather than the session-env ContextVar on
    purpose: this is the DAEMON's revive posture for the process, decided
    by the operator who started it, and the session being revived has no
    env context yet — resolving its workspace ``.env`` is downstream of
    the very decision this makes.

    Returns:
        :data:`PERSISTED` (default) or :data:`DISK`.
    """
    return _resolve(ENV_REVIVE_PROFILE, os.environ.get(ENV_REVIVE_PROFILE))


def persona_source() -> str:
    """Where a revived session's rendered system instruction comes from.

    Returns:
        :data:`PERSISTED` (default) or :data:`DISK`.  ``DISK`` re-renders,
        which re-runs the persona's prefetch scripts — see the module
        docstring; that is not necessarily a read-only operation.
    """
    return _resolve(ENV_REVIVE_PERSONA, os.environ.get(ENV_REVIVE_PERSONA))
