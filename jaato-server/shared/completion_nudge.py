"""The completion-nudge budget: one default, and the profile knob that moves it.

WHAT A NUDGE IS.  A session whose tool surface carries ``signal_completion``
is expected to call it before its loop ends.  When the loop settles without
that call, the framework re-prompts the model — a *nudge* — and re-enters
the loop.  The budget bounds how many times it will do that before giving up
and producing the ``NudgeExhausted`` terminal, so a model that keeps
narrating completion instead of invoking it eventually halts.

WHY THIS MODULE EXISTS.  The number was a function-local ``MAX_COMPLETION_
NUDGES = 2`` in three unrelated files — ``server/core.py`` (the daemon's
top-level guard), ``jaato_embedded/client.py`` (the in-process lead) and
``shared/plugins/subagent/plugin.py`` (the subagent loop).  Nothing kept the
three equal, and none of them was reachable from a profile, which made the
nudge budget the one bound in the completion path a deployment could not
express: ``max_turns``, ``runtime_limits`` and a processor's ``max_refusals``
all are (#919).

Two is a good default and stays one — for a strong tool-caller it is right,
and raising it globally would make weak models loop longer for everyone.
What a deployment that knows its model needs four attempts gains is a way to
say so:

.. code-block:: yaml

    # .jaato/profiles/<agent>.yaml
    max_completion_nudges: 4      # default 2, unchanged when unset

The value is per SESSION, not per turn — the same claim ``_completion_nudges_
fired`` has always made (#767): a turn start does not refund a nudge, or the
subagent loop's ``while`` could not terminate at all.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: How many times the framework re-prompts a session that settled without
#: calling ``signal_completion`` before giving up.  The framework default,
#: used whenever a profile declares no ``max_completion_nudges`` — and the
#: single definition the daemon, the embedded lead and the subagent loop
#: all read, so the three cannot drift.
#:
#: ``jaato_eval.sign_off.MAX_COMPLETION_NUDGES`` restates it deliberately
#: (the eval engine must not import ``server.*`` / ``shared.*``); that copy
#: is a reporting ceiling, not a budget, and is stale in exactly one
#: direction against a profile that raised its own.
DEFAULT_MAX_COMPLETION_NUDGES = 2


def resolve_max_completion_nudges(profile: Any) -> int:
    """The nudge budget this profile asks for, or the framework default.

    The single reader of the profile knob.  Every nudge site calls this
    rather than reading ``profile.max_completion_nudges`` itself, so a
    profile object that predates the field (an older snapshot, a duck-typed
    stand-in, ``None`` for a session created without a profile) resolves to
    the default instead of raising.

    Args:
        profile: The resolved
            :class:`~shared.plugins.subagent.config.SubagentProfile`, or
            ``None`` for a session created without one.

    Returns:
        A positive int: the profile's declared budget, else
        :data:`DEFAULT_MAX_COMPLETION_NUDGES`.
    """
    if profile is None:
        return DEFAULT_MAX_COMPLETION_NUDGES
    return coerce_max_completion_nudges(
        getattr(profile, "max_completion_nudges", None),
        where=repr(getattr(profile, "name", "<unnamed>")),
    )


def coerce_max_completion_nudges(
    declared: Any,
    *,
    where: str = "<caller>",
) -> int:
    """The same resolution against a RAW declared value.

    The sibling of :func:`resolve_max_completion_nudges` for callers that
    hold the number rather than the profile — the embedded facade, which
    receives a profile *spec dict* across its boundary and never
    reconstructs the dataclass.  Splitting it this way keeps one
    implementation of the rule instead of asking such a caller to
    fabricate a stand-in object with the right attribute.

    Defensive about the VALUE as well as its presence: ``validate_profile``
    rejects a non-positive or non-integer ``max_completion_nudges`` at
    load, so a bad value arriving here means a hand-built profile object or
    a spec dict assembled in code rather than a profile file.  Such a value
    is announced at WARNING and replaced by the default — the alternative,
    honouring ``0``, silently disarms the nudge AND (because the give-up
    predicate is ``nudges_fired >= max``) reports ``NudgeExhausted`` on
    sessions that completed cleanly.

    Args:
        declared: The raw value, or ``None`` for "not declared".
        where: Identifies the declaring profile in the WARNING; purely
            diagnostic.

    Returns:
        A positive int: *declared* when it is one, else
        :data:`DEFAULT_MAX_COMPLETION_NUDGES`.
    """
    if declared is None:
        return DEFAULT_MAX_COMPLETION_NUDGES
    if (isinstance(declared, bool)
            or not isinstance(declared, int)
            or declared < 1):
        logger.warning(
            "profile %s declares max_completion_nudges=%r, which is not a "
            "positive integer — using the framework default (%d)",
            where, declared, DEFAULT_MAX_COMPLETION_NUDGES,
        )
        return DEFAULT_MAX_COMPLETION_NUDGES
    return declared
