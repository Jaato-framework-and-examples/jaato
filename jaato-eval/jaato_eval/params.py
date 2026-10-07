"""A task's inputs, exported as environment — one encoding, two consumers.

A shell command cannot read a Python mapping, so anything the engine
starts as a PROCESS receives the task's inputs as environment variables.
Two things are started that way:

* a ``script`` grader (:mod:`jaato_eval.graders.script`), which receives
  the arm's ``agent_params`` so an input-dependent check can follow the
  input by construction instead of by remembering it (jaato #762);
* a driver (``harness.kind: driver``, :mod:`jaato_eval.driver`), which
  receives ``input.params`` — for a driver arm that mapping IS the input.

The encoding lived in the script grader while it was the only consumer.
It moved here rather than being copied, because the whole value of the
export is that a grader and the thing it grades read the SAME variable
for the same input: a driver told ``JAATO_EVAL_PARAM_TICKER=NVDA`` and a
scorer told ``JAATO_EVAL_PARAM_TICKER=NVDA`` are talking about one arm,
and two encoders could disagree about the name or the value.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Optional, Tuple

#: Prefix for the per-parameter variables.  Namespaced so a task's inputs
#: cannot shadow anything the surrounding environment already means — a
#: parameter called ``path`` must not become ``$PATH``.
PARAM_PREFIX = "JAATO_EVAL_PARAM_"

#: Variable carrying the whole mapping as JSON, under the authors' own key
#: spellings.  It is what distinguishes "this task has no such parameter"
#: from "the parameter is there and empty" — the per-key variables cannot,
#: because an unset variable and an empty one read alike in a shell.  A
#: consumer that must be strict about a parameter it depends on can assert
#: against this rather than passing vacuously.
PARAMS_JSON = "JAATO_EVAL_PARAMS"

#: Characters an environment variable name cannot carry.  Parameter keys
#: are YAML identifiers by convention but nothing enforces it, so
#: ``issue-id`` and ``issue.id`` have to become something a shell can name.
_NON_IDENTIFIER = re.compile(r"[^A-Z0-9_]")


def env_name(key: str) -> str:
    """Environment variable name for one parameter key."""
    return PARAM_PREFIX + _NON_IDENTIFIER.sub("_", key.upper())


def encode(value: Any) -> str:
    """Render one parameter value for a shell.

    Strings pass through verbatim — quoting them would mean every consumer
    had to unquote, and the common case is a bare identifier.  Everything
    else is JSON, which is the only encoding that survives the round trip
    for the values a manifest actually holds: a nested dict or list has no
    other textual form a consumer could parse back, ``True`` becomes the
    ``true`` the manifest author wrote rather than Python's ``True``, and
    ``None`` becomes ``null`` instead of the empty string that would make
    an explicit null indistinguishable from an absent key.

    ``default=str`` keeps an exotic value (a ``Path``, say) from raising
    out of a grader — the adapter's contract is to describe what it can
    do, never to explode inside the sweep driver.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def param_env(params: Mapping[str, Any]) -> Tuple[Dict[str, str], Optional[str]]:
    """Exported environment for ``params``, or a reason it cannot be.

    Returns ``(env, None)`` normally, and ``({}, reason)`` when two
    parameter keys collapse onto one variable name (``issue-id`` and
    ``issue_id`` both want ``$JAATO_EVAL_PARAM_ISSUE_ID``).  Picking a
    winner there would hand the consumer one input while the arm ran with
    the other — precisely the silent disagreement this export exists to
    remove — so the ambiguity is surfaced and the task author renames one
    key.  The manifest parser refuses a ``kind: driver`` task on it before
    anything is spent; the script grader returns it as BLOCKED.
    """
    env = {PARAMS_JSON: json.dumps(dict(params), sort_keys=True, default=str)}
    origin: Dict[str, str] = {}
    for key, value in params.items():
        name = env_name(key)
        if name in origin:
            return {}, (f"parameter keys {origin[name]!r} and {key!r} both map "
                        f"to ${name}; rename one — grading against whichever "
                        "won would be arbitrary")
        origin[name] = key
        env[name] = encode(value)
    return env, None
