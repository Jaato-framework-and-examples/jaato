"""Whether a person approved THIS tool call, as the tool body sees it.

The permission gate decides before a tool runs, and a tool's executor never
saw the decision: ``ToolExecutor`` injected a ``_permission`` block into the
RESULT afterwards, which is too late for a tool that wants to record the
approval in what it writes.  ``proposeReference`` is that tool: a claim a
person approved at the prompt is worth more than one nobody saw, and the
claim file is written during the call.

So the executor binds a **witness** for the duration of the tool body and a
tool reads it with :func:`current_call_witness`.  A witness exists only when
the call REACHED the approval channel and was allowed -- the permission
plugin's ``asked`` flag (#968), which is what separates "a person said yes"
from "the policy said yes".  ``method`` alone cannot: ``allow_all`` and the
two suspensions are produced both by a person answering ``a`` / ``t`` / ``i``
and by a pre-approval short-circuit that asked nobody.

Shape (only the keys that are known are present)::

    {"via": "permission-prompt", "method": "user_approved",
     "user": "app:alice", "approver": "...", "edited": True}

``user`` is the identity the daemon authenticated for the responder and
``approver`` the name an external approval system attached (#859).  Neither
is invented: an approval answered on an unauthenticated channel yields a
witness with neither, which still says "somebody was asked and said yes".

Stdlib only, and holds no state beyond one ``ContextVar``: the executor runs
tool bodies on pool threads, and a ContextVar set on the thread that runs
the body is visible to exactly that body.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

#: The ``via`` value of every witness this module builds.
WITNESS_VIA_PERMISSION_PROMPT = "permission-prompt"

_WITNESS: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "jaato_call_witness", default=None,
)


def witness_from_verdict(
    allowed: bool, perm_info: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """The witness one permission verdict yields, or ``None``.

    ``None`` unless the call was ALLOWED and the plugin reports
    ``asked=True``.  A verdict from a plugin that does not report ``asked``
    at all (an out-of-tree policy engine) yields ``None`` too: absence of
    evidence is not a witness.
    """
    if not allowed or not isinstance(perm_info, dict):
        return None
    if perm_info.get("asked") is not True:
        return None
    witness: Dict[str, Any] = {
        "via": WITNESS_VIA_PERMISSION_PROMPT,
        "method": str(perm_info.get("method") or "unknown"),
    }
    for src, dst in (("user_id", "user"), ("approver", "approver")):
        value = perm_info.get(src)
        if isinstance(value, str) and value:
            witness[dst] = value
    if perm_info.get("was_edited"):
        witness["edited"] = True
    return witness


def current_call_witness() -> Optional[Dict[str, Any]]:
    """The witness of the tool call running on this context, as a copy."""
    witness = _WITNESS.get()
    return dict(witness) if witness else None


@contextlib.contextmanager
def bound_call_witness(witness: Optional[Dict[str, Any]]) -> Iterator[None]:
    """Make ``witness`` the current call's for the duration of the block.

    Always binds, ``None`` included, so a body run on a pool thread that
    last ran an approved call cannot read that call's witness.
    """
    token = _WITNESS.set(dict(witness) if witness else None)
    try:
        yield
    finally:
        _WITNESS.reset(token)
