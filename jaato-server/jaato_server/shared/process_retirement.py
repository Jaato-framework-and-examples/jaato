"""Reasons THIS process may not serve another session (#1565, decision models).

``slot_retire_reason()`` lets a registry PLUGIN say that a pool slot holds
something no reset frees (references, after it loaded an embedding
model).  A model provider is not a registry plugin: a session builds its
decision providers itself (``JaatoSession.decision_provider``), so a
provider that loads weights into the runner (``laya`` with ``mode:
local``) has nowhere to say so.  It notes the reason here instead, and
the runner's ``session.end`` (``slot_retire_reasons`` in
``server/runner/rpc.py``) reports these beside the plugins' reasons.

A note lasts for the life of the process and is never cleared: what it
describes (torch's heap, the weights) is returned only by process exit.
A process that never loads anything notes nothing, and a pool slot is
returned exactly as before.
"""

from __future__ import annotations

import threading
from typing import List

_lock = threading.Lock()
_reasons: List[str] = []


def note_process_retirement(reason: str) -> None:
    """Record that this process holds something no reset frees.

    The same reason noted twice is kept once.
    """
    if not reason:
        return
    with _lock:
        if reason not in _reasons:
            _reasons.append(reason)


def process_retirement_reasons() -> List[str]:
    """The reasons noted so far, oldest first."""
    with _lock:
        return list(_reasons)
