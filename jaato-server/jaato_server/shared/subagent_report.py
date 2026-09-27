"""Which user-role messages are subagent reports, not a person speaking.

A subagent reports to its parent by ``inject_prompt(..., source_type=
SourceType.CHILD)``: its completion, its errors, its requests for a
permission or a clarification.  The parent's queue knows the message is a
CHILD, but that tag does not survive the trip to a client:

- **live**, the report reaches the parent as a continuation turn
  (``_on_continuation_needed`` hands the daemon the TEXT only), and the
  daemon echoed nothing for it, so a client saw no sign of it at all;
- **on replay**, it is a ``Role.USER`` message in the stored history like
  any other, so an attaching client drew it as the user's own turn.

A report is for the PARENT AGENT's awareness.  If a person needs to know
something from it, the parent says so.  So a client should be able to
show it apart from the user's turns, collapsed.  This module is the one
rule that decides which messages those are, and it is used on both paths:
the live echo in ``JaatoServer`` (``source="child"``) and the history
replay in ``history_pages`` (``origin="subagent"``).

**The rule is the marker, and why that is enough.**  The stored history
carries no source type (``Message`` has none), so the only thing both
paths can read is the text.  Every CHILD report in this tree begins with
:data:`SUBAGENT_REPORT_MARKER` (``test_subagent_reports_are_marked.py``
fails the build on a CHILD ``inject_prompt`` that does not), and the
subagent plugin tells the model it must never type that marker itself.
The drain collects high-priority messages (a person, a parent) BEFORE
idle-only ones, so a merged batch that begins with the marker holds no
person's message; one that begins with a person's message is shown as
theirs, whole.

A user who types the marker gets their own message drawn as a report.
That changes presentation only: the text still reaches the model the
same way, with the same authority.
"""

from __future__ import annotations

import re
from typing import Optional

#: How every subagent report begins.  Also the start of the header the
#: subagent plugin's instructions describe to the parent model.
SUBAGENT_REPORT_MARKER = "[SUBAGENT agent_id="

#: ``AgentOutputEvent.source`` of a report echoed to clients, live or in the
#: full replay.  Named after ``SourceType.CHILD``, the queue's own word.
SUBAGENT_REPORT_SOURCE = "child"

#: ``HistoryPageEvent`` unit ``origin`` of a report (the unit stays
#: ``kind: "user"``, so a client that predates the field draws it as before).
SUBAGENT_REPORT_ORIGIN = "subagent"

_HIDDEN_RE = re.compile(r"<hidden>.*?</hidden>", re.DOTALL)


def is_subagent_report(text: Optional[str]) -> bool:
    """Whether a user-role message is a subagent report.

    Args:
        text: The message text, as stored or as handed to a continuation.
            ``<hidden>`` spans are ignored, as the replay ignores them.

    Returns:
        ``True`` when the visible text begins (after leading whitespace)
        with :data:`SUBAGENT_REPORT_MARKER`.
    """
    if not text:
        return False
    visible = _HIDDEN_RE.sub("", text) if "<hidden>" in text else text
    return visible.lstrip().startswith(SUBAGENT_REPORT_MARKER)
