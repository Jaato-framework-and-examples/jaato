"""Client-side handling of batched clarification requests.

WHY THIS EXISTS.  ``request_clarification`` reaches a client on two
different shapes of wire traffic, and until #704 the TUI only knew one
of them:

* **Daemon-local sessions** stream the request question by question —
  an ``AgentOutputEvent`` carrying the rendered question text, then a
  ``ClarificationInputModeEvent`` telling the client to take input.  The
  daemon also emits a ``ClarificationBatchEvent`` up front as an optional
  preview (``batch_only=False``); a client using the per-question flow
  ignores it.
* **Runner-tier sessions** (the pre-warm pool / confined runner, on by
  default) relay the whole request in ONE ``ClarificationBatchEvent``
  stamped ``batch_only=True``.  Nothing else follows it: no question
  text, no input-mode event, no ``ClarificationResolvedEvent``.  A client
  that ignores it never prompts, never answers, and the tool call — and
  the turn behind it — blocks forever.

So a ``batch_only`` batch makes the client responsible for everything the
daemon does on the other path: rendering each question, walking them in
order, resolving the tool tree entry when the last one is answered, and
sending exactly one ``ClarificationBatchResponseEvent`` back.  That is
what this module implements, kept out of ``rich_client`` so the event and
input loops there stay flat.

RENDERING mirrors ``shared.plugins.clarification.channels.QueueChannel``
line for line, so a clarification looks the same whichever path carried
it — plus a ``cancel`` hint on choice questions, since cancelling is the
only way out of a question the user cannot answer.

ANSWER PARSING here is DISPLAY ONLY.  ``answer_summary`` mirrors
``ClarificationChannel._parse_answer`` well enough to name the choice a
user picked in the tool tree; the server re-parses the same raw strings
and remains the authority on what the model is told.
"""

from typing import Any, Callable, Dict, List, Optional

from jaato_sdk.events import ClarificationBatchEvent


# Typed by duck-typing rather than imported: this module is loaded by the
# TUI, which must not drag the server package in.
Question = Dict[str, Any]


def _choices(question: Question) -> List[Dict[str, Any]]:
    """Return the question's choice dicts (``[]`` for free text)."""
    return question.get("choices") or []


def format_question_lines(
    question: Question,
    index: int,
    total: int,
    context: str = "",
) -> List[str]:
    """Render one question the way ``QueueChannel`` renders it.

    Args:
        question: One entry of ``ClarificationBatchEvent.questions``.
        index: 1-based position of this question.
        total: How many questions the batch carries.
        context: The request's context blurb, shown above question 1 only.

    Returns:
        Display lines, ready to be joined with newlines and appended to
        an ``OutputBuffer`` under the ``clarification`` source.
    """
    lines: List[str] = []
    if index == 1 and context:
        lines.append(f"Context: {context}")
        lines.append("")

    req_status = "*required" if question.get("required") else "optional"
    lines.append(f"Question {index}/{total} [{req_status}]")
    lines.append(f"  {question.get('text', '')}")

    choices = _choices(question)
    default_choice = question.get("default_choice")
    for j, choice in enumerate(choices, 1):
        marker = " (default)" if default_choice == j else ""
        lines.append(f"    {j}. {choice.get('text', '')}{marker}")

    qtype = question.get("question_type", "free_text")
    if qtype == "single_choice":
        lines.append(f"  Enter choice [1-{len(choices)}], or 'cancel':")
    elif qtype == "multiple_choice":
        lines.append("  Enter choices (comma-separated, e.g., 1,3), or 'cancel':")
    elif question.get("required"):
        lines.append("  (type 'cancel' to cancel)")
    else:
        lines.append("  (press Enter to skip, or type 'cancel' to cancel)")
    return lines


def _selected_indices(question: Question, raw: str) -> List[int]:
    """Best-effort read of which choices *raw* selects, for display.

    Mirrors ``ClarificationChannel._parse_answer``'s choice branches
    including its fallbacks (empty falls back to ``default_choice``,
    unparseable falls back to choice 1) so the tool tree names the same
    choice the model will be told about.  An empty list means "skipped".
    """
    raw = (raw or "").strip()
    choices = _choices(question)
    if not raw:
        default_choice = question.get("default_choice")
        if default_choice and question.get("question_type") == "single_choice":
            return [default_choice]
        return [] if not question.get("required") else [1]

    selected: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part.isdigit():
            continue
        num = int(part)
        if 1 <= num <= len(choices) and num not in selected:
            selected.append(num)
    if question.get("question_type") == "single_choice":
        selected = selected[:1]
    return selected or [1]


def answer_summary(question: Question, raw: str) -> str:
    """Describe *raw* as an answer to *question*, for the tool tree.

    Matches the strings the clarification plugin puts in a
    ``ClarificationResolvedEvent``'s ``qa_pairs``: choice answers resolve
    to their choice text, free text shows verbatim, and an unanswered
    optional question shows as ``(skipped)``.  An empty answer to a
    *required* free-text question is an empty answer, not a skip — the
    server records it as one too.
    """
    qtype = question.get("question_type", "free_text")
    if qtype in ("single_choice", "multiple_choice"):
        indices = _selected_indices(question, raw)
        if not indices:
            return "(skipped)"
        choices = _choices(question)
        return ", ".join(
            choices[i - 1].get("text", f"choice {i}")
            if 1 <= i <= len(choices) else f"choice {i}"
            for i in indices
        )
    text = (raw or "").strip()
    if text:
        return text
    return "" if question.get("required") else "(skipped)"


def build_pending(event: Any) -> Dict[str, Any]:
    """Build the ``pending_clarification_request`` state for a batch.

    The dict is the same one the per-question flow uses (so the input
    loop's ``request_id`` lookup is unchanged) plus the batch-only keys:
    ``batch`` marks which flow we are in, ``questions`` is the payload to
    walk, ``answers`` accumulates the raw strings, and ``index`` is the
    0-based position of the question currently being asked.
    """
    questions = list(event.questions or [])
    return {
        "request_id": event.request_id,
        "agent_id": event.agent_id,
        "tool_name": event.tool_name or "request_clarification",
        "batch": True,
        "context": event.context or "",
        "questions": questions,
        "answers": [],
        "index": 0,
        "current_question": 1,
        "total_questions": len(questions),
    }


def resolve_buffer(agent_registry: Any, agent_id: str) -> Any:
    """Find the output buffer for *agent_id*, falling back to the selected one.

    Used by the BATCH path only.  A batch event relayed from a runner
    carries the clarification plugin's agent name rather than an agent id
    — often empty for the main agent — so an id that resolves to nothing
    must not swallow the prompt.  The per-question path keeps its
    stricter lookup (see :func:`enter_clarification_input_mode`), where
    the id comes from the daemon's own agent registry and a miss means
    something is wrong rather than merely unnamed.
    """
    buffer = agent_registry.get_buffer(agent_id) if agent_id else None
    return buffer if buffer is not None else agent_registry.get_selected_buffer()


def render_current_question(
    pending: Dict[str, Any],
    agent_registry: Any,
    display: Any,
) -> None:
    """Show the question at ``pending["index"]`` and take input for it.

    Writes the rendered lines into the agent's buffer under the
    ``clarification`` source — the same source the daemon's per-question
    ``AgentOutputEvent`` uses, so ``set_tool_awaiting_clarification``
    attaches them to the tool tree entry exactly as it does there.
    """
    index = pending["index"]
    questions = pending["questions"]
    total = len(questions)
    lines = format_question_lines(
        questions[index], index + 1, total, pending.get("context", "")
    )
    buffer = resolve_buffer(agent_registry, pending.get("agent_id", ""))
    if buffer is not None:
        buffer.append("clarification", "\n".join(lines), "write")
        buffer.set_tool_awaiting_clarification(
            pending["tool_name"], index + 1, total
        )
    display.set_waiting_for_channel_input(True)
    display.refresh()


def start(
    event: Any,
    agent_registry: Any,
    display: Any,
) -> Optional[Dict[str, Any]]:
    """Begin a ``batch_only`` clarification; return its pending state.

    Returns ``None`` for a batch with no questions — there is nothing to
    ask, and the caller answers it immediately rather than prompting.
    """
    pending = build_pending(event)
    if not pending["questions"]:
        return None
    render_current_question(pending, agent_registry, display)
    return pending


def _finish(
    pending: Dict[str, Any],
    agent_registry: Any,
    display: Any,
    *,
    cancelled: bool,
) -> None:
    """Close out the tool tree entry and leave clarification input mode.

    The client does this itself because a ``batch_only`` clarification
    gets no ``ClarificationResolvedEvent`` — the runner-side plugin has
    no daemon-side hooks to emit one.  Without it the tool would sit at
    "awaiting clarification" for the rest of the session.
    """
    buffer = resolve_buffer(agent_registry, pending.get("agent_id", ""))
    if buffer is not None:
        qa_pairs = None if cancelled else [
            (q.get("text", ""), answer_summary(q, a))
            for q, a in zip(pending["questions"], pending["answers"])
        ]
        buffer.set_tool_clarification_resolved(pending["tool_name"], qa_pairs)
    display.set_waiting_for_channel_input(False)
    display.refresh()


async def submit_answer(
    client: Any,
    pending: Dict[str, Any],
    text: str,
    agent_registry: Any,
    display: Any,
) -> Optional[Dict[str, Any]]:
    """Take *text* as the answer to the current question.

    Returns the pending state to keep waiting on, or ``None`` once the
    clarification is done — answered in full or cancelled.  Answers are
    held client-side and sent as ONE
    ``ClarificationBatchResponseEvent``; there is no per-question reply
    on this path.
    """
    if text.strip().lower() == "cancel":
        await client.respond_to_clarification_batch(
            pending["request_id"], [], cancelled=True
        )
        _finish(pending, agent_registry, display, cancelled=True)
        return None

    pending["answers"].append(text)
    pending["index"] += 1
    pending["current_question"] = pending["index"] + 1

    if pending["index"] < len(pending["questions"]):
        render_current_question(pending, agent_registry, display)
        return pending

    await client.respond_to_clarification_batch(
        pending["request_id"], pending["answers"]
    )
    _finish(pending, agent_registry, display, cancelled=False)
    return None


async def enter_clarification_input_mode(
    event: Any,
    pending: Optional[Dict[str, Any]],
    client: Any,
    agent_registry: Any,
    display: Any,
    trace: Callable[[str], None],
) -> Optional[Dict[str, Any]]:
    """Take input for a clarification, whichever event asked for it.

    Handles both events a clarification can arrive on, because what the
    client owes differs and the difference is not visible from the event
    type alone:

    * ``ClarificationInputModeEvent`` — the daemon already sent the
      rendered question as output and is waiting on the input queue.  All
      that is left is to note which question is live and switch the input
      pane into clarification mode.
    * ``ClarificationBatchEvent`` with ``batch_only`` — nothing else is
      coming.  Render question one here and let ``submit_answer`` walk
      the rest.
    * ``ClarificationBatchEvent`` without ``batch_only`` — a preview of a
      per-question flow that follows.  Ignored, or the client would
      prompt twice for the same request.

    Returns the pending-clarification state the caller should hold.
    """
    if isinstance(event, ClarificationBatchEvent):
        trace(
            f"  ClarificationBatchEvent: tool={event.tool_name}, "
            f"questions={len(event.questions or [])}, "
            f"batch_only={event.batch_only}"
        )
        if not event.batch_only:
            return pending
        batch = start(event, agent_registry, display)
        if batch is None:
            # Nothing to ask, but the runner is still blocked on an answer.
            # Guarded because this is the one reply sent from inside the
            # event loop: a send that raises here (mid-reconnect, say) would
            # kill the loop and take every other event with it, which is a
            # worse outcome than the stuck tool call it is trying to avoid.
            try:
                await client.respond_to_clarification_batch(event.request_id, [])
            except Exception as exc:  # noqa: BLE001 — transport boundary
                trace(f"  clarification batch auto-answer failed: {exc!r}")
            return pending
        return batch

    trace(
        f"  ClarificationInputModeEvent: tool={event.tool_name}, "
        f"q{event.question_index}/{event.total_questions}"
    )
    if not pending:
        pending = {"request_id": event.request_id, "agent_id": event.agent_id}
    pending["current_question"] = event.question_index
    pending["total_questions"] = event.total_questions
    pending["tool_name"] = event.tool_name
    buffer = (agent_registry.get_buffer(event.agent_id) if event.agent_id
              else agent_registry.get_selected_buffer())
    if buffer is not None:
        buffer.set_tool_awaiting_clarification(
            event.tool_name, event.question_index, event.total_questions
        )
    display.set_waiting_for_channel_input(True)
    display.refresh()
    return pending


async def submit_clarification_answer(
    client: Any,
    pending: Dict[str, Any],
    text: str,
    agent_registry: Any,
    display: Any,
) -> Optional[Dict[str, Any]]:
    """Send *text* as the answer to the live clarification question.

    Per-question requests go straight back to the daemon, which drives
    the next question and clears the state via
    ``ClarificationResolvedEvent``, so the pending state is returned
    unchanged.  Batched requests are walked client-side by
    :func:`submit_answer`, which returns ``None`` when the batch is done.
    """
    if not pending.get("batch"):
        await client.respond_to_clarification(pending["request_id"], text)
        return pending
    return await submit_answer(client, pending, text, agent_registry, display)
