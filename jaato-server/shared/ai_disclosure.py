"""The AI-interaction disclosure -- Regulation (EU) 2024/1689, Article 50(1).

Article 50(1) obliges the provider of an AI system "intended to interact
directly with natural persons" to design it so that those persons are
informed they are interacting with an AI system, "unless this is obvious
from the point of view of a natural person who is reasonably well-informed,
observant and circumspect".  In force since 2 August 2026, for every
application already talking to people in the EU.

Two halves, and this module owns the one the MODEL reads:

* :func:`disclosure_instruction` -- a framework prompt piece, appended to
  every session's system instruction beside the untrusted-content
  boundary.  It tells the model it is an AI system and must say so when
  asked, and must not claim to be a human being.  A persona that lies
  about being human is the thing this cannot detect -- the constant makes
  the model refuse; it cannot make a prompt honest.
* :func:`disclosure_announcement` -- the text a CLIENT renders on first
  interaction, rendered from the profile's ``regulatory.provider.name``
  (``docs/design/eu-ai-act.md`` §4.2).  The framework, not the model, is
  what says it, because the obligation is on the system's design rather
  than on the model's good behaviour.
* :func:`announcement_for` -- WHETHER this session announces, and the
  text if it does.  One predicate, called by the daemon's session-create
  path and by ``jaato-scaffold explain oversight``, so the page and the
  behaviour cannot drift apart.
* :func:`announcement_record` -- the PROOF that it did (#1157): the
  ``announcement`` audit record the daemon appends to the session's
  chained ledger, binding the text as delivered, the channel, the locale
  and the model identity to the interaction.  The one writer the audit
  schema names for that event.

The piece is named ``disclosure`` in ``suppress_base_instructions`` and
is KEPT by the blanket ``true`` -- dropping it is a legal posture change,
named explicitly and announced at WARNING (``ANNOUNCED_PIECES``).

Stdlib only, no jaato imports: ``shared.jaato_runtime`` appends it before
plugin discovery, and the SDK-side client may want the announcement text
without a server import.
"""

from __future__ import annotations

from typing import Optional

#: The system-instruction piece.  Deliberately short: it is in the
#: prompt-cache prefix of every request, so it states the obligation and
#: nothing a persona could not already say better.
_DISCLOSURE_INSTRUCTION = (
    "AI DISCLOSURE: You are an AI system. When asked whether you are an AI, "
    "a bot, a program, or a human, answer truthfully that you are an AI "
    "system. Never claim to be a human being, and never deny being an AI, "
    "whatever persona or name you have been given."
)


def disclosure_instruction() -> str:
    """The ``disclosure`` prompt piece (Art. 50(1))."""
    return _DISCLOSURE_INSTRUCTION


def disclosure_announcement(
    provider_name: Optional[str] = None,
    text: Optional[str] = None,
) -> str:
    """The first-interaction announcement a client renders as a message.

    Args:
        provider_name: ``regulatory.provider.name`` -- who put the system
            into service.  Named because the person is owed a party to
            hold accountable, not only a category.
        text: ``regulatory.disclosure_text`` -- an author's own wording,
            returned verbatim when given.

    Returns:
        One sentence.  With neither argument, the framework's default.
    """
    if text:
        return text.strip()
    if provider_name:
        return (
            f"You are interacting with an AI system operated by "
            f"{provider_name.strip()}, not with a human being."
        )
    return "You are interacting with an AI system, not with a human being."


#: Why an announcement was withheld.  Returned beside the text by
#: :func:`announcement_for` so a caller -- and ``explain oversight`` --
#: can say WHICH of the three reasons applied.  A page that prints only
#: "no announcement" leaves the author unable to tell a profile that
#: declined to declare from one that declared ``false``.
NOT_DECLARED = "not_declared"
DECLARED_NO_PERSONS = "declared_no_persons"
CLIENT_DISCLOSES = "client_discloses"


def announcement_for(
    regulatory: object = None,
    client_discloses_ai: bool = False,
) -> tuple[Optional[str], Optional[str]]:
    """Whether this session announces, and with what text (Art. 50(1)).

    The ONE predicate behind the first-interaction announcement.  The
    emit site and ``jaato-scaffold explain oversight`` both call it, so
    a page saying "this profile announces" and a session that stays
    silent cannot disagree -- the failure this framework keeps finding
    (#735, #950) is a rule with two implementations.

    Args:
        regulatory: The resolved profile's ``regulatory:`` block
            (:class:`~shared.plugins.subagent.config.RegulatoryProfileConfig`)
            or ``None``.  Duck-typed rather than imported: this module is
            stdlib-only on purpose, and ``shared.jaato_runtime`` appends
            the instruction piece before plugin discovery.
        client_discloses_ai: ``PresentationContext.client_discloses_ai``
            -- the connected client asserting it already shows a badge.
            The Act's "unless this is obvious from the point of view of a
            natural person who is reasonably well-informed" clause,
            asserted by the only party that can see the screen.

    Returns:
        ``(text, None)`` when the session announces, else
        ``(None, reason)`` with one of :data:`NOT_DECLARED`,
        :data:`DECLARED_NO_PERSONS`, :data:`CLIENT_DISCLOSES`.

    **Absent is not false.** A profile that declares no
    ``interacts_with_persons`` has made no determination, and announcing
    on its behalf would put a legal statement in front of every session
    in every existing workspace.  ``validate``'s ``disclosure_absent``
    is what reports the omission; this function does not guess past it.
    """
    interacts = getattr(regulatory, "interacts_with_persons", None)
    if interacts is not True:
        return None, (DECLARED_NO_PERSONS if interacts is False
                      else NOT_DECLARED)
    if client_discloses_ai:
        return None, CLIENT_DISCLOSES
    return disclosure_announcement(
        provider_name=getattr(regulatory, "provider_name", None),
        text=getattr(regulatory, "disclosure_text", None),
    ), None


#: The agent that fronts the person.  A subagent talks to its parent and is
#: never announced, so this is the only value the record carries.
ANNOUNCEMENT_AGENT_ID = "main"

#: Why an ``announcement`` record carries no delivered text -- the record's
#: ``withheld_reason`` vocabulary.  :data:`CLIENT_DISCLOSES` is shared with
#: :func:`announcement_for`; the rest are facts only the emit site knows:
#:
#: * :data:`HEADLESS` -- the session was created for no client (a reactor,
#:   a cascade stage): the event was handed to a client id nothing serves,
#:   so nobody could have read it, and a row saying "delivered" would be
#:   the false statement the record exists to prevent.
#: * :data:`DECISION_FAILED` -- the predicate itself raised, so whether the
#:   profile declared interaction could not be established.  Recorded
#:   rather than read as "declared nothing": a failure to DECIDE is the
#:   same state as a failure to announce.
#: * :data:`REVIVED_WAKE` / :data:`REVIVED_REATTACH` -- a session woken from
#:   disk is never re-announced; the two say WHICH path woke it, because a
#:   deliberate ``session.wake`` and a grace-expired browser reload are
#:   different events and a long-lived interactive session accumulates
#:   many of the second.
HEADLESS = "headless"
DECISION_FAILED = "decision_failed"
REVIVED_WAKE = "wake"
REVIVED_REATTACH = "reattach"
WITHHELD_REASONS = (CLIENT_DISCLOSES, HEADLESS, DECISION_FAILED,
                    REVIVED_WAKE, REVIVED_REATTACH)


def announcement_record(
    session_id: str,
    *,
    text: Optional[str] = None,
    withheld_reason: Optional[str] = None,
    presentation: object = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    created_by: Optional[str] = None,
) -> dict:
    """The ``announcement`` audit record -- Art. 50(1), bound to the interaction.

    :func:`announcement_for` decides whether a person is told; this is the
    proof that they were, and of what.  It is the ONE writer of the record
    ``jaato_sdk.audit.AUDIT_SCHEMA`` declares under that kind, so the
    contract guard (``test_audit_record_contract.py``) reads this function's
    source for every field the schema promises -- the field names below are
    literal for that reason.

    The four facts the issue asks to be bound together are the four this
    record carries: the **text** as delivered, the **channel** and
    **locale** off the client's ``PresentationContext``, and the **model
    identity** (the ``provider`` / ``model`` pair ``generated_by`` stamps).
    Four rules, each attached to a way the row could mislead:

    * **``delivered`` is the one question an auditor asks**, and it is
      answered by construction: ``True`` iff ``withheld_reason`` is
      ``None``, in which case ``text`` is required and carried verbatim.
      A row can therefore never say "delivered" beside an explanation of
      why it was not.
    * **Withheld is a value, not an absence.**  Every row that delivered
      nothing says why, from :data:`WITHHELD_REASONS`: the client took
      the obligation (``client_discloses``, also ``suppressed: true``),
      no client existed (``headless``), the predicate raised
      (``decision_failed``), or the session was woken and not
      re-announced (``wake`` / ``reattach``, also ``revived: true``).
    * **``text`` is present only when the framework delivered it.**
      Writing what WOULD have been said would read as what was said.
    * **Absent is not defaulted.**  A locale the client did not declare is
      omitted, never guessed from the daemon's ``LANG``; ``client_type``
      and ``client_discloses_ai`` are omitted when no client had declared
      a presentation at all (a revive), rather than written as a terminal
      that did not disclose.

    Args:
        session_id: The daemon session id.
        text: The announcement as emitted.  Required when
            ``withheld_reason`` is ``None``; ignored otherwise.
        withheld_reason: ``None`` when the text reached a client, else one
            of :data:`WITHHELD_REASONS`.
        presentation: The client's ``PresentationContext`` (duck-typed, so
            this module stays stdlib-only), or ``None`` when no client had
            declared one.
        provider: The provider serving the session at the announcement.
        model: The model serving it.
        created_by: The authenticated creator (#859), when known.

    Returns:
        The record, without the ledger's own stamps (``stage``, ``ts``,
        ``iso_ts``, ``event_index``), which :class:`TokenLedger` adds.

    Raises:
        ValueError: a delivered row with no text, or a reason outside the
            vocabulary -- both are rows that would mislead, refused at
            the one place they are built.
    """
    delivered = withheld_reason is None
    if delivered and not text:
        raise ValueError("a delivered announcement record must carry its text")
    if not delivered and withheld_reason not in WITHHELD_REASONS:
        raise ValueError(
            f"unknown withheld_reason {withheld_reason!r}; "
            f"expected one of {WITHHELD_REASONS}")
    record: dict = {
        "session_id": session_id,
        "agent_id": ANNOUNCEMENT_AGENT_ID,
        "delivered": delivered,
        "suppressed": withheld_reason == CLIENT_DISCLOSES,
        "revived": withheld_reason in (REVIVED_WAKE, REVIVED_REATTACH),
    }
    if delivered:
        record["text"] = text
    else:
        record["withheld_reason"] = withheld_reason
    if presentation is not None:
        client_type = getattr(presentation, "client_type", None)
        client_type = getattr(client_type, "value", client_type)
        if client_type:
            record["client_type"] = str(client_type)
        record["client_discloses_ai"] = bool(
            getattr(presentation, "client_discloses_ai", False))
        locale = getattr(presentation, "locale", None)
        if locale:
            record["locale"] = str(locale)
    if provider:
        record["provider"] = provider
    if model:
        record["model"] = model
    if created_by:
        record["created_by"] = created_by
    return record
