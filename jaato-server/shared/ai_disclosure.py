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
