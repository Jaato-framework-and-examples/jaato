"""The ``disclosure`` instruction piece -- EU AI Act, Article 50(1).

A provider of an AI system that interacts directly with natural persons
must design it so that those persons are informed they are dealing with an
AI (Regulation (EU) 2024/1689, Art. 50(1); in force since 2 August 2026).
Nothing in the framework's prompt said so: the constants cover task
completion, parallelism and summaries, and a persona is free to be
anyone.  ``docs/design/eu-ai-act.md`` §4.2 is the design; this module pins
the half that reaches the model.

Three properties, each attached to a way it could silently stop holding:

A. the piece is in every assembled system instruction by default, and is
   removed ONLY by naming it -- the blanket ``suppress_base_instructions:
   true`` keeps it, exactly as it keeps ``security``;
B. removing it is announced at WARNING by the session that applies it, the
   posture ``scrub_secret_env: none`` and ``--ws-unsafe-no-auth`` take;
C. the announcement text a client renders names the provider when the
   profile declares one, and an author's own wording wins verbatim.
"""

from __future__ import annotations

import logging

from shared.ai_disclosure import disclosure_announcement, disclosure_instruction
from shared.instruction_suppression import (
    ANNOUNCED_PIECES,
    PIECE_DISCLOSURE,
    PIECE_SECURITY,
    normalize_suppression,
)
from shared.jaato_session import _announce_dropped_pieces
from shared.tests.reversion import Reversion

_RUNTIME = "jaato-server/shared/jaato_runtime.py"
_SUPPRESSION = "jaato-server/shared/instruction_suppression.py"

REVERSIONS = [
    Reversion(
        target=_RUNTIME,
        find="    if include_disclosure:\n        from shared.ai_disclosure import disclosure_instruction",
        replace="    if False:\n        from shared.ai_disclosure import disclosure_instruction",
        because=(
            "the piece must actually reach the assembled system instruction; "
            "a constant nobody appends discloses nothing"
        ),
        test="test_the_piece_is_in_the_assembled_prompt_by_default",
    ),
    Reversion(
        target=_SUPPRESSION,
        find="_TRUE_PIECES: FrozenSet[str] = frozenset({PIECE_DISK, PIECE_CONSTANTS})",
        replace="_TRUE_PIECES: FrozenSet[str] = frozenset({PIECE_DISK, PIECE_CONSTANTS, PIECE_DISCLOSURE})",
        because=(
            "the blanket `true` is a token-saving flag and must not drop a "
            "legal posture -- disclosure is dropped only by name"
        ),
        test="test_the_blanket_true_keeps_it",
    ),
]


def _runtime():
    from shared.jaato_runtime import JaatoRuntime
    rt = JaatoRuntime.__new__(JaatoRuntime)
    rt._registry = None
    rt._system_instructions = None
    rt._permission_plugin = None
    rt._formatter_pipeline = None
    rt.get_base_system_instructions = lambda: "BASE"
    return rt


def _assemble(**kw):
    return _runtime().get_system_instructions(plugin_names=[], additional="PERSONA", **kw)


# ------------------------------------------------------------------ A. piece

def test_the_piece_says_what_the_article_requires():
    text = disclosure_instruction()
    assert "AI system" in text
    assert "human" in text.lower()


def test_the_piece_is_in_the_assembled_prompt_by_default():
    assert disclosure_instruction() in _assemble()


def test_it_is_removed_only_by_naming_it():
    assert disclosure_instruction() not in _assemble(include_disclosure=False)
    assert normalize_suppression({"disclosure": True}) == frozenset({PIECE_DISCLOSURE})
    assert normalize_suppression(["all"]) >= {PIECE_DISCLOSURE, PIECE_SECURITY}


def test_the_blanket_true_keeps_it():
    supp = normalize_suppression(True)
    assert PIECE_DISCLOSURE not in supp
    assert PIECE_SECURITY not in supp, "the two posture pieces are treated alike"


# ------------------------------------------------------------- B. announced

def test_dropping_it_is_announced_at_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="shared.jaato_session"):
        _announce_dropped_pieces(frozenset({PIECE_DISCLOSURE}), "main")
    assert any("disclosure" in r.getMessage() and "50(1)" in r.getMessage()
               for r in caplog.records if r.levelno == logging.WARNING)


def test_a_token_saving_drop_is_not_announced(caplog):
    with caplog.at_level(logging.WARNING, logger="shared.jaato_session"):
        _announce_dropped_pieces(normalize_suppression(True), "main")
    assert not caplog.records
    assert ANNOUNCED_PIECES == frozenset({PIECE_SECURITY, PIECE_DISCLOSURE})


# ---------------------------------------------------------- C. announcement

def test_the_announcement_names_the_provider_when_declared():
    assert "Acme Talent GmbH" in disclosure_announcement("Acme Talent GmbH")
    assert "AI system" in disclosure_announcement(None)


def test_an_authors_own_wording_wins_verbatim():
    assert disclosure_announcement("Acme", text="  Hi, I am a bot.  ") == "Hi, I am a bot."
