"""A derived profile field is never NAMED without the spelling that sets it.

``preloaded_plugins`` and ``tool_scopes`` are ``SubagentProfile`` fields and
NOT profile-file keys (:data:`~shared.plugins.subagent.config.PROFILE_DERIVED_FIELDS`):
both are derived by ``parse_plugin_list`` from the ``plugins`` entries' own
modifiers.  Writing either as a top-level YAML key parses, resolves, validates
as structurally fine, and is then read by nobody.

#1040 annotated the two rows of ``explain profile``'s FIELD LISTING.  It did
not touch the other place the page names them -- the **inheritance merge
table** -- where ``tool_scopes`` sat in a row with ``env`` and ``quirks``::

    tool_scopes, env, quirks     per-KEY dict-merge -- child wins on keys it
                                 sets; the parent's other keys survive.

Every other row of that table names a key an author writes, and ``env`` /
``quirks`` genuinely are.  So one render said both "NOT a profile-file key"
(field row) and "merges per-key like ``env``" (table row), 700 lines apart.
The skill reference shipped at
``scaffold/integrations/claude-code/payload/references/profiles.md`` carried
the same four-key list with no correcting clause **anywhere in the file**,
followed immediately by a worked *top-level-YAML* example using one of the
co-listed keys -- and it is the surface an agent reads first, because it is
symlinked into ``.claude/skills/jaato-sdk`` and loads without running
anything.

Two agents wrote ``tool_scopes:`` as a top-level key on the same day.

THE ASSERTION THAT COULD NOT CATCH IT.  ``test_explain_profile_inheritance``
asserted ``"tool_scopes" in text``, which passes on every broken version
above -- including the one that produced the incident.  Presence was never
the property.  What has to hold is that the term never appears **alone**:
wherever a surface names a derived field, the modifier that actually sets it
is within reading distance, so the two cannot be read apart.
"""
from pathlib import Path

import pytest

from shared.scaffold import explain

#: How far from a mention the spelling may sit and still be read with it.
#: Six lines is about one rendered table row -- long enough for a multi-line
#: cell, short enough that no reader scrolls past it.
WINDOW = 6

#: derived field -> the substring that spells how a profile FILE sets it.
SPELLINGS = {
    "tool_scopes": "tools:[",
    "preloaded_plugins": "(preload)",
}

_FIELD_CASES = [pytest.param(t, s, id=t) for t, s in sorted(SPELLINGS.items())]


def _skill_reference() -> Path:
    """The profiles reference shipped in the claude-code skill payload.

    Located from ``explain.__file__`` rather than from the repo root so the
    meta-guard's sandboxed copy is the one that resolves, like every other
    import in a sandboxed run.
    """
    here = Path(explain.__file__).resolve().parent
    return (here / "integrations" / "claude-code" / "payload"
            / "references" / "profiles.md")


def _orphan_mentions(text: str, term: str, spelling: str):
    """Lines naming ``term`` with no ``spelling`` within ``WINDOW`` lines."""
    lines = text.splitlines()
    orphans = []
    for i, line in enumerate(lines):
        if term not in line:
            continue
        near = "\n".join(lines[max(0, i - WINDOW):i + WINDOW + 1])
        if spelling not in near:
            orphans.append(f"  line {i + 1}: {line.strip()}")
    return orphans


@pytest.mark.parametrize("term,spelling", _FIELD_CASES)
def test_explain_profile_never_names_a_derived_field_alone(term, spelling):
    _data, text = explain.profile()
    orphans = _orphan_mentions(text, term, spelling)
    assert not orphans, (
        f"`explain profile` names the DERIVED field {term!r} with no "
        f"{spelling!r} within {WINDOW} lines:\n" + "\n".join(orphans) +
        f"\n\n{term} is not a profile-file key.  A mention that does not "
        f"carry its spelling reads as an instruction to write `{term}:` at "
        f"the top level, which parses and is silently inert."
    )


@pytest.mark.parametrize("term,spelling", _FIELD_CASES)
def test_skill_reference_never_names_a_derived_field_alone(term, spelling):
    ref = _skill_reference()
    assert ref.is_file(), f"skill reference missing at {ref}"
    orphans = _orphan_mentions(ref.read_text(encoding="utf-8"), term, spelling)
    assert not orphans, (
        f"{ref.name} names the DERIVED field {term!r} with no {spelling!r} "
        f"within {WINDOW} lines:\n" + "\n".join(orphans) +
        "\n\nThis file is symlinked into .claude/skills/jaato-sdk and is "
        "read before anything is run, so a bare mention here is the "
        "cheapest path to the mistake."
    )


def test_the_derived_fields_this_guard_polices_are_the_declared_ones():
    """The guard's field list is the framework's, not a second opinion.

    A third entry added to ``PROFILE_DERIVED_FIELDS`` fails here until
    someone decides how a FILE spells it -- which is the question nobody had
    answered for ``tool_scopes``.
    """
    from shared.plugins.subagent.config import PROFILE_DERIVED_FIELDS

    assert set(SPELLINGS) == set(PROFILE_DERIVED_FIELDS), (
        "PROFILE_DERIVED_FIELDS and this guard's SPELLINGS have diverged: "
        f"{sorted(set(PROFILE_DERIVED_FIELDS) ^ set(SPELLINGS))}"
    )


def test_explain_profile_points_at_the_validate_finding():
    """The page names the finding an author will actually see.

    ``derived_profile_key`` (#1040) is what ``validate`` reports for this
    mistake.  Naming it in ``explain`` is what connects the two surfaces for
    a reader who arrived at either one.
    """
    _data, text = explain.profile()
    assert "derived_profile_key" in text


# ---------------------------------------------------------------------------
# Reversions: put each surface back the way it shipped in jaato-server 0.14.0
# -- the release the two agents were reading -- and name the one test that
# must notice.  See test_every_guard_detects_its_own_reversion.
# ---------------------------------------------------------------------------

from shared.tests.reversion import (  # noqa: E402
    Reversion,
)

_SKILL_REF = ("jaato-server/shared/scaffold/integrations/claude-code/"
              "payload/references/profiles.md")

REVERSIONS = [
    Reversion(
        target="jaato-server/shared/scaffold/explain.py",
        find='"    env, quirks                  per-KEY dict-merge \u2014 child wins on keys it sets;\\n"',
        replace='"    tool_scopes, env, quirks     per-KEY dict-merge \u2014 child wins on keys it sets;\\n"',
        because=(
            "the inheritance merge table listing tool_scopes as a peer of env "
            "and quirks -- every other row of that table names a key an "
            "author writes, so the row reads as 'tool_scopes: is a top-level "
            "dict'"
        ),
        test="test_explain_profile_never_names_a_derived_field_alone[tool_scopes]",
    ),
    Reversion(
        target=_SKILL_REF,
        find=(
            "  surface, re-list the plugin carrying an allow-list \u2014\n"
            "  `plugins: [memory(tools:[a,b])]` \u2014 or use the permission whitelist, or do\n"
            "  not inherit.\n"
            "- `completion_processors` **concatenate**"
        ),
        replace=(
            "  surface: `tool_scopes`, the permission whitelist, or do not inherit.\n"
            "- `completion_processors` **concatenate**"
        ),
        because=(
            "the skill reference naming tool_scopes as the way to scope down "
            "while the file spells it nowhere -- the surface an agent reads "
            "before it runs anything"
        ),
        test="test_skill_reference_never_names_a_derived_field_alone[tool_scopes]",
    ),
]
