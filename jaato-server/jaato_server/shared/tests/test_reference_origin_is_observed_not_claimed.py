"""A reference records where it ARRIVED from, and only what was observed.

``Memory`` carries four provenance fields (``source_agent``,
``source_session``, and since #1123 ``generated_by`` / ``curated_by``).
``ReferenceSource`` carried **none** -- so the rule that provenance gates
placement, which the wikiLLM brainstorm states in §8, was not merely
unimplemented for references but *unrepresentable*: there was no field for
a fence to read.

What is added is deliberately narrower than memory's, and the narrowing is
the design rather than a shortfall.  ``generated_by`` names the model that
wrote a memory because a model writes memories; **nothing in this tree
writes a reference**.  A reference is authored by a human, emitted by
``gen-references``, or copied in from another workspace's bundle -- and
only the last is an event the framework is present for.  A field naming an
author would therefore have no stamper: an inert mechanism, which this
repository has already had to review out once (``85c3bfd``, "three inert
mechanisms").  So ``origin`` records ARRIVAL.

Three properties, each its own failure if dropped:

1. **Observed, not claimed.**  The stamp is written over whatever the
   incoming file said.  An ``origin`` already in a foreign JSON is an
   assertion by the party being judged; a document claiming to have been
   authored locally would otherwise launder itself into this catalog as
   native.  This is the one with a security consequence.

2. **Absent means unobserved.**  A hand-authored reference, one predating
   the field, and one installed by ``bundle unpack`` (which copies whole
   directories rather than rewriting each reference) all carry ``None``,
   and none of the three may be read as "authored here".  Positive
   evidence only -- the posture #1014 and #1023 take about confinement
   labels.

3. **It reaches a reader.**  A field nothing renders is a field nobody can
   act on.  ``to_instruction`` names it where the model reads the
   reference, and ``listReferences`` carries it in the catalog entry.

Per-REFERENCE rather than per-bundle, which is where the federation unit
otherwise lives: ``merge_bundle`` copies source references *into the target
bundle's own directory*, so after a merge the source bundle boundary is
gone.  Bundle-level provenance would be erased by exactly the operation
that creates foreign references.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaato_server.shared.plugins.references.models import (
    ORIGIN_IMPORTED,
    InjectionMode,
    ReferenceOrigin,
    ReferenceSource,
    SourceType,
)
from jaato_server.shared.plugins.references.merge import _copy_reference_json
from jaato_server.shared.tests.reversion import Reversion

_MERGE = "jaato-server/jaato_server/shared/plugins/references/merge.py"
_MODELS = "jaato-server/jaato_server/shared/plugins/references/models.py"

REVERSIONS = [
    Reversion(
        target=_MERGE,
        find='    if origin is not None:\n        raw["origin"] = origin.to_dict()\n',
        replace='    if origin is not None and "origin" not in raw:\n        raw["origin"] = origin.to_dict()\n',
        because=(
            "deferring to an origin already in the incoming file lets a "
            "foreign reference claim its own provenance -- a document "
            "asserting it was authored locally launders itself into the "
            "catalog as native"
        ),
        test=("TestObservedNotClaimed::"
              "test_a_foreign_claim_is_overwritten_by_what_was_observed"),
    ),
    Reversion(
        target=_MODELS,
        find='        kind = data.get("kind")\n'
             '        if not isinstance(kind, str) or not kind:\n'
             '            return None\n',
        replace='        kind = data.get("kind") or ORIGIN_IMPORTED\n',
        because=(
            "inventing a kind for a payload that does not state one makes "
            "the field say what was observed when nothing was"
        ),
        test=("TestAbsentMeansUnobserved::"
              "test_a_payload_with_no_kind_reads_as_no_origin"),
    ),
]


def _ref(**kw) -> ReferenceSource:
    base = dict(
        id="adr-001", name="ADR 001", description="a decision",
        type=SourceType.LOCAL, mode=InjectionMode.SELECTABLE,
        path="docs/adr-001.md",
    )
    base.update(kw)
    return ReferenceSource(**base)


class TestRoundTrip:
    """The field survives the catalog's own serialization."""

    def test_an_origin_round_trips(self):
        origin = ReferenceOrigin(
            kind=ORIGIN_IMPORTED, bundle="teammate",
            source_id="adr-9", at="2026-09-29T10:00:00+00:00",
        )
        revived = ReferenceSource.from_dict(_ref(origin=origin).to_dict())
        assert revived.origin is not None
        assert revived.origin.kind == ORIGIN_IMPORTED
        assert revived.origin.bundle == "teammate"
        assert revived.origin.source_id == "adr-9"
        assert revived.origin.at == "2026-09-29T10:00:00+00:00"

    def test_an_unset_key_is_omitted_rather_than_nulled(self):
        """Absent is "not measured"; ``null`` invites reading it as a
        measured negative.  The rule ``ai_generated_by`` already follows."""
        payload = ReferenceOrigin(kind=ORIGIN_IMPORTED).to_dict()
        assert payload == {"kind": ORIGIN_IMPORTED}


class TestAbsentMeansUnobserved:
    """No origin is never a claim that the reference was authored here."""

    def test_a_reference_with_no_origin_serializes_without_the_key(self):
        assert "origin" not in _ref().to_dict()

    def test_a_record_predating_the_field_revives_with_none(self):
        legacy = {"id": "x", "name": "X", "description": "",
                  "type": "local", "mode": "selectable", "path": "x.md"}
        assert ReferenceSource.from_dict(legacy).origin is None

    def test_a_payload_with_no_kind_reads_as_no_origin(self):
        assert ReferenceOrigin.from_dict({"bundle": "teammate"}) is None
        assert ReferenceOrigin.from_dict({"kind": ""}) is None
        assert ReferenceOrigin.from_dict("imported") is None
        assert ReferenceOrigin.from_dict(None) is None


class TestObservedNotClaimed:
    """The stamp records what this workspace saw, not what it was told."""

    def _copy(self, tmp_path: Path, raw: dict) -> dict:
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        src.mkdir()
        tgt.mkdir()
        (src / "adr-001.json").write_text(json.dumps(raw), encoding="utf-8")
        assert _copy_reference_json(
            src, tgt, "adr-001", "adr-001", None,
            ReferenceOrigin(kind=ORIGIN_IMPORTED, bundle="teammate",
                            source_id="adr-001", at="2026-09-29T10:00:00+00:00"),
        )
        return json.loads((tgt / "adr-001.json").read_text(encoding="utf-8"))

    def test_a_copied_reference_is_stamped(self, tmp_path):
        out = self._copy(tmp_path, {"id": "adr-001", "name": "ADR 001"})
        assert out["origin"]["kind"] == ORIGIN_IMPORTED
        assert out["origin"]["bundle"] == "teammate"

    def test_a_foreign_claim_is_overwritten_by_what_was_observed(self, tmp_path):
        """The incoming file is the party being judged.  Its own account of
        where it came from is hearsay, and a claim of local authorship is
        the one that would do damage."""
        out = self._copy(tmp_path, {
            "id": "adr-001", "name": "ADR 001",
            "origin": {"kind": "authored-here", "bundle": "trust-me"},
        })
        assert out["origin"]["kind"] == ORIGIN_IMPORTED
        assert out["origin"]["bundle"] == "teammate"

    def test_the_id_it_was_known_by_elsewhere_is_kept(self, tmp_path):
        """``bundle merge --prefix`` renames on collision, so the local id
        is not the one the other workspace knows it by."""
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        src.mkdir()
        tgt.mkdir()
        (src / "adr-001.json").write_text(
            json.dumps({"id": "adr-001", "name": "X"}), encoding="utf-8")
        _copy_reference_json(
            src, tgt, "adr-001", "teammate--adr-001", None,
            ReferenceOrigin(kind=ORIGIN_IMPORTED, bundle="teammate",
                            source_id="adr-001"),
        )
        out = json.loads(
            (tgt / "teammate--adr-001.json").read_text(encoding="utf-8"))
        assert out["id"] == "teammate--adr-001"
        assert out["origin"]["source_id"] == "adr-001"

    def test_an_unstamped_copy_writes_no_origin(self, tmp_path):
        """``origin=None`` is the pre-#1123-shaped caller: it must not
        invent one."""
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        src.mkdir()
        tgt.mkdir()
        (src / "a.json").write_text(json.dumps({"id": "a"}), encoding="utf-8")
        _copy_reference_json(src, tgt, "a", "a", None)
        assert "origin" not in json.loads(
            (tgt / "a.json").read_text(encoding="utf-8"))


class TestItReachesAReader:
    """A field nothing renders is a field nobody can act on."""

    def test_the_model_is_told_where_a_selected_reference_came_from(self):
        text = _ref(origin=ReferenceOrigin(
            kind=ORIGIN_IMPORTED, bundle="teammate",
            at="2026-09-29T10:00:00+00:00")).to_instruction()
        assert "**Origin**" in text
        assert "teammate" in text

    def test_a_locally_authored_reference_says_nothing(self):
        assert "**Origin**" not in _ref().to_instruction()

    def test_describe_names_the_alias_when_the_id_was_rewritten(self):
        described = ReferenceOrigin(
            kind=ORIGIN_IMPORTED, bundle="teammate",
            source_id="adr-9").describe()
        assert "teammate" in described and "adr-9" in described
