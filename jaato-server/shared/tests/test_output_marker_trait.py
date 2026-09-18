"""Marking AI-generated output -- EU AI Act, Art. 50(2) (#1117).

``test_generated_by_stamp.py`` pins the machine-readable half: model
media crosses the wire carrying ``{"kind": "ai", provider, model,
session_id, agent_id}``.  That field travels with the DELIVERY EVENT and
is gone the instant a client writes the bytes down -- and Article 50(2)
is about the output, not about the event that carried it.  Nothing in
the tree put a marking in or beside a payload.

This module pins the hook that does, and the two rules the framework
enforces so a marker cannot get them wrong.

Six properties, each attached to a way it could silently stop holding:

A. the trait exists and the in-tree marker declares it;
B. **a relayed payload is never marked** -- the gate is on
   ``generated_by``, in the framework, once, so two seams cannot
   disagree about it;
C. **a failing marker must not lose the payload** -- log, deliver
   unmarked, record that marking failed.  Refusing to deliver would be
   a stronger posture than the Article asks for and would take down
   every voice session on a transient disk error;
D. which marker ran is recorded -- a DECLINED marking and NO MARKER
   CONFIGURED are different facts, and a deployment that cannot tell
   them apart believes its output is marked because a marker is
   installed;
E. the sidecar names the stamp, and does not claim to be a signed C2PA
   manifest;
F. **any future generator stamps.**  No in-tree tool produces an
   AI-generated ``Attachment`` today (``clarification`` builds the only
   ones, and those are a person's voice note).  The guard is what stops
   the next image-generation plugin shipping unstamped.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from jaato_sdk.output_marking import (
    IPTC_TRAINED_ALGORITHMIC_MEDIA,
    MarkResult,
    OutputPayload,
    sidecar_path_for,
)
from jaato_sdk.plugins.base import TRAIT_OUTPUT_MARKER
from shared.plugins.output_marker import create_plugin
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

_SESSION = "jaato-server/shared/jaato_session.py"
_MARKER = "jaato-server/shared/plugins/output_marker/plugin.py"

REVERSIONS = [
    Reversion(
        target=_SESSION,
        find="        if not data or not generated_by:\n            return data",
        replace="        if not data:\n            return data",
        because=(
            "a relayed payload is never marked: a fetched image is not "
            "AI-generated because an agent fetched it, and marking one "
            "would put a false provenance record on somebody else's file"
        ),
        test="test_a_relayed_payload_is_never_marked",
    ),
    Reversion(
        target=_SESSION,
        find=("            except Exception as exc:  # noqa: BLE001 -- see the docstring\n"
              "                self._trace(\n"
              "                    f\"OUTPUT_MARKER: {getattr(marker, 'name', marker)!r} \""),
        replace=("            except ValueError as exc:  # reversion: a raising marker escapes\n"
                 "                self._trace(\n"
                 "                    f\"OUTPUT_MARKER: {getattr(marker, 'name', marker)!r} \""),
        because=(
            "a marker that fails must not lose the payload -- refusing to "
            "deliver is a stronger posture than Art. 50(2) asks for and "
            "would break every voice session on a transient error"
        ),
        test="test_a_raising_marker_does_not_lose_the_payload",
    ),
    Reversion(
        target=_MARKER,
        find='                data=payload.data, marked=False, marker=self.name,\n                detail=("no filesystem destination',
        replace='                data=payload.data, marked=True, marker=self.name,\n                detail=("no filesystem destination',
        because=(
            "a marker that declined must not report a marking: 'marked' is "
            "what a dossier and an audit record read, and a false one is "
            "the certify-what-you-did-not-find failure"
        ),
        test="test_a_marker_with_nowhere_to_write_declines_rather_than_claiming",
    ),
    Reversion(
        target=_MARKER,
        find='        return MarkResult(\n            data=payload.data, marked=True, marker=self.name,\n            detail=f"provenance sidecar written to {target}",',
        replace='        return MarkResult(\n            data=payload.data, marked=False, marker=self.name,\n            detail=f"provenance sidecar written to {target}",',
        because=(
            "a sidecar that WAS written must be reported as a marking, or "
            "the one thing the plugin does is invisible to everything that "
            "reads the result"
        ),
        test="test_the_sidecar_names_the_stamp",
    ),
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find='''        lister = (getattr(registry, "list_enabled", None)
                  or getattr(registry, "list_exposed", None))''',
        replace='''        lister = (getattr(registry, "list_exposed", None)
                  or getattr(registry, "list_enabled", None))''',
        because=(
            "the marker looked up in the TOOL-bearing set, which an "
            "enrichment plugin can never be in -- every AI-generated "
            "payload delivered unmarked while `explain oversight` reports "
            "the marker as installed"
        ),
        test="test_a_real_registry_surfaces_an_enrichment_marker",
    ),
]


# --------------------------------------------------------------- helpers

class _Registry:
    """A registry stub that answers the accessor the seam actually reads.

    It deliberately does NOT implement ``list_exposed``: a marker is an
    enrichment plugin and can never be in that set, so a stub offering it
    would let the seam read the wrong one and still pass -- which is how
    the defect this file guards against survived its own test suite.
    ``test_a_real_registry_surfaces_an_enrichment_marker`` is the other
    half: it drives a real ``PluginRegistry`` with the real plugin.
    """

    def __init__(self, plugins):
        self._plugins = plugins

    def list_enabled(self):
        return list(self._plugins)

    def get_plugin(self, name):
        return self._plugins.get(name)


def _session(*markers):
    """A bare session carrying just the marker seam's two inputs.

    The METHOD under test is the framework's; only the registry it walks
    is fabricated.  Constructing a real ``JaatoSession`` needs a
    provider, a runtime and a workspace, none of which this question
    involves.
    """
    from shared.jaato_session import JaatoSession

    sess = JaatoSession.__new__(JaatoSession)
    named = {getattr(m, "name", f"m{i}"): m for i, m in enumerate(markers)}
    sess._runtime = type("RT", (), {"registry": _Registry(named)})()
    sess.traced = []
    sess._trace = sess.traced.append
    return sess


_STAMP = {"kind": "ai", "provider": "openrouter", "model": "openai/gpt-5.1"}


class _Recorder:
    """A marker that records what it was handed and rewrites the bytes."""

    plugin_traits = frozenset({TRAIT_OUTPUT_MARKER})
    name = "recorder"

    def __init__(self):
        self.seen = []

    def mark_output(self, payload):
        self.seen.append(payload)
        return MarkResult(data=payload.data + b"!", marked=True,
                          marker=self.name, detail="rewrote the bytes")


# --------------------------------------------------------------- A. trait

def test_the_in_tree_marker_declares_the_trait():
    plugin = create_plugin()
    assert TRAIT_OUTPUT_MARKER in plugin.plugin_traits
    assert callable(plugin.mark_output)


def test_the_trait_lives_with_the_other_plugin_traits():
    # A PLUGIN capability, not a tool behaviour, so it belongs beside
    # TRAIT_AUTH_PROVIDER / TRAIT_SLOT_SCOPED rather than beside the
    # ToolSchema traits.  #1117's text proposes model_provider/types.py,
    # which is where the TOOL traits live; following that would have put
    # a plugin trait where nothing reads plugin traits.
    from jaato_sdk.plugins import base

    assert base.TRAIT_OUTPUT_MARKER == "output_marker"
    assert hasattr(base, "TRAIT_AUTH_PROVIDER")


# ------------------------------------------------------- B. relays only

def test_a_relayed_payload_is_never_marked():
    rec = _Recorder()
    sess = _session(rec)
    out = sess._mark_generated_output(b"BYTES", "image/png", None)
    assert out == b"BYTES", "an unstamped payload must be delivered untouched"
    assert rec.seen == [], "the marker must not even be asked"


def test_a_stamped_payload_reaches_the_marker():
    rec = _Recorder()
    sess = _session(rec)
    out = sess._mark_generated_output(
        b"BYTES", "image/png", _STAMP, display_name="chart.png")
    assert out == b"BYTES!"
    assert len(rec.seen) == 1
    assert rec.seen[0].generated_by == _STAMP
    assert rec.seen[0].mime_type == "image/png"
    assert rec.seen[0].display_name == "chart.png"


def test_the_gate_is_in_one_place():
    """Both seams read the same gate, so they cannot disagree.

    Source-level, because the property is that neither seam grew its own
    copy: a second ``if generated_by`` in ``_deliver_model_media`` or in
    ``_emit_withheld_attachments_to_clients`` is how the two start
    meaning different things about what "relayed" is.
    """
    tree = ast.parse(Path(_SESSION).read_text())
    fn = next(f for f in ast.walk(tree)
              if isinstance(f, ast.FunctionDef)
              and f.name == "_mark_generated_output")
    src = ast.unparse(fn)
    assert "not generated_by" in src, (
        "the relays-only gate must live in the one dispatcher")


def test_no_marker_configured_changes_nothing():
    sess = _session()
    assert sess._mark_generated_output(b"BYTES", "image/png", _STAMP) == b"BYTES"
    assert sess.traced == [], "nothing happened, so nothing is recorded"


# --------------------------------------------------- C. failure keeps bytes

def test_a_raising_marker_does_not_lose_the_payload():
    class Exploding:
        plugin_traits = frozenset({TRAIT_OUTPUT_MARKER})
        name = "exploding"

        def mark_output(self, payload):
            raise RuntimeError("the watermarker is down")

    sess = _session(Exploding())
    assert sess._mark_generated_output(b"BYTES", "image/png", _STAMP) == b"BYTES"
    assert any("RAISED" in line for line in sess.traced), (
        "a marker that blew up must be recorded, not swallowed")


def test_a_failing_sidecar_write_declines_rather_than_raising(tmp_path):
    plugin = create_plugin()
    plugin.initialize({})
    # A path whose parent is a FILE: makedirs cannot create it.
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"x")
    result = plugin.mark_output(OutputPayload(
        data=b"PNG", mime_type="image/png", generated_by=_STAMP,
        path=str(blocker / "nested" / "chart.png")))
    assert result.marked is False
    assert result.data == b"PNG"
    assert "failed" in result.detail


# ------------------------------------------------------- D. it is recorded

def test_a_decline_is_recorded_distinctly_from_a_marking():
    plugin = create_plugin()
    plugin.initialize({})
    sess = _session(plugin)
    sess._mark_generated_output(b"AUDIO", "audio/wav", _STAMP)
    assert len(sess.traced) == 1
    line = sess.traced[0]
    assert "declined" in line and "output_marker" in line
    assert "no filesystem destination" in line


def test_a_marker_with_nowhere_to_write_declines_rather_than_claiming():
    plugin = create_plugin()
    plugin.initialize({})
    result = plugin.mark_output(OutputPayload(
        data=b"AUDIO", mime_type="audio/wav", generated_by=_STAMP))
    assert result.marked is False
    assert result.data == b"AUDIO"
    assert "sidecar" in result.detail


def test_a_mime_outside_the_markers_set_is_declined_by_name():
    plugin = create_plugin()
    plugin.initialize({})
    result = plugin.mark_output(OutputPayload(
        data=b"hello", mime_type="text/plain", generated_by=_STAMP,
        path="/tmp/a.txt"))
    assert result.marked is False
    assert "text/plain" in result.detail


# ---------------------------------------------------------- E. the sidecar

def test_the_sidecar_names_the_stamp(tmp_path):
    plugin = create_plugin()
    plugin.initialize({})
    target = tmp_path / "chart.png"
    target.write_bytes(b"PNGBYTES")

    result = plugin.mark_output(OutputPayload(
        data=b"PNGBYTES", mime_type="image/png", generated_by=_STAMP,
        path=str(target), display_name="chart.png"))

    assert result.marked is True
    assert result.sidecar_path == sidecar_path_for(str(target))
    manifest = json.loads(Path(result.sidecar_path).read_text())

    assert manifest["generated_by"] == _STAMP
    assert manifest["subject"]["media_type"] == "image/png"
    assert manifest["subject"]["size"] == len(b"PNGBYTES")
    assert manifest["assertions"][0]["data"]["actions"][0][
        "digitalSourceType"] == IPTC_TRAINED_ALGORITHMIC_MEDIA


def test_the_sidecar_does_not_claim_to_be_a_signed_c2pa_manifest(tmp_path):
    """The name and the document both refuse the claim.

    A file called ``.c2pa.json`` carrying no signature would be read as a
    C2PA manifest by anything that looks for one, and rejected by every
    verifier -- while telling a deployer their output is C2PA-marked.
    The whole value of a provenance record is that it is hard to forge,
    so one that overstates itself is worse than none.
    """
    plugin = create_plugin()
    plugin.initialize({})
    target = tmp_path / "chart.png"
    target.write_bytes(b"PNG")
    result = plugin.mark_output(OutputPayload(
        data=b"PNG", mime_type="image/png", generated_by=_STAMP,
        path=str(target)))

    assert result.sidecar_path.endswith(".provenance.json")
    assert not result.sidecar_path.endswith(".c2pa.json")
    manifest = json.loads(Path(result.sidecar_path).read_text())
    assert manifest["signature"] is None
    assert manifest["conformance"] == "c2pa-shaped-unsigned"


def test_two_markings_of_one_payload_produce_the_same_bytes(tmp_path):
    """Sorted keys, so a regenerated sidecar is not a diff."""
    from datetime import datetime, timezone

    from jaato_sdk.output_marking import manifest_bytes, provenance_manifest

    payload = OutputPayload(data=b"PNG", mime_type="image/png",
                            generated_by=_STAMP, display_name="c.png")
    now = datetime(2026, 9, 18, 15, 4, 11, tzinfo=timezone.utc)
    first = manifest_bytes(provenance_manifest(payload, "0.16.0", now))
    second = manifest_bytes(provenance_manifest(payload, "0.16.0", now))
    assert first == second


def test_the_sidecar_name_does_not_collide_across_extensions():
    # chart.png and chart.pdf in one directory must not share a manifest.
    assert sidecar_path_for("/w/chart.png") != sidecar_path_for("/w/chart.pdf")
    assert sidecar_path_for("/w/chart.png") == "/w/chart.png.provenance.json"


# --------------------------------------------------- F. the next generator

#: Modules allowed to build an ``Attachment`` with no ``generated_by``.
#: Each entry is a RELAY or a person's own content -- never something a
#: model produced.  Adding a module here is a claim, and the claim is
#: reviewable because it is written down beside the reason.
_RELAY_ONLY = {
    # A person's voice note or screenshot answering a clarification.
    # Emphatically not AI-generated: stamping it would attribute the
    # person's own words to the model.
    "shared/plugins/clarification/attachments.py",
}


def _attachment_construction_sites():
    """Every non-test module that constructs an ``Attachment``."""
    root = Path("jaato-server/shared/plugins")
    sites = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to("jaato-server").as_posix()
        if "/tests/" in rel or path.name.startswith("test_"):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "Attachment"):
                sites.append((rel, node))
    return sites


def test_every_attachment_producer_either_stamps_or_is_a_declared_relay():
    """The contract that forces the NEXT generator tool to stamp.

    There is no image-generation tool in this tree, so
    ``Attachment.generated_by`` has no producer and this guard currently
    protects a set of one relay.  That is the point: #1117 exists because
    nothing forced the stamp, and a guard written only when the first
    generator ships is a guard written after the first unmarked output.

    An allow-LIST rather than a skip-list, the ``created_by`` AST guard's
    rule: a module this test does not recognise FAILS and must be
    classified, because the failure being guarded against is a producer
    nobody thought about.
    """
    unstamped = []
    for rel, node in _attachment_construction_sites():
        stamps = any(kw.arg == "generated_by" for kw in node.keywords)
        if not stamps and rel not in _RELAY_ONLY:
            unstamped.append(f"{rel}:{node.lineno}")

    assert not unstamped, (
        "these build an Attachment without `generated_by` and are not in "
        f"_RELAY_ONLY: {unstamped}.\n"
        "If the bytes came from a model, stamp them with "
        "jaato_sdk.events.ai_generated_by(...) -- EU AI Act Art. 50(2).\n"
        "If the tool merely relays or reads them, add the module to "
        "_RELAY_ONLY with the reason.")


def test_the_relay_list_names_only_modules_that_exist():
    # A stale entry is an exemption nobody can see is dead, which is how
    # an allow-list stops being an allow-list.
    for rel in _RELAY_ONLY:
        assert (Path("jaato-server") / rel).is_file(), f"stale entry: {rel}"


def test_the_guard_is_not_vacuous():
    # If nothing constructs an Attachment at all, the test above passes
    # for the wrong reason.
    assert _attachment_construction_sites(), (
        "no Attachment construction site found -- the scan is broken")


# ------------------------------------------- the seam, through a REAL registry

def test_a_real_registry_surfaces_an_enrichment_marker():
    """The marker must be findable through the registry a session holds.

    Every other test here fabricates the registry, which is right for
    asking what the seam DOES with a marker and useless for asking
    whether it can FIND one.  The defect this closes lived exactly in
    that gap: ``output_marker`` declares ``PLUGIN_KIND = "enrichment"``,
    so ``PluginRegistry`` files it under ``_enrichment_only`` and it is
    absent from ``list_exposed()`` by construction -- a profile enabling
    it got an empty marker list, forever, with no error and no trace
    line.

    So this one builds the real registry, registers the real plugin, and
    asks the real accessor.
    """
    from shared.jaato_session import JaatoSession
    from shared.plugins.registry import PluginRegistry

    registry = PluginRegistry()
    plugin = create_plugin()
    registry.register_plugin(plugin, enrichment_only=True)
    registry.expose_tool(plugin.name)

    assert plugin.name not in registry.list_exposed(), (
        "an enrichment plugin in the tool-bearing set would mean the "
        "registry changed and this test is now asking nothing")
    assert plugin.name in registry.list_enabled()

    sess = JaatoSession.__new__(JaatoSession)
    sess._runtime = type("RT", (), {"registry": registry})()
    sess.traced = []
    sess._trace = sess.traced.append

    found = sess._output_markers()
    assert [getattr(m, "name", None) for m in found] == [plugin.name], (
        "the marker a profile enabled was not found -- Article 50(2) "
        "marking is inert in the one configuration that asks for it")
