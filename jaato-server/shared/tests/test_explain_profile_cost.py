"""``jaato-scaffold explain profile <name>`` — what a profile INHERITS.

A profile file states what it ADDS.  It never states what it INHERITS, and
the inherited instruction layers are prepended to every session in the
workspace.  The cost is invisible at authoring time and surfaces later as a
cascade budget refusal, several sessions downstream of the decision.

Measured live by the cascade-coordination peer: 29,987 prompt tokens per turn
as shipped, 5,679 with ``suppress_base_instructions`` — 81% of the prompt was
inherited boilerplate their personas already replaced.

Also detects the case their number actually revealed: two tiers holding
IDENTICAL content.  The runtime LAYERS premium and workspace/user, it does
not dedupe, so a copy of the premium instructions left in the user tier
reaches the model twice.  That is why the delta was ~24k and not ~12k.
"""

import pathlib

import pytest

from shared.scaffold import explain


def _ws(tmp_path, profile_body="name: coder\ndescription: d\n", instr=None,
        persona=None):
    pdir = tmp_path / ".jaato" / "profiles" / "set1"
    pdir.mkdir(parents=True)
    (pdir / "coder.yaml").write_text(profile_body, encoding="utf-8")
    if instr is not None:
        idir = tmp_path / ".jaato" / "instructions"
        idir.mkdir(parents=True)
        (idir / "00-base.md").write_text(instr, encoding="utf-8")
    if persona is not None:
        adir = tmp_path / ".jaato" / "agents"
        adir.mkdir(parents=True)
        (adir / "coder.md").write_text(persona, encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def _no_premium_or_home(monkeypatch, tmp_path):
    """Isolate from the developer's real premium install and HOME.

    Without this the numbers under test would include whatever this machine
    happens to have — the tests-that-read-the-developer's-machine family, in
    a test whose entire subject is a byte count.
    """
    monkeypatch.setattr(explain, "_instruction_search_order",
                        lambda ws: [("workspace", ws / ".jaato" / "instructions"),
                                    ("user", tmp_path / "nonexistent-home")])


def test_it_reports_what_the_profile_inherits(tmp_path):
    ws = _ws(tmp_path, instr="x" * 40_000)
    data, text = explain.profile_cost("coder", str(ws))
    assert data["found"] is True
    assert data["inherited_bytes"] == 40_000
    assert data["approx_total_tokens"] == 10_000
    assert "inherited on EVERY turn" in text


def test_the_persona_is_counted_but_named_separately(tmp_path):
    """A persona is the profile's OWN cost; inheritance is the surprise."""
    ws = _ws(tmp_path, instr="x" * 1_000, persona="y" * 500)
    data, _ = explain.profile_cost("coder", str(ws))
    assert data["inherited_bytes"] == 1_000
    assert data["persona_bytes"] == 500
    assert data["total_bytes"] == 1_500


def test_a_suppressing_profile_reports_zero_inherited(tmp_path):
    """The knob's whole point — and the report must reflect it."""
    ws = _ws(tmp_path,
             profile_body="name: coder\ndescription: d\n"
                          "suppress_base_instructions: {disk: true}\n",
             instr="x" * 40_000)
    data, text = explain.profile_cost("coder", str(ws))
    assert data["disk_layer_suppressed"] is True
    assert data["inherited_bytes"] == 0
    assert "SUPPRESSED" in text


def test_a_suppressing_profile_is_not_nagged(tmp_path):
    """Advice a reader has already taken trains them to skim the rest."""
    ws = _ws(tmp_path,
             profile_body="name: coder\ndescription: d\n"
                          "suppress_base_instructions: {disk: true}\n",
             instr="x" * 40_000)
    _data, text = explain.profile_cost("coder", str(ws))
    assert "opt out with" not in text


def test_identical_tiers_are_reported_as_a_duplicate(tmp_path, monkeypatch):
    """Two tiers, same content — loaded TWICE, not deduped.

    Reported by CONTENT digest, not by size: two different 48KB files are a
    coincidence, the same 48KB file twice is a fixable mistake, and a size
    comparison cannot tell them apart.
    """
    ws = _ws(tmp_path, instr="x" * 30_000)
    twin = tmp_path / "twin"
    twin.mkdir()
    (twin / "00-base.md").write_text("x" * 30_000, encoding="utf-8")
    monkeypatch.setattr(explain, "_instruction_search_order",
                        lambda w: [("premium", twin),
                                   ("workspace", w / ".jaato" / "instructions")])

    data, text = explain.profile_cost("coder", str(ws))
    assert data["duplicate_layers"] == [["premium", "workspace"]]
    assert "DUPLICATE" in text
    assert "TWICE" in text


def test_different_content_of_the_same_size_is_not_a_duplicate(tmp_path, monkeypatch):
    """The contrast that gives the duplicate check its meaning."""
    ws = _ws(tmp_path, instr="x" * 30_000)
    other = tmp_path / "other"
    other.mkdir()
    (other / "00-base.md").write_text("y" * 30_000, encoding="utf-8")
    monkeypatch.setattr(explain, "_instruction_search_order",
                        lambda w: [("premium", other),
                                   ("workspace", w / ".jaato" / "instructions")])

    data, text = explain.profile_cost("coder", str(ws))
    assert data["duplicate_layers"] == []
    assert "DUPLICATE" not in text


def test_the_runtime_stops_at_the_first_of_workspace_or_user(tmp_path, monkeypatch):
    """Mirrors ``_load_base_system_instructions``: first match wins.

    Reporting both would overstate the cost by a layer the runtime never
    loads — a report that disagrees with the runtime is worse than none.
    """
    ws = _ws(tmp_path, instr="x" * 1_000)
    home = tmp_path / "home-instr"
    home.mkdir()
    (home / "00-base.md").write_text("y" * 9_000, encoding="utf-8")
    monkeypatch.setattr(explain, "_instruction_search_order",
                        lambda w: [("workspace", w / ".jaato" / "instructions"),
                                   ("user", home)])

    data, _ = explain.profile_cost("coder", str(ws))
    assert [l["layer"] for l in data["layers"]] == ["workspace"]
    assert data["inherited_bytes"] == 1_000


def test_an_unknown_profile_says_so(tmp_path):
    ws = _ws(tmp_path)
    data, text = explain.profile_cost("ghost", str(ws))
    assert data["found"] is False
    assert "ghost" in text


def test_the_estimate_is_labelled_as_one(tmp_path):
    """bytes/4 is a heuristic; presenting it as a count invites false precision."""
    ws = _ws(tmp_path, instr="x" * 1_000)
    data, text = explain.profile_cost("coder", str(ws))
    assert "estimate" in data["note"].lower()
    assert "ESTIMATES" in text
