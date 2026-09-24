"""The authoring surface must know the EU AI Act keys, all three verbs of it.

``explain`` documented ``regulatory:``, ``trace.ledger`` and
``record_keeping:``, and ``validate`` checked them -- once declared.  ``new``
emitted none of them, and a workspace scaffolded the documented way got a
clean bill from ``validate``, so the author who most needed to hear the keys
existed was the one nothing told.  Three guards, one per verb:

* ``new profile-set`` emits the block, COMMENTED (declared, never inferred);
* ``validate`` says once, per workspace, that nothing declares it;
* the Claude Code integration skill -- the surface an author working from
  Claude Code reads first -- lists the two compliance scopes and the dossier
  archetype, and every topic and archetype it lists is one the CLI has.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from jaato_server.shared.tests.reversion import Reversion

_BUILD = "jaato-server/jaato_server/shared/scaffold/build.py"
_VALIDATE = "jaato-server/jaato_server/shared/scaffold/validate.py"
_SKILL = "jaato-server/jaato_server/shared/scaffold/integrations/claude-code/payload/SKILL.md"

REVERSIONS = [
    Reversion(
        target=_BUILD,
        find='        + "\\n".join(_compliance_example()) + "\\n"\n',
        replace="",
        because=("a scaffolded workspace that carries no trace of the compliance "
                 "keys is the state validate could not warn about"),
        test="test_the_base_profile_carries_the_keys_commented",
    ),
    Reversion(
        target=_VALIDATE,
        find="    _check_regulatory_declared(result.profiles, out)\n",
        replace="",
        because="a workspace declaring nothing under the Act got a clean bill",
        # The end-to-end test, not the unit one: the unit test calls the
        # checker directly and passes whether or not validate_workspace does.
        test="test_the_finding_reaches_validate_workspace_on_a_scaffolded_set",
    ),
    Reversion(
        target=_SKILL,
        find="jaato-scaffold explain oversight [<profile>]",
        replace="jaato-scaffold explain prefetch",
        because=("the skill is the authoring surface for Claude Code users, and "
                 "it listed the CLI's topics without the compliance scopes"),
        test="test_the_skill_names_the_compliance_scopes_and_the_dossier",
    ),
]


# --------------------------------------------------------------- new

def test_the_base_profile_carries_the_keys_commented():
    from jaato_server.shared.scaffold.build import _base_profile_yaml
    text = _base_profile_yaml("triage")
    for key in ("regulatory:", "trace:", "record_keeping:", "intended_purpose",
                "interacts_with_persons", "ledger:", "retention_days", "integrity:"):
        assert key in text, f"{key!r} is not in the scaffolded base profile"
    # Commented, every line of it: a live regulatory block with no fields is a
    # determination nobody made, and a live record_keeping block changes what
    # DELETE means for the workspace.
    for line in text.splitlines():
        if any(k in line for k in ("regulatory", "record_keeping", "ledger", "retention")):
            assert line.startswith("#"), f"emitted LIVE: {line!r}"


def test_the_scaffolded_base_still_loads_and_declares_nothing(tmp_path):
    """Commented is inert: the loader sees no regulatory block."""
    from jaato_server.shared.plugins.subagent.config import discover_profiles
    from jaato_server.shared.scaffold.build import _base_profile_yaml
    (tmp_path / ".jaato" / "profiles").mkdir(parents=True)
    (tmp_path / ".jaato" / "profiles" / "_base_triage.yaml").write_text(
        _base_profile_yaml("triage"), encoding="utf-8")
    result = discover_profiles(profiles_dir=".jaato/profiles", base_path=str(tmp_path),
                               config_root=str(tmp_path / ".jaato"))
    prof = result.profiles["_base_triage"]
    assert getattr(prof, "regulatory", None) is None
    assert getattr(prof, "record_keeping", None) is None


# ---------------------------------------------------------- validate

def _prof(name, *, model="m", regulatory=None):
    return SimpleNamespace(name=name, model=model, model_tiers=None, regulatory=regulatory)


def _codes(out):
    return [d.code for d in out]


def test_validate_nudges_a_workspace_that_declares_nothing():
    from jaato_server.shared.scaffold.validate import _check_regulatory_declared
    out = []
    _check_regulatory_declared({"a": _prof("a"), "b": _prof("b")}, out)
    assert _codes(out) == ["regulatory_undeclared"]
    assert out[0].severity == "warn", "a nudge, the budget_control_absent posture"
    assert out[0].profile is None, "once per workspace, not per profile"
    assert "minimal" in out[0].message and "UNDECLARED" in out[0].message, (
        "the message must say absent is undeclared, never inferred as minimal")


def test_one_declaration_anywhere_in_the_tree_is_enough():
    from jaato_server.shared.scaffold.validate import _check_regulatory_declared
    out = []
    _check_regulatory_declared({"a": _prof("a", regulatory=object()), "b": _prof("b")}, out)
    assert out == []


def test_silent_for_no_profiles_but_not_for_bases():
    """A base-only tree is what `validate <workspace>` sees with no --set,
    and the base is where the block belongs -- so bases are NOT exempt."""
    from jaato_server.shared.scaffold.validate import _check_regulatory_declared
    out = []
    _check_regulatory_declared({}, out)
    assert out == []
    _check_regulatory_declared({"_base": _prof("_base", model=None)}, out)
    assert _codes(out) == ["regulatory_undeclared"]


def test_the_finding_reaches_validate_workspace_on_a_scaffolded_set(tmp_path):
    """End to end: the set `new profile-set` writes, validated as a workspace."""
    from jaato_server.shared.scaffold.build import _base_profile_yaml
    from jaato_server.shared.scaffold.validate import validate_workspace
    pdir = tmp_path / ".jaato" / "profiles"
    (pdir / "acme").mkdir(parents=True)
    (pdir / "_base_triage.yaml").write_text(_base_profile_yaml("triage"), encoding="utf-8")
    (pdir / "acme" / "triage.yaml").write_text(
        "name: triage\ndescription: triage stage\ninherits: [_base_triage]\n"
        "plugins: []\nmodel: claude-sonnet-4-6\nprovider: anthropic\n", encoding="utf-8")
    codes = {d.code for d in validate_workspace(str(tmp_path), profile_set="acme")}
    assert "regulatory_undeclared" in codes


# --------------------------------------------------------------- skill

def _skill_text() -> str:
    import jaato_server.shared.scaffold as scaffold
    return (Path(scaffold.__file__).parent / "integrations" / "claude-code" / "payload"
            / "SKILL.md").read_text(encoding="utf-8")


def test_the_skill_names_the_compliance_scopes_and_the_dossier():
    text = _skill_text()
    assert "jaato-scaffold explain oversight" in text
    assert "jaato-scaffold explain audit" in text
    assert "`dossier`" in text
    assert "regulatory:" in text and "record_keeping:" in text


def test_every_topic_the_skill_lists_is_one_the_cli_has():
    """The skill's own rule: if it contradicts `explain`, `explain` is right.
    A topic listed here that the CLI does not have is that contradiction."""
    from jaato_server.shared.scaffold.__main__ import _SCOPES
    listed = set(re.findall(r"^jaato-scaffold explain ([a-z]+)", _skill_text(), re.M))
    listed.discard("dependencies")   # a FACET, accepted after any topic, not a scope
    unknown = sorted(listed - set(_SCOPES))
    assert not unknown, f"the skill lists explain topics the CLI has not got: {unknown}"


def test_every_archetype_the_skill_lists_is_one_new_accepts():
    from jaato_server.shared.scaffold import archetypes
    m = re.search(r"Archetypes: \*\*(.*?)\*\*", _skill_text(), re.S)
    assert m, "the archetype list moved; update the guard"
    names = re.findall(r"`([a-z-]+)`", m.group(1))
    unknown = [n for n in names if archetypes.resolve(n) is None]
    assert not unknown, f"the skill lists archetypes `new` does not accept: {unknown}"
