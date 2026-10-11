"""The authoring surface knows the ``decision_models:`` block.

``explain profile`` documents it and ``validate`` checks it, but nothing in
a generated profile hinted that a stage can bind a decision model, and the
``decide`` tool is hidden until one is bound.  So ``new profile-set`` emits
the block COMMENTED in each set profile (a live entry would send workspace
content to a third party), and the Claude Code integration skill names it.

The uncommented example must be a block ``validate`` accepts, or the
generator teaches a profile it then refuses.
"""
from __future__ import annotations

from pathlib import Path

from jaato_server.shared.tests.reversion import Reversion

_BUILD = "jaato-sdk/jaato_sdk/scaffold/build.py"
_SKILL = "jaato-sdk/jaato_sdk/scaffold/integrations/claude-code/payload/SKILL.md"

REVERSIONS = [
    Reversion(
        target=_BUILD,
        find="        *_decision_models_example(),\n",
        replace="",
        because="a scaffolded set carries no trace of decision_models, so "
                "the decide tool is undiscoverable from the authoring surface",
        test="test_the_set_profile_carries_the_block_commented",
    ),
    Reversion(
        target=_SKILL,
        find="**A decision model is bound by name, never as a tier.**",
        replace="**A model is bound.**",
        because="the skill an author reads first never names decision_models",
        test="test_the_skill_names_decision_models_and_the_tool",
    ),
]


def _set_yaml() -> str:
    from jaato_server.shared.scaffold.build import _set_profile_yaml
    return _set_profile_yaml("triage", "anthropic", "claude-sonnet-4-6")


def test_the_set_profile_carries_the_block_commented():
    text = _set_yaml()
    lines = [ln for ln in text.splitlines() if "decision_models" in ln
             or "typesafe/jev" in ln]
    assert any("decision_models:" in ln for ln in lines), text
    for line in lines:
        assert line.lstrip().startswith("#"), f"emitted LIVE: {line!r}"


def _uncommented_block(text: str) -> str:
    out, inside = [], False
    for line in text.splitlines():
        if line.startswith("# decision_models:"):
            inside = True
        elif inside and not line.startswith("#   "):
            break
        if inside:
            out.append(line[2:])
    return "\n".join(out) + "\n"


def _workspace(tmp_path: Path, set_text: str) -> Path:
    from jaato_server.shared.scaffold.build import _base_profile_yaml
    pdir = tmp_path / ".jaato" / "profiles"
    (pdir / "acme").mkdir(parents=True)
    (pdir / "_base_triage.yaml").write_text(
        _base_profile_yaml("triage"), encoding="utf-8")
    (pdir / "acme" / "triage.yaml").write_text(set_text, encoding="utf-8")
    return tmp_path


def test_the_uncommented_example_validates(tmp_path):
    from jaato_server.shared.scaffold.validate import validate_workspace
    text = _set_yaml()
    block = _uncommented_block(text)
    assert block.startswith("decision_models:") and "triage:" in block
    ws = _workspace(tmp_path, text + block)
    findings = validate_workspace(str(ws), profile_set="acme")
    bad = [d for d in findings
           if d.code.startswith("decision") and d.severity == "error"]
    assert bad == [], [d.message for d in bad]


def test_the_commented_example_binds_nothing():
    import yaml
    assert "decision_models" not in (yaml.safe_load(_set_yaml()) or {})


def test_the_skill_names_decision_models_and_the_tool():
    import jaato_sdk.scaffold as scaffold
    text = (Path(scaffold.__file__).parent / "integrations" / "claude-code"
            / "payload" / "SKILL.md").read_text(encoding="utf-8")
    assert "**A decision model is bound by name, never as a tier.**" in text
    assert "decision_models:" in text and "`decide`" in text
    assert "decision_tool_without_models" in text
