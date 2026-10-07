"""``sibling_tools_moved``: a persona naming the sibling tools needs ``courier``.

``send_to_sibling`` / ``list_siblings`` left the ``subagent`` plugin for
``courier`` (session group messaging).  A profile that listed ``subagent``
for them still loads, so the only runtime symptom is a model hunting for a
tool its instructions promise; the finding says so at authoring time.
"""
from pathlib import Path
from types import SimpleNamespace

from jaato_server.shared.scaffold.validate import _check_sibling_tools_moved


def _run(tmp_path: Path, *, plugins, persona=None, inline=None):
    cr = tmp_path / ".jaato"
    (cr / "agents").mkdir(parents=True, exist_ok=True)
    kw = {"plugins": list(plugins)}
    if persona is not None:
        (cr / "agents" / "peer.md").write_text(persona, encoding="utf-8")
        kw["default_agent"] = "peer"
    if inline is not None:
        kw["system_instructions"] = inline
    prof = SimpleNamespace(**kw)
    out: list = []
    _check_sibling_tools_moved({"stage": prof}, tmp_path, str(cr), out)
    return out


def test_a_persona_naming_the_tools_without_courier_is_an_error(tmp_path):
    out = _run(tmp_path, plugins=["subagent"],
               persona="Use list_siblings, then send_to_sibling to nudge.")
    assert [d.code for d in out] == ["sibling_tools_moved"]
    assert out[0].severity == "error"
    assert out[0].profile == "stage" and out[0].where == "default_agent"
    assert "courier" in out[0].message and "send_to_sibling" in out[0].message


def test_courier_in_plugins_silences_it(tmp_path):
    out = _run(tmp_path, plugins=["subagent", "courier"],
               persona="Use send_to_sibling.")
    assert out == []


def test_the_deprecated_inline_instructions_are_checked_too(tmp_path):
    out = _run(tmp_path, plugins=["subagent"], inline="call list_siblings first")
    assert [d.where for d in out] == ["system_instructions"]


def test_a_persona_naming_neither_tool_is_not_judged(tmp_path):
    """Listing ``subagent`` alone is not the finding: most subagent users
    never touch the peer tools."""
    out = _run(tmp_path, plugins=["subagent"], persona="Spawn a worker.")
    assert out == []


def test_one_finding_per_profile(tmp_path):
    out = _run(tmp_path, plugins=[], persona="send_to_sibling",
               inline="list_siblings")
    assert len(out) == 1
