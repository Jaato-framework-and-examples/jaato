"""jaato-scaffold explain profile documents the inheritance semantics — the
conclusion of the 2026-06-24 plugins-inheritance investigation (union/additive,
empty []=none, scope-down via tool_scopes/whitelist), so the next author doesn't
re-derive it from config.py."""
from shared.scaffold import explain


def test_profile_documents_inheritance_semantics():
    _data, text = explain.profile()
    # plugins is union/additive, NOT override; can't scope down via the list
    assert "UNION" in text and "additive" in text
    assert "CANNOT scope DOWN" in text
    # empty [] = none (the corrected behaviour, not the old ~30-tools falsy bug)
    assert "plugins: []" in text and "tools=[]" in text
    assert "NONE of the registry tool plugins" in text
    # replace fields + the real scope-down path
    assert "child REPLACES" in text
    assert "tool_scopes" in text and "whitelist" in text
