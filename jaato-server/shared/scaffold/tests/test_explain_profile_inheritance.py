"""jaato-scaffold explain profile documents the inheritance semantics — the
conclusion of the 2026-06-24 plugins-inheritance investigation (union/additive,
empty []=none, scope-down via tool_scopes/whitelist), so the next author doesn't
re-derive it from config.py.

#791 added the two completion fields to that documented contract.  They were
the ones a profile author could only learn by reading ``_merge_profiles``, and
getting them wrong cost an "interrogate" profile its budget ceiling: it stopped
inheriting in order to drop one processor and silently lost ``budget_control``,
``max_turns``, ``runtime_limits``, ``env`` and ``plugin_configs`` with it."""
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


def test_profile_documents_completion_field_inheritance():
    """#791: the two completion fields follow rules the doc never named."""
    _data, text = explain.profile()
    # completion_processors concatenate, and `[]` in the child is not a reset
    assert "completion_processors" in text
    assert "CONCATENATED" in text
    assert "ADDS NOTHING" in text
    # ...with the one opt-out, and its failure mode
    assert "suppress_inherited_processors" in text
    assert "load ERROR" in text
    # the payload schemas replace, and {} is a value (null is not)
    assert "completion_payload_schema" in text
    assert "spawn_payload_schema" in text
    assert "empty dict `{}` IS a" in text


def test_profile_warns_against_dropping_inherits_to_lose_a_processor():
    """The wrong turn #791 documents: escaping one processor by declaring the
    profile from scratch, and losing every safety ceiling with it."""
    _data, text = explain.profile()
    assert "Don't stop inheriting just to drop a processor" in text
    for lost in ("budget_control", "max_turns", "runtime_limits"):
        assert lost in text
