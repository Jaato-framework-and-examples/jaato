"""allowed_scopes write-side gate: deployment policy for which memory scopes
the model may store. Default permissive; a disallowed scope is hard-rejected
(not silently down-scoped), so a deployment can set ["project"] to stay off the
HOME/global tier entirely.
"""
from unittest.mock import MagicMock

from shared.plugins.memory.plugin import MemoryPlugin
from shared.plugins.memory.models import VALID_SCOPES, SCOPE_PROJECT, SCOPE_UNIVERSAL


def test_resolve_allowed_scopes_parsing():
    f = MemoryPlugin._resolve_allowed_scopes
    assert f(None) == VALID_SCOPES                                  # default permissive
    assert f([SCOPE_PROJECT]) == frozenset({SCOPE_PROJECT})
    assert f([SCOPE_PROJECT, SCOPE_UNIVERSAL]) == VALID_SCOPES
    assert f(["bogus"]) == frozenset()                             # all invalid -> empty
    assert f([SCOPE_PROJECT, "bogus"]) == frozenset({SCOPE_PROJECT})
    assert f([]) == frozenset()                                    # explicit empty honored
    assert f(["PROJECT", " universal "]) == VALID_SCOPES           # normalized (strip/lower)


def test_default_is_permissive():
    assert MemoryPlugin()._allowed_scopes == VALID_SCOPES


def test_initialize_applies_allowed_scopes(tmp_path):
    p = MemoryPlugin()
    p.initialize({"allowed_scopes": [SCOPE_PROJECT],
                  "global_storage_path": str(tmp_path / "g.jsonl")})
    assert p._allowed_scopes == frozenset({SCOPE_PROJECT})


def test_store_hard_rejects_disallowed_scope():
    p = MemoryPlugin()
    p._allowed_scopes = frozenset({SCOPE_PROJECT})   # bot's deployment policy
    res = p._execute_store({"content": "c", "description": "d",
                            "tags": ["topic"], "scope": SCOPE_UNIVERSAL})
    assert res["status"] == "rejected"               # NOT a silent down-scope
    assert SCOPE_UNIVERSAL in res["error"]
    assert res["allowed_scopes"] == [SCOPE_PROJECT]


def test_gate_fires_before_init_and_tags_checks():
    # disallowed scope rejected even with no storage / no tags — the policy gate
    # precedes the per-request content validations.
    p = MemoryPlugin()
    p._allowed_scopes = frozenset({SCOPE_PROJECT})
    res = p._execute_store({"content": "c", "scope": SCOPE_UNIVERSAL})
    assert res["status"] == "rejected"


def test_store_allows_permitted_scope_passes_gate():
    p = MemoryPlugin()
    p._allowed_scopes = VALID_SCOPES
    p._storage = MagicMock()
    p._indexer = MagicMock()
    p._get_session_id = lambda: None
    res = p._execute_store({"content": "c", "description": "d",
                            "tags": ["topic"], "scope": SCOPE_PROJECT})
    assert res["status"] == "success"
    p._storage.save.assert_called_once()
