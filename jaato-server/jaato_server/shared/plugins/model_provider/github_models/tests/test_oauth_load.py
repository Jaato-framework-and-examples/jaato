"""Tests for GitHub Models OAuth token loading with error reason surfacing.

``load_tokens`` historically collapsed every failure to ``None``, so a
broken token file looked identical to "never logged in" at
``verify_auth`` time.  The new ``try_load_tokens_with_reason`` helper
returns ``(tokens, reason)`` so callers can surface the specific
failure instead of emitting a misleading "no credentials" message.

The GitHub auth file layout is a little different from other providers
— it may hold an OAuth section, a Copilot section, or both.  A file
that contains only a Copilot token (no ``oauth`` section and no
top-level ``access_token``) is not an error; the tests below lock that
in so we don't regress.
"""

from pathlib import Path
import pytest
import json
import time

from ..oauth import try_load_tokens_with_reason


def _write_auth_file(tmp_path, payload: str):
    jaato_dir = tmp_path / ".jaato"
    jaato_dir.mkdir()
    auth_file = jaato_dir / "github_oauth.json"
    auth_file.write_text(payload)
    return auth_file


@pytest.fixture(autouse=True)
def _isolate_credential_tiers(empty_project_tier):
    """Keep the developer's real credentials out of these tests.

    Resolution walks a project tier (``<workspace>/.jaato/``, falling
    back to the working directory) and then a HOME tier
    (``~/.jaato/``).  Tests that clear ``os.environ`` or point
    ``workspace_path`` at a tmpdir still hit both fallbacks, so on a
    machine where the developer has actually authenticated they load
    REAL credentials: "no token" tests found one, and an env-token test
    saw the stored token win on precedence.  A clean CI box has no such
    file, which is why it never showed there.

    The HOME tier is now isolated for every test in the tree by
    ``jaato-server/conftest.py`` (#721).  This fixture adds the tier
    that cannot be closed globally — the working directory — via the
    shared ``empty_project_tier`` fixture in the model_provider
    conftest.
    """

class TestTryLoadTokensWithReason:
    def test_file_missing_returns_none_and_no_reason(self, tmp_path):
        tokens, reason = try_load_tokens_with_reason(workspace_path=str(tmp_path))
        assert tokens is None
        assert reason is None

    def test_valid_nested_oauth_section(self, tmp_path):
        _write_auth_file(tmp_path, json.dumps({
            "oauth": {
                "access_token": "gho_valid",
                "token_type": "bearer",
                "scope": "read:user",
                "created_at": time.time(),
            },
        }))
        tokens, reason = try_load_tokens_with_reason(workspace_path=str(tmp_path))
        assert tokens is not None
        assert tokens.access_token == "gho_valid"
        assert reason is None

    def test_valid_legacy_top_level_token(self, tmp_path):
        _write_auth_file(tmp_path, json.dumps({
            "access_token": "gho_legacy",
            "token_type": "bearer",
            "scope": "read:user",
            "created_at": time.time(),
        }))
        tokens, reason = try_load_tokens_with_reason(workspace_path=str(tmp_path))
        assert tokens is not None
        assert tokens.access_token == "gho_legacy"
        assert reason is None

    def test_copilot_only_file_is_not_an_error(self, tmp_path):
        """A file holding a Copilot token but no OAuth section is not a
        broken file — it's just "no OAuth token configured".  Must not
        produce a reason string, otherwise verify_auth would wrongly
        emit "auth file is broken"."""
        _write_auth_file(tmp_path, json.dumps({
            "copilot": {
                "token": "ghu_copilot",
                "expires_at": time.time() + 3600,
            },
        }))
        tokens, reason = try_load_tokens_with_reason(workspace_path=str(tmp_path))
        assert tokens is None
        assert reason is None

    def test_corrupt_json_surfaces_reason(self, tmp_path):
        path = _write_auth_file(tmp_path, "{not valid json")
        tokens, reason = try_load_tokens_with_reason(workspace_path=str(tmp_path))
        assert tokens is None
        assert reason is not None
        assert "invalid JSON" in reason
        assert str(path) in reason

    def test_malformed_oauth_section_surfaces_reason(self, tmp_path):
        """Nested ``oauth`` section missing required fields surfaces a
        malformed reason."""
        _write_auth_file(tmp_path, json.dumps({"oauth": {"random": "payload"}}))
        tokens, reason = try_load_tokens_with_reason(workspace_path=str(tmp_path))
        assert tokens is None
        assert reason is not None
        assert "malformed" in reason or "missing" in reason
