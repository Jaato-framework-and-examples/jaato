"""Tests for Anthropic OAuth token loading with error reason surfacing.

``load_tokens`` historically collapsed every failure (corrupt JSON,
permission error, missing field) to ``None``, so a broken token file
looked identical to "never logged in" at ``verify_auth`` time.  The new
``try_load_tokens_with_reason`` helper returns ``(tokens, reason)`` so
callers can surface the specific failure instead of emitting a
misleading "no credentials" message.
"""

import json
import time

from ..oauth import OAuthTokens, try_load_tokens_with_reason


def _write_token_file(tmp_path, contents: str):
    jaato_dir = tmp_path / ".jaato"
    jaato_dir.mkdir()
    auth_file = jaato_dir / "anthropic_oauth.json"
    auth_file.write_text(contents)
    return auth_file


class TestTryLoadTokensWithReason:
    def test_file_missing_returns_none_and_no_reason(self, tmp_path):
        tokens, reason = try_load_tokens_with_reason(workspace_path=str(tmp_path))
        assert tokens is None
        assert reason is None

    def test_valid_file_loads_tokens(self, tmp_path):
        _write_token_file(
            tmp_path,
            json.dumps({
                "access_token": "sk-ant-oat01-valid",
                "refresh_token": "rt-valid",
                "expires_at": time.time() + 3600,
                "token_type": "Bearer",
            }),
        )
        tokens, reason = try_load_tokens_with_reason(workspace_path=str(tmp_path))
        assert tokens is not None
        assert tokens.access_token == "sk-ant-oat01-valid"
        assert reason is None

    def test_corrupt_json_surfaces_reason(self, tmp_path):
        path = _write_token_file(tmp_path, "{not valid json")
        tokens, reason = try_load_tokens_with_reason(workspace_path=str(tmp_path))
        assert tokens is None
        assert reason is not None
        assert "invalid JSON" in reason
        assert str(path) in reason

    def test_missing_fields_surfaces_reason(self, tmp_path):
        _write_token_file(tmp_path, json.dumps({"random": "payload"}))
        tokens, reason = try_load_tokens_with_reason(workspace_path=str(tmp_path))
        assert tokens is None
        assert reason is not None
        assert "malformed" in reason or "missing" in reason
