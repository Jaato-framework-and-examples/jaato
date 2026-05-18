"""Tests for ``SessionManager._resolve_profile`` env_file overlay.

Pin the wire-gap fix discovered 2026-05-18 (kb-enablement-2.0
handoff_test profile-set refactor): when a workspace declares
``JAATO_PROFILE_SET=<name>`` in its ``.env``, the profile-set subdir
``profiles/<name>/`` must be discovered.  Pre-0.6.124 the
session-scoped ContextVar wasn't populated when ``_resolve_profile``
fired, so ``discover_profiles`` read ``None`` for
``JAATO_PROFILE_SET`` and silently skipped the subdir scan.

Symmetric to:
- PR #139 (completion_validators wire-gap)
- PR #140 (plugin_configs pass:// wire-gap)

See ``project_backlog_profile_field_propagation_duplication`` memory
for the bug class — "config readers fire before session_env is
populated, silent failure".  This PR closes a third instance.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from server.session_manager import SessionManager


def _write_profile(dir_path: Path, name: str, model: str = "test-model") -> None:
    """Write a minimal valid profile JSON file."""
    import json
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{name}.json").write_text(json.dumps({
        "name": name,
        "description": f"{name} profile",
        "model": model,
        "plugins": [],
    }))


def _make_manager() -> SessionManager:
    """Build a SessionManager with no daemon plumbing — we only need
    ``_resolve_profile``."""
    # SessionManager() takes no required args today — but the
    # constructor may evolve.  Use a MagicMock-spec'd subclass if
    # the signature changes.
    mgr = object.__new__(SessionManager)
    return mgr


class TestEnvFileOverlay:
    """Pin: env_file is parsed and overlaid onto the session contextvar
    BEFORE discover_profiles fires."""

    def test_no_env_file_baseline_behavior(self, tmp_path: Path) -> None:
        """Without env_file: profile-set subdir is NOT scanned (legacy
        behaviour preserved)."""
        config_root = tmp_path / ".jaato"
        # Base profile in profiles/
        _write_profile(config_root / "profiles", "codegen")
        # Profile-set in profiles/zhipuai_glm5/ — should NOT be picked
        # up without the env var.
        _write_profile(
            config_root / "profiles" / "zhipuai_glm5", "codegen",
            model="overriden-model",
        )

        mgr = _make_manager()
        profile, err = mgr._resolve_profile(
            "codegen",
            workspace_path=str(tmp_path),
            config_root=str(config_root),
            env_file=None,
        )
        assert err is None
        assert profile is not None
        assert profile.model == "test-model", (
            "Without env_file, the profile-set override should NOT apply"
        )

    def test_env_file_with_profile_set_picks_up_subdir(
        self, tmp_path: Path,
    ) -> None:
        """Pin (the fix): env_file declaring JAATO_PROFILE_SET=<name>
        causes profiles/<name>/ to be scanned, and entries there
        override the base profiles/ entries."""
        config_root = tmp_path / ".jaato"
        _write_profile(config_root / "profiles", "codegen")
        _write_profile(
            config_root / "profiles" / "zhipuai_glm5", "codegen",
            model="overridden-model",
        )
        env_file = tmp_path / ".env"
        env_file.write_text("JAATO_PROFILE_SET=zhipuai_glm5\n")

        mgr = _make_manager()
        profile, err = mgr._resolve_profile(
            "codegen",
            workspace_path=str(tmp_path),
            config_root=str(config_root),
            env_file=str(env_file),
        )
        assert err is None
        assert profile is not None
        assert profile.model == "overridden-model", (
            "With JAATO_PROFILE_SET in env_file, the profile-set "
            "subdir entry should win"
        )

    def test_env_file_parse_failure_is_nonfatal(
        self, tmp_path: Path,
    ) -> None:
        """A malformed env_file logs a warning but still allows
        profile resolution to proceed against the base tier."""
        config_root = tmp_path / ".jaato"
        _write_profile(config_root / "profiles", "codegen")
        # Path that doesn't exist — dotenv_values returns {}
        env_file_missing = tmp_path / "nonexistent.env"

        mgr = _make_manager()
        profile, err = mgr._resolve_profile(
            "codegen",
            workspace_path=str(tmp_path),
            config_root=str(config_root),
            env_file=str(env_file_missing),
        )
        # dotenv_values on a missing file silently returns {} — no
        # exception path; profile resolves against the base tier.
        assert err is None
        assert profile is not None
        assert profile.model == "test-model"

    def test_overlay_does_not_persist_after_resolve(
        self, tmp_path: Path,
    ) -> None:
        """The contextvar overlay is scoped to the discover_profiles
        call and restored on return — concurrent callers don't see
        each other's env_file values."""
        from shared.session_context import _session_env as _ctx_var, get_session_env

        config_root = tmp_path / ".jaato"
        _write_profile(config_root / "profiles", "codegen")
        env_file = tmp_path / ".env"
        env_file.write_text("JAATO_PROFILE_SET=some_set\n")

        # Snapshot the contextvar's value BEFORE resolve
        before = _ctx_var.get()

        mgr = _make_manager()
        _, _ = mgr._resolve_profile(
            "codegen",
            workspace_path=str(tmp_path),
            config_root=str(config_root),
            env_file=str(env_file),
        )

        # After resolve, the contextvar must be back to its previous
        # value (None, since no parent set anything).
        after = _ctx_var.get()
        assert after == before, (
            f"Contextvar leaked after _resolve_profile: "
            f"before={before!r}, after={after!r}"
        )
        # And get_session_env shouldn't see the overlay
        assert get_session_env("JAATO_PROFILE_SET") is None

    def test_env_file_pass_uri_resolution(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Pin: env_file values with pass:// URIs resolve daemon-side
        via expand_variables — same trust posture as
        JaatoServer._resolve_session_env.

        Stubs the secret resolver so the test doesn't touch a real
        pass store.
        """
        from shared.plugins.subagent import config as _subagent_config

        config_root = tmp_path / ".jaato"
        _write_profile(config_root / "profiles", "codegen")
        # The "real" set name is what the pass URI resolves to —
        # secret resolver stub returns this verbatim.
        _write_profile(
            config_root / "profiles" / "resolved_set", "codegen",
            model="from-resolved-set",
        )

        def _stub_resolve_secret_uri(value):
            if isinstance(value, str) and value.startswith("pass://"):
                return "resolved_set"
            return value

        monkeypatch.setattr(
            _subagent_config, "_resolve_secret_uri",
            _stub_resolve_secret_uri,
        )

        env_file = tmp_path / ".env"
        env_file.write_text(
            "JAATO_PROFILE_SET=pass://jaato/profile-sets/active\n"
        )

        mgr = _make_manager()
        profile, err = mgr._resolve_profile(
            "codegen",
            workspace_path=str(tmp_path),
            config_root=str(config_root),
            env_file=str(env_file),
        )
        assert err is None
        assert profile is not None
        assert profile.model == "from-resolved-set", (
            "pass:// in env_file must resolve daemon-side before "
            "discover_profiles reads JAATO_PROFILE_SET"
        )
