"""Tests for AppArmor sub-profile provisioning (Phase 4 §4.3.4).

Sub-profile machinery lets an isolated subagent's runner self-
confine to a distinct kernel-level profile.  Standalone profile
with ``//`` in its name (Audit 6's structural decision); separate
file from the parent.

Test surfaces:

1. **Subagent_id sanitization** — strict allow-list; rejects
   newlines, path-traversal characters, empty, oversize.
2. **Profile name + filename templates** — pinned bit-exact so
   ``SessionManager._spawn_isolated_runner``'s isolated session id
   stays in sync.
3. **Provision** — writes file + invokes ``apparmor_parser``;
   returns ``(True, profile_name)`` on success.
4. **Provision failure cleanup** — file unlinked on parser failure
   (mirrors ``_provision_profile_impl``'s pattern).
5. **Teardown** — symmetric to ``teardown_profile``; idempotent
   when already gone.
6. **Sub-profile body** — workspace allows present, integrity
   denies present, fragment-admit dropped, no tool_hat nesting.

CONFUSED-DEPUTY: these tests are load-bearing — sanitization
failures would let a malicious runner inject profile rules or
escape the profile directory.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# Same import shim as test_apparmor.py — avoid pulling heavy server/__init__.
if "server" not in sys.modules:
    _stub = types.ModuleType("server")
    _stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "server")]
    sys.modules["server"] = _stub

import server.apparmor as _apparmor_mod  # noqa: E402

AppArmorManager = _apparmor_mod.AppArmorManager


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def workspace_root(tmp_path):
    root = tmp_path / "workspaces"
    root.mkdir()
    (root / "sessions").mkdir()
    return root


@pytest.fixture
def profile_dir(tmp_path):
    d = tmp_path / "apparmor_profiles"
    d.mkdir()
    return d


@pytest.fixture
def manager(workspace_root, profile_dir):
    return AppArmorManager(
        workspace_root=str(workspace_root),
        venv_path="/usr/local/venv",
        profile_dir=str(profile_dir),
    )


# ──────────────────────────────────────────────────────────────────────
# subagent_id sanitization
# ──────────────────────────────────────────────────────────────────────


class TestSubagentIdSanitization:
    """``_validate_subagent_id`` is the confused-deputy guard for
    sub-profile names.  Strict allow-list, length cap, non-empty."""

    def test_alphanumeric_is_valid(self):
        assert AppArmorManager._validate_subagent_id("agent1") is None

    def test_with_hyphens_is_valid(self):
        assert AppArmorManager._validate_subagent_id("agent-1") is None

    def test_with_underscores_is_valid(self):
        assert AppArmorManager._validate_subagent_id("agent_1") is None

    def test_empty_string_rejected(self):
        err = AppArmorManager._validate_subagent_id("")
        assert err is not None
        assert "empty" in err.lower()

    def test_non_str_rejected(self):
        err = AppArmorManager._validate_subagent_id(42)
        assert err is not None
        assert "must be a str" in err

    def test_none_rejected(self):
        err = AppArmorManager._validate_subagent_id(None)
        assert err is not None

    def test_oversize_rejected(self):
        """65-char id exceeds the 64-char cap."""
        err = AppArmorManager._validate_subagent_id("a" * 65)
        assert err is not None
        assert "length" in err.lower()
        assert "cap" in err.lower()

    def test_exactly_max_len_accepted(self):
        """64 chars is the boundary; must be accepted."""
        assert AppArmorManager._validate_subagent_id("a" * 64) is None

    def test_newline_rejected(self):
        """Newline injection would let an attacker break out of the
        profile-name string into the profile-file body."""
        err = AppArmorManager._validate_subagent_id("agent\nprofile evil { }")
        assert err is not None
        assert "outside" in err

    def test_path_traversal_rejected(self):
        """Path separators / dots would let an attacker target
        ``../../etc/apparmor.d/``-style filenames."""
        err = AppArmorManager._validate_subagent_id("../escape")
        assert err is not None

    def test_slash_rejected(self):
        """Slash would clash with the profile's `//` separator."""
        err = AppArmorManager._validate_subagent_id("agent/1")
        assert err is not None

    def test_space_rejected(self):
        """Space is not in the allow-list — strict rejection."""
        err = AppArmorManager._validate_subagent_id("my agent")
        assert err is not None

    def test_dot_rejected(self):
        """Dot is allowed in fragment filenames but NOT in subagent_id
        (Audit 6: strict allow-list ``[A-Za-z0-9_-]`` only)."""
        err = AppArmorManager._validate_subagent_id("agent.1")
        assert err is not None


# ──────────────────────────────────────────────────────────────────────
# Name + filename templates
# ──────────────────────────────────────────────────────────────────────


class TestNameAndFilenameTemplates:
    """Templates are pinned bit-exact — both
    ``SessionManager._spawn_isolated_runner``'s
    ``isolated_session_id`` and ``_spawn_session_runner_unconditional``'s
    eventual ``profile_name`` arg derive from these."""

    def test_sub_profile_name_template(self):
        name = AppArmorManager.get_sub_profile_name("sess-A", "agent-1")
        assert name == "jaato-ws-sess-A//agent-1"

    def test_sub_profile_filename_template(self):
        filename = AppArmorManager.get_sub_profile_filename("sess-A", "agent-1")
        assert filename == "jaato-ws-sess-A__sub_agent-1"

    def test_filename_has_no_slashes(self):
        """Filenames must not contain ``/`` (path separator).  The
        ``__sub_`` infix replaces the profile name's ``//``."""
        filename = AppArmorManager.get_sub_profile_filename("sess-A", "agent-1")
        assert "/" not in filename


# ──────────────────────────────────────────────────────────────────────
# provision_sub_profile — happy path + failures
# ──────────────────────────────────────────────────────────────────────


class TestProvisionSubProfile:
    """``provision_sub_profile`` writes the rendered body to a file
    in ``profile_dir`` and invokes ``apparmor_parser -r`` to load it.
    Returns ``(True, profile_name)`` on success."""

    def test_happy_path_writes_and_loads(self, manager, profile_dir):
        manager._available = True
        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            ok, result = manager.provision_sub_profile(
                parent_session_id="sess-A",
                subagent_id="agent-1",
                workspace_path="/workspace",
            )

        assert ok is True
        assert result == "jaato-ws-sess-A//agent-1"

        sub_path = profile_dir / "jaato-ws-sess-A__sub_agent-1"
        assert sub_path.exists()
        content = sub_path.read_text()
        # Sub-profile name shows up in the file body.
        assert "jaato-ws-sess-A//agent-1" in content

        # apparmor_parser invoked with -r and the file path.
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "sudo"
        assert cmd[1] == "apparmor_parser"
        assert "-r" in cmd
        assert str(sub_path) in cmd

    def test_invalid_subagent_id_rejected_before_disk_io(
        self, manager, profile_dir,
    ):
        """Sanitization rejects BEFORE any file is written — confused-
        deputy guard.  Pin: profile_dir stays empty."""
        manager._available = True
        with patch("server.apparmor.subprocess.run") as mock_run:
            ok, err = manager.provision_sub_profile(
                parent_session_id="sess-A",
                subagent_id="agent\n../etc/passwd",
                workspace_path="/workspace",
            )

        assert ok is False
        assert "subagent_id rejected" in err
        # No file written.
        assert list(profile_dir.iterdir()) == []
        # No parser call.
        mock_run.assert_not_called()

    def test_cleans_up_file_on_parser_failure(self, manager, profile_dir):
        """Mirrors ``_provision_profile_impl``'s cleanup pattern —
        on parser failure, unlink the written file so it doesn't
        accumulate as junk in ``profile_dir``."""
        manager._available = True
        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="profile syntax error",
            )
            ok, err = manager.provision_sub_profile(
                parent_session_id="sess-A",
                subagent_id="agent-1",
                workspace_path="/workspace",
            )

        assert ok is False
        assert "apparmor_parser exit=1" in err
        assert "profile syntax error" in err
        # File cleaned up.
        assert not (profile_dir / "jaato-ws-sess-A__sub_agent-1").exists()

    def test_cleans_up_on_parser_timeout(self, manager, profile_dir):
        import subprocess
        manager._available = True
        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="apparmor_parser", timeout=30,
            )
            ok, err = manager.provision_sub_profile(
                parent_session_id="sess-A",
                subagent_id="agent-1",
                workspace_path="/workspace",
            )

        assert ok is False
        assert "timed out" in err.lower()
        assert not (profile_dir / "jaato-ws-sess-A__sub_agent-1").exists()

    def test_returns_false_when_unavailable(self, manager):
        manager._available = False
        ok, err = manager.provision_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
        )
        assert ok is False
        assert "unavailable" in err.lower()


# ──────────────────────────────────────────────────────────────────────
# teardown_sub_profile
# ──────────────────────────────────────────────────────────────────────


class TestTeardownSubProfile:
    """Symmetric to ``provision_sub_profile`` — runs
    ``apparmor_parser -R`` + unlinks the file."""

    def test_unloads_and_removes(self, manager, profile_dir):
        manager._available = True
        sub_file = profile_dir / "jaato-ws-sess-A__sub_agent-1"
        sub_file.write_text("# sub-profile")

        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = manager.teardown_sub_profile("sess-A", "agent-1")

        assert result is True
        assert not sub_file.exists()
        # apparmor_parser -R invoked.
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "-R" in cmd

    def test_returns_true_if_already_gone(self, manager):
        """Idempotent: tearing down a nonexistent sub-profile is OK."""
        manager._available = True
        result = manager.teardown_sub_profile("sess-A", "nonexistent")
        assert result is True

    def test_returns_false_when_unavailable(self, manager):
        manager._available = False
        result = manager.teardown_sub_profile("sess-A", "agent-1")
        assert result is False

    def test_invalid_subagent_id_returns_false(self, manager):
        """Invalid subagent_id can't be sanitized → can't be torn
        down.  Returns False (logged WARNING)."""
        manager._available = True
        result = manager.teardown_sub_profile("sess-A", "agent\n../escape")
        assert result is False


# ──────────────────────────────────────────────────────────────────────
# _render_sub_profile body shape
# ──────────────────────────────────────────────────────────────────────


class TestRenderSubProfile:
    """Pin the sub-profile body's rule-set posture per Audit 6's
    conservative-tightening decision."""

    def test_workspace_allows_present(self, manager):
        body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
        )
        assert "/workspace/   rw," in body
        assert "/workspace/** rwkl," in body

    def test_integrity_denies_present(self, manager):
        body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
        )
        # Mirror parent base — write-denies on .jaato/agents/**, etc.
        assert "audit deny /workspace/.jaato/agents/**" in body
        assert "audit deny /workspace/.jaato/profiles/**" in body
        assert "audit deny /workspace/.jaato/reactors.json" in body

    def test_read_denies_present(self, manager):
        """Read-denies mirror parent's tool_hat — information-isolation
        between agents."""
        body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
        )
        assert "audit deny /workspace/.jaato/agents/**             r," in body
        assert "audit deny /workspace/.jaato/profiles/**           r," in body

    def test_no_tool_hat_subprofile_nested(self, manager):
        """Audit 6: flat profile — no further nesting."""
        body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
        )
        # No nested ``profile tool_hat`` declaration.
        assert "profile tool_hat" not in body

    def test_no_fragment_include_directive(self, manager):
        """Audit 6: drop fragment-admit — sub-profile can't have
        external refs added via add_reference_fragment."""
        body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
        )
        assert "include if exists" not in body

    def test_profile_name_in_body(self, manager):
        """The rendered body must declare the full sub-profile name
        with the ``//`` separator."""
        body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
        )
        assert 'profile "jaato-ws-sess-A//agent-1"' in body

    def test_parent_session_id_metadata_present(self, manager):
        """Body comments include parent + subagent id for auditing."""
        body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
        )
        assert "Parent session: sess-A" in body
        assert "Subagent id:    agent-1" in body
