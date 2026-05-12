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


# ──────────────────────────────────────────────────────────────────────
# Phase 5 §5.9: supervisor-declared sub-profile tightenings
# ──────────────────────────────────────────────────────────────────────


class TestSubProfileTighteningsRender:
    """Pin the rendered sub-profile body's behavior under each
    supervisor-declared tightening flag.

    These tests are the load-bearing snapshot pins for §5.9: the
    validator only checks structural safety of the supervisor's
    input, but the renderer is what actually narrows the
    sub-profile's authority on the kernel.  If the renderer
    silently ignored a tightening, the validator wouldn't catch
    it — that's the regression these tests guard against."""

    def test_no_tightenings_renders_default_body(self, manager):
        """Pin: ``None`` / empty / omitted tightenings produce
        byte-identical output to the legacy 3-arg call.
        Required for backward compatibility with all pre-§5.9
        callers (every test in this file, every production
        caller via session_manager that didn't pass the new
        kwarg)."""
        default_body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
        )
        none_body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
            tightenings=None,
        )
        empty_body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
            tightenings={},
        )
        assert default_body == none_body == empty_body

    def test_workspace_subpath_narrows_allow_rule(self, manager):
        """Pin: ``isolated_workspace_subpath: "scratch"`` narrows
        the workspace allow from ``/workspace/**`` to
        ``/workspace/scratch/**``.

        Load-bearing security pin — without this the renderer
        silently ignores the supervisor's subpath declaration and
        the subagent gets default (broader) access."""
        body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
            tightenings={"isolated_workspace_subpath": "scratch"},
        )
        # New scoped allow MUST be present.
        assert "/workspace/scratch/   rw," in body
        assert "/workspace/scratch/** rwkl," in body
        # Default broader allow MUST be absent.
        assert "/workspace/   rw," not in body
        assert "/workspace/** rwkl," not in body

    def test_workspace_subpath_with_nested_dir(self, manager):
        """Pin: subpath with internal ``/`` is rendered
        verbatim (validator already vetted it for path
        traversal)."""
        body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
            tightenings={
                "isolated_workspace_subpath": "scratch/agent-1",
            },
        )
        assert "/workspace/scratch/agent-1/   rw," in body
        assert "/workspace/scratch/agent-1/** rwkl," in body

    def test_read_only_workspace_downgrades_perms(self, manager):
        """Pin: ``isolated_read_only_workspace: True`` downgrades
        workspace ``rwkl`` to ``r`` and dir ``rw`` to ``r``.
        No write/lock/link capability anywhere under the
        workspace tree."""
        body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
            tightenings={"isolated_read_only_workspace": True},
        )
        # Workspace allow rules — read-only forms present.
        assert "/workspace/   r," in body
        assert "/workspace/** r," in body
        # Default write forms absent.
        assert "/workspace/   rw," not in body
        assert "/workspace/** rwkl," not in body

    def test_combined_subpath_plus_read_only(self, manager):
        """Pin: subpath + read-only compose — read-only allow
        rules scoped to the subpath, no broader allow rules at
        all."""
        body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
            tightenings={
                "isolated_workspace_subpath": "scratch",
                "isolated_read_only_workspace": True,
            },
        )
        assert "/workspace/scratch/   r," in body
        assert "/workspace/scratch/** r," in body
        # No other workspace allows at all.
        assert "/workspace/   rw," not in body
        assert "/workspace/   r,\n" not in body  # bare workspace
        assert "/workspace/** rwkl," not in body

    def test_tightenings_do_not_erode_integrity_denies(self, manager):
        """Pin: integrity-deny block on ``.jaato/**`` is
        baseline-invariant — tightenings can only strengthen
        security, never weaken it.  This test guards against an
        accidental refactor that templates the deny block
        through the tightening branch."""
        body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
            tightenings={
                "isolated_workspace_subpath": "scratch",
                "isolated_read_only_workspace": True,
            },
        )
        assert "audit deny /workspace/.jaato/agents/**" in body
        assert "audit deny /workspace/.jaato/profiles/**" in body
        assert "audit deny /workspace/.jaato/reactors.json" in body

    def test_tightenings_do_not_erode_drop_invariants(self, manager):
        """Pin: §4.3.4 + v15 + §5.10e DROP invariants stay
        intact under tightenings.  No ``change_profile``, no
        fragment-admit ``include if exists``, no nested
        ``profile tool_hat`` — these are layered guarantees that
        tightenings only add to, never remove from."""
        body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
            tightenings={
                "isolated_workspace_subpath": "scratch",
                "isolated_read_only_workspace": True,
            },
        )
        assert "change_profile -> unconfined" not in body
        assert "include if exists" not in body
        assert "profile tool_hat" not in body

    def test_profile_name_unchanged_by_tightenings(self, manager):
        """Pin: tightenings affect rule bodies, NOT the profile
        name.  Sub-runner self-confines to
        ``jaato-ws-{parent}//{subagent}`` regardless of
        tightenings."""
        body = manager._render_sub_profile(
            parent_session_id="sess-A",
            subagent_id="agent-1",
            workspace_path="/workspace",
            tightenings={
                "isolated_workspace_subpath": "scratch",
                "isolated_read_only_workspace": True,
            },
        )
        assert 'profile "jaato-ws-sess-A//agent-1"' in body


class TestProvisionSubProfileWithTightenings:
    """Pin that ``provision_sub_profile`` threads the
    ``tightenings`` kwarg through to ``_render_sub_profile``."""

    def test_provision_forwards_tightenings_to_render(
        self, manager, profile_dir,
    ):
        """Pin: ``tightenings`` kwarg passed to
        ``provision_sub_profile`` reaches the rendered profile
        file on disk.

        Asserts on the written-file content rather than on the
        render-method invocation — that way the test exercises
        the full provision → write → load pipeline (matching
        the existing happy-path test's style)."""
        manager._available = True
        tightenings = {"isolated_workspace_subpath": "scratch"}
        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            ok, _ = manager.provision_sub_profile(
                parent_session_id="sess-A",
                subagent_id="agent-1",
                workspace_path="/workspace",
                tightenings=tightenings,
            )
        assert ok is True

        sub_path = profile_dir / "jaato-ws-sess-A__sub_agent-1"
        content = sub_path.read_text()

        # The tightening reached the rendered body: narrowed
        # allow + no default broader allow.
        assert "/workspace/scratch/   rw," in content
        assert "/workspace/scratch/** rwkl," in content
        assert "/workspace/   rw," not in content
        assert "/workspace/** rwkl," not in content

    def test_provision_default_tightenings_render_unchanged(
        self, manager, profile_dir,
    ):
        """Pin: when caller omits ``tightenings``, the rendered
        body retains the pre-§5.9 default workspace allow rules
        — backward-compat with all pre-§5.9 production callers
        + test fixtures."""
        manager._available = True
        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            ok, _ = manager.provision_sub_profile(
                parent_session_id="sess-A",
                subagent_id="agent-1",
                workspace_path="/workspace",
            )
        assert ok is True

        sub_path = profile_dir / "jaato-ws-sess-A__sub_agent-1"
        content = sub_path.read_text()
        assert "/workspace/   rw," in content
        assert "/workspace/** rwkl," in content
