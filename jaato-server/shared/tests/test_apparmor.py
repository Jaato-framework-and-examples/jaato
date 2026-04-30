"""Tests for AppArmorManager."""

import os
import platform
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Avoid importing server.__init__ which pulls heavy deps (google, etc.)
import types
if "server" not in sys.modules:
    _stub = types.ModuleType("server")
    _stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "server")]
    sys.modules["server"] = _stub

import server.apparmor as _apparmor_mod
AppArmorManager = _apparmor_mod.AppArmorManager


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


class TestAvailability:
    def test_not_available_on_non_linux(self, manager):
        with patch("server.apparmor.platform.system", return_value="Darwin"):
            manager._available = None  # Reset cache
            assert manager.is_available() is False

    def test_not_available_without_apparmor_parser(self, manager):
        with patch("server.apparmor.platform.system", return_value="Linux"), \
             patch("server.apparmor.shutil.which", return_value=None):
            manager._available = None
            assert manager.is_available() is False

    def test_not_available_without_kernel_module(self, manager):
        with patch("server.apparmor.platform.system", return_value="Linux"), \
             patch("server.apparmor.shutil.which", return_value="/usr/sbin/apparmor_parser"), \
             patch("server.apparmor.Path.exists", return_value=False):
            manager._available = None
            assert manager.is_available() is False

    def test_caches_result(self, manager):
        manager._available = True
        assert manager.is_available() is True
        # Should not re-check
        manager._available = False
        assert manager.is_available() is False


class TestProfileName:
    def test_profile_name_format(self, manager):
        assert manager.get_profile_name("20250101_120000") == "jaato-ws-20250101_120000"

    def test_profile_name_with_special_chars(self, manager):
        assert manager.get_profile_name("session_1") == "jaato-ws-session_1"


class TestRenderProfile:
    def test_contains_workspace_path(self, manager):
        profile = manager._render_profile("s1", "/srv/workspaces/sessions/s1")
        assert "/srv/workspaces/sessions/s1/" in profile
        assert "/srv/workspaces/sessions/s1/**" in profile

    def test_contains_venv_path(self, manager):
        profile = manager._render_profile("s1", "/workspace")
        assert "/usr/local/venv/" in profile

    def test_sibling_workspaces_implicitly_denied(self, manager, workspace_root):
        """Sibling workspaces are denied by AppArmor's default-deny policy.

        We must NOT emit an explicit deny on sessions_root because in
        AppArmor a deny rule overrides an allow rule of equal specificity
        (both end with /**), which would block the agent's own workspace.
        """
        profile = manager._render_profile("s1", "/workspace")
        sessions_root = str(workspace_root / "sessions")
        # No explicit deny on sessions_root — implicit deny is sufficient
        assert f"deny {sessions_root}" not in profile
        # Only the session's own workspace is allowed
        assert "/workspace/** rwkl" in profile

    def test_profile_name_in_output(self, manager):
        profile = manager._render_profile("test_session", "/workspace")
        assert "jaato-ws-test_session" in profile

    def test_allows_writing_attr_current_for_restore(self, manager):
        """Regression: restore-to-unconfined on context-manager exit
        writes to ``/proc/self/task/<tid>/attr/current``.  If the profile
        doesn't grant write access to that file, the kernel denies the
        file-write and the thread stays trapped in the enforce-mode
        profile — even though ``change_profile -> unconfined`` authorizes
        the semantic transition.  Trapped workers leak across sessions
        and cause EACCES on subsequent reads of ``~/.jaato/*_auth.json``
        and any external sandbox-added paths."""
        profile = manager._render_profile("s1", "/workspace")
        # Per-thread variant (used by apparmor_confine since it keys
        # on threading.get_native_id())
        assert "/proc/self/task/*/attr/current w" in profile
        # Process-level variant — harmless to include and covers
        # code paths that might write to the process-level attr file.
        assert "/proc/self/attr/current" in profile
        # The semantic capability rule must still be present (file-
        # write alone doesn't authorize the profile transition).
        assert "change_profile -> unconfined" in profile

    def test_template_version_bumped(self, manager):
        """Template changes affecting confinement correctness (like the
        attr/current write rule, or new allow rules such as
        ~/.jaato/services/) must bump _TEMPLATE_VERSION so
        ``apparmor_parser`` recompiles from source instead of reusing a
        stale cached binary."""
        assert manager._TEMPLATE_VERSION >= 4
        profile = manager._render_profile("s1", "/workspace")
        assert f"jaato-apparmor-template-version: {manager._TEMPLATE_VERSION}" in profile

    def test_allows_reading_user_tier_services(self, manager):
        """Regression: SchemaStore's tiered lookup reads
        ``~/.jaato/services/`` as a user-tier fallback when the
        workspace tier doesn't have the service.  Confined WS sessions
        need AppArmor read access to that path, otherwise tiered lookup
        is invisible to any model call coming from a confined tool."""
        profile = manager._render_profile("s1", "/workspace")
        assert "@{HOME}/.jaato/services/" in profile
        assert "@{HOME}/.jaato/services/**" in profile


class TestMakeConfineContext:
    def test_returns_callable(self):
        from server.apparmor import make_confine_context
        ctx_factory = make_confine_context("jaato-ws-test")
        assert callable(ctx_factory)
        # Calling it returns a context manager
        ctx = ctx_factory()
        assert hasattr(ctx, "__enter__")
        assert hasattr(ctx, "__exit__")

    def test_confine_unavailable_profile_no_raise(self):
        """apparmor_confine() degrades gracefully when the profile is missing."""
        from server.apparmor import apparmor_confine
        # Use a profile name that doesn't exist — should not raise
        with apparmor_confine("nonexistent-profile-xyz"):
            pass  # body runs unconfined


class TestProvisionProfile:
    def test_writes_and_loads_profile(self, manager, profile_dir):
        manager._available = True
        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = manager.provision_profile("s1", "/workspace/sessions/s1")

        assert result is True
        profile_path = profile_dir / "jaato-ws-s1"
        assert profile_path.exists()
        content = profile_path.read_text()
        assert "jaato-ws-s1" in content

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "sudo"
        assert cmd[1] == "apparmor_parser"

    def test_returns_false_when_unavailable(self, manager):
        manager._available = False
        assert manager.provision_profile("s1", "/workspace") is False

    def test_cleans_up_on_parser_failure(self, manager, profile_dir):
        manager._available = True
        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="parse error")
            result = manager.provision_profile("s1", "/workspace")

        assert result is False
        assert not (profile_dir / "jaato-ws-s1").exists()


class TestTeardownProfile:
    def test_unloads_and_removes_profile(self, manager, profile_dir):
        manager._available = True
        # Create a profile file
        profile_file = profile_dir / "jaato-ws-s1"
        profile_file.write_text("# profile")

        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = manager.teardown_profile("s1")

        assert result is True
        assert not profile_file.exists()

    def test_returns_true_if_already_gone(self, manager):
        manager._available = True
        assert manager.teardown_profile("nonexistent") is True

    def test_returns_false_when_unavailable(self, manager):
        manager._available = False
        assert manager.teardown_profile("s1") is False


class TestProfileTemplateIncludesRefsDir:
    """The base profile must reference the per-session refs.d directory
    via ``include if exists`` so add_reference_fragment() can splice
    fragments without ever editing the base file again."""

    def test_render_emits_include_directive(self, manager):
        rendered = manager._render_profile("sess123", "/workspace")
        # The directive should reference the per-session refs dir under
        # the configured profile_dir.
        refs_glob = f"{manager._refs_dir('sess123')}/*"
        assert "include if exists" in rendered
        assert refs_glob in rendered

    def test_template_version_header_present(self, manager):
        # The rendered profile must always carry a
        # ``jaato-apparmor-template-version: <N>`` header.  The number
        # is whatever ``AppArmorManager._TEMPLATE_VERSION`` currently
        # is — bumping the version is what invalidates apparmor_parser's
        # cache for confined sessions, so the header MUST match the
        # source-of-truth constant exactly (no drift between the
        # constant and the rendered output).
        rendered = manager._render_profile("sess123", "/workspace")
        expected = f"jaato-apparmor-template-version: {manager._TEMPLATE_VERSION}"
        assert expected in rendered


class TestPathValidation:
    """Reference paths must be encodable as bare AppArmor rules; the
    validator catches the cases that would otherwise silently mean
    something different from what the path says."""

    def test_relative_path_rejected(self, manager):
        err = manager._validate_path_for_fragment("relative/path")
        assert err and "must be absolute" in err

    def test_empty_path_rejected(self, manager):
        err = manager._validate_path_for_fragment("")
        assert err is not None

    def test_glob_metacharacters_rejected(self, manager):
        for ch in "[]{}*?\\":
            err = manager._validate_path_for_fragment(f"/some/{ch}/path")
            assert err and "glob metacharacter" in err, (
                f"failed to reject {ch!r}"
            )

    def test_newline_rejected(self, manager):
        err = manager._validate_path_for_fragment("/some/path\nevil rule")
        assert err and ("newline" in err or "CR" in err)

    def test_normal_path_accepted(self, manager):
        assert manager._validate_path_for_fragment("/home/user/docs") is None

    def test_path_with_spaces_accepted(self, manager):
        # Spaces are not glob metachars; the fragment writer wraps the
        # path in double quotes so the parser handles them.
        assert manager._validate_path_for_fragment("/Users/me/My Docs") is None


class TestSafeFragmentFilename:
    """Fragment files share a directory; arbitrary ref_id strings must
    not produce filenames that escape the dir or collide cross-session."""

    def test_alphanumeric_passes_through(self, manager):
        assert manager._safe_fragment_filename("ref-001_v2.json") == "ref-001_v2.json"

    def test_path_separator_collapsed(self, manager):
        assert "/" not in manager._safe_fragment_filename("foo/bar")
        assert manager._safe_fragment_filename("foo/bar") == "foo_bar"

    def test_empty_id_falls_back(self, manager):
        # Empty string would otherwise produce a fragment file named ""
        # which is not a valid filename on most filesystems.
        assert manager._safe_fragment_filename("") == "ref"

    def test_unicode_collapsed(self, manager):
        # Non-ASCII characters are sanitized to keep the filename
        # portable across filesystems.
        assert manager._safe_fragment_filename("café") == "caf_"


class TestFragmentContent:
    """The content the fragment writer emits must be syntactically
    valid AppArmor and grant the right permissions."""

    def test_file_emits_single_readonly_rule(self, manager, tmp_path):
        target = tmp_path / "doc.md"
        target.write_text("hello")
        body = manager._fragment_content(str(target))
        # File rule: just the path with `r,`.
        assert f'"{target}" r,' in body
        # Should NOT include a directory glob for a file.
        assert "**" not in body

    def test_directory_emits_recursive_rules(self, manager, tmp_path):
        target = tmp_path / "docs"
        target.mkdir()
        body = manager._fragment_content(str(target))
        # Directory rule: trailing-slash for the dir itself plus **
        # for descendants — matches the workspace pattern at the top
        # of the base template.
        assert f'"{target}/"   r,' in body
        assert f'"{target}/**" r,' in body


class TestAddRemoveReferenceFragment:
    """End-to-end fragment lifecycle, with the parser invocation
    mocked.  Exercises the threading lock, atomic write, rollback on
    parser failure, and the no-op-when-unavailable contract."""

    def test_unavailable_returns_true_without_writing(self, manager, profile_dir):
        manager._available = False
        ok = manager.add_reference_fragment("s1", "ref1", "/some/path")
        assert ok is True
        # Nothing should have been written when AppArmor is off — the
        # references plugin treats this as "no kernel layer to mutate".
        refs_dir = manager._refs_dir("s1")
        assert not refs_dir.exists()

    def test_add_writes_fragment_and_reloads(self, manager, profile_dir, tmp_path):
        manager._available = True
        # Create a base profile so add_reference_fragment finds it.
        (profile_dir / "jaato-ws-s1").write_text("# base profile")

        target = tmp_path / "doc.md"
        target.write_text("hello")

        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            ok = manager.add_reference_fragment("s1", "ref-A", str(target))

        assert ok is True
        fragment_path = manager._refs_dir("s1") / "ref-A"
        assert fragment_path.exists()
        # Content should encode the path as a single readonly rule.
        assert f'"{target}" r,' in fragment_path.read_text()
        # apparmor_parser was invoked exactly once with -r.
        assert mock_run.call_count == 1
        cmd = mock_run.call_args.args[0]
        assert "apparmor_parser" in cmd[1] or cmd[1].endswith("apparmor_parser")
        assert "-r" in cmd

    def test_add_rejects_glob_in_path_without_writing(self, manager, profile_dir):
        manager._available = True
        (profile_dir / "jaato-ws-s1").write_text("# base profile")

        with patch("server.apparmor.subprocess.run") as mock_run:
            ok = manager.add_reference_fragment("s1", "ref-glob", "/path/with/*/glob")

        assert ok is False
        # Nothing reaches the parser when validation rejects the path.
        mock_run.assert_not_called()
        assert not (manager._refs_dir("s1") / "ref-glob").exists()

    def test_add_rolls_back_fragment_when_reload_fails(self, manager, profile_dir, tmp_path):
        manager._available = True
        (profile_dir / "jaato-ws-s1").write_text("# base profile")
        target = tmp_path / "doc.md"
        target.write_text("hello")

        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="parse error")
            ok = manager.add_reference_fragment("s1", "ref-bad", str(target))

        assert ok is False
        # The bad fragment must be unlinked so the next reload doesn't
        # keep failing on it.
        assert not (manager._refs_dir("s1") / "ref-bad").exists()

    def test_add_fails_when_base_profile_missing(self, manager, profile_dir, tmp_path):
        # Without a base profile to reload, add can't work — returning
        # True here would silently leave the kernel without the rule.
        manager._available = True
        target = tmp_path / "doc.md"
        target.write_text("x")

        with patch("server.apparmor.subprocess.run") as mock_run:
            ok = manager.add_reference_fragment("s1", "ref", str(target))

        assert ok is False
        mock_run.assert_not_called()

    def test_remove_deletes_fragment_and_reloads(self, manager, profile_dir, tmp_path):
        manager._available = True
        (profile_dir / "jaato-ws-s1").write_text("# base profile")
        # Pre-create a fragment as if add_reference_fragment had run.
        refs_dir = manager._refs_dir("s1")
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "ref-A").write_text('"/path" r,\n')

        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            ok = manager.remove_reference_fragment("s1", "ref-A")

        assert ok is True
        assert not (refs_dir / "ref-A").exists()
        assert mock_run.call_count == 1

    def test_remove_is_idempotent(self, manager, profile_dir):
        manager._available = True
        with patch("server.apparmor.subprocess.run") as mock_run:
            ok = manager.remove_reference_fragment("s1", "never-existed")
        assert ok is True
        # No reload when there was nothing to remove.
        mock_run.assert_not_called()

    def test_teardown_profile_clears_refs_dir(self, manager, profile_dir):
        manager._available = True
        (profile_dir / "jaato-ws-s1").write_text("# base profile")
        refs_dir = manager._refs_dir("s1")
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "ref-A").write_text('"/p" r,\n')
        (refs_dir / "ref-B").write_text('"/q" r,\n')

        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            manager.teardown_profile("s1")

        # Both the base profile file and the entire refs dir should be
        # gone — leaking either across session_id reuse would resurrect
        # stale rules in the next session.
        assert not (profile_dir / "jaato-ws-s1").exists()
        assert not refs_dir.exists()

    def test_teardown_clears_session_lock(self, manager, profile_dir):
        manager._available = True
        (profile_dir / "jaato-ws-s1").write_text("# base profile")
        # Touch the lock dict by acquiring once.
        _ = manager._session_lock("s1")
        assert "s1" in manager._session_locks

        with patch("server.apparmor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            manager.teardown_profile("s1")

        assert "s1" not in manager._session_locks


class TestReferenceAuthorizer:
    """The thin handle handed across the server→shared boundary so
    plugins can mutate the kernel profile without importing
    server.apparmor."""

    def test_authorize_delegates_to_manager(self, manager):
        from server.apparmor import ReferenceAuthorizer
        authorizer = ReferenceAuthorizer(manager, "s1")

        with patch.object(manager, "add_reference_fragment", return_value=True) as add:
            ok = authorizer.authorize("ref-A", "/some/path")

        assert ok is True
        add.assert_called_once_with("s1", "ref-A", "/some/path")

    def test_deauthorize_delegates_to_manager(self, manager):
        from server.apparmor import ReferenceAuthorizer
        authorizer = ReferenceAuthorizer(manager, "s1")

        with patch.object(manager, "remove_reference_fragment", return_value=True) as rm:
            ok = authorizer.deauthorize("ref-A")

        assert ok is True
        rm.assert_called_once_with("s1", "ref-A")
