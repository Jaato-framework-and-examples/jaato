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

    def test_workspace_dotjaato_narrow_writes_denied(self, manager):
        """Server 0.6.55+ (template v13): broad ``.jaato/** w,l,k`` deny
        replaced with narrow per-subpath denies on user-authored config
        only.  Tenant-runtime subpaths (sessions/, logs/, cache/, etc.)
        are NOT denied — they fall through to the broad workspace rwkl.

        Pre-v13 (v12) the broad deny was empirically shown to win over
        more-specific allow carve-outs on AppArmor 4.0 — the carve-out
        approach didn't work as designed.  v13 sidesteps the deny-vs-
        allow specificity conflict by only denying paths we genuinely
        want denied.

        Combined ``wlk`` permission flags: w = write integrity, l =
        link bypass via hardlink, k = lock blocking daemon reads.
        """
        profile = manager._render_profile("s1", "/workspace")
        # Workspace rw stays open
        assert "/workspace/   rw" in profile
        assert "/workspace/** rwkl" in profile
        # Narrow user-authored config denies (each gets w,l,k combined):
        assert "audit deny /workspace/.jaato/agents/**" in profile
        assert "audit deny /workspace/.jaato/profiles/**" in profile
        assert "audit deny /workspace/.jaato/prompts/**" in profile
        assert "audit deny /workspace/.jaato/scripts/**" in profile
        assert "audit deny /workspace/.jaato/services/*/" in profile
        assert "audit deny /workspace/.jaato/reactors.json" in profile
        assert "audit deny /workspace/.jaato/completion_schemas/**" in profile
        assert "audit deny /workspace/.jaato/spawn_schemas/**" in profile
        assert "audit deny /workspace/.jaato/instructions/**" in profile
        assert "audit deny /workspace/.jaato/references/**" in profile
        # NO broad deny anymore — that's what was breaking carve-outs.
        assert "audit deny /workspace/.jaato/**  w" not in profile

    def test_workspace_dotjaato_reads_allowed_in_base(self, manager):
        """In the BASE profile (not the tool_hat sub-profile),
        read access to ``<workspace>/.jaato/`` stays open — agents,
        reactors, and session-init need to read profiles, prompts,
        schemas, agent .md files, etc.

        The user-authored config write-denies are write-only (w,l,k);
        reads flow through the surrounding ``rwkl`` allow rule.  The
        sub-profile is what adds read-denies (verified separately).
        """
        profile = manager._render_profile("s1", "/workspace")
        # Find the BASE profile body (everything before the tool_hat sub-profile).
        base_body = profile.split("profile tool_hat")[0]
        # Base must NOT deny reads on user-authored config — reactor
        # dispatch, prefetch, and session-init all need them.
        assert "audit deny /workspace/.jaato/agents/**             r" not in base_body
        assert "audit deny /workspace/.jaato/profiles/**           r" not in base_body
        # Reads flow through workspace rwkl.
        assert "/workspace/** rwkl" in base_body

    def test_template_version_bumped_to_13(self, manager):
        """v13 template ships the narrow per-subpath denies + tool_hat
        sub-profile.  apparmor_parser recompiles from source on version
        bump rather than reusing a stale v12 cached binary.
        """
        assert manager._TEMPLATE_VERSION >= 13
        profile = manager._render_profile("s1", "/workspace")
        assert (
            f"jaato-apparmor-template-version: {manager._TEMPLATE_VERSION}"
            in profile
        )

    def test_template_version_bumped_to_14(self, manager):
        """v14 template ships the ``profile child`` sub-profile that
        closes the verified escape vector at apparmor.py:413-449.
        Subprocesses transition into ``//child`` between fork() and
        exec() via preexec_fn — Phase 5 §5.10."""
        assert manager._TEMPLATE_VERSION >= 14
        profile = manager._render_profile("s1", "/workspace")
        assert (
            f"jaato-apparmor-template-version: {manager._TEMPLATE_VERSION}"
            in profile
        )

    def test_tenant_runtime_paths_are_writable(self, manager):
        """Tenant-runtime subpaths under ``.jaato/`` (sessions/, logs/,
        cache/, vision/, services/_discovered/, memory/, todos/,
        waypoints.json, *_auth.json) are NOT under any deny rule in
        v13.  They flow through the broad workspace rwkl and writes
        succeed regardless of confinement context.

        This is the key correctness property that pre-v13 broke:
        confined reactor-spawned session-journal saves at
        ``.jaato/sessions/<id>.json.tmp`` were EACCES because the
        broad v12 deny dominated the carve-out allow.
        """
        profile = manager._render_profile("s1", "/workspace")
        # No deny for tenant-runtime paths.
        for tenant_path in (
            "/workspace/.jaato/sessions/",
            "/workspace/.jaato/logs/",
            "/workspace/.jaato/cache/",
            "/workspace/.jaato/vision/",
            "/workspace/.jaato/services/_discovered/",
            "/workspace/.jaato/memory/",
            "/workspace/.jaato/todos/",
            "/workspace/.jaato/waypoints.json",
        ):
            assert f"audit deny {tenant_path}" not in profile, (
                f"tenant-runtime path {tenant_path} must NOT be under deny"
            )

    def test_tool_hat_subprofile_present(self, manager):
        """v13 introduces the ``profile tool_hat { ... }`` sub-profile.
        Tool execution enters it via ``change_profile -> jaato-ws-X//tool_hat``;
        prefetch / reactor dispatch / session-init stay in BASE.
        """
        profile = manager._render_profile("s1", "/workspace")
        assert "profile tool_hat" in profile
        # Sub-profile redeclares workspace allow (sub-profiles don't
        # inherit base rules).
        tool_hat_body = profile.split("profile tool_hat")[1]
        assert "/workspace/   rw," in tool_hat_body
        assert "/workspace/** rwkl," in tool_hat_body

    def test_tool_hat_adds_read_denies_on_user_authored_config(self, manager):
        """The whole point of the sub-profile: tool execution can't
        read other agents' personas, profile JSON, prompts, schemas,
        instructions, scripts, or reactors.json.  Information-isolation
        between agents in a cascade.

        Each user-authored config subpath gets an explicit ``r`` deny
        in the sub-profile body (in addition to the integrity wlk
        denies that mirror base).
        """
        import re
        profile = manager._render_profile("s1", "/workspace")
        tool_hat_body = profile.split("profile tool_hat")[1]
        for path in (
            "/workspace/.jaato/agents/**",
            "/workspace/.jaato/profiles/**",
            "/workspace/.jaato/prompts/**",
            "/workspace/.jaato/scripts/**",
            "/workspace/.jaato/completion_schemas/**",
            "/workspace/.jaato/spawn_schemas/**",
            "/workspace/.jaato/instructions/**",
            "/workspace/.jaato/reactors.json",
        ):
            # Match ``audit deny <path>  ... r,`` with arbitrary
            # whitespace before the permission flag.
            pattern = (
                r"audit\s+deny\s+"
                + re.escape(path)
                + r"\s+r,"
            )
            assert re.search(pattern, tool_hat_body), (
                f"hat must deny reads on user-authored config: {path}"
            )

    def test_make_tool_confine_context_yields_subprofile_path(self):
        """The tool-confinement factory produces a callable that
        confines to the per-session sub-profile (``parent//tool_hat``).
        """
        from server.apparmor import make_tool_confine_context
        factory = make_tool_confine_context("jaato-ws-test_session")
        assert callable(factory)
        # Calling it returns a context manager (the actual confinement
        # write would be ``changeprofile jaato-ws-test_session//tool_hat``).
        ctx = factory()
        assert hasattr(ctx, "__enter__")
        assert hasattr(ctx, "__exit__")

    def test_child_subprofile_present(self, manager):
        """Phase 5 §5.10a: v14 introduces the ``profile child {...}``
        sub-profile.  Subprocesses transition into it via
        ``change_profile -> jaato-ws-X//child`` between fork() and
        exec() so the dangerous escape rules (writable
        attr/current + change_profile -> unconfined) don't apply to
        model-controlled subprocess content.

        Body mirrors tool_hat's workspace + venv reads."""
        profile = manager._render_profile("s1", "/workspace")
        assert "profile child" in profile
        child_body = profile.split("profile child")[1]
        # Mirrors tool_hat workspace allow.
        assert "/workspace/   rw," in child_body
        assert "/workspace/** rwkl," in child_body

    def test_child_subprofile_drops_escape_rules(self, manager):
        """The whole point of //child: the three escape-vector rules
        from base + tool_hat are absent so a process in //child
        cannot escape to unconfined.  This is the kernel-enforced
        primitive that closes the Phase 4 known-escape-vector class
        for subprocess spawn."""
        import re
        profile = manager._render_profile("s1", "/workspace")
        child_body = profile.split("profile child")[1]
        # Match the rule SHAPE (apparmor_parser ignores `#` comments
        # so any non-comment line ending with `,` is a real rule).
        # Test must distinguish rule lines from comment mentions —
        # the audit's "drops these rules" comment block lives in the
        # body and contains the rule strings as documentation.
        def _has_uncommented_rule(body: str, rule_pattern: str) -> bool:
            for line in body.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if re.search(rule_pattern, stripped):
                    return True
            return False

        assert not _has_uncommented_rule(
            child_body, r"^change_profile\s+->\s+unconfined,$",
        ), "child must not authorize change_profile -> unconfined"
        # The writable proc rules — both base + task variants — are
        # absent.  Subprocess cannot effect ANY change_profile (even
        # to another sub-profile) because the file write itself is
        # denied.
        assert not _has_uncommented_rule(
            child_body, r"^/proc/self/attr/current\s+w,$",
        ), "child must not authorize writes to /proc/self/attr/current"
        assert not _has_uncommented_rule(
            child_body, r"^/proc/self/task/\*/attr/current\s+w,$",
        ), "child must not authorize writes to /proc/self/task/*/attr/current"

    def test_child_subprofile_keeps_tool_hat_read_denies(self, manager):
        """Per the user's design pick (option A — tool_hat body minus
        the three escape rules), //child preserves the
        information-isolation read-denies on user-authored config.
        A subprocess can't read other agents' personas/profiles any
        more than the in-process tool_hat path can."""
        import re
        profile = manager._render_profile("s1", "/workspace")
        child_body = profile.split("profile child")[1]
        for path in (
            "/workspace/.jaato/agents/**",
            "/workspace/.jaato/profiles/**",
            "/workspace/.jaato/prompts/**",
            "/workspace/.jaato/scripts/**",
            "/workspace/.jaato/completion_schemas/**",
            "/workspace/.jaato/spawn_schemas/**",
            "/workspace/.jaato/instructions/**",
            "/workspace/.jaato/reactors.json",
        ):
            pattern = (
                r"audit\s+deny\s+"
                + re.escape(path)
                + r"\s+r,"
            )
            assert re.search(pattern, child_body), (
                f"child must deny reads on user-authored config: {path}"
            )


class TestMakeChildTransitionCallback:
    """Phase 5 §5.10b — the preexec_fn-style transition callback
    that subprocess-spawning plugins use in ``preexec_fn`` to enter
    the per-session ``//child`` sub-profile between fork() and exec()."""

    def test_returns_zero_arg_callable(self):
        from server.apparmor import make_child_transition_callback
        cb = make_child_transition_callback("jaato-ws-test")
        assert callable(cb)

    def test_writes_changeprofile_target_to_attr_current(self, tmp_path, monkeypatch):
        """The callback writes ``changeprofile <profile>//child`` to
        /proc/self/attr/current.  We swap in a tmp file to capture
        the write without touching the real kernel-pseudofile."""
        from server import apparmor as ap

        captured = tmp_path / "attr_current"
        captured.write_text("")  # ensure file exists

        # Replace os.open in the apparmor module so our callback
        # targets the fake path instead of /proc/self/attr/current.
        real_open = ap.os.open

        def _fake_open(path, *args, **kwargs):
            if path == "/proc/self/attr/current":
                return real_open(str(captured), *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(ap.os, "open", _fake_open)

        cb = ap.make_child_transition_callback("jaato-ws-myses")
        cb()

        assert captured.read_bytes() == b"changeprofile jaato-ws-myses//child"

    def test_session_profile_name_used_verbatim(self, tmp_path, monkeypatch):
        """The session profile name passed at factory time appears
        verbatim in the write payload.  Subagent isolated-runner
        callers can pass a sub-profile name (e.g.
        ``jaato-ws-parent//subagent``) and get
        ``jaato-ws-parent//subagent//child`` — three-level nesting
        is preserved (Phase 5 §5.10e preview)."""
        from server import apparmor as ap

        captured = tmp_path / "attr_current"
        captured.write_text("")
        real_open = ap.os.open

        def _fake_open(path, *args, **kwargs):
            if path == "/proc/self/attr/current":
                return real_open(str(captured), *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(ap.os, "open", _fake_open)

        cb = ap.make_child_transition_callback(
            "jaato-ws-parent//subagent",
        )
        cb()

        assert captured.read_bytes() == (
            b"changeprofile jaato-ws-parent//subagent//child"
        )

    def test_fails_closed_when_write_path_missing(self, tmp_path):
        """If /proc/self/attr/current is missing or unwritable,
        the callback raises (subprocess.Popen surfaces this as a
        spawn failure — the new process never starts).  Fail-closed
        is the correct posture: a silent transition failure would
        leave the child in the parent profile with the escape
        rules intact."""
        import os as _os
        from server.apparmor import make_child_transition_callback

        # Force os.open to fail with ENOENT.
        original_open = _os.open

        def _broken_open(path, *args, **kwargs):
            if path == "/proc/self/attr/current":
                raise FileNotFoundError(2, "no such file", path)
            return original_open(path, *args, **kwargs)

        try:
            _os.open = _broken_open  # type: ignore[assignment]
            cb = make_child_transition_callback("jaato-ws-X")
            try:
                cb()
            except FileNotFoundError:
                pass
            else:
                raise AssertionError(
                    "expected FileNotFoundError from broken /proc write"
                )
        finally:
            _os.open = original_open  # type: ignore[assignment]


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


class TestApparmorConfineDefensiveReset:
    """Tests for the defensive ``changeprofile unconfined`` write that
    apparmor_confine performs on entry, recovering from a prior
    session's stuck-confinement state.  See apparmor.py docstring for
    the full bug rationale.
    """

    def test_entry_writes_unconfined_before_target_profile(self):
        """On entry, apparmor_confine MUST write 'unconfined' before
        writing the target profile name — the defensive reset that
        breaks cross-session state poisoning."""
        from server.apparmor import apparmor_confine
        from unittest.mock import mock_open, patch

        m = mock_open()
        with patch("builtins.open", m):
            with apparmor_confine("jaato-ws-test"):
                pass
        # write() called twice on entry (defensive unconfined + target)
        # plus once on exit (restore unconfined).  Order matters.
        write_calls = m().write.call_args_list
        assert len(write_calls) >= 3, (
            f"expected ≥3 attr/current writes (defensive-reset, entry, exit), "
            f"got {len(write_calls)}: {[c.args[0] for c in write_calls]}"
        )
        # First write is defensive reset to unconfined
        assert write_calls[0].args[0] == "changeprofile unconfined", (
            f"first write must be the defensive unconfined reset, "
            f"got {write_calls[0].args[0]!r}"
        )
        # Second write is the target profile entry
        assert write_calls[1].args[0] == "changeprofile jaato-ws-test", (
            f"second write must be the target profile entry, "
            f"got {write_calls[1].args[0]!r}"
        )
        # Third (final) write is the exit restoration
        assert write_calls[-1].args[0] == "changeprofile unconfined"

    def test_defensive_reset_failure_does_not_block_entry(self):
        """If the defensive unconfined write fails (already-unconfined
        edge case may PermissionError), the entry write must still be
        attempted — failures here are non-fatal."""
        from server.apparmor import apparmor_confine
        from unittest.mock import patch

        # First open() call (defensive reset) raises; subsequent calls succeed.
        call_count = [0]

        def fake_open(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise PermissionError("simulated kernel transient")
            from unittest.mock import mock_open
            return mock_open()()

        with patch("builtins.open", side_effect=fake_open):
            with apparmor_confine("jaato-ws-test"):
                pass
        # Three open() calls total: defensive (failed), entry, exit.
        assert call_count[0] >= 2, (
            f"entry write must be attempted even when defensive reset failed; "
            f"got {call_count[0]} open() calls"
        )

    def test_exit_restoration_failure_logged_at_error(self, caplog):
        """When exit-time restoration fails, the message MUST be logged
        at ERROR level (not warning) so it surfaces — the next entry's
        defensive reset will recover the thread, but the underlying
        cause (kernel state, missing rule, etc.) deserves attention."""
        from server.apparmor import apparmor_confine
        from unittest.mock import patch

        # Defensive reset + entry succeed; exit raises.
        call_count = [0]

        def fake_open(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 3:  # exit-time restoration write
                raise PermissionError("simulated stuck thread")
            from unittest.mock import mock_open
            return mock_open()()

        with caplog.at_level("ERROR", logger="server.apparmor"):
            with patch("builtins.open", side_effect=fake_open):
                with apparmor_confine("jaato-ws-test"):
                    pass
        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert any(
            "could not restore unconfined" in r.getMessage().lower()
            for r in error_records
        ), (
            f"expected ERROR-level log naming the restoration failure; "
            f"got: {[r.getMessage() for r in caplog.records]}"
        )


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
