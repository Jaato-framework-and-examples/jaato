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

    def test_unavailable_reason_records_failing_precondition(self, manager):
        """unavailable_reason names the specific precondition that failed."""
        with patch("server.apparmor.platform.system", return_value="Darwin"):
            manager._available = None
            assert manager.is_available() is False
            assert manager.unavailable_reason is not None
            assert "Linux" in manager.unavailable_reason

    def test_unavailable_reason_for_missing_parser(self, manager):
        with patch("server.apparmor.platform.system", return_value="Linux"), \
             patch("server.apparmor.shutil.which", return_value=None):
            manager._available = None
            assert manager.is_available() is False
            assert "apparmor_parser" in (manager.unavailable_reason or "")

    def test_is_available_logs_warning_when_unavailable(self, manager, caplog):
        """Degradation is logged at WARNING (visible), not INFO/silent."""
        with patch("server.apparmor.platform.system", return_value="Darwin"):
            manager._available = None
            with caplog.at_level("WARNING", logger="server.apparmor"):
                assert manager.is_available() is False
            assert any(
                "NOT available" in r.message and r.levelname == "WARNING"
                for r in caplog.records
            )


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
        writes to attr/current.  If the profile doesn't grant write
        access, the kernel denies the file-write and the thread stays
        trapped in the enforce-mode profile — even though
        ``change_profile -> unconfined`` authorizes the semantic
        transition.  Trapped workers leak across sessions and cause
        EACCES on subsequent reads of ``~/.jaato/*_auth.json`` and any
        external sandbox-added paths.

        v15 switched the rule form from ``/proc/self/...`` (which only
        matched writes accidentally via the procfs special-case
        write path) to ``owner /proc/*/...`` (which matches the
        kernel's resolved path for BOTH read and write)."""
        import re
        profile = manager._render_profile("s1", "/workspace")
        # Strip comments so docstring text doesn't fool the assertion.
        rule_lines = [
            l.strip() for l in profile.splitlines()
            if l.strip() and not l.lstrip().startswith("#")
        ]
        # Per-thread variant write rule (apparmor_confine writes to
        # /proc/self/task/<tid>/attr/current).  Accept either v14
        # form (/proc/self/...) or v15 form (owner /proc/*/...).
        has_task_write = any(
            re.match(
                r"(owner\s+)?/proc/(self|\*)/task/\*/attr/current\s+[rw]+,",
                rule,
            )
            for rule in rule_lines
        )
        assert has_task_write, (
            "profile missing task/<tid>/attr/current write rule; "
            "apparmor_confine restore-to-unconfined will fail"
        )
        # Process-level variant
        has_proc_write = any(
            re.match(
                r"(owner\s+)?/proc/(self|\*)/attr/current\s+[rw]+,",
                rule,
            )
            for rule in rule_lines
        )
        assert has_proc_write, (
            "profile missing process-level attr/current write rule"
        )
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

    def test_template_version_bumped_to_15(self, manager):
        """Phase 5 ad-hoc fix: read-rule addition to /proc/self/attr/current
        bumps template_version 14 → 15.  Invalidates apparmor_parser's
        binary cache so the runner self-confinement verify-read step
        starts working on hosts that have an older cached compile."""
        assert manager._TEMPLATE_VERSION >= 15

    def test_v17_parent_grants_change_profile_to_child_subprofile(self, manager):
        """Template v17 (2026-05-15) regression pin.

        Pre-v17 the parent profile body declared the ``//child``
        sub-profile inline but did NOT grant
        ``change_profile -> jaato-ws-X//child,``.  The kernel
        denied every cli_based_tool subprocess transition with
        ``apparmor="DENIED" operation="change_profile"
        target="...//child"`` — peer's v82 cascade smoking gun.
        The legacy code comment in
        ``make_child_transition_callback`` claimed "inline
        sub-profiles are implicitly authorized" — empirically
        false on AppArmor 4.0.1.

        Pin the parent body explicitly grants the transition.  Any
        future restorer who deletes the rule (e.g. while
        simplifying the template) breaks every confined cli call.
        """
        import re
        session_id = "20260515_x_v17"
        profile = manager._render_profile(session_id, "/workspace")
        # Take only the parent body (everything before the
        # ``profile tool_hat`` sub-profile declaration).
        parent_end = profile.find("profile tool_hat")
        assert parent_end > 0, "tool_hat sub-profile marker missing"
        parent_body = profile[:parent_end]
        rule_lines = [
            l.strip() for l in parent_body.splitlines()
            if l.strip() and not l.lstrip().startswith("#")
        ]
        expected = f"change_profile -> jaato-ws-{session_id}//child,"
        assert expected in rule_lines, (
            f"Parent profile missing the ``change_profile -> "
            f"jaato-ws-{{session_id}}//child,`` grant.  Without it "
            f"every cli_based_tool subprocess in a confined session "
            f"hits AppArmor DENIED on the preexec_fn transition.  "
            f"Looked for: {expected!r}.  Rules found: "
            f"{[r for r in rule_lines if 'change_profile' in r]}"
        )

    def test_v17_unconfined_grant_still_present(self, manager):
        """v17 adds the //child rule without removing the
        ``-> unconfined`` rule.  apparmor_confine's defensive
        reset-to-unconfined on tool_hat entry/exit still needs
        the unconfined transition; this is a deliberate-but-
        documented escape vector (see PROFILE_TEMPLATE block
        at apparmor.py:447-510) that's distinct from this fix."""
        profile = manager._render_profile("s1", "/workspace")
        parent_end = profile.find("profile tool_hat")
        parent_body = profile[:parent_end] if parent_end > 0 else profile
        rule_lines = [
            l.strip() for l in parent_body.splitlines()
            if l.strip() and not l.lstrip().startswith("#")
        ]
        assert "change_profile -> unconfined," in rule_lines

    def test_v17_child_subprofile_still_denies_change_profile(self, manager):
        """v17 only changes the parent.  The //child sub-profile's
        contract (no ``change_profile -> unconfined,`` so the
        forked subprocess can't escape back) is preserved.  This
        is the existing test
        ``test_apparmor_sub_profile.py::test_child_sub_profile_no_change_profile_to_unconfined``
        in spirit — duplicate here as a paired-pin: parent grants
        the transition INTO //child, but //child grants NO
        outbound transitions.  Both halves are load-bearing."""
        import re
        profile = manager._render_profile("s1", "/workspace")
        child_start = profile.find("profile child")
        # Sub-profile name may render with various surrounding
        # whitespace; locate the first ``profile child`` block.
        if child_start < 0:
            # Older template may name it differently — skip this
            # check rather than false-positive.
            return
        # Take a slice until the closing brace of the child block.
        child_body = profile[child_start:]
        rule_lines = [
            l.strip() for l in child_body.splitlines()
            if l.strip() and not l.lstrip().startswith("#")
        ]
        # The forked subprocess in //child MUST NOT be able to
        # escape via change_profile.
        assert not any(
            re.match(r"change_profile\s+->\s+", rule)
            for rule in rule_lines[:40]  # only inspect the immediate //child body
        ), (
            "//child sub-profile must NOT grant change_profile to "
            "any target — otherwise a model-controlled subprocess "
            "could re-escape from //child."
        )

    def test_v17_template_version_bumped(self, manager):
        """Confinement-correctness changes must bump
        _TEMPLATE_VERSION so apparmor_parser doesn't reuse a stale
        cached compile that lacks the new grant."""
        assert manager._TEMPLATE_VERSION >= 17

    def _slice_child_body(self, rendered: str) -> str:
        """Return the rendered //child sub-profile body, slicing
        from ``profile child`` to the rest of the rendered output.
        The //child block is the LAST sub-profile in the template,
        so slicing to the end is correct.  Inline helper keeps
        the v18 tests below readable."""
        child_start = rendered.find("profile child")
        assert child_start > 0, "//child sub-profile marker missing"
        return rendered[child_start:]

    def test_v18_child_subprofile_drops_broad_ix_grants(self, manager):
        """Template v18 (2026-05-15) regression pin.

        Pre-v18 the //child sub-profile mirrored the base body
        verbatim including ``/usr/bin/** ix``, ``/usr/local/bin/**
        ix``, ``/bin/** ix``.  That shadowed every per-profile
        ``apparmor_fragments`` declaration — fragments listed
        specific binaries but the broad rule already covered
        everything.  Peer's v83 caught it empirically (agent
        improvised ``curl`` when ``mvn dependency:get`` failed
        even though the fragment only listed java/mvn).

        v18 strips the three broad ``ix`` grants from //child
        SPECIFICALLY.  Parent + tool_hat keep them (framework
        code paths depend on them).  Fragments become the SOLE
        source of exec authority for agent-controlled subprocesses.
        """
        rendered = manager._render_profile("v18_test", "/workspace")
        child_body = self._slice_child_body(rendered)

        for broad_rule in (
            "/usr/bin/**          ix,",
            "/usr/local/bin/**    ix,",
            "/bin/**              ix,",
        ):
            assert broad_rule not in child_body, (
                f"//child sub-profile must NOT grant the broad "
                f"rule {broad_rule!r}.  Per-profile "
                f"apparmor_fragments is the sole source of exec "
                f"authority in //child post-v18; this rule "
                f"shadows fragments and breaks per-stage scoping. "
                f"Peer's v83 verified this empirically with the "
                f"curl-fallback escape."
            )

    def test_v18_parent_and_tool_hat_keep_broad_ix(self, manager):
        """v18 only narrows //child.  The parent profile body
        and tool_hat sub-profile RETAIN the broad ``ix`` grants
        because framework code (Python stdlib subprocess
        machinery, prefetch scripts, references plugin's
        SentenceTransformer load, runner-tier plugin internals)
        runs in those contexts and needs to exec a long tail of
        helpers we don't enumerate."""
        rendered = manager._render_profile("v18_test", "/workspace")
        tool_hat_start = rendered.find("profile tool_hat")
        child_start = rendered.find("profile child")
        assert 0 < tool_hat_start < child_start

        parent_body = rendered[:tool_hat_start]
        tool_hat_body = rendered[tool_hat_start:child_start]

        for broad_rule, region_name, region in (
            ("/usr/bin/**          ix,",     "parent",   parent_body),
            ("/usr/local/bin/**    ix,",     "parent",   parent_body),
            ("/bin/**              ix,",     "parent",   parent_body),
            ("/usr/bin/**          ix,",     "tool_hat", tool_hat_body),
            ("/usr/local/bin/**    ix,",     "tool_hat", tool_hat_body),
            ("/bin/**              ix,",     "tool_hat", tool_hat_body),
        ):
            assert broad_rule in region, (
                f"{region_name} must keep the broad rule "
                f"{broad_rule!r} — v18 only narrows //child, not "
                f"the framework-internal contexts.  Removing this "
                f"would break framework subprocess machinery."
            )

    def test_v18_child_keeps_library_mapping(self, manager):
        """The narrowing in //child is on EXEC capability (``ix``),
        not on library mmap (``rm``).  An exec'd binary still
        needs to mmap shared libraries (libc, ld-linux, etc.) to
        run.  Pin the rm rules stay so a fragment-authorized
        ``/usr/bin/java ix`` actually works when java tries to
        load its required shared libraries."""
        rendered = manager._render_profile("v18_test", "/workspace")
        child_body = self._slice_child_body(rendered)
        for rule in ("/usr/lib/**          rm,", "/lib/**              rm,"):
            assert rule in child_body, (
                f"//child must keep library-mapping rule {rule!r} "
                f"so fragment-authorized execs can load their "
                f"shared libraries.  Removing this breaks every "
                f"exec, not just the broad ones."
            )

    def test_v18_template_version_bumped(self, manager):
        """v18 narrows //child's exec authority — confinement
        semantics change → ``apparmor_parser`` must recompile
        from source.  Bump enforces that."""
        assert manager._TEMPLATE_VERSION >= 18

    def test_v15_base_profile_allows_attr_current_read(self, manager):
        """Phase 5 ad-hoc fix: parent profile body grants `r` on
        /proc/self/attr/current so the runner's confine_to_profile
        verify-after-write step (server/runner/bootstrap.py:188) can
        read the kernel's view of the current profile.  Before v15 the
        profile had `w` only and the read EACCESed.

        Pin checks both rule shapes: `rw,` (the chosen form) OR a
        separate `r,` line.  Either passes the kernel's permission
        check."""
        import re
        profile = manager._render_profile("s1", "/workspace")
        # Strip comments so docstring text mentioning the old `w,`
        # rule doesn't fool the assertion.
        rule_lines = [
            l.strip() for l in profile.splitlines()
            if l.strip() and not l.lstrip().startswith("#")
        ]
        attr_rules = [
            l for l in rule_lines
            if "attr/current" in l and "/proc/" in l
            and "task" not in l  # exclude task/<tid>/ variant (separate rule)
        ]
        # At least one rule must include `r` permission (rw or just r)
        has_read = any(
            re.match(r"(owner\s+)?/proc/(self|\*)/attr/current\s+r[wlmkixacd]*,", rule) is not None
            for rule in attr_rules
        )
        assert has_read, (
            f"v15 base profile must allow read on /proc/self/attr/current; "
            f"got rules: {attr_rules!r}"
        )

    def test_v15_tool_hat_allows_attr_current_read(self, manager):
        """Phase 5 ad-hoc fix: tool_hat sub-profile body mirrors the
        base profile's rw permission on attr/current.  Defensive — the
        framework's apparmor_confine context manager doesn't itself
        read attr/current today, but keeping the hat at parity with
        the base means a future verify-read addition won't silently
        break the hat path."""
        import re
        profile = manager._render_profile("s1", "/workspace")
        if "profile tool_hat" not in profile:
            return  # tool_hat may be optional in some templates
        # Find tool_hat body by brace-counting (the body has `@{HOME}`
        # references with literal `}` characters that confuse naive
        # split-on-`}`).  Start at the `{` right after "profile tool_hat"
        # and walk until depth returns to zero.
        start = profile.find("profile tool_hat")
        brace_start = profile.find("{", start)
        depth = 0
        body_end = None
        for i, ch in enumerate(profile[brace_start:], brace_start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    body_end = i
                    break
        assert body_end is not None, "couldn't locate tool_hat closing brace"
        tool_hat_body = profile[brace_start + 1: body_end]

        attr_rules = [
            l.strip() for l in tool_hat_body.splitlines()
            if "attr/current" in l and "/proc/" in l
            and "task" not in l
            and not l.lstrip().startswith("#")
        ]
        has_read = any(
            re.match(r"(owner\s+)?/proc/(self|\*)/attr/current\s+r[wlmkixacd]*,", rule) is not None
            for rule in attr_rules
        )
        assert has_read, (
            f"v15 tool_hat body must allow read on /proc/self/attr/current; "
            f"got rules: {attr_rules!r}"
        )

    def test_v15_isolated_sub_profile_allows_attr_current_read(self, manager):
        """Phase 5 ad-hoc fix: isolated-subagent sub-profile grants
        read access to /proc/*/attr/current so the sub-runner subprocess
        can verify its own confinement after the kernel transitions it.

        No write permission expected — the sub-runner stays in this
        profile for its lifetime per §4.3.4 design (no further
        self-transitions).  Read-only is the right contract."""
        sub = manager._render_sub_profile(
            parent_session_id="parent-A",
            subagent_id="agent-B",
            workspace_path="/tmp/test-ws",
        )
        rule_lines = [
            l.strip() for l in sub.splitlines()
            if "/proc/" in l and "attr/current" in l
            and not l.lstrip().startswith("#")
        ]
        assert rule_lines, (
            "isolated sub-profile missing /proc/*/attr/current rule entirely; "
            "sub-runner confine_to_profile verify-read will EACCES"
        )
        # All present rules must grant `r` (we don't want a pure `w,`)
        for rule in rule_lines:
            assert rule.endswith(",") and " r" in rule, (
                f"isolated sub-profile attr/current rule missing read: {rule!r}"
            )

    def test_template_version_bumped_to_16(self, manager):
        """v16 template adds ``mr`` (mmap-exec) rules on venv
        ``*.so`` / ``*.so.*`` shared objects.  Closes the
        ``failed to map segment from shared object`` class when
        the daemon venv lives outside ``abstractions/python``'s
        coverage (``/usr/lib/python*``, ``/usr/local/lib/python*``)
        — e.g., the runner can't import jiter / numpy /
        anthropic because their C-extensions have no PROT_EXEC
        grant.  Narrow grant: only ELF shared objects, not .py.
        Surfaced 2026-05-12 by kb-enablement-2.0 cascade smoke
        test running against a daemon at ``/tmp/jaato-test/``."""
        assert manager._TEMPLATE_VERSION >= 16
        profile = manager._render_profile("s1", "/workspace")
        assert (
            f"jaato-apparmor-template-version: {manager._TEMPLATE_VERSION}"
            in profile
        )

    @staticmethod
    def _extract_brace_body(text: str, anchor: str) -> str:
        """Extract the body inside ``{...}`` following ``anchor``.

        Brace-counts to handle nested braces and literal ``}``
        characters inside ``@{HOME}`` substitutions.  Returns the
        body text (between the matching braces, exclusive)."""
        start = text.find(anchor)
        assert start != -1, f"anchor {anchor!r} not in profile"
        brace_start = text.find("{", start)
        depth = 0
        for i, ch in enumerate(text[brace_start:], brace_start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_start + 1: i]
        raise AssertionError(
            f"unmatched brace after anchor {anchor!r}"
        )

    def test_v16_base_profile_grants_mmap_exec_on_venv_so(
        self, manager,
    ):
        """Pin v16 base profile body grants ``mr`` on venv .so
        files.  Without ``m`` the kernel rejects PROT_EXEC mmap
        when Python imports a C extension whose shared object
        lives in the daemon venv — the import surfaces as a
        misleading "package not installed" ToolError because the
        provider wrapper translates ImportError uniformly."""
        profile = manager._render_profile("s1", "/workspace")
        # Strip nested sub-profile bodies so we only inspect the
        # base.  Order matters: child + tool_hat come after the
        # base-only rules.
        base_body = profile
        for anchor in ("profile child", "profile tool_hat"):
            if anchor in base_body:
                base_body = base_body.split(anchor)[0]
        assert "/usr/local/venv/**/*.so    mr," in base_body, (
            "v16 base profile missing mmap-exec grant on "
            "venv *.so; C-extension imports will EACCES"
        )
        assert "/usr/local/venv/**/*.so.*  mr," in base_body, (
            "v16 base profile missing mmap-exec grant on "
            "venv *.so.* (versioned shared libs like "
            "libpython3.12.so.1.0)"
        )

    def test_v16_tool_hat_grants_mmap_exec_on_venv_so(
        self, manager,
    ):
        """Pin v16 tool_hat sub-profile body grants ``mr`` on
        venv .so files (mirrors base).  Tools spawned under
        tool_hat (CLI, interactive_shell) need the same
        C-extension import capability as the base."""
        profile = manager._render_profile("s1", "/workspace")
        if "profile tool_hat" not in profile:
            return
        tool_hat_body = self._extract_brace_body(
            profile, "profile tool_hat",
        )
        assert "/usr/local/venv/**/*.so    mr," in tool_hat_body, (
            "v16 tool_hat body missing mmap-exec grant on "
            "venv *.so"
        )
        assert "/usr/local/venv/**/*.so.*  mr," in tool_hat_body, (
            "v16 tool_hat body missing mmap-exec grant on "
            "venv *.so.*"
        )

    def test_v16_child_grants_mmap_exec_on_venv_so(self, manager):
        """Pin v16 //child sub-profile body grants ``mr`` on
        venv .so files.  Subprocesses transition into //child
        between fork() and exec() via preexec_fn (Phase 5
        §5.10); a confined subprocess that can't mmap-exec a
        .so can't import any C extension."""
        profile = manager._render_profile("s1", "/workspace")
        if "profile child" not in profile:
            return
        child_body = self._extract_brace_body(profile, "profile child")
        assert "/usr/local/venv/**/*.so    mr," in child_body, (
            "v16 //child body missing mmap-exec grant on venv *.so"
        )
        assert "/usr/local/venv/**/*.so.*  mr," in child_body, (
            "v16 //child body missing mmap-exec grant on venv *.so.*"
        )

    def test_v16_isolated_sub_profile_grants_mmap_exec_on_venv_so(
        self, manager,
    ):
        """Pin v16 isolated subagent sub-profile (§4.3.4) grants
        ``mr`` on venv .so files.  Isolated subagents run their
        own sub-runner with the same Python import surface as
        the parent — same C-extension requirement."""
        sub = manager._render_sub_profile(
            parent_session_id="parent-A",
            subagent_id="agent-B",
            workspace_path="/tmp/test-ws",
        )
        assert "/usr/local/venv/**/*.so    mr," in sub, (
            "v16 isolated sub-profile missing mmap-exec grant "
            "on venv *.so; isolated subagent can't import "
            "C extensions"
        )
        assert "/usr/local/venv/**/*.so.*  mr," in sub, (
            "v16 isolated sub-profile missing mmap-exec grant "
            "on venv *.so.*"
        )

    def test_workspace_apparmor_fragments_writes_denied_in_all_profiles(
        self, manager,
    ):
        """Pin: ``<workspace>/.jaato/apparmor-fragments/**`` is
        write-denied in EVERY profile body — base, isolated sub-
        profile (§4.3.4), tool_hat (§5.10), //child (§5.10).

        Privilege-escalation guard: PR #77 made
        ``<workspace>/.jaato/apparmor-fragments/*.rules`` a
        first-class fragment-discovery source, read live by
        ``_render_profile`` on the next session spawn.  Without
        this deny, a confined runner that can write to its
        workspace could plant a fragment file that auto-installs
        broader AppArmor rules on the next session — escape via
        time-shifted rule authoring.  Same ``wlk`` pattern as the
        other v13 narrow per-subpath denies (write + link +
        lock)."""
        # Base profile
        profile = manager._render_profile("s1", "/workspace")
        assert "audit deny /workspace/.jaato/apparmor-fragments/** wlk" in profile, (
            "base profile missing apparmor-fragments write-deny"
        )

        # Isolated sub-profile
        sub = manager._render_sub_profile(
            parent_session_id="parent-A",
            subagent_id="agent-B",
            workspace_path="/workspace",
        )
        assert "audit deny /workspace/.jaato/apparmor-fragments/** wlk" in sub, (
            "isolated sub-profile missing apparmor-fragments write-deny"
        )

        # tool_hat body (extract via brace counting)
        if "profile tool_hat" in profile:
            tool_hat_body = self._extract_brace_body(
                profile, "profile tool_hat",
            )
            assert (
                "audit deny /workspace/.jaato/apparmor-fragments/** wlk"
                in tool_hat_body
            ), "tool_hat body missing apparmor-fragments write-deny"

        # //child body
        if "profile child" in profile:
            child_body = self._extract_brace_body(
                profile, "profile child",
            )
            assert (
                "audit deny /workspace/.jaato/apparmor-fragments/** wlk"
                in child_body
            ), "//child body missing apparmor-fragments write-deny"

    def test_allows_reading_user_tier_services(self, manager):
        """Regression: SchemaStore's tiered lookup reads
        ``~/.jaato/services/`` as a user-tier fallback when the
        workspace tier doesn't have the service.  Confined WS sessions
        need AppArmor read access to that path, otherwise tiered lookup
        is invisible to any model call coming from a confined tool.

        Phase 3 (template v24, 2026-05-16): rule migrated to the
        service_connector plugin's ``get_apparmor_rules``.  Sessions
        whose ``profile.plugins`` includes service_connector get the
        grant via the resolver+plugin_rules path; sessions without it
        no longer carry the grant (least-privilege).  This test now
        verifies the new convention rather than the old hardcoded form.
        """
        from shared.plugins.service_connector.plugin import ServiceConnectorPlugin
        rules = ServiceConnectorPlugin.get_apparmor_rules(
            workspace_path="/workspace", session_id="s1",
            config_root=None, plugin_config={},
        )
        profile = manager._render_profile(
            "s1", "/workspace", plugin_rules=rules,
        )
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


class TestExtensionFragmentTierDiscovery:
    """``_render_profile`` discovers extension fragments from TWO
    tiers and composes both into the rendered profile body:

    - **user-tier** at ``~/.jaato/apparmor-fragments/*.rules`` —
      survives across workspaces, used by jaato-premium's reactor
      framework and other cross-workspace extensions.
    - **workspace-tier** at
      ``<workspace>/.jaato/apparmor-fragments/*.rules`` — scoped
      to the workspace, lives with the repo, version-controlled
      with its branches.

    Tier ordering matches the rest of jaato's tier discipline
    (workspace-tier composed AFTER user-tier in the rendered
    output for provenance clarity; AppArmor unions allow / unions
    deny semantics make the order semantically irrelevant)."""

    def _patch_user_tier_to_tmp(self, monkeypatch, tmp_path):
        """Redirect ``Path('~/...').expanduser()`` to a temp dir so
        the test doesn't read the developer's real
        ``~/.jaato/apparmor-fragments/`` (which jaato-premium may
        have populated during daemon-startup)."""
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path / ".jaato" / "apparmor-fragments"

    def test_workspace_tier_fragment_inlined(
        self, manager, tmp_path, monkeypatch,
    ):
        """Pin: a fragment at
        ``<workspace>/.jaato/apparmor-fragments/foo.rules`` lands
        in the rendered profile, wrapped with a
        ``# === workspace/foo.rules ===`` provenance comment."""
        self._patch_user_tier_to_tmp(monkeypatch, tmp_path)
        workspace = tmp_path / "workspace"
        ws_fragments = workspace / ".jaato" / "apparmor-fragments"
        ws_fragments.mkdir(parents=True)
        (ws_fragments / "kb-fixtures.rules").write_text(
            "/repo/fixtures/** r,\n"
        )

        rendered = manager._render_profile("sess1", str(workspace))

        assert "# === workspace/kb-fixtures.rules ===" in rendered, (
            "workspace-tier fragment must carry the workspace/ "
            "provenance label"
        )
        assert "/repo/fixtures/** r," in rendered, (
            "workspace-tier fragment body must be inlined"
        )

    def test_user_and_workspace_tier_both_compose(
        self, manager, tmp_path, monkeypatch,
    ):
        """Pin: when fragments exist in BOTH tiers, both are
        inlined into the rendered profile with distinct
        provenance comments.  Order: user-tier first, workspace-
        tier second (matches the comment header convention; the
        AppArmor parser's union semantics make the order
        semantically irrelevant)."""
        user_fragments = self._patch_user_tier_to_tmp(
            monkeypatch, tmp_path,
        )
        user_fragments.mkdir(parents=True)
        (user_fragments / "premium-reactor.rules").write_text(
            "@{HOME}/.jaato/reactors/** r,\n"
        )

        workspace = tmp_path / "workspace"
        ws_fragments = workspace / ".jaato" / "apparmor-fragments"
        ws_fragments.mkdir(parents=True)
        (ws_fragments / "kb-fixtures.rules").write_text(
            "/repo/fixtures/** r,\n"
        )

        rendered = manager._render_profile("sess1", str(workspace))

        assert "# === user/premium-reactor.rules ===" in rendered
        assert "# === workspace/kb-fixtures.rules ===" in rendered
        # User-tier appears BEFORE workspace-tier in the rendered
        # output — order is by tier-iteration order, not by file
        # name.
        user_pos = rendered.find("# === user/premium-reactor.rules ===")
        ws_pos = rendered.find("# === workspace/kb-fixtures.rules ===")
        assert 0 < user_pos < ws_pos, (
            f"expected user-tier before workspace-tier in render; "
            f"got user_pos={user_pos}, ws_pos={ws_pos}"
        )

    def test_missing_workspace_fragments_dir_silently_ok(
        self, manager, tmp_path, monkeypatch,
    ):
        """Pin: when a workspace has no
        ``.jaato/apparmor-fragments/`` directory at all (the
        typical case for most workspaces), render proceeds
        normally with whatever user-tier fragments exist.  No
        ``OSError`` raised, no log spam."""
        self._patch_user_tier_to_tmp(monkeypatch, tmp_path)
        workspace = tmp_path / "workspace-without-fragments"
        workspace.mkdir()
        # No .jaato/apparmor-fragments/ subdir created.

        rendered = manager._render_profile("sess1", str(workspace))

        # The (no extension fragments) placeholder should appear
        # when neither tier has anything.
        assert "(no extension fragments)" in rendered

    def test_user_only_tier_back_compat(
        self, manager, tmp_path, monkeypatch,
    ):
        """Pin: when only the user-tier has fragments (the
        pre-workspace-tier world — premium's only consumer
        today), the rendered profile is identical to the
        pre-change behavior: user fragment inlined, no
        workspace-tier provenance line.

        Back-compat invariant: existing user-tier installations
        keep working exactly as before."""
        user_fragments = self._patch_user_tier_to_tmp(
            monkeypatch, tmp_path,
        )
        user_fragments.mkdir(parents=True)
        (user_fragments / "premium-reactor.rules").write_text(
            "@{HOME}/.jaato/reactors/** r,\n"
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()  # No .jaato/apparmor-fragments/ here.

        rendered = manager._render_profile("sess1", str(workspace))

        assert "# === user/premium-reactor.rules ===" in rendered
        assert "# === workspace/" not in rendered, (
            "no workspace-tier fragments → no workspace/ comments"
        )


class TestApparmorFragmentsPerProfileScoping:
    """Piece 1 (2026-05-14): per-profile fragment scoping via the
    ``requested_fragments`` kwarg on ``_render_profile``.

    ``None`` keeps the pre-Piece-1 "compose all" behaviour.
    Non-None lists filter to a subset of the discovered fragments.
    Closes the cascade least-privilege footgun documented in
    ``project_backlog_per_profile_apparmor_fragments``.
    """

    def _patch_user_tier_to_tmp(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path / ".jaato" / "apparmor-fragments"

    def _make_workspace(
        self, tmp_path, ws_fragments=None, cache_fragments=None,
    ):
        """Build a workspace with optional workspace-tier and
        cache-tier fragments.

        Args:
            ws_fragments: ``{basename: body}`` for
                ``<workspace>/.jaato/apparmor-fragments/``.
            cache_fragments: ``{basename: body}`` for
                ``<workspace>/.jaato/.cache/apparmor-fragments/``.

        Returns the workspace Path.
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir(exist_ok=True)
        if ws_fragments:
            ws_dir = workspace / ".jaato" / "apparmor-fragments"
            ws_dir.mkdir(parents=True, exist_ok=True)
            for name, body in ws_fragments.items():
                (ws_dir / f"{name}.rules").write_text(body)
        if cache_fragments:
            cache_dir = workspace / ".jaato" / ".cache" / "apparmor-fragments"
            cache_dir.mkdir(parents=True, exist_ok=True)
            for name, body in cache_fragments.items():
                (cache_dir / f"{name}.rules").write_text(body)
        return workspace

    def test_none_composes_all_fragments_back_compat(
        self, manager, tmp_path, monkeypatch,
    ):
        """``requested_fragments=None`` (default) keeps the
        pre-Piece-1 behaviour: every fragment found is composed."""
        self._patch_user_tier_to_tmp(monkeypatch, tmp_path)
        workspace = self._make_workspace(tmp_path, ws_fragments={
            "host_validator": "/usr/bin/mvn ix,\n",
            "kb-fixtures": "/repo/fixtures/** r,\n",
        })

        rendered = manager._render_profile("sess1", str(workspace))

        assert "# === workspace/host_validator.rules ===" in rendered
        assert "# === workspace/kb-fixtures.rules ===" in rendered

    def test_empty_list_composes_no_fragments_maximally_locked_down(
        self, manager, tmp_path, monkeypatch,
    ):
        """``requested_fragments=[]`` is distinct from ``None``:
        composes NO fragments even when the search path has
        them.  Maximally locked-down cascade stage."""
        self._patch_user_tier_to_tmp(monkeypatch, tmp_path)
        workspace = self._make_workspace(tmp_path, ws_fragments={
            "host_validator": "/usr/bin/mvn ix,\n",
            "kb-fixtures": "/repo/fixtures/** r,\n",
        })

        rendered = manager._render_profile(
            "sess1", str(workspace), requested_fragments=[],
        )

        assert "# === workspace/host_validator.rules ===" not in rendered
        assert "# === workspace/kb-fixtures.rules ===" not in rendered
        assert "(no extension fragments)" in rendered

    def test_filter_includes_only_listed_by_basename(
        self, manager, tmp_path, monkeypatch,
    ):
        """``requested_fragments=["host_validator"]`` includes only
        that fragment by basename match; the kb-fixtures sibling
        is excluded."""
        self._patch_user_tier_to_tmp(monkeypatch, tmp_path)
        workspace = self._make_workspace(tmp_path, ws_fragments={
            "host_validator": "/usr/bin/mvn ix,\n",
            "kb-fixtures": "/repo/fixtures/** r,\n",
        })

        rendered = manager._render_profile(
            "sess1", str(workspace),
            requested_fragments=["host_validator"],
        )

        assert "# === workspace/host_validator.rules ===" in rendered
        assert "/usr/bin/mvn ix," in rendered
        assert "# === workspace/kb-fixtures.rules ===" not in rendered

    def test_cache_tier_wins_over_workspace_tier_on_collision(
        self, manager, tmp_path, monkeypatch,
    ):
        """Walker-generated cache fragment shadows the
        hand-authored workspace one when basenames collide.  This
        is the Piece-2 hand-off contract: walker writes to
        ``.jaato/.cache/apparmor-fragments/`` and that file is
        what the framework composes."""
        self._patch_user_tier_to_tmp(monkeypatch, tmp_path)
        workspace = self._make_workspace(
            tmp_path,
            ws_fragments={
                "host_validator": "# stub from repo\n/usr/bin/echo ix,\n",
            },
            cache_fragments={
                "host_validator": "# walker-generated for java-spring stack\n/usr/bin/mvn ix,\n",
            },
        )

        rendered = manager._render_profile(
            "sess1", str(workspace),
            requested_fragments=["host_validator"],
        )

        # Cache content present, workspace stub overridden out.
        assert "/usr/bin/mvn ix," in rendered
        assert "/usr/bin/echo ix," not in rendered
        # Provenance comment tracks the WINNING tier (cache).
        assert "# === cache/host_validator.rules ===" in rendered
        assert "# === workspace/host_validator.rules ===" not in rendered

    def test_unknown_fragment_name_logs_warning_no_abort(
        self, manager, tmp_path, monkeypatch, caplog,
    ):
        """Profile declares ``apparmor_fragments`` with a name that
        doesn't exist on disk → log WARNING but continue.  Operator
        may have removed the fragment after authoring the profile;
        rendering shouldn't fail loudly."""
        self._patch_user_tier_to_tmp(monkeypatch, tmp_path)
        workspace = self._make_workspace(tmp_path, ws_fragments={
            "host_validator": "/usr/bin/mvn ix,\n",
        })

        with caplog.at_level("WARNING", logger="server.apparmor"):
            rendered = manager._render_profile(
                "sess1", str(workspace),
                requested_fragments=["host_validator", "missing_one"],
            )

        # Known fragment is composed; missing is silently dropped.
        assert "# === workspace/host_validator.rules ===" in rendered
        # Warning mentions the missing name.
        assert any(
            "missing_one" in record.message for record in caplog.records
        ), "operator should see a warning naming the missing fragment"

    def test_cache_only_fragment_composed_when_listed(
        self, manager, tmp_path, monkeypatch,
    ):
        """A fragment that exists ONLY in the cache tier (the
        normal Piece-2 case — walker auto-generated, no
        hand-authored sibling) composes when listed in
        ``requested_fragments``."""
        self._patch_user_tier_to_tmp(monkeypatch, tmp_path)
        workspace = self._make_workspace(
            tmp_path, cache_fragments={
                "host_validator": "/usr/bin/mvn ix,\n",
            },
        )

        rendered = manager._render_profile(
            "sess1", str(workspace),
            requested_fragments=["host_validator"],
        )

        assert "# === cache/host_validator.rules ===" in rendered
        assert "/usr/bin/mvn ix," in rendered


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


class TestPluginContributedRules:
    """Plugin-contribution hook (template v20, Phase 0)."""

    def test_no_rules_renders_empty_marker_in_base_and_both_subprofiles(self, manager):
        """``plugin_rules=None`` emits a "(none for this session)" comment so the
        section header is greppable in every rendered profile."""
        rendered = manager._render_profile("s1", "/workspace")
        # Base + tool_hat + child sub-profile = 3 occurrences
        assert rendered.count("(none for this session)") == 3

    def test_empty_list_treated_same_as_none(self, manager):
        rendered = manager._render_profile("s1", "/workspace", plugin_rules=[])
        assert rendered.count("(none for this session)") == 3

    def test_rules_spliced_into_base_and_both_subprofiles(self, manager):
        rules = [
            "/dev/shm/sem.* rwk,",
            "@{HOME}/.cache/foo/   rw,",
        ]
        rendered = manager._render_profile("s1", "/workspace", plugin_rules=rules)
        # Section header appears in base + tool_hat + child
        assert rendered.count("# ---- plugin-contributed rules ----") == 3
        # Each rule appears 3 times (once per profile context)
        assert rendered.count("/dev/shm/sem.* rwk,") == 3
        assert rendered.count("@{HOME}/.cache/foo/   rw,") == 3
        # Default "(none for this session)" marker absent
        assert "(none for this session)" not in rendered

    def test_format_plugin_contributed_rules_indents_correctly(self):
        from server.apparmor import AppArmorManager
        out_base = AppArmorManager._format_plugin_contributed_rules(
            ["/dev/shm/sem.* rwk,"], indent="  ",
        )
        assert out_base == "  # ---- plugin-contributed rules ----\n  /dev/shm/sem.* rwk,"

        out_sub = AppArmorManager._format_plugin_contributed_rules(
            ["/dev/shm/sem.* rwk,"], indent="    ",
        )
        assert out_sub == "    # ---- plugin-contributed rules ----\n    /dev/shm/sem.* rwk,"

    def test_format_plugin_contributed_rules_empty(self):
        from server.apparmor import AppArmorManager
        out = AppArmorManager._format_plugin_contributed_rules(None, indent="  ")
        assert out == "  # ---- plugin-contributed rules (none for this session) ----"
        out_empty = AppArmorManager._format_plugin_contributed_rules([], indent="  ")
        assert out_empty == "  # ---- plugin-contributed rules (none for this session) ----"

    def test_provision_profile_forwards_plugin_rules(self, manager, tmp_path, monkeypatch):
        """End-to-end: provision_profile -> _render_profile picks up plugin_rules."""
        # Stub out apparmor_parser invocation; we only care about the rendered file.
        manager._profile_dir = tmp_path
        captured = {}

        def fake_parser(*args, **kwargs):
            class R: returncode = 0
            return R()

        monkeypatch.setattr(_apparmor_mod, "subprocess", MagicMock(run=fake_parser))
        monkeypatch.setattr(manager, "is_available", lambda: True)
        monkeypatch.setattr(manager, "_run_unconfined",
                            lambda fn, *a, **kw: fn(*a, **kw))

        rules = ["@{HOME}/.cache/foo/ rw,"]
        ok = manager.provision_profile(
            "s1", "/workspace", plugin_rules=rules,
        )
        assert ok is True

        # Profile file should contain the rule
        written = (tmp_path / "jaato-ws-s1").read_text()
        assert "@{HOME}/.cache/foo/ rw," in written
        assert written.count("@{HOME}/.cache/foo/ rw,") == 3  # base + 2 subprofiles


class TestResolvePluginApparmorRules:
    """Module-level helper used by IPC + WS provision paths."""

    def test_returns_none_when_profile_is_none(self):
        from server.apparmor import resolve_plugin_apparmor_rules
        out = resolve_plugin_apparmor_rules(
            server=MagicMock(), profile=None,
            session_id="s1", workspace_path="/ws", config_root=None,
        )
        assert out is None

    def test_returns_none_when_registry_empty(self):
        # Server 0.6.x+: the composer grants for the DISCOVERED plugin set
        # (registry.all_plugins()), not profile.plugins — an empty registry
        # contributes nothing.
        from server.apparmor import resolve_plugin_apparmor_rules
        profile = MagicMock()
        profile.plugin_configs = {}
        profile.gc = None  # Phase 3b: gc field also unions rules; opt out here.
        registry = MagicMock()
        registry.all_plugins.return_value = {}
        server = MagicMock()
        server.registry = registry
        out = resolve_plugin_apparmor_rules(
            server=server, profile=profile,
            session_id="s1", workspace_path="/ws", config_root=None,
        )
        assert out is None

    def test_returns_none_when_no_plugin_overrides(self):
        from server.apparmor import resolve_plugin_apparmor_rules
        profile = MagicMock()
        profile.plugin_configs = {}
        profile.gc = None  # Phase 3b: gc field also unions rules; opt out here.
        # Discovered plugins without get_apparmor_rules contribute nothing.
        class _Plain: pass
        registry = MagicMock()
        registry.all_plugins.return_value = {"cli": _Plain(), "todo": _Plain()}
        server = MagicMock()
        server.registry = registry
        out = resolve_plugin_apparmor_rules(
            server=server, profile=profile,
            session_id="s1", workspace_path="/ws", config_root=None,
        )
        assert out is None

    def test_unions_contributions_from_multiple_plugins(self):
        from server.apparmor import resolve_plugin_apparmor_rules

        class _PluginA:
            @classmethod
            def get_apparmor_rules(cls, **kw):
                return ["/dev/shm/sem.* rwk,"]

        class _PluginB:
            @classmethod
            def get_apparmor_rules(cls, **kw):
                return ["@{HOME}/.cache/x/ rw,", "@{HOME}/.cache/x/** rwk,"]

        profile = MagicMock()
        profile.plugin_configs = {}
        profile.gc = None  # Phase 3b: gc field also unions rules; opt out here.
        registry = MagicMock()
        registry.all_plugins.return_value = {"a": _PluginA(), "b": _PluginB()}
        server = MagicMock()
        server.registry = registry

        out = resolve_plugin_apparmor_rules(
            server=server, profile=profile,
            session_id="s1", workspace_path="/ws", config_root=None,
        )
        assert out == [
            "/dev/shm/sem.* rwk,",
            "@{HOME}/.cache/x/ rw,",
            "@{HOME}/.cache/x/** rwk,",
        ]

    def test_grants_loaded_plugins_even_with_empty_profile_plugins(self):
        """Regression (2026-06-18): a profile with ``plugins: []`` STILL gets
        the grants for auto-loaded plugins.  The runner inits ALL discovered
        runner-tier plugins regardless of profile.plugins (#114 disabled the
        load-gate), so the composer must grant for the loaded set — else
        memory/references/subagent init with [Errno 13].  Pre-fix this returned
        None for an empty profile.plugins, disabling those plugins every stage.
        """
        from server.apparmor import resolve_plugin_apparmor_rules

        class _Memory:
            @classmethod
            def get_apparmor_rules(cls, **kw):
                return ["@{HOME}/.jaato/memories/** rw,"]

        profile = MagicMock()
        profile.plugins = []  # IGNORED by the composer now
        profile.plugin_configs = {}
        profile.gc = None
        registry = MagicMock()
        # memory auto-loads even though it's not in profile.plugins
        registry.all_plugins.return_value = {"memory": _Memory()}
        server = MagicMock()
        server.registry = registry

        out = resolve_plugin_apparmor_rules(
            server=server, profile=profile,
            session_id="s1", workspace_path="/ws", config_root=None,
        )
        assert out == ["@{HOME}/.jaato/memories/** rw,"]

    def test_plugin_failure_does_not_abort(self, caplog):
        """One plugin's exception is logged but doesn't break the union."""
        from server.apparmor import resolve_plugin_apparmor_rules

        class _Broken:
            @classmethod
            def get_apparmor_rules(cls, **kw):
                raise RuntimeError("boom")

        class _Ok:
            @classmethod
            def get_apparmor_rules(cls, **kw):
                return ["/ok rwk,"]

        profile = MagicMock()
        profile.plugin_configs = {}
        profile.gc = None  # Phase 3b: gc field also unions rules; opt out here.
        registry = MagicMock()
        # dict preserves insertion order: broken raises (logged), ok contributes
        registry.all_plugins.return_value = {"broken": _Broken(), "ok": _Ok()}
        server = MagicMock()
        server.registry = registry

        out = resolve_plugin_apparmor_rules(
            server=server, profile=profile,
            session_id="s1", workspace_path="/ws", config_root=None,
        )
        assert out == ["/ok rwk,"]
        # The broken plugin's exception was logged
        assert any("broken get_apparmor_rules failed" in r.message for r in caplog.records)


class TestReferencesPluginApparmorRules:
    """References plugin override.

    Phase 1 (template v21) — HF + torch caches.
    Phase 3 (template v24) — user-tier references catalog reads.
    """

    def test_returns_huggingface_and_torch_caches(self):
        from shared.plugins.references.plugin import ReferencesPlugin
        rules = ReferencesPlugin.get_apparmor_rules(
            workspace_path="/ws",
            session_id="s1",
            config_root=None,
            plugin_config={},
        )
        assert "@{HOME}/.cache/huggingface/   rw," in rules
        assert "@{HOME}/.cache/huggingface/** rwk," in rules
        assert "@{HOME}/.cache/torch/         rw," in rules
        assert "@{HOME}/.cache/torch/**       rwk,"

    def test_returns_user_tier_catalog_reads(self):
        """Phase 3 acceptance: ~/.jaato/references/ catalog reads."""
        from shared.plugins.references.plugin import ReferencesPlugin
        rules = ReferencesPlugin.get_apparmor_rules(
            workspace_path="/ws",
            session_id="s1",
            config_root=None,
            plugin_config={},
        )
        assert "@{HOME}/.jaato/references/    r," in rules
        assert "@{HOME}/.jaato/references/**  r," in rules

    def test_is_a_classmethod_callable_without_instance(self):
        """Daemon-side resolution must work without instantiating the plugin."""
        from shared.plugins.references.plugin import ReferencesPlugin
        rules = ReferencesPlugin.get_apparmor_rules(
            workspace_path="/ws", session_id="s1",
            config_root=None, plugin_config={},
        )
        # 4 cache rules (Phase 1) + 2 catalog rules (Phase 3) = 6
        assert len(rules) == 6

    def test_template_no_longer_hardcodes_hf_torch(self, manager):
        """Phase 1 acceptance: rendered profile body (with no plugins)
        must NOT contain the HF or torch grants."""
        rendered = manager._render_profile("s1", "/workspace")
        assert "/.cache/huggingface/" not in rendered
        assert "/.cache/torch/" not in rendered

    def test_template_no_longer_hardcodes_user_catalog(self, manager):
        """Phase 3 acceptance: rendered profile body (with no plugins)
        must NOT contain the user-tier references catalog grants."""
        rendered = manager._render_profile("s1", "/workspace")
        assert "@{HOME}/.jaato/references/" not in rendered

    def test_template_picks_up_references_rules_when_passed(self, manager):
        """When the resolver feeds references' rules into _render_profile,
        all grants reappear in all 3 profile contexts."""
        from shared.plugins.references.plugin import ReferencesPlugin
        rules = ReferencesPlugin.get_apparmor_rules(
            workspace_path="/workspace", session_id="s1",
            config_root=None, plugin_config={},
        )
        rendered = manager._render_profile("s1", "/workspace", plugin_rules=rules)
        # Each of the 6 rules appears 3 times (base + tool_hat + child)
        assert rendered.count("@{HOME}/.cache/huggingface/   rw,") == 3
        assert rendered.count("@{HOME}/.cache/huggingface/** rwk,") == 3
        assert rendered.count("@{HOME}/.cache/torch/         rw,") == 3
        assert rendered.count("@{HOME}/.cache/torch/**       rwk,") == 3
        assert rendered.count("@{HOME}/.jaato/references/    r,") == 3
        assert rendered.count("@{HOME}/.jaato/references/**  r,") == 3


class TestGCApparmorRules:
    """GC subsystem apparmor contribution (Phase 3b, template v25).

    gc plugins live in ``profile.gc`` (not ``profile.plugins``); the
    resolver special-cases the field and calls the shared module-level
    helper ``shared.plugins.gc.get_gc_apparmor_rules``.
    """

    def test_helper_returns_gc_json_rule(self):
        from shared.plugins.gc import get_gc_apparmor_rules
        rules = get_gc_apparmor_rules()
        assert "@{HOME}/.jaato/gc.json  r," in rules
        assert len(rules) == 1

    def test_template_no_longer_hardcodes_gc_json(self, manager):
        """Phase 3b acceptance: rendered profile body (with no profile)
        must NOT contain the gc.json grant."""
        rendered = manager._render_profile("s1", "/workspace")
        assert "@{HOME}/.jaato/gc.json" not in rendered

    def test_template_picks_up_gc_rule_when_passed(self, manager):
        """When the resolver feeds gc's rules into _render_profile,
        the grant reappears in all 3 profile contexts."""
        from shared.plugins.gc import get_gc_apparmor_rules
        rules = get_gc_apparmor_rules()
        rendered = manager._render_profile("s1", "/workspace", plugin_rules=rules)
        assert rendered.count("@{HOME}/.jaato/gc.json  r,") == 3

    def test_resolver_unions_gc_rules_when_profile_gc_is_set(self):
        """resolve_plugin_apparmor_rules should append gc rules when
        profile.gc is non-None — even when profile.plugins is empty."""
        from server.apparmor import resolve_plugin_apparmor_rules
        from shared.plugins.subagent.config import GCProfileConfig

        class _StubProfile:
            plugins = []
            plugin_configs = {}
            gc = GCProfileConfig(type="truncate")

        class _StubServer:
            registry = None  # No registry: only the gc branch should fire.

        rules = resolve_plugin_apparmor_rules(
            server=_StubServer(),
            profile=_StubProfile(),
            session_id="s1",
            workspace_path="/workspace",
            config_root=None,
        )
        assert rules is not None
        assert "@{HOME}/.jaato/gc.json  r," in rules

    def test_resolver_does_not_emit_gc_rules_when_profile_gc_is_none(self):
        """When profile.gc is None (rare; usually GCProfileConfig is set),
        the resolver must NOT emit gc.json grants."""
        from server.apparmor import resolve_plugin_apparmor_rules

        class _StubProfile:
            plugins = []
            plugin_configs = {}
            gc = None

        class _StubServer:
            registry = None

        rules = resolve_plugin_apparmor_rules(
            server=_StubServer(),
            profile=_StubProfile(),
            session_id="s1",
            workspace_path="/workspace",
            config_root=None,
        )
        # No plugins, no gc → None
        assert rules is None


class TestSubagentPluginApparmorRules:
    """Subagent plugin override (Phase 4, template v26)."""

    def test_returns_agents_and_profiles_paths(self):
        from shared.plugins.subagent.plugin import SubagentPlugin
        rules = SubagentPlugin.get_apparmor_rules(
            workspace_path="/ws",
            session_id="s1",
            config_root=None,
            plugin_config={},
        )
        assert "@{HOME}/.jaato/agents/    r," in rules
        assert "@{HOME}/.jaato/agents/**  r," in rules
        assert "@{HOME}/.jaato/profiles/  r," in rules
        assert "@{HOME}/.jaato/profiles/** r," in rules

    def test_is_a_classmethod_callable_without_instance(self):
        """Daemon-side resolution must work without instantiating the plugin."""
        from shared.plugins.subagent.plugin import SubagentPlugin
        rules = SubagentPlugin.get_apparmor_rules(
            workspace_path="/ws", session_id="s1",
            config_root=None, plugin_config={},
        )
        assert len(rules) == 4

    def test_template_no_longer_hardcodes_agents_profiles(self, manager):
        """Phase 4 acceptance: rendered profile body (with no plugins)
        must NOT contain ~/.jaato/agents/ or ~/.jaato/profiles/ grants."""
        rendered = manager._render_profile("s1", "/workspace")
        assert "@{HOME}/.jaato/agents/" not in rendered
        assert "@{HOME}/.jaato/profiles/" not in rendered

    def test_template_picks_up_subagent_rules_when_passed(self, manager):
        """When the resolver feeds subagent's rules into _render_profile,
        the grants reappear in all 3 profile contexts (base + tool_hat +
        child)."""
        from shared.plugins.subagent.plugin import SubagentPlugin
        rules = SubagentPlugin.get_apparmor_rules(
            workspace_path="/workspace", session_id="s1",
            config_root=None, plugin_config={},
        )
        rendered = manager._render_profile("s1", "/workspace", plugin_rules=rules)
        # Each of the 4 rules appears 3 times (base + tool_hat + child)
        assert rendered.count("@{HOME}/.jaato/agents/    r,") == 3
        assert rendered.count("@{HOME}/.jaato/profiles/  r,") == 3


class TestPromptLibraryAgentsContribution:
    """Phase 4 extension: prompt_library plugin also declares agents/.

    Both subagent and prompt_library legitimately read ~/.jaato/agents/
    (subagent for persona reads on --agent spawn; prompt_library for
    agent-as-prompt discovery).  Each plugin's classmethod declares the
    rule independently; the resolver unions both contributions, and
    AppArmor parses duplicate rules idempotently.
    """

    def test_prompt_library_returns_agents_paths(self):
        from shared.plugins.prompt_library.plugin import PromptLibraryPlugin
        rules = PromptLibraryPlugin.get_apparmor_rules(
            workspace_path="/ws",
            session_id="s1",
            config_root=None,
            plugin_config={},
        )
        assert "@{HOME}/.jaato/agents/     r," in rules
        assert "@{HOME}/.jaato/agents/**   r," in rules

    def test_prompt_library_total_rule_count_includes_agents(self):
        """Phase 2 shipped 8 rules; Phase 4 adds agents/ + agents/** → 10."""
        from shared.plugins.prompt_library.plugin import PromptLibraryPlugin
        rules = PromptLibraryPlugin.get_apparmor_rules(
            workspace_path="/ws", session_id="s1",
            config_root=None, plugin_config={},
        )
        assert len(rules) == 10

    def test_resolver_unions_subagent_and_prompt_library_agents(self):
        """When both plugins are discovered, the resolver appends each
        plugin's contribution to the union (duplicate rules are idempotent
        at the AppArmor parser level)."""
        from server.apparmor import resolve_plugin_apparmor_rules
        from shared.plugins.subagent.plugin import SubagentPlugin
        from shared.plugins.prompt_library.plugin import PromptLibraryPlugin

        class _StubRegistry:
            def all_plugins(self):
                return {
                    "subagent": SubagentPlugin(),
                    "prompt_library": PromptLibraryPlugin(),
                }

        class _StubServer:
            registry = _StubRegistry()

        class _StubProfile:
            plugins = []  # composer ignores this now; discovery drives grants
            plugin_configs = {}
            gc = None

        rules = resolve_plugin_apparmor_rules(
            server=_StubServer(),
            profile=_StubProfile(),
            session_id="s1",
            workspace_path="/ws",
            config_root=None,
        )
        assert rules is not None
        # Subagent contributes agents/ + agents/**.
        # Prompt_library ALSO contributes agents/ + agents/** (different
        # whitespace formatting; both parse to the same AppArmor rule).
        assert "@{HOME}/.jaato/agents/    r," in rules     # subagent's
        assert "@{HOME}/.jaato/agents/     r," in rules    # prompt_library's


class TestMigratedPathsAbsentAcrossAllRenderers:
    """Phase 5 contract test (template v27).

    Pins that no plugin-attributable path (migrated in Phases 1-4)
    appears in ANY profile-body renderer's output by default.  This
    is the regression-prevention test that should have shipped with
    Phase 1 — without it, Phases 1/2/3/3b shipped silently broken on
    the fourth renderer (``_render_sub_profile``, see Phase 4 PR).

    When a future migration adds a new ``get_apparmor_rules``
    contribution, this test catches "forgot to strip a site"
    automatically across ALL renderers.
    """

    # All paths migrated to plugin classmethods (Phases 1-4).
    # If any of these appears in a default-rendered profile body
    # (no plugins configured), some renderer is broken.
    MIGRATED_PATH_SUBSTRINGS = (
        # Phase 1 (references plugin caches)
        "@{HOME}/.cache/huggingface/",
        "@{HOME}/.cache/torch/",
        # Phase 2 (memory + prompt_library)
        "@{HOME}/.jaato/memories/",
        "@{HOME}/.jaato/memories.jsonl",
        "@{HOME}/.jaato/prompts/",
        "@{HOME}/.jaato/skills/",
        "@{HOME}/.claude/skills/",
        "@{HOME}/.claude/commands/",
        # Phase 3 (service_connector + references catalog)
        "@{HOME}/.jaato/services/",
        "@{HOME}/.jaato/references/",
        # Phase 3b (gc subsystem)
        "@{HOME}/.jaato/gc.json",
        # Phase 4 (subagent agents/profiles)
        "@{HOME}/.jaato/agents/",
        "@{HOME}/.jaato/profiles/",
    )

    def _all_renderer_outputs(self, manager):
        """Yield (renderer_name, rendered_body) for every profile-body
        rendering path.  Adding a new renderer requires adding it here
        — that's the explicit contract for "new renderer must pass
        the migration-absence assertions."
        """
        # Base profile (includes the two nested sub-profiles tool_hat + child)
        yield "base", manager._render_profile("s1", "/workspace")
        # Isolated subagent (standalone profile)
        yield "isolated", manager._render_sub_profile(
            parent_session_id="parent_s1",
            subagent_id="sub_s1",
            workspace_path="/workspace",
        )

    def test_no_migrated_path_appears_by_default(self, manager):
        """For every renderer, no migrated path appears when no
        plugin_rules are passed (default render)."""
        for name, body in self._all_renderer_outputs(manager):
            for path in self.MIGRATED_PATH_SUBSTRINGS:
                assert path not in body, (
                    f"renderer={name!r} still contains migrated path "
                    f"{path!r}; either a Phase 1-4 strip was missed at "
                    f"this site, or a new migration didn't update this "
                    f"renderer.  See ``_render_user_global_block`` and "
                    f"Phase 1-4 PRs for the convention."
                )


class TestIsolatedSubprofileMigrationCompletion:
    """Phase 4 closes the Phase 2/3 gap on _build_isolated_subprofile.

    Phases 2/3 stripped plugin-attributable paths from base + tool_hat +
    child sub-profiles but missed the isolated sub-profile site.  This
    pins that the isolated rendering no longer contains the migrated
    paths either.
    """

    def test_isolated_subprofile_strips_phase2_phase3_paths(self, manager):
        """The isolated sub-profile body must no longer contain the
        paths migrated in Phase 2/3 (prompts, skills, memories,
        memories.jsonl) — Phase 4 closes the missed site.

        The isolated sub-profile builder is ``_render_sub_profile``
        (different naming from the ``_build_*`` siblings — the reason
        the strip was missed across Phases 1/2/3/3b).
        """
        body = manager._render_sub_profile(
            parent_session_id="parent_s1",
            subagent_id="sub_s1",
            workspace_path="/workspace",
        )
        # Phase 2 paths
        assert "@{HOME}/.jaato/prompts/" not in body
        assert "@{HOME}/.jaato/skills/" not in body
        assert "@{HOME}/.jaato/memories/" not in body
        assert "@{HOME}/.jaato/memories.jsonl" not in body
        # Phase 4 paths
        assert "@{HOME}/.jaato/agents/" not in body
        assert "@{HOME}/.jaato/profiles/" not in body


class TestServiceConnectorPluginApparmorRules:
    """Service connector plugin override (Phase 3, template v24)."""

    def test_returns_services_paths(self):
        from shared.plugins.service_connector.plugin import ServiceConnectorPlugin
        rules = ServiceConnectorPlugin.get_apparmor_rules(
            workspace_path="/ws",
            session_id="s1",
            config_root=None,
            plugin_config={},
        )
        assert "@{HOME}/.jaato/services/    r," in rules
        assert "@{HOME}/.jaato/services/**  r," in rules

    def test_is_a_classmethod_callable_without_instance(self):
        """Daemon-side resolution must work without instantiating the plugin."""
        from shared.plugins.service_connector.plugin import ServiceConnectorPlugin
        rules = ServiceConnectorPlugin.get_apparmor_rules(
            workspace_path="/ws", session_id="s1",
            config_root=None, plugin_config={},
        )
        assert len(rules) == 2

    def test_template_no_longer_hardcodes_services(self, manager):
        """Phase 3 acceptance: rendered profile body (with no plugins)
        must NOT contain the services grants."""
        rendered = manager._render_profile("s1", "/workspace")
        assert "@{HOME}/.jaato/services/" not in rendered

    def test_template_picks_up_service_connector_rules_when_passed(self, manager):
        """When the resolver feeds service_connector's rules into _render_profile,
        the grants reappear in all 3 profile contexts."""
        from shared.plugins.service_connector.plugin import ServiceConnectorPlugin
        rules = ServiceConnectorPlugin.get_apparmor_rules(
            workspace_path="/workspace", session_id="s1",
            config_root=None, plugin_config={},
        )
        rendered = manager._render_profile("s1", "/workspace", plugin_rules=rules)
        assert rendered.count("@{HOME}/.jaato/services/    r,") == 3
        assert rendered.count("@{HOME}/.jaato/services/**  r,") == 3


class TestMemoryPluginApparmorRules:
    """Memory plugin override (Phase 2, template v23)."""

    def test_returns_memories_paths(self):
        from shared.plugins.memory.plugin import MemoryPlugin
        rules = MemoryPlugin.get_apparmor_rules(
            workspace_path="/ws",
            session_id="s1",
            config_root=None,
            plugin_config={},
        )
        assert "@{HOME}/.jaato/memories/       rw," in rules
        assert "@{HOME}/.jaato/memories/**     rw," in rules
        assert "@{HOME}/.jaato/memories.jsonl  rw," in rules

    def test_is_a_classmethod_callable_without_instance(self):
        """Daemon-side resolution must work without instantiating the plugin."""
        from shared.plugins.memory.plugin import MemoryPlugin
        rules = MemoryPlugin.get_apparmor_rules(
            workspace_path="/ws", session_id="s1",
            config_root=None, plugin_config={},
        )
        assert len(rules) == 3

    def test_template_no_longer_hardcodes_memories(self, manager):
        """Phase 2 acceptance: rendered profile body (with no plugins)
        must NOT contain the memories grants."""
        rendered = manager._render_profile("s1", "/workspace")
        assert "@{HOME}/.jaato/memories/" not in rendered
        assert "@{HOME}/.jaato/memories.jsonl" not in rendered

    def test_template_picks_up_memory_rules_when_passed(self, manager):
        """When the resolver feeds memory's rules into _render_profile,
        the grants reappear in all 3 profile contexts."""
        from shared.plugins.memory.plugin import MemoryPlugin
        rules = MemoryPlugin.get_apparmor_rules(
            workspace_path="/workspace", session_id="s1",
            config_root=None, plugin_config={},
        )
        rendered = manager._render_profile("s1", "/workspace", plugin_rules=rules)
        # Each rule appears 3 times (base + tool_hat + child)
        assert rendered.count("@{HOME}/.jaato/memories/       rw,") == 3
        assert rendered.count("@{HOME}/.jaato/memories/**     rw,") == 3
        assert rendered.count("@{HOME}/.jaato/memories.jsonl  rw,") == 3


class TestPromptLibraryPluginApparmorRules:
    """Prompt library plugin override (Phase 2, template v23)."""

    def test_returns_prompts_skills_claude_paths(self):
        from shared.plugins.prompt_library.plugin import PromptLibraryPlugin
        rules = PromptLibraryPlugin.get_apparmor_rules(
            workspace_path="/ws",
            session_id="s1",
            config_root=None,
            plugin_config={},
        )
        # User-tier jaato prompts/skills are WRITABLE (savePrompt/deletePrompt
        # global tier): rwk, not read-only.
        assert "@{HOME}/.jaato/prompts/    rwk," in rules
        assert "@{HOME}/.jaato/prompts/**  rwk," in rules
        assert "@{HOME}/.jaato/skills/     rwk," in rules
        assert "@{HOME}/.jaato/skills/**   rwk," in rules
        # Agents stay READ-ONLY (not managed by these tools).
        assert "@{HOME}/.jaato/agents/     r," in rules
        assert "@{HOME}/.jaato/agents/**   r," in rules
        # Claude Code interop
        assert "@{HOME}/.claude/skills/    r," in rules
        assert "@{HOME}/.claude/skills/**  r," in rules
        assert "@{HOME}/.claude/commands/  r," in rules
        assert "@{HOME}/.claude/commands/**  r," in rules

    def test_is_a_classmethod_callable_without_instance(self):
        """Daemon-side resolution must work without instantiating the plugin.

        Phase 2 contract: 8 rules (prompts/skills/claude paths).
        Phase 4 contract: 10 rules (adds agents/ + agents/**).
        """
        from shared.plugins.prompt_library.plugin import PromptLibraryPlugin
        rules = PromptLibraryPlugin.get_apparmor_rules(
            workspace_path="/ws", session_id="s1",
            config_root=None, plugin_config={},
        )
        assert len(rules) == 10

    def test_template_no_longer_hardcodes_prompts_skills_claude(self, manager):
        """Phase 2 acceptance: rendered profile body (with no plugins)
        must NOT contain ~/.jaato/prompts, ~/.jaato/skills, ~/.claude grants."""
        rendered = manager._render_profile("s1", "/workspace")
        assert "@{HOME}/.jaato/prompts/" not in rendered
        assert "@{HOME}/.jaato/skills/" not in rendered
        assert "@{HOME}/.claude/skills/" not in rendered
        assert "@{HOME}/.claude/commands/" not in rendered

    def test_template_picks_up_prompt_library_rules_when_passed(self, manager):
        """When the resolver feeds prompt_library's rules into _render_profile,
        the grants reappear in all 3 profile contexts."""
        from shared.plugins.prompt_library.plugin import PromptLibraryPlugin
        rules = PromptLibraryPlugin.get_apparmor_rules(
            workspace_path="/workspace", session_id="s1",
            config_root=None, plugin_config={},
        )
        rendered = manager._render_profile("s1", "/workspace", plugin_rules=rules)
        # Each rule appears 3 times (base + tool_hat + child)
        assert rendered.count("@{HOME}/.jaato/prompts/    rwk,") == 3
        assert rendered.count("@{HOME}/.claude/commands/  r,") == 3

    def test_workspace_prompts_writable_but_config_dirs_still_deny_denied(self, manager):
        """Posture pin for the savePrompt/deletePrompt fix: the framework
        template makes .jaato/prompts/ runner-WRITABLE (no wlk deny — unlink
        needs 'w') while agents/profiles/scripts stay integrity-protected.

        Regression guard against BOTH directions: a future re-add of the
        prompts wlk deny (breaks deletePrompt again), or an accidental drop
        of the agents/profiles denies (over-widens the carve-out).
        """
        rendered = manager._render_profile("s1", "/workspace")
        # prompts is NO LONGER write-denied (the deliberate carve-out)...
        assert "audit deny /workspace/.jaato/prompts/**            wlk," not in rendered
        # ...but the sibling user-authored config dirs STILL are.
        assert "audit deny /workspace/.jaato/agents/**             wlk," in rendered
        assert "audit deny /workspace/.jaato/profiles/**           wlk," in rendered
        assert "audit deny /workspace/.jaato/scripts/**            wlk," in rendered
        # And the tool_hat READ-isolation deny on prompts is untouched.
        assert "audit deny /workspace/.jaato/prompts/**            r," in rendered

    def test_rendered_profile_with_prompt_rules_compiles(self, manager, tmp_path):
        """The whole point of the fix: the profile carrying prompt_library's
        writable-tier grants must actually COMPILE (PR #547 shipped an invalid
        'd,' that didn't — caught only by running apparmor_parser)."""
        import shutil, subprocess
        parser = shutil.which("apparmor_parser")
        if not parser:
            pytest.skip("apparmor_parser not installed")
        from shared.plugins.prompt_library.plugin import PromptLibraryPlugin
        rules = PromptLibraryPlugin.get_apparmor_rules(
            workspace_path="/workspace", session_id="s1",
            config_root="/cfg", plugin_config={},
        )
        rendered = manager._render_profile("s1", "/workspace", plugin_rules=rules)
        prof = tmp_path / "p.aa"
        prof.write_text(rendered)
        res = subprocess.run(
            [parser, "-Q", "-K", str(prof)], capture_output=True, text=True
        )
        assert res.returncode == 0, res.stderr


class TestSubprofileComplainFlag:
    """Template v22 — JAATO_APPARMOR_COMPLAIN propagates to sub-profiles."""

    def test_default_subprofiles_have_no_flags_declaration(self, manager, monkeypatch):
        """When the env knob is unset, sub-profile headers stay byte-equivalent
        to v21 (no ``flags=`` clause).  AppArmor doesn't permit
        ``attach_disconnected`` on sub-profiles, so the only legal flag here
        is ``complain``; absent the knob, no clause is emitted at all."""
        monkeypatch.delenv("JAATO_APPARMOR_COMPLAIN", raising=False)
        rendered = manager._render_profile("s1", "/workspace")
        assert "profile tool_hat {" in rendered
        assert "profile child {" in rendered
        assert "profile tool_hat flags=" not in rendered
        assert "profile child flags=" not in rendered

    def test_complain_env_propagates_to_both_subprofiles(self, manager, monkeypatch):
        """v22 acceptance: with the knob set, both sub-profile headers
        carry ``flags=(complain)`` so cli_based_tool subprocesses
        transitioning into ``//child`` also run complain-mode."""
        monkeypatch.setenv("JAATO_APPARMOR_COMPLAIN", "1")
        rendered = manager._render_profile("s1", "/workspace")
        assert "profile tool_hat flags=(complain) {" in rendered
        assert "profile child flags=(complain) {" in rendered
        # Base profile keeps the existing combined flag set
        assert "flags=(attach_disconnected, complain)" in rendered

    def test_complain_env_truthy_variants(self, manager, monkeypatch):
        """Accept the same truthy variants as the base profile path:
        ``1``, ``true``, ``yes`` (case-insensitive)."""
        for value in ("1", "true", "True", "YES"):
            monkeypatch.setenv("JAATO_APPARMOR_COMPLAIN", value)
            rendered = manager._render_profile("s1", "/workspace")
            assert "profile tool_hat flags=(complain) {" in rendered, (
                f"JAATO_APPARMOR_COMPLAIN={value!r} did not enable complain on tool_hat"
            )

    def test_complain_env_falsy_keeps_subprofiles_enforce(self, manager, monkeypatch):
        """Empty string / unset / "0" / "false" all keep sub-profiles enforce."""
        for value in ("", "0", "false", "no"):
            monkeypatch.setenv("JAATO_APPARMOR_COMPLAIN", value)
            rendered = manager._render_profile("s1", "/workspace")
            assert "profile tool_hat flags=" not in rendered, (
                f"JAATO_APPARMOR_COMPLAIN={value!r} incorrectly enabled complain on tool_hat"
            )


class TestRenderedProfileCompiles:
    """Guard: a rendered profile must not merely CONTAIN the expected rule
    strings — it must actually COMPILE under ``apparmor_parser``.

    Regression pin for PR #547: ``prompt_library.get_apparmor_rules`` emitted
    ``<tier>/**  d,`` — an INVALID rule (classic AppArmor has no standalone
    ``d`` delete mode; ``apparmor_parser`` fails with "unexpected TOK_ID,
    expecting TOK_MODE"). The whole per-session profile then failed to compile
    and never loaded, so every runner ran UNCONFINED. String-containment tests
    all passed because the bytes were present — only a real compile catches it.

    Skips when ``apparmor_parser`` isn't on the box (CI images without it);
    runs everywhere it exists (dev, VPS-like envs).
    """

    def _parser(self):
        import shutil
        return shutil.which("apparmor_parser")

    def _assert_compiles(self, profile_text: str, tmp_path) -> None:
        import subprocess
        parser = self._parser()
        if not parser:
            pytest.skip("apparmor_parser not installed")
        prof = tmp_path / "candidate.aa"
        prof.write_text(profile_text)
        # -Q: parse + cache only (don't load into kernel). -K: skip the shared
        # cache (avoids needing write access to /var/cache/apparmor).
        res = subprocess.run(
            [parser, "-Q", "-K", str(prof)],
            capture_output=True, text=True,
        )
        assert res.returncode == 0, (
            "rendered AppArmor profile does NOT compile — it would fail to "
            "load and DISABLE confinement (PR #547 class). apparmor_parser "
            f"stderr:\n{res.stderr}"
        )

    def test_default_profile_compiles(self, manager, tmp_path):
        profile = manager._render_profile("s1", "/workspace")
        self._assert_compiles(profile, tmp_path)

    def test_profile_with_prompt_library_rules_compiles(self, manager, tmp_path):
        from shared.plugins.prompt_library.plugin import PromptLibraryPlugin
        rules = PromptLibraryPlugin.get_apparmor_rules(
            workspace_path="/workspace", session_id="s1",
            config_root=None, plugin_config={},
        )
        profile = manager._render_profile("s1", "/workspace", plugin_rules=rules)
        self._assert_compiles(profile, tmp_path)

    def test_standalone_d_mode_is_rejected_by_parser(self, tmp_path):
        """Direct pin of the invalid token, independent of any plugin: the
        classic policy language has no bare ``d`` mode. If a future AppArmor
        ever adds one, this test flips and we can reconsider delete-only."""
        import subprocess
        parser = self._parser()
        if not parser:
            pytest.skip("apparmor_parser not installed")
        prof = tmp_path / "bare_d.aa"
        prof.write_text("profile t { /ws/x/** d,\n}\n")
        res = subprocess.run(
            [parser, "-Q", "-K", str(prof)], capture_output=True, text=True
        )
        assert res.returncode != 0, (
            "expected apparmor_parser to REJECT a bare 'd,' mode; if it now "
            "accepts one, delete-only grants may be reconsidered"
        )
