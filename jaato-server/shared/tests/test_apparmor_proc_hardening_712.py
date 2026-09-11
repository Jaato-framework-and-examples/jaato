"""Template v30 (#712): procfs credential hardening in every profile body.

``shared/secret_scrub.py`` strips secret names from the environment *given
to* a model-driven subprocess and deliberately leaves the runner's own
``os.environ`` intact — so the runner stays a live credential store
(provider API keys, OAuth tokens), and anything able to read
``/proc/<pid>/environ`` walks straight around the scrub.  #863 turned that
scrub on by default, which makes its one bypass matter more rather than
less.

Before v30 the only control on this path was application-layer workspace
containment (``check_path_with_jaato_containment`` rejects ``/proc/...`` as
outside the workspace), and #668 establishes that layer is porous —
``sh -c '...'`` hides a whole command in one quoted token the path
extractor never inspects.  AppArmor exists to be the backstop when the
application layer is wrong; for this class it was not one.

**What these tests can and cannot prove.**  They assert the rendered
profile TEXT.  A generated rule and an ENFORCED rule are different claims:
the second needs a host with AppArmor loaded and
``AppArmorManager.is_available()`` returning True.
``test_rendered_profiles_still_compile`` is the strongest check available
off such a host — it proves ``apparmor_parser`` accepts the profile — and
it skips where that binary is absent.

The application-layer half of #712 lives in
``test_proc_denylist_712.py``.
"""

import os
import re
import sys
import types

import pytest

# Avoid importing server.__init__ which pulls heavy deps (google, etc.) —
# mirrors the stub in test_apparmor.py.
if "server" not in sys.modules:
    _stub = types.ModuleType("server")
    _stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "server")]
    sys.modules["server"] = _stub

import server.apparmor as _apparmor_mod

AppArmorManager = _apparmor_mod.AppArmorManager


@pytest.fixture
def manager(tmp_path):
    workspace_root = tmp_path / "workspaces"
    (workspace_root / "sessions").mkdir(parents=True)
    profile_dir = tmp_path / "apparmor_profiles"
    profile_dir.mkdir()
    return AppArmorManager(
        workspace_root=str(workspace_root),
        venv_path="/usr/local/venv",
        profile_dir=str(profile_dir),
    )


def _extract_brace_body(text: str, anchor: str) -> str:
    """Extract the body inside ``{...}`` following ``anchor``.

    Brace-counts so a nested sub-profile and the literal ``}`` of an
    ``@{HOME}`` substitution don't terminate the scan early.
    """
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
    raise AssertionError(f"unmatched brace after anchor {anchor!r}")


def _bodies(manager):
    """Every rule body a confined process can be running under.

    Sub-profiles do NOT inherit base rules in AppArmor (the v13 note in
    ``apparmor.py`` records the empirical finding), so a deny present only
    in the base leaves ``tool_hat`` — the scope every in-process tool runs
    in — and ``//child`` — where every model-driven subprocess runs —
    uncovered.  The isolated sub-runner profile is a fourth, flat body.
    """
    profile = manager._render_profile("s1", "/workspace")
    sub = manager._render_sub_profile(
        parent_session_id="parent-A",
        subagent_id="agent-B",
        workspace_path="/workspace",
    )
    return {
        "base": profile,
        "tool_hat": _extract_brace_body(profile, "profile tool_hat"),
        "child": _extract_brace_body(profile, "profile child"),
        "isolated_sub_runner": sub,
    }


# Entries denied in every body, with the mode each is denied for.
DENIED = (
    ("environ", "r"),
    ("mem", "rw"),
    ("pagemap", "r"),
    ("auxv", "r"),
    ("cmdline", "r"),
)


class TestProcCredentialHardening712:
    def test_every_leaky_proc_entry_is_denied_in_every_body(self, manager):
        for label, body in _bodies(manager).items():
            for entry, mode in DENIED:
                pattern = re.compile(
                    r"audit deny /proc/\*/" + entry + r"\s+" + mode + r","
                )
                assert pattern.search(body), (
                    f"{label} body is missing "
                    f"'audit deny /proc/*/{entry} {mode},' — a process "
                    f"confined by it could read /proc/<pid>/{entry}"
                )

    def test_per_thread_twins_are_denied_too(self, manager):
        """``/proc/<pid>/task/<tid>/environ`` is the same file under
        another name, and ``/proc/thread-self/environ`` resolves to
        exactly that path — so a deny naming only the process-level entry
        is bypassed by spelling it per-thread.
        """
        for label, body in _bodies(manager).items():
            for entry, mode in DENIED:
                pattern = re.compile(
                    r"audit deny /proc/\*/task/\*/" + entry + r"\s+" + mode + r","
                )
                assert pattern.search(body), (
                    f"{label} body is missing the per-thread twin "
                    f"'audit deny /proc/*/task/*/{entry} {mode},'"
                )

    def test_denies_use_the_pid_glob_never_proc_self(self, manager):
        """The rule FORM is load-bearing, not cosmetic.

        AppArmor resolves the ``/proc/self`` symlink to ``/proc/<pid>/``
        BEFORE matching — the empirical finding recorded as the v15 note
        in ``apparmor.py``, which is why ``/proc/self/** r,`` matches
        nothing and the v14 ``/proc/self/attr/current w,`` rule only ever
        worked through a procfs special case.  A deny written
        ``/proc/self/environ`` would never fire against the read it exists
        to stop, and would satisfy every string-containment test.
        """
        self_form = re.compile(r"audit deny\s+/proc/self/")
        for label, body in _bodies(manager).items():
            assert not self_form.search(body), (
                f"{label} body denies a /proc/self/ path; AppArmor resolves "
                "that symlink before matching, so the rule can never fire. "
                "Use the /proc/*/ form."
            )

    def test_no_body_still_grants_cmdline(self, manager):
        """``--ws-token TOKEN`` is a documented way to start the daemon and
        puts the bearer token in its argv.  The isolated sub-runner profile
        granted ``/proc/*/cmdline r,`` for ANY pid until v30, so every such
        session could read the daemon's command line.  ``--ws-token-file``
        is the spelling that avoids the exposure.
        """
        grant = re.compile(r"^\s*(owner\s+)?/proc/\*/cmdline\s+r,", re.M)
        for label, body in _bodies(manager).items():
            assert not grant.search(body), (
                f"{label} body still grants read on /proc/*/cmdline"
            )

    def test_benign_process_metadata_stays_readable(self, manager):
        """The deny set stops at secrets and memory.

        ``status`` / ``stat`` / ``comm`` are ordinary process metadata —
        the ``environment`` plugin reads ``/proc/<ppid>/comm``, and the
        isolated sub-runner profile grants all three deliberately under its
        self-introspection block.  Denying them would break a legitimate
        read for no credential-confidentiality gain.
        """
        sub = manager._render_sub_profile(
            parent_session_id="parent-A",
            subagent_id="agent-B",
            workspace_path="/workspace",
        )
        for entry in ("status", "stat", "comm"):
            assert re.search(r"^\s*/proc/\*/" + entry + r"\s+r,", sub, re.M), (
                f"isolated sub-runner profile lost its /proc/*/{entry} grant"
            )
            assert not re.search(r"audit deny /proc/\*/" + entry, sub), (
                f"/proc/*/{entry} is benign metadata and must not be denied"
            )

    def test_proc_fd_directory_stays_readable(self, manager):
        """CPython's ``close_fds`` path enumerates ``/proc/self/fd`` at
        every subprocess spawn, so this grant is load-bearing and is
        deliberately left in place.  #712 names it as a smaller exposure of
        the same kind; AppArmor has no rule form for "my own pid only", so
        narrowing it is not expressible here.
        """
        sub = manager._render_sub_profile(
            parent_session_id="parent-A",
            subagent_id="agent-B",
            workspace_path="/workspace",
        )
        assert re.search(r"^\s*/proc/\*/fd/\s+r,", sub, re.M), (
            "isolated sub-runner profile lost its /proc/*/fd/ grant; "
            "subprocess spawn enumerates it"
        )

    def test_attr_current_transition_rules_survive(self, manager):
        """The hardening must not collide with confinement itself.

        ``apparmor_confine().__exit__`` restores unconfined by WRITING to
        ``attr/current``, and the runner's bootstrap verify-after-write
        READS it.  ``attr`` is not in the deny set, and a deny that
        over-matched it would trap every thread-pool worker in the
        session's profile.
        """
        profile = manager._render_profile("s1", "/workspace")
        assert re.search(
            r"owner /proc/\*/attr/current\s+rw,", profile
        ), "base profile lost its attr/current transition grant"
        assert not re.search(r"audit deny /proc/\*/attr", profile), (
            "attr/current must stay writable — denying it traps threads "
            "in the session profile"
        )

    def test_template_version_bumped_to_30(self, manager):
        """Confined sessions must pick the new rules up rather than a
        cached compile of v29 — the version comment changes the content
        hash, which is what forces ``apparmor_parser`` to recompile
        instead of reusing its cache entry.
        """
        assert manager._TEMPLATE_VERSION >= 30
        profile = manager._render_profile("s1", "/workspace")
        assert (
            f"jaato-apparmor-template-version: {manager._TEMPLATE_VERSION}"
            in profile
        )

    def test_rendered_profiles_still_compile(self, manager, tmp_path):
        """The v30 block must PARSE, not merely be present.

        Same class of guard as ``TestRenderedProfileCompiles`` (PR #547): a
        profile that fails to compile never loads, and every runner then
        runs UNCONFINED — a hardening change that silently removes all
        confinement is worse than no change at all.  Skips where
        ``apparmor_parser`` is absent, which is every CI image without
        AppArmor userspace installed.
        """
        import shutil
        import subprocess

        parser = shutil.which("apparmor_parser")
        if not parser:
            pytest.skip("apparmor_parser not installed")
        renders = {
            "base": manager._render_profile("s1", "/workspace"),
            "isolated": manager._render_sub_profile(
                parent_session_id="parent-A",
                subagent_id="agent-B",
                workspace_path="/workspace",
            ),
        }
        for label, text in renders.items():
            prof = tmp_path / f"{label}.aa"
            prof.write_text(text)
            res = subprocess.run(
                [parser, "-Q", "-K", str(prof)],
                capture_output=True,
                text=True,
            )
            assert res.returncode == 0, (
                f"{label} profile does not compile after the v30 procfs "
                f"deny block:\n{res.stderr}"
            )
