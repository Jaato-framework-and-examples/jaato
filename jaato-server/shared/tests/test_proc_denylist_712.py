"""The application-layer half of #712: procfs paths a tool may not name.

``sandbox_utils`` carried a hard denylist scoped to ``/proc/**/attr/**``
alone — a targeted anti-escape rule (writing ``changeprofile unconfined``
to ``attr/current`` leaves AppArmor entirely), well-reasoned and not a
``/proc`` policy.  ``environ``, ``mem``, ``pagemap``, ``auxv``, ``maps``
and ``smaps`` fell straight through it.

That mattered because ``/proc/self/environ`` read from an **in-process**
path-taking tool (``readFile``, ``glob_files``, ``file_edit``) is the
RUNNER's own environment: every provider API key and OAuth token the
session holds.  ``shared/secret_scrub.py`` scrubs the environment handed
to a model-driven SUBPROCESS and deliberately leaves the runner's intact,
so this read is the scrub's one bypass — and #863 made the scrub the
default, which raises the stakes rather than lowering them.

What stopped it before was workspace containment: ``/proc/...`` is outside
the workspace.  Two configurations left that argument without force —
``workspace_root`` unset (the denylist's own comment calls out that it
"applies even when workspace_root is unset", implying nothing else does),
and the degraded directory-sandbox-only posture of #504, where AppArmor is
unavailable and this gate is the ONLY control.

The kernel-profile half lives in ``test_apparmor_proc_hardening_712.py``.
Unlike that one, everything here is enforced in-process and is therefore
verified by these tests rather than merely rendered.
"""

import os

import pytest

from shared.plugins.sandbox_utils import (
    check_path_with_jaato_containment,
    is_proc_attr_path,
    is_sensitive_proc_path,
)


# Every spelling of the same file the kernel offers.  ``/proc/thread-self``
# resolves to ``/proc/<pid>/task/<tid>``, so the per-thread twin is not a
# theoretical form — it is what one of the three literal spellings becomes.
SPELLINGS = (
    "/proc/self/{entry}",
    "/proc/thread-self/{entry}",
    "/proc/1234/{entry}",
    "/proc/1234/task/5678/{entry}",
    "/proc/self/task/5678/{entry}",
)

LEAKY_ENTRIES = ("environ", "mem", "pagemap", "auxv", "maps", "smaps")


class TestSensitiveProcPaths:
    @pytest.mark.parametrize("entry", LEAKY_ENTRIES)
    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_leaky_entry_is_sensitive_under_every_spelling(
        self, entry, spelling
    ):
        path = spelling.format(entry=entry)
        assert is_sensitive_proc_path(path), f"{path} must be denied"

    def test_attr_paths_are_still_covered(self):
        """The pre-existing anti-escape rule is subsumed, not replaced.

        Writing ``changeprofile unconfined`` to ``attr/current`` from an
        in-process tool escapes the session's AppArmor profile outright,
        which is a strictly worse outcome than reading a credential.
        """
        for path in (
            "/proc/self/attr/current",
            "/proc/thread-self/attr/current",
            "/proc/1234/attr/current",
            "/proc/1234/task/5678/attr/exec",
        ):
            assert is_proc_attr_path(path)
            assert is_sensitive_proc_path(path)

    @pytest.mark.parametrize(
        "path",
        [
            "/proc/self/status",
            "/proc/1234/stat",
            "/proc/1234/comm",
            "/proc/1234/cmdline",
            "/proc/cpuinfo",
            "/proc/meminfo",
            "/proc/",
            "/proc",
        ],
    )
    def test_benign_proc_paths_are_not_denied(self, path):
        """The list stops at secrets and memory.

        ``status`` / ``stat`` / ``comm`` are ordinary metadata the
        framework itself reads (``environment`` reads
        ``/proc/<ppid>/comm``).  ``cmdline`` IS denied in the AppArmor
        template — the ``--ws-token TOKEN`` exposure — and deliberately
        not here: this gate governs paths handed to FILE tools, where the
        leak it enables is narrower than the ``ps``-shaped workflows a
        denial would break.
        """
        assert not is_sensitive_proc_path(path)

    @pytest.mark.parametrize(
        "path",
        [
            "/home/user/environ",
            "/workspace/src/maps",
            "/etc/passwd",
            "/tmp/mem",
            "environ",
        ],
    )
    def test_non_proc_paths_named_like_proc_entries_are_untouched(self, path):
        """A workspace file called ``maps`` or ``environ`` is an ordinary
        file.  The match is anchored at ``/proc/``, not on the basename.
        """
        assert not is_sensitive_proc_path(path)


class TestContainmentGate:
    """The denylist as the caller sees it.

    ``check_path_with_jaato_containment`` is what every path-taking
    in-process tool routes through, and the procfs check runs BEFORE the
    ``workspace_root`` early-return, so an unsandboxed session is covered
    too.
    """

    @pytest.mark.parametrize("entry", LEAKY_ENTRIES)
    def test_denied_with_no_workspace_configured(self, entry):
        """The case the old denylist comment singled out: with no
        ``workspace_root`` the function returns True for everything it has
        not explicitly denied, so a leaky procfs path reached the tool.
        """
        assert not check_path_with_jaato_containment(f"/proc/self/{entry}", "")

    @pytest.mark.parametrize("entry", LEAKY_ENTRIES)
    def test_denied_with_a_workspace_configured(self, entry, tmp_path):
        assert not check_path_with_jaato_containment(
            f"/proc/self/{entry}", str(tmp_path)
        )

    @pytest.mark.parametrize("entry", LEAKY_ENTRIES)
    def test_denied_through_a_symlink_in_the_workspace(self, entry, tmp_path):
        """A link committed into the workspace resolves into ``/proc``.

        The gate checks the literal AND the ``realpath``-resolved form for
        exactly this: ``ws/notes -> /proc/self/environ`` is a workspace
        path by every syntactic measure.
        """
        link = tmp_path / f"link_{entry}"
        link.symlink_to(f"/proc/self/{entry}")
        assert not check_path_with_jaato_containment(
            str(link), str(tmp_path)
        )

    def test_ordinary_workspace_file_still_allowed(self, tmp_path):
        """Guard against over-blocking: the denylist must not cost the
        gate its normal verdicts.
        """
        target = tmp_path / "notes.md"
        target.write_text("hello")
        assert check_path_with_jaato_containment(str(target), str(tmp_path))

    def test_dev_pseudo_devices_still_allowed(self, tmp_path):
        """``/dev/stdin`` and friends are symlinks THROUGH
        ``/proc/self/fd/<n>`` (jaato issue #784).  ``fd`` is not in the
        denylist, and the pseudo-device allowance is decided on the
        literal path, so ``2>/dev/null`` keeps working.
        """
        for path in ("/dev/null", "/dev/stdout", "/dev/fd/1"):
            assert check_path_with_jaato_containment(path, str(tmp_path)), (
                f"{path} must stay allowed"
            )


class TestDenialIsNotBypassable:
    def test_traversal_into_a_leaky_entry_is_denied(self, tmp_path):
        """``..`` traversal is normalised by ``os.path.abspath`` before the
        check, so a path that merely LOOKS like it leaves ``/proc`` does
        not evade the match.
        """
        assert not check_path_with_jaato_containment(
            "/proc/self/task/../environ", str(tmp_path)
        )

    def test_relative_path_resolving_into_proc_is_denied(self, tmp_path):
        """The gate makes the path absolute against the process cwd before
        matching, so a relative spelling reaches the same file and the same
        verdict.
        """
        cwd = os.getcwd()
        try:
            os.chdir("/proc")
            assert not check_path_with_jaato_containment(
                "self/environ", str(tmp_path)
            )
        finally:
            os.chdir(cwd)
