"""A root daemon announces itself, and a umask is expressible (#1168).

THE STATE THIS GUARDS AGAINST.  The runner executes as the daemon's uid —
``RunnerSpawner.spawn`` is ``os.fork()`` + ``os.execvpe`` with no
``setuid`` anywhere on the path — so on a root daemon every file the
agent writes into a workspace is root-owned, and its owner needs ``sudo``
to overwrite or delete it.  ``grep -rn 'geteuid'`` across ``server/``
returned only the egress proxy's sudo decision and a cgroups writability
message: **the daemon never noticed it was root**, while every other
weakened posture in this tree announces itself at WARNING
(``scrub_secret_env: none``, ``--ws-unsafe-no-auth``, complain-mode
AppArmor).  And ``grep -rn 'os.umask'`` returned nothing tree-wide, so
the cheap mitigation — ``umask 002`` plus a setgid workspace, which makes
those files group-writable without touching ownership — could not be
expressed at all.

WHAT IS **NOT** GUARDED HERE, because it is not implemented: privilege
dropping (step 3 of the issue).  Nothing in this file asserts that any
process changes uid, and the module under test says so in its own
docstring.

THE UID IS SUBSTITUTED IN BOTH DIRECTIONS, DELIBERATELY.  ``os.geteuid``
is patched to 0 for the warning cases and to a non-zero uid for the
control.  Patching only one side would make the suite's verdict depend on
the uid it happens to run under: on a normal CI runner an *unconditional*
warning would sail through the control, and in a root container (which is
where this was written) a *never-firing* one would sail through the
warning cases.  Patched both ways, both cases are decided by the code
rather than by the environment.

What that proves is the *decision* and the *message* — that the daemon
notices euid 0 and says what it costs.  It proves nothing about the
kernel's behaviour under uid 0, and nothing here needs to: the property
under test is exactly the predicate being substituted for.

THE CALL-SITE TESTS ARE THE LOAD-BEARING ONES.  Everything else here
exercises ``process_posture`` directly, which says the mechanism works and
says nothing about whether anything invokes it — and a mechanism that
exists and is never called is precisely the pre-#1133 shape (a GC
strategy resolved, rendered in the UI, and installed on nobody) and the
#735 shape (``tool_timeout_seconds`` enforced at a point nothing handed a
value to).  So the daemon's ``start()`` and the standalone WS ``main()``
are checked by walking their AST for the call.
"""

from __future__ import annotations

import ast
import inspect
import logging

import pytest

from jaato_server.shared.tests.reversion import Reversion


REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/server/process_posture.py",
        find="    return geteuid() == 0\n",
        replace="    return False\n",
        test="TestTheRootWarning::test_a_root_daemon_warns",
        because="the daemon going back to never noticing it is root, "
                "which is the whole of #1168's first half: every file the "
                "agent writes lands root-owned in somebody else's "
                "workspace and nothing says so",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/process_posture.py",
        find="    _root_announced.set()\n",
        replace="    pass  # _root_announced.set()\n",
        test="TestTheRootWarning::test_the_warning_fires_once_per_process",
        because="the once-per-daemon latch going away, so the line repeats "
                "per session until an operator filters it out -- which is "
                "the same outcome as silence, and the reason "
                "announce_complain_mode_once latches too",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/process_posture.py",
        find=(
            '            "umask this process inherited.", raw,\n'
            "        )\n"
            "        return None\n"
            "    return value\n"
        ),
        replace=(
            '            "umask this process inherited.", raw,\n'
            "        )\n"
            "        return 0o022\n"
            "    return value\n"
        ),
        test="TestTheUmaskKnob::test_a_malformed_umask_changes_nothing",
        because="a typo in --umask silently applying an invented mask, "
                "changing the mode of every file the daemon and its "
                "runners write for a reason no operator could find",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/process_posture.py",
        find="    if explicit is not None:\n        return parse_umask(explicit)\n",
        replace="    if False:\n        return parse_umask(explicit)\n",
        test="TestTheUmaskKnob::test_the_flag_outranks_the_env_var",
        because="--umask being silently overruled by an exported "
                "JAATO_UMASK, so the daemon runs a posture the operator "
                "did not ask for on the command line they can see",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/__main__.py",
        find="        apply_process_posture(self._umask)\n",
        replace="        pass  # apply_process_posture(self._umask)\n",
        test="TestTheCallSites::test_the_daemon_start_applies_the_posture",
        because="the mechanism existing and the daemon never invoking it "
                "-- the #1133 shape, where a test that imports the helper "
                "and calls it passes on the broken tree",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/websocket.py",
        find="    apply_process_posture()\n",
        replace="    pass  # apply_process_posture()\n",
        test="TestTheCallSites::test_the_standalone_ws_server_applies_it_too",
        because="one of the two documented daemon entry points going back "
                "to silence, which is worse than neither warning: a check "
                "that fires on some root daemons and not others teaches "
                "an operator to disbelieve it",
    ),
]


@pytest.fixture(autouse=True)
def _fresh_latch():
    """Clear the once-per-process latch around every test.

    The module deliberately holds it for the life of the process, which is
    right for a daemon and wrong for a test session that asserts the first
    call and the second in different tests.  Cleared on the way in *and*
    out so neither ordering nor a failure mid-test leaks into a sibling.
    """
    from jaato_server.server import process_posture

    process_posture._root_announced.clear()
    yield
    process_posture._root_announced.clear()


def _as_root(monkeypatch):
    """Make :func:`running_as_root` answer yes without becoming root."""
    monkeypatch.setattr("os.geteuid", lambda: 0, raising=False)


def _as_ordinary_user(monkeypatch):
    """Make it answer no, whatever uid the suite is really running as.

    Needed because this is a symmetric substitution: see the module
    docstring.  1000 is the first ordinary uid on every distribution this
    tree is deployed on and is not special to the code, which only asks
    whether the number is 0.
    """
    monkeypatch.setattr("os.geteuid", lambda: 1000, raising=False)


class TestTheRootWarning:
    """The daemon notices euid 0, says what it costs, and says it once."""

    def test_a_root_daemon_warns(self, monkeypatch, caplog):
        from jaato_server.server import process_posture

        _as_root(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=process_posture.__name__):
            process_posture.announce_root_daemon_once()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, (
            "a daemon running as root emitted no WARNING. That is the "
            "reported state: nothing in server/ ever read geteuid, so the "
            "posture was invisible."
        )

    def test_the_warning_names_the_consequence_and_both_remedies(
        self, monkeypatch, caplog,
    ):
        """A warning that does not say what to do is noise.

        Asserted on substance rather than wording: the files being
        root-owned (the consequence), a service user (the fix the
        deployment guides already assume), and the umask knob (the
        mitigation where that is impossible).
        """
        from jaato_server.server import process_posture

        _as_root(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=process_posture.__name__):
            process_posture.announce_root_daemon_once()

        text = "\n".join(r.getMessage() for r in caplog.records).lower()
        assert "root-owned" in text
        assert "sudo" in text
        assert "service user" in text
        assert "umask" in text
        assert "apparmor-setup.md" in text
        assert "runtime-limits-setup.md" in text

    def test_the_warning_fires_once_per_process(self, monkeypatch, caplog):
        from jaato_server.server import process_posture

        _as_root(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=process_posture.__name__):
            process_posture.announce_root_daemon_once()
            process_posture.announce_root_daemon_once()
            process_posture.announce_root_daemon_once()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, (
            f"the root warning fired {len(warnings)} times. It is a daemon "
            f"posture, not a session event -- one line per session is noise "
            f"an operator filters out, which is the same outcome as silence."
        )

    def test_an_ordinary_daemon_is_told_nothing(self, monkeypatch, caplog):
        """The control — and the reason it cannot be skipped.

        Every other test in this class reports euid 0. If the
        announcement were unconditional they would all still pass, and
        every non-root deployment would carry a warning describing
        something that is not happening to it — which is how a warning
        stops being read.
        """
        from jaato_server.server import process_posture

        _as_ordinary_user(monkeypatch)
        assert process_posture.running_as_root() is False
        with caplog.at_level(logging.WARNING, logger=process_posture.__name__):
            process_posture.announce_root_daemon_once()

        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_a_platform_without_geteuid_is_not_guessed_at(self, monkeypatch):
        """Windows has no euid, and inventing one would be a claim.

        ``False`` is not a convenience default here: it is the statement
        that the POSIX ownership problem this reports cannot arise on a
        platform that cannot be asked.
        """
        from jaato_server.server import process_posture

        monkeypatch.delattr("os.geteuid", raising=False)
        assert process_posture.running_as_root() is False


class TestTheUmaskKnob:
    """A umask is resolvable, applied, and never invented."""

    def test_the_flag_outranks_the_env_var(self, monkeypatch):
        from jaato_server.server import process_posture

        monkeypatch.setenv(process_posture.UMASK_ENV_VAR, "027")
        assert process_posture.resolve_umask("002") == 0o002

    def test_the_env_var_answers_when_no_flag_was_passed(self, monkeypatch):
        from jaato_server.server import process_posture

        monkeypatch.setenv(process_posture.UMASK_ENV_VAR, "002")
        assert process_posture.resolve_umask(None) == 0o002

    def test_nothing_configured_is_not_a_framework_default(self, monkeypatch):
        """``None`` means "leave the inherited umask alone".

        Substituting a framework default would change the mode of every
        file on every existing deployment, which is the opposite of what
        an opt-in mitigation may do.
        """
        from jaato_server.server import process_posture

        monkeypatch.delenv(process_posture.UMASK_ENV_VAR, raising=False)
        assert process_posture.resolve_umask(None) is None

    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_a_blank_value_is_not_a_umask_of_zero(self, raw, monkeypatch):
        """``umask 000`` is world-writable — the worst thing a blank
        string could be read as, and ``int("", 8)`` would not even get
        that far."""
        from jaato_server.server import process_posture

        monkeypatch.delenv(process_posture.UMASK_ENV_VAR, raising=False)
        assert process_posture.resolve_umask(raw) is None

    @pytest.mark.parametrize("raw", ["not-octal", "99", "1000", "-1", "777777", "0x1ff"])
    def test_a_malformed_umask_changes_nothing(self, raw, caplog):
        from jaato_server.server import process_posture

        with caplog.at_level(logging.ERROR, logger=process_posture.__name__):
            assert process_posture.parse_umask(raw) is None, (
                f"{raw!r} produced a umask. A malformed value must leave "
                f"the inherited umask alone -- applying an invented mask "
                f"changes every file's mode for a reason no operator can "
                f"find in their own configuration."
            )
        assert [r for r in caplog.records if r.levelno == logging.ERROR], (
            f"{raw!r} was ignored silently, which is the silent-ignore "
            f"family this tree reports rather than performs"
        )

    @pytest.mark.parametrize(
        "raw,expected",
        [("002", 0o002), ("22", 0o022), ("0022", 0o022), ("777", 0o777),
         ("0", 0), (" 002 ", 0o002),
         # ``int(text, 8)`` also takes Python's own ``0o`` prefix. Pinned
         # rather than filtered out: it denotes the same mask the shell
         # spelling does, so refusing it would reject a value whose
         # meaning is unambiguous.
         ("0o22", 0o022)],
    )
    def test_octal_is_read_the_way_the_shell_builtin_writes_it(self, raw, expected):
        from jaato_server.server import process_posture

        assert process_posture.parse_umask(raw) == expected

    def test_applying_a_umask_sets_the_process_umask(self):
        """The one test that touches real process state, and restores it.

        ``os`` has no umask getter, so the restore uses the value
        ``apply_umask`` returns — which is also what makes that return
        value worth having.
        """
        from jaato_server.server import process_posture

        original = os_umask_snapshot()
        try:
            previous = process_posture.apply_umask(0o002)
            assert previous is not None
            assert os_umask_snapshot() == 0o002
        finally:
            import os

            os.umask(original)

    def test_no_umask_configured_leaves_the_process_alone(self):
        from jaato_server.server import process_posture

        original = os_umask_snapshot()
        try:
            assert process_posture.apply_umask(None) is None
            assert os_umask_snapshot() == original
        finally:
            import os

            os.umask(original)


def os_umask_snapshot() -> int:
    """Read the current umask the only way the stdlib allows.

    ``os`` exposes no getter, so this is a set-and-restore pair — which is
    exactly why :func:`server.process_posture.apply_umask` logs the value
    it wrote instead of reading one back in a daemon full of threads.
    """
    import os

    current = os.umask(0o022)
    os.umask(current)
    return current


class TestTheCallSites:
    """The mechanism is invoked, not merely present.

    Source inspection rather than behaviour, because the failure being
    guarded against is *nobody calls it* — which a test that calls it
    cannot see.  The sibling guards ``test_envelope_carries_gc.py``,
    ``test_budget_mid_turn_955.py`` and
    ``test_orphan_bound_observes_attachment_812.py`` assert their call
    sites the same way, for the same reason.
    """

    @staticmethod
    def _functions_calling(module, name: str) -> set:
        tree = ast.parse(inspect.getsource(module))
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == name
                for c in ast.walk(node)
            )
        }

    def test_the_daemon_start_applies_the_posture(self):
        from jaato_server.server import __main__ as daemon_main

        callers = self._functions_calling(daemon_main, "apply_process_posture")
        assert "start" in callers, (
            "JaatoDaemon.start() does not call apply_process_posture(). "
            "The umask would then never reach the pre-warm template this "
            "process forks -- so not the pool slots that serve the default "
            "path -- and a root daemon would go back to saying nothing."
        )

    def test_the_standalone_ws_server_applies_it_too(self):
        from jaato_server.server import websocket as ws

        callers = self._functions_calling(ws, "apply_process_posture")
        assert "main" in callers, (
            "server/websocket.py::main() does not call "
            "apply_process_posture(), yet it is the entry point both "
            "deployment guides print and it spawns runners under its own "
            "uid exactly as the daemon does."
        )

    def test_the_umask_flag_reaches_the_daemon(self):
        """The flag, the constructor and the round-trip are one chain.

        A ``--umask`` argparse entry that no constructor receives is the
        advertised-but-unread shape ``allow_inline`` was in (#944):
        documented, offered to the operator, and read nowhere.
        """
        from jaato_server.server import __main__ as daemon_main

        source = inspect.getsource(daemon_main)
        assert '"--umask"' in source, "the CLI flag is gone"
        assert "umask=args.umask" in source, (
            "main() parses --umask and does not hand it to JaatoDaemon, so "
            "the flag decides nothing"
        )
        assert '"umask": self._umask' in source, (
            "_write_config() does not persist the umask, so --restart "
            "silently reverts the posture the operator chose"
        )
        signature = inspect.signature(daemon_main.JaatoDaemon.__init__)
        assert "umask" in signature.parameters


class TestWhatThisDeliberatelyDoesNotDo:
    """Step 3 of #1168 is out of scope, and stays out of scope."""

    def test_nothing_here_drops_privileges(self):
        """Privilege dropping needs a uid on ``SlotKey``, a policy for
        which uid, and an answer for a transport that has no OS principal
        to read.  A guard that quietly grew one would be shipping that
        decision without it being made — so the absence is asserted.
        """
        from jaato_server.server import process_posture

        source = inspect.getsource(process_posture)
        for call in ("setuid(", "setgid(", "seteuid(", "setegid(",
                     "initgroups(", "setresuid("):
            assert call not in source, (
                f"{call} appeared in process_posture.py. Privilege "
                f"dropping is step 3 of #1168 and is deliberately not "
                f"implemented here; it belongs in its own change, with "
                f"the SlotKey uid field and the WS uid policy that make "
                f"it correct."
            )
