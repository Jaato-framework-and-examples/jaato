"""``--pool-size`` / ``--pool-max`` size a daemon the command starts.

The flags shipped with protocol 1.35 as a RESIZE of a running daemon, and a
start command carrying them did not start anything: ``_exit_on_pool_flags``
ran before the start path, sent the request, and exited -- with "could not
resize" when no daemon was up.  The only way to start with N warm runners
was ``JAATO_RUNNER_POOL_SIZE``.  What each test here holds:

* a command that starts a daemon (``--daemon``, ``--web-socket``,
  ``--restart``) takes the flags as its starting sizes and sends no resize;
* any other command still resizes the running daemon;
* each flag outranks its env knob and leaves the other knob alone, so
  ``--pool-size 6`` keeps a ``JAATO_RUNNER_POOL_MAX_SIZE`` the operator set;
* a flag on ``--restart`` outranks the size a runtime resize recorded;
* the started daemon records the sizes, so ``--restart`` keeps them;
* a negative size is refused before anything is sent or started.
"""

from __future__ import annotations

import argparse
import json

import pytest

from jaato_server.server import __main__ as main_mod
from jaato_server.shared.tests.reversion import Reversion

_MAIN = "jaato-server/jaato_server/server/__main__.py"


def _args(**over) -> argparse.Namespace:
    base = dict(pool_size=None, pool_max=None, daemon=False,
                web_socket=None, restart=False, ipc_socket=None)
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv("JAATO_RUNNER_POOL_SIZE", raising=False)
    monkeypatch.delenv("JAATO_RUNNER_POOL_MAX_SIZE", raising=False)


@pytest.fixture
def no_resize(monkeypatch):
    """Fail the test if anything asks a running daemon to resize."""
    async def _refuse(*_a, **_k):
        raise AssertionError("a start command sent a pool resize")
    monkeypatch.setattr(main_mod, "_pool_request", _refuse)


@pytest.mark.parametrize("start", [
    {"daemon": True}, {"web_socket": ":8080"}, {"restart": True},
])
def test_a_start_command_takes_the_flags_and_resizes_nothing(
        start, no_resize) -> None:
    # Returns (no SystemExit): the start path goes on with these sizes.
    main_mod._exit_on_pool_flags(_args(pool_size=6, **start))


def test_a_command_that_starts_nothing_still_resizes(monkeypatch) -> None:
    sent = []

    async def _fake(socket_path, target, ceiling, timeout=10.0):
        sent.append((socket_path, target, ceiling))
        from jaato_sdk.events import PoolStatusEvent
        return PoolStatusEvent(target_size=target, max_size=12)
    monkeypatch.setattr(main_mod, "_pool_request", _fake)
    with pytest.raises(SystemExit) as exc:
        main_mod._exit_on_pool_flags(
            _args(pool_size=6, ipc_socket="/tmp/x.sock"))
    assert exc.value.code == 0
    assert sent == [("/tmp/x.sock", 6, None)]


@pytest.mark.parametrize("flags", [{"pool_size": -1}, {"pool_max": -3}])
def test_a_negative_size_is_refused_before_anything(flags, no_resize) -> None:
    with pytest.raises(SystemExit) as exc:
        main_mod._exit_on_pool_flags(_args(daemon=True, **flags))
    assert exc.value.code == 2


def test_no_flags_leave_the_env_to_the_daemon(no_env) -> None:
    assert main_mod._startup_pool_sizes(_args(daemon=True)) == (None, None)


def test_each_flag_outranks_only_its_own_env_knob(monkeypatch) -> None:
    monkeypatch.setenv("JAATO_RUNNER_POOL_SIZE", "3")
    monkeypatch.setenv("JAATO_RUNNER_POOL_MAX_SIZE", "12")
    assert main_mod._startup_pool_sizes(
        _args(daemon=True, pool_size=6)) == (6, 12)
    assert main_mod._startup_pool_sizes(
        _args(daemon=True, pool_max=20)) == (3, 20)
    assert main_mod._startup_pool_sizes(
        _args(daemon=True, pool_size=6, pool_max=8)) == (6, 8)


def test_a_flag_on_restart_outranks_the_recorded_resize(no_env) -> None:
    args = _args(restart=True, pool_size=7)
    args.restart_pool_size, args.restart_pool_max_size = 4, 9
    assert main_mod._startup_pool_sizes(args) == (7, 9)
    args = _args(restart=True)
    args.restart_pool_size, args.restart_pool_max_size = 4, 9
    assert main_mod._startup_pool_sizes(args) == (4, 9)


def test_the_started_daemon_has_the_sizes_and_restart_keeps_them(
        tmp_path, no_env) -> None:
    size, ceiling = main_mod._startup_pool_sizes(
        _args(daemon=True, pool_size=6))
    config = tmp_path / "config.json"
    daemon = main_mod.JaatoDaemon(config_file=str(config),
                                  pid_file=str(tmp_path / "pid"),
                                  pool_size=size, pool_max_size=ceiling)
    assert (daemon._pool_manager.target_size,
            daemon._pool_manager.max_size) == (6, 12)
    assert daemon._write_config() is True
    saved = json.loads(config.read_text())
    assert (saved["pool_size"], saved["pool_max_size"]) == (6, None)


def test_env_pool_sizes_reads_and_defaults(monkeypatch) -> None:
    monkeypatch.setenv("JAATO_RUNNER_POOL_SIZE", "not-a-number")
    monkeypatch.setenv("JAATO_RUNNER_POOL_MAX_SIZE", "")
    assert main_mod.env_pool_sizes() == (2, None)
    monkeypatch.setenv("JAATO_RUNNER_POOL_SIZE", "5")
    monkeypatch.setenv("JAATO_RUNNER_POOL_MAX_SIZE", "11")
    assert main_mod.env_pool_sizes() == (5, 11)


REVERSIONS = [
    Reversion(
        target=_MAIN,
        find="    return bool(args.daemon or args.web_socket or args.restart)",
        replace="    return False",
        test="test_a_start_command_takes_the_flags_and_resizes_nothing",
        because=("a start command carrying --pool-size resizing a daemon "
                 "that does not exist yet, then exiting without starting"),
    ),
    Reversion(
        target=_MAIN,
        find="""    if args.pool_size is not None:
        size = args.pool_size
    if args.pool_max is not None:""",
        replace="""    if args.pool_max is not None:""",
        test="test_each_flag_outranks_only_its_own_env_knob",
        because="--pool-size on a start command ignored in favour of the env",
    ),
    Reversion(
        target=_MAIN,
        find="        size, ceiling = env_pool_sizes()\n    if args.pool_size",
        replace="        size, ceiling = env_pool_sizes()[0], None\n    if args.pool_size",
        test="test_each_flag_outranks_only_its_own_env_knob",
        because=("--pool-size silently replacing the operator's "
                 "JAATO_RUNNER_POOL_MAX_SIZE with a derived ceiling"),
    ),
]
