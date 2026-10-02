"""``jaato-scaffold explain pool`` describes the pool the daemon runs.

Protocol 1.35 made the runner pool resizable on a running daemon
(``--pool-size``, ``pool.resize``, ``IPCClient.resize_pool``), and no
``explain`` topic said so: an operator found the flags only in
``--help`` and the verbs not at all.

The topic READS ``server/pool_admin.py`` and the SDK floor rather than
restating them, and these tests hold the parts that could still drift:
the env knobs it names are the ones the daemon reads, the CLI flags are
the ones argparse accepts, every refusal the admin can produce is one
the topic lists, and the sizing counters exist.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

from jaato_sdk.client.ipc import IPCClient
from jaato_server.server import pool_admin as pa
from jaato_server.server.runner_pool import PoolManager
from jaato_server.shared.scaffold import explain
from jaato_server.shared.scaffold.introspection_verbs import _SCOPES
from jaato_server.shared.tests.reversion import Reversion

_SERVER = pathlib.Path(pa.__file__).resolve().parent

REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/introspection_verbs.py",
        find='    "pool": ExplainScope(_explain.pool,\n',
        replace='    "pool-gone": ExplainScope(_explain.pool,\n',
        test="test_the_topic_is_registered",
        because="the topic leaves the explain dispatch, so resizing the "
                "pool is documented nowhere an operator looks",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/__main__.py",
        find='os.environ.get("JAATO_RUNNER_POOL_MAX_SIZE", "")',
        replace='os.environ.get("JAATO_RUNNER_POOL_CEILING", "")',
        test="test_every_startup_knob_is_one_the_daemon_reads",
        because="the daemon reads a ceiling the topic does not name, so an "
                "operator sets a variable nothing reads",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/pool_admin.py",
        find="    (NO_POOL, ",
        replace="    ('no-pool-listed', ",
        test="test_every_refusal_the_admin_gives_is_listed",
        because="the admin answers a category the topic does not explain",
    ),
]


def test_the_topic_is_registered():
    assert "pool" in _SCOPES


def _env_literals_read(path: pathlib.Path) -> set:
    """Env names read as a string literal (``os.environ.get`` / ``getenv``)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = ast.unparse(node.func)
        if func in ("os.environ.get", "os.getenv") and isinstance(
                node.args[0], ast.Constant):
            found.add(node.args[0].value)
    return found


def test_every_startup_knob_is_one_the_daemon_reads():
    read = (_env_literals_read(_SERVER / "__main__.py")
            | _env_literals_read(_SERVER / "runner_spawn.py"))
    data, _text = explain.pool()
    for knob in data["startup_knobs"]:
        assert knob["env"] in read, knob["env"]


def test_the_cli_flags_are_the_ones_argparse_accepts():
    out = subprocess.run(
        [sys.executable, "-m", "jaato_server", "--help"],
        capture_output=True, text=True, timeout=60,
    ).stdout
    _data, text = explain.pool()
    for flag in (pa.CLI_SIZE_FLAG, pa.CLI_MAX_FLAG):
        assert flag in out, flag
        assert flag in text, flag


def test_every_refusal_the_admin_gives_is_listed():
    data, _text = explain.pool()
    own = SimpleNamespace(uid=os.getuid(), identity="me")
    pool = PoolManager(None, target_size=0)
    answers = [
        pa.PoolAdmin(pool, routing_enabled=lambda: True).answer(None),
        pa.PoolAdmin(None, routing_enabled=lambda: True).answer(own),
        pa.PoolAdmin(pool, routing_enabled=lambda: True).answer(
            own, target_size=-1),
    ]
    categories = {a.category for a in answers}
    assert categories == {pa.NOT_AUTHORIZED, pa.NO_POOL, pa.INVALID_REQUEST}
    assert categories <= set(data["refusals"])


def test_the_sdk_floor_and_verbs_are_the_live_ones():
    data, text = explain.pool()
    assert data["resize"]["min_protocol"] == IPCClient.MIN_POOL_ADMIN_PROTOCOL
    assert pa.VERB_STATUS in text and pa.VERB_RESIZE in text
    assert set(data["sizing_signals"]) <= set(
        PoolManager(None, target_size=0).get_telemetry())
