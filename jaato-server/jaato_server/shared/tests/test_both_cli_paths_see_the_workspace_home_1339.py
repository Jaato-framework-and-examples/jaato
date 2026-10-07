"""Both cli execution paths give a command the same environment (#1339).

``CLIToolPlugin`` runs a command through one of two paths:

- ``_execute_streaming``, the auto-background path, hands ``Popen`` the whole
  environment ``_build_subprocess_env`` built;
- ``_execute``, the synchronous path, hands ``run_command`` an ``extra_env``
  that is merged over ``os.environ``.

``extra_env`` used to be a hand-kept list: ``PATH``, ``VIRTUAL_ENV`` and
``PYTHONPATH``.  It predated #1225, which added ``HOME`` and the four
``XDG_*`` keys to the builder, so on the synchronous path a daemon-managed
workspace's command ran with the daemon's own HOME.  #1338 made that visible,
by running every call through both paths at once.

The list is now derived from what the builder changed, so these tests run
one command through both paths and compare what it saw.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin
from jaato_server.shared.tests.reversion import Reversion

_CLI = "jaato-server/jaato_server/shared/plugins/cli/plugin.py"

REVERSIONS = [
    Reversion(
        target=_CLI,
        find="            extra_env = _env_changes(env)\n",
        replace=(
            "            extra_env = {\n"
            "                key: env[key]\n"
            "                for key in ('PATH', 'VIRTUAL_ENV', 'PYTHONPATH')\n"
            "                if key in env\n"
            "            } or None\n"
        ),
        test="test_the_synchronous_path_gets_the_workspace_home",
        because="the synchronous path forwards a hand-kept list that has no HOME",
    ),
    Reversion(
        target=_CLI,
        find=(
            "        key: value for key, value in env.items()\n"
            "        if os.environ.get(key) != value\n"
        ),
        replace="        key: value for key, value in {**os.environ, **env}.items()\n",
        test="test_the_synchronous_path_does_not_restore_a_scrubbed_secret",
        because="a key the builder scrubbed is carried back in from os.environ",
    ),
]

_PROBE = (
    'printf \'{"HOME":"%s","XDG_CONFIG_HOME":"%s","SECRET":"%s"}\' '
    '"$HOME" "$XDG_CONFIG_HOME" "$JAATO_1339_API_KEY"'
)


@pytest.fixture
def cli(tmp_path: Path):
    plugin = CLIToolPlugin()
    plugin.initialize({
        "workspace_root": str(tmp_path),
        "workspace_home": ".home",
        "scrub_secret_env": "default",
    })
    yield plugin
    plugin.shutdown()


def _sync(cli: CLIToolPlugin) -> Dict[str, str]:
    result = cli._execute({"command": _PROBE})
    assert result.get("returncode") == 0, result
    return json.loads(result["stdout"])


def _streaming(cli: CLIToolPlugin) -> Dict[str, str]:
    out: List[bytes] = []
    result = cli._execute_streaming(
        {"command": _PROBE},
        on_stdout=out.append,
        on_stderr=lambda _b: None,
        on_returncode=lambda _c: None,
    )
    assert result.get("returncode") == 0, result
    return json.loads(b"".join(out).decode())


def test_the_synchronous_path_gets_the_workspace_home(cli, tmp_path):
    seen = _sync(cli)
    assert seen["HOME"] == str(tmp_path / ".home")
    assert seen["XDG_CONFIG_HOME"] == str(tmp_path / ".home" / ".config")


def test_both_paths_see_the_same_environment(cli):
    assert _sync(cli) == _streaming(cli)


def test_the_synchronous_path_does_not_restore_a_scrubbed_secret(cli, tmp_path, monkeypatch):
    """``extra_env`` carries the builder's changes and never a scrubbed key.

    ``run_command`` rebuilds from ``os.environ`` and re-scrubs, so the child
    is safe either way; what this pins is that ``extra_env`` itself does not
    carry a secret the builder removed back in.
    """
    import jaato_server.shared.plugins.cli.plugin as cli_plugin
    from jaato_server.shared.subprocess_runner import run_command as real_run

    monkeypatch.setenv("JAATO_1339_API_KEY", "sk-not-for-the-shell")
    seen: Dict[str, Any] = {}

    def spy(command, **kwargs):
        seen.update(kwargs.get("extra_env") or {})
        return real_run(command, **kwargs)

    monkeypatch.setattr(cli_plugin, "run_command", spy)
    assert _sync(cli)["SECRET"] == ""
    assert "JAATO_1339_API_KEY" not in seen
    assert seen.get("HOME") == str(tmp_path / ".home")
