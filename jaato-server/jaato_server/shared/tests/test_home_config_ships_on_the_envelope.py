"""``~/.config/jaato`` reaches a runner on the envelope; the runner never probes it.

The SELinux phase 3 kernel run (79a0506a): an isolated sub-runner dropped
to a user lost ``todo`` and ``filesystem_query``. Their loaders asked
``~/.config/jaato/<plugin>.json`` ``exists()``, the isolated domains are
refused even a ``search`` on a user's home (105 AVCs per run), pathlib
re-raises ``EACCES``, and the registry dropped the plugin. As root the
same sub-runner kept both, because ``/root`` is searchable: the boundary
differed by uid.

The daemon now ships the directory in the #1465 user-tier snapshot, as it
ships ``~/.claude/skills``, and the three loaders that read it ask
``user_tier.home_path``.
"""

import pathlib

import pytest

from jaato_server.shared import user_tier
from jaato_server.shared.tests.reversion import Reversion

_UT = "jaato-server/jaato_server/shared/user_tier.py"
_PLUGINS = "jaato-server/jaato_server/shared/plugins"

REVERSIONS = [
    Reversion(
        target=_UT,
        find='    ".config/jaato",\n)\n',
        replace=")\n",
        test="test_the_snapshot_carries_config_jaato",
        because="the daemon would not ship the user's plugin config",
    ),
    Reversion(
        target=f"{_PLUGINS}/todo/config_loader.py",
        find='            user_tier.home_path(".config/jaato/todo.json"),\n',
        replace='            Path.home() / ".config" / "jaato" / "todo.json",\n',
        test="test_todo_never_probes_the_real_home",
        because="a dropped isolated sub-runner would lose the todo plugin",
    ),
    Reversion(
        target=f"{_PLUGINS}/filesystem_query/config_loader.py",
        find='                user_tier.home_path(".config/jaato/filesystem_query.json"),\n',
        replace='                Path.home() / ".config" / "jaato" / "filesystem_query.json",\n',
        test="test_filesystem_query_never_probes_the_real_home",
        because="a dropped isolated sub-runner would lose filesystem_query",
    ),
    Reversion(
        target=f"{_PLUGINS}/references/config_loader.py",
        find='    out.append(str(user_tier.home_path(".config/jaato/references.json")))\n',
        replace='    out.append(str(Path.home() / ".config" / "jaato" / "references.json"))\n',
        test="test_references_never_probes_the_real_home",
        because="references would probe a home directory the runner is denied",
    ),
]


@pytest.fixture(autouse=True)
def _no_installed_snapshot():
    yield
    user_tier.install(None, "/nonexistent", "x")


def _home(tmp_path):
    home = tmp_path / "home"
    (home / ".jaato").mkdir(parents=True)
    cfg = home / ".config" / "jaato"
    cfg.mkdir(parents=True)
    (cfg / "todo.json").write_text("{}")
    (cfg / "antigravity_accounts.json").write_text("{}")
    return home


def test_the_snapshot_carries_config_jaato(tmp_path):
    snap = user_tier.collect(str(_home(tmp_path) / ".jaato"))
    assert snap["@home/.config/jaato/todo.json"] == "{}"


def test_a_credential_in_config_jaato_is_not_shipped(tmp_path):
    snap = user_tier.collect(str(_home(tmp_path) / ".jaato"))
    assert not any(k.endswith("_accounts.json") for k in snap)


@pytest.fixture
def denied_home(tmp_path, monkeypatch):
    """The kernel's refusal: any probe under the real home raises EACCES."""
    home = tmp_path / "real-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    real = pathlib.Path.exists

    def exists(self, *a, **kw):
        if str(self).startswith(str(home)):
            raise PermissionError(13, "Permission denied", str(self))
        return real(self, *a, **kw)

    monkeypatch.setattr(pathlib.Path, "exists", exists)
    user_tier.install({}, str(tmp_path), "s1")
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def test_todo_never_probes_the_real_home(denied_home, monkeypatch):
    from jaato_server.shared.plugins.todo.config_loader import load_config

    monkeypatch.delenv("TODO_CONFIG_PATH", raising=False)
    assert load_config(base_path=str(denied_home)) is not None


def test_filesystem_query_never_probes_the_real_home(denied_home, monkeypatch):
    from jaato_server.shared.plugins.filesystem_query.config_loader import load_config

    monkeypatch.delenv("FILESYSTEM_QUERY_CONFIG_PATH", raising=False)
    assert load_config(base_path=str(denied_home)) is not None


def test_references_never_probes_the_real_home(denied_home, monkeypatch):
    from jaato_server.shared.plugins.references.config_loader import load_config

    monkeypatch.delenv("REFERENCES_CONFIG_PATH", raising=False)
    assert load_config(None, workspace_path=str(denied_home)) is not None
