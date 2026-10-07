"""``session.profiles`` lists the CLIENT's profiles, never a busy neighbour's (#1593).

Reported from a shared daemon: an IPC client declared
``working_dir=/home/kbwiki/mcp/template`` and asked for its profiles five
times, and five times got the 43 profiles of ANOTHER workspace (plus that
workspace's ``minimax_m3`` set) and not its own ``librarian_mcp``.  That
other workspace's sessions were mid-turn.

``CommandRouter`` called ``list_profiles(workspace_path=...)`` with no
config root and no env file, so ``discover_profiles`` resolved both from
``get_config_root()`` / ``get_session_env()``.  With no ContextVar on the
handler's thread, those fall back to ``os.environ``, which
``JaatoServer._in_workspace`` / ``_with_session_env`` overlay
process-wide during another session's turn.

Pinned here against the real router, manager and discovery, with that
overlay simulated in ``os.environ``:

- the client's own profiles are listed, the busy workspace's are not;
- a client that named no workspace is not shown the busy one either;
- the profile set is the one the client's own ``.env`` selects, and a
  set selected only by the overlay is not scanned.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jaato_sdk.events import SessionProfilesEvent
from jaato_server.server.command_router import CommandRouter
from jaato_server.server.session_manager import SessionManager
from jaato_server.shared.tests.reversion import Reversion

_COMMAND_ROUTER = "jaato-server/jaato_server/server/command_router.py"
_CONFIG = "jaato-server/jaato_server/shared/plugins/subagent/config.py"


class _Sink:
    def __init__(self):
        self.sent = []

    def send_event(self, client_id, event):
        self.sent.append(event)

    def get_client_user(self, client_id):
        return None

    def get_client_workspace(self, client_id):
        return None

    def visible_workspace_paths(self, client_id):
        return None


def _profile(directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.yaml").write_text(
        f"name: {name}\ndescription: {name}\nplugins: [cli]\nmodel: m\n")


@pytest.fixture()
def daemon(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    mine = tmp_path / "template"
    busy = tmp_path / "extractor"
    _profile(mine / ".jaato" / "profiles", "librarian_mcp")
    _profile(mine / ".jaato" / "profiles" / "mine_set", "librarian_set_only")
    _profile(busy / ".jaato" / "profiles", "extractor_stage")
    _profile(busy / ".jaato" / "profiles" / "minimax_m3", "extractor_m3")
    _profile(mine / ".jaato" / "profiles" / "minimax_m3", "leaked_set_only")
    # What another session's turn leaves in the process environment.
    monkeypatch.setenv("JAATO_CONFIG_ROOT", str(busy / ".jaato"))
    monkeypatch.setenv("JAATO_WORKSPACE_ROOT", str(busy))
    monkeypatch.setenv("JAATO_PROFILE_SET", "minimax_m3")
    sm = SessionManager()
    sink = _Sink()
    router = CommandRouter(session_manager=sm, event_sink=sink, daemon_plugins={})
    return SimpleNamespace(sm=sm, sink=sink, router=router, mine=mine, busy=busy)


def _listed(d, client_id="c1"):
    from jaato_sdk.events import CommandRequest
    d.sink.sent.clear()
    d.router.handle_request(client_id, None, CommandRequest(command="session.profiles"))
    [event] = [e for e in d.sink.sent if isinstance(e, SessionProfilesEvent)]
    return {p.name for p in event.profiles}


def test_the_client_sees_its_own_profiles_not_the_busy_workspaces(daemon):
    daemon.sm._client_config["c1"] = {
        "working_dir": str(daemon.mine),
        "config_root": str(daemon.mine / ".jaato"),
    }
    names = _listed(daemon)
    assert "librarian_mcp" in names
    assert "extractor_stage" not in names


def test_a_working_dir_alone_is_enough(daemon):
    daemon.sm._client_config["c1"] = {"working_dir": str(daemon.mine)}
    names = _listed(daemon)
    assert "librarian_mcp" in names
    assert "extractor_stage" not in names


def test_a_client_that_named_no_workspace_is_not_shown_the_busy_one(
        daemon, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = _listed(daemon)
    assert "extractor_stage" not in names
    assert "extractor_m3" not in names


def test_the_profile_set_is_the_clients_own_env_files(daemon):
    daemon.sm._client_config["c1"] = {"working_dir": str(daemon.mine)}
    assert "leaked_set_only" not in _listed(daemon)
    (daemon.mine / ".env").write_text("JAATO_PROFILE_SET=mine_set\n")
    names = _listed(daemon)
    assert "librarian_set_only" in names
    assert "leaked_set_only" not in names


REVERSIONS = [
    Reversion(
        target=_COMMAND_ROUTER,
        find="                    **self._profile_listing_scope(client_id, workspace_path),\n",
        replace="                    workspace_path=workspace_path,\n",
        test="test_the_client_sees_its_own_profiles_not_the_busy_workspaces",
        because="the router naming no config root, so discovery read the busy one",
    ),
    Reversion(
        target=_CONFIG,
        find="        return (base_path if base_path is not None else os.getcwd(), None,\n",
        replace="        return (base_path if base_path is not None else os.getcwd(), get_config_root(),\n",
        test="test_a_client_that_named_no_workspace_is_not_shown_the_busy_one",
        because="discovery filling an omitted root from the overlaid os.environ",
    ),
    Reversion(
        target=_CONFIG,
        find="                session_env.get(PROFILE_SET_ENV_VAR))\n",
        replace="                os.environ.get(PROFILE_SET_ENV_VAR))\n",
        test="test_the_profile_set_is_the_clients_own_env_files",
        because="the profile set read from another session's env overlay",
    ),
]
