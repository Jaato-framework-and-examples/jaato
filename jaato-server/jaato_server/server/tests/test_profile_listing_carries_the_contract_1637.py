"""``session.profiles`` carries a profile's input contract and permission default (#1637).

``jaato-mcp`` publishes each profile as an MCP tool from the listing, before
any session exists.  The listing carried ``completion_payload_schema`` (the
output contract) and nothing about the input: every tool published
``{"prompt": string}``.  And ``plugin_configs.permission.policy`` is only
layer 4 of the effective policy (#1474), so a client could not tell that a
workspace ``permissions.json`` made a profile ``ask``.

Pinned here against the real router, manager, discovery, spawn-schema
lookup and permission layering:

- a path ``spawn_payload_schema`` is listed RESOLVED, through the tier the
  daemon validates ``agent_params`` against;
- a declared schema that resolves nowhere is an error, not ``None``;
- the permission default is the merged one, and says which layer won;
- an unreadable ``permissions.json`` is an error, and names no path.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jaato_sdk.events import CommandRequest, SessionProfilesEvent
from jaato_server.server.command_router import CommandRouter
from jaato_server.server.session_manager import SessionManager
from jaato_server.shared.tests.reversion import Reversion

_SESSION_MANAGER = "jaato-server/jaato_server/server/session_manager.py"
_LOADER = "jaato-server/jaato_server/shared/spawn_schema_loader.py"
_LAYERS = "jaato-server/jaato_server/shared/plugins/permission/policy_layers.py"

_SCHEMA = {
    "type": "object",
    "properties": {"ticket": {"type": "string", "pattern": "^T-[0-9]+$"}},
    "required": ["ticket"],
}


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


def _write_profile(ws: Path, name: str, body: str) -> None:
    profiles = ws / ".jaato" / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / f"{name}.yaml").write_text(
        f"name: {name}\ndescription: {name}\nplugins: [cli]\nmodel: m\n{body}")


@pytest.fixture()
def daemon(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PERMISSION_CONFIG_PATH", raising=False)
    ws = tmp_path / "ws"
    schemas = ws / ".jaato" / "spawn_schemas"
    schemas.mkdir(parents=True)
    (schemas / "ticket.json").write_text(json.dumps(_SCHEMA))
    _write_profile(ws, "by_path",
                   "spawn_payload_schema: spawn_schemas/ticket.json\n")
    _write_profile(ws, "missing", "spawn_payload_schema: spawn_schemas/nope.json\n")
    _write_profile(ws, "plain", "")
    _write_profile(ws, "profile_allows", (
        "plugin_configs:\n  permission:\n    policy:\n"
        "      defaultPolicy: allow\n"))
    sm = SessionManager()
    sink = _Sink()
    router = CommandRouter(session_manager=sm, event_sink=sink, daemon_plugins={})
    sm._client_config["c1"] = {"working_dir": str(ws)}
    return SimpleNamespace(sm=sm, sink=sink, router=router, ws=ws, home=home)


def _listed(d):
    d.sink.sent.clear()
    d.router.handle_request("c1", None, CommandRequest(command="session.profiles"))
    [event] = [e for e in d.sink.sent if isinstance(e, SessionProfilesEvent)]
    # Round-trip through the wire shape, as a client reads it.
    event = SessionProfilesEvent.model_validate_json(event.model_dump_json())
    return {p.name: p for p in event.profiles}


def test_a_path_spawn_schema_is_listed_resolved(daemon):
    listed = _listed(daemon)
    assert listed["by_path"].spawn_payload_schema == _SCHEMA
    assert listed["by_path"].spawn_payload_schema_error is None
    assert listed["plain"].spawn_payload_schema is None
    assert listed["plain"].spawn_payload_schema_error is None


def test_a_schema_that_resolves_nowhere_is_an_error_not_none(daemon):
    entry = _listed(daemon)["missing"]
    assert entry.spawn_payload_schema is None
    assert entry.spawn_payload_schema_error == "not found in any tier"


def test_a_workspace_file_decides_the_listed_default(daemon):
    listed = _listed(daemon)
    assert listed["plain"].permission_default.model_dump() == {
        "policy": "ask", "source": "framework"}
    (daemon.ws / ".jaato" / "permissions.json").write_text(
        json.dumps({"version": "1.0", "defaultPolicy": "deny"}))
    listed = _listed(daemon)
    assert listed["plain"].permission_default.model_dump() == {
        "policy": "deny", "source": "workspace"}
    # The profile's own policy still outranks the file.
    assert listed["profile_allows"].permission_default.model_dump() == {
        "policy": "allow", "source": "profile"}


def test_an_unreadable_permissions_file_is_an_error_naming_no_path(daemon):
    (daemon.ws / ".jaato" / "permissions.json").write_text("{not json")
    entry = _listed(daemon)["plain"]
    assert entry.permission_default is None
    assert entry.permission_default_error
    assert str(daemon.ws) not in entry.permission_default_error


REVERSIONS = [
    Reversion(
        target=_SESSION_MANAGER,
        find="                spawn_payload_schema=spawn_schema,\n",
        replace="                spawn_payload_schema=None,\n",
        test="test_a_path_spawn_schema_is_listed_resolved",
        because="the listing not carrying the resolved input contract",
    ),
    Reversion(
        target=_LOADER,
        find="        return None, SPAWN_SCHEMA_NOT_FOUND\n",
        replace="        return None, None\n",
        test="test_a_schema_that_resolves_nowhere_is_an_error_not_none",
        because="a missing declared schema reading as 'declares none'",
    ),
    Reversion(
        target=_LAYERS,
        find="            workspace_path=workspace_path,\n            config_root=config_root,\n            config_path=config_path,\n        )\n    except",
        replace="            workspace_path=None,\n            config_root=None,\n            config_path=config_path,\n        )\n    except",
        test="test_a_workspace_file_decides_the_listed_default",
        because="the listed default ignoring the workspace permissions.json",
    ),
]
