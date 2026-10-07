"""``session.reload_env`` -- the runner half of refreshing a live session's credentials.

A session applies ``envelope.session_env`` once, at bootstrap, and its
provider resolves a credential once, in ``initialize()``.  A key stored with
``<provider>-auth key`` while the session is open therefore never reaches it
(observed live: a session on a stale daemon-wide ``JAATO_ZHIPUAI_API_KEY``
failing every turn while the workspace held a valid stored key).  This
handler re-applies a freshly resolved env and rebuilds the provider.

Pinned here:

- the env reaches ``os.environ`` as a REPLACEMENT of the previous session
  env (a key the old dict set and the new one lacks is gone), and the
  session's ``_session_env`` accessor sees the new dict;
- the provider is rebuilt AFTER the env is applied, and a rebuild failure
  leaves the env applied and answers ``stage="provider"``;
- a running turn is refused with ``stage="busy"`` and nothing changed;
- no host answers the ``no_host`` shape every session handler shares;
- the dispatcher routes the method (so the lane guard sees it).
"""
from __future__ import annotations

import os
import socket
from typing import Any

import pytest

from jaato_server.server.runner import session as runner_session
from jaato_server.server.runner.envelope import RequestEnvelope
from jaato_server.server.runner.rpc import RunnerRPC
from jaato_server.server.runner.session import RunnerSessionHost
from jaato_server.shared.session_envelope import SessionInitEnvelope


def _make_lone_runner() -> RunnerRPC:
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()
    return RunnerRPC(a, lambda name, args: (False, {"error": "no executor"}))


class _Session:
    """The three things the handler touches on a JaatoSession."""

    def __init__(self, *, running: bool = False, fail: Exception | None = None):
        self.is_running = running
        self._session_env: dict = {}
        self.reloads = 0
        self._fail = fail
        self.env_seen_at_reload: dict | None = None

    def reload_provider(self):
        self.reloads += 1
        # what the provider's initialize() would read
        self.env_seen_at_reload = {
            k: v for k, v in os.environ.items() if k.startswith("JAATO_ZHIPUAI")
        }
        if self._fail:
            raise self._fail
        return {"provider": "zhipuai", "model": "glm-5",
                "auth_info": "API key from /ws/.jaato/zhipuai_auth.json"}


def _rpc_with(session: Any) -> RunnerRPC:
    rpc = _make_lone_runner()
    rpc._session_host = RunnerSessionHost(
        envelope=SessionInitEnvelope(
            session_id="s-reload", workspace_path="/ws", profile_name="p",
            provider_name="zhipuai", model_name="glm-5", plugins=[],
        ),
        runtime=None, session=session,
    )
    return rpc


@pytest.fixture
def clean_environ(monkeypatch):
    """Isolate the slot's pristine-environment snapshot and os.environ."""
    monkeypatch.setattr(runner_session, "_PRISTINE_ENVIRON", None)
    for key in list(os.environ):
        if key.startswith("JAATO_ZHIPUAI") or key == "RELOAD_ONLY":
            monkeypatch.delenv(key, raising=False)
    yield
    monkeypatch.setattr(runner_session, "_PRISTINE_ENVIRON", None)


def test_reload_replaces_the_session_env_then_rebuilds_the_provider(clean_environ):
    session = _Session()
    rpc = _rpc_with(session)
    # bootstrap-time env: the key the session is stuck on
    runner_session.apply_session_env({"JAATO_ZHIPUAI_API_KEY": "stale", "RELOAD_ONLY": "old"})

    ok, result = rpc._handle_session_reload_env(
        {"session_env": {"JAATO_ZHIPUAI_API_KEY": "fresh"}})

    assert ok is True
    assert result["applied"] == 1
    assert result["provider"] == "zhipuai" and result["model"] == "glm-5"
    assert result["auth_info"].startswith("API key from")
    # a replacement, not a merge: the key the new dict does not name is gone
    assert os.environ["JAATO_ZHIPUAI_API_KEY"] == "fresh"
    assert "RELOAD_ONLY" not in os.environ
    # the provider was rebuilt AFTER the env landed
    assert session.reloads == 1
    assert session.env_seen_at_reload == {"JAATO_ZHIPUAI_API_KEY": "fresh"}
    assert session._session_env == {"JAATO_ZHIPUAI_API_KEY": "fresh"}


def test_a_running_turn_is_refused_and_nothing_changes(clean_environ):
    session = _Session(running=True)
    rpc = _rpc_with(session)
    runner_session.apply_session_env({"JAATO_ZHIPUAI_API_KEY": "stale"})

    ok, result = rpc._handle_session_reload_env(
        {"session_env": {"JAATO_ZHIPUAI_API_KEY": "fresh"}})

    assert ok is False and result["stage"] == "busy"
    assert os.environ["JAATO_ZHIPUAI_API_KEY"] == "stale"
    assert session.reloads == 0


def test_a_provider_that_will_not_rebuild_leaves_the_env_applied(clean_environ):
    session = _Session(fail=RuntimeError("No Zhipu AI API key found"))
    rpc = _rpc_with(session)

    ok, result = rpc._handle_session_reload_env(
        {"session_env": {"JAATO_ZHIPUAI_API_KEY": "fresh"}})

    assert ok is False and result["stage"] == "provider"
    assert "No Zhipu AI API key found" in result["error"]
    assert result["applied"] == 1
    # the next turn's lazy creation sees the new environment regardless
    assert os.environ["JAATO_ZHIPUAI_API_KEY"] == "fresh"


def test_no_host_is_the_shared_refusal(clean_environ):
    rpc = _make_lone_runner()
    ok, result = rpc._handle_session_reload_env({"session_env": {"A": "b"}})
    assert ok is False and result["stage"] == "no_host"


def test_the_dispatcher_routes_the_method(clean_environ):
    session = _Session()
    rpc = _rpc_with(session)
    env = RequestEnvelope(id=7, method="session.reload_env",
                          args={"session_env": {"JAATO_ZHIPUAI_API_KEY": "fresh"}})
    ok, result = rpc._dispatch_method(env)
    assert ok is True and result["applied"] == 1
    assert session.reloads == 1
