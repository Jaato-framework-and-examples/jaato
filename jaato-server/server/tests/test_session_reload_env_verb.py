"""``session.reload_env`` on the daemon: the router verb, the manager, and the auth hook.

The runner half is covered by ``server/runner/tests/test_session_reload_env.py``;
here the collaborators are mocked and what is pinned is the daemon's own
contract:

- the verb defaults to the CALLER's session and takes an explicit id;
- each of the four outcomes is one confirmation line an operator can act on
  (rebuilt + credential source / mid-turn / not loaded / provider failure);
- ``SessionManager.reload_session_env`` refuses a mid-turn session BEFORE
  paying the runner RPC, and reports rather than raises;
- ``JaatoServer.reload_session_env`` drops the once-only resolution flag,
  resolves again and pushes the WHOLE dict to the runner;
- after ``<provider>-auth key|login`` the router reloads the caller's live
  session only when it runs that plugin's provider, and never on ``status``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from jaato_sdk.events import ErrorEvent, SystemMessageEvent

from server.command_router import CommandRouter
from server.session_manager import SessionManager


def _make_router():
    session_manager = MagicMock()
    event_sink = MagicMock()
    event_sink.get_client_workspace.return_value = "/ws"
    router = CommandRouter.__new__(CommandRouter)
    router._session_manager = session_manager
    router._event_sink = event_sink
    return router, session_manager, event_sink


def _sent(event_sink):
    return [call[0][1] for call in event_sink.send_event.call_args_list]


def _outcome(**over):
    base = {"session_id": "s1", "found": True, "ok": True, "was_processing": False,
            "error": None, "result": {"applied": 3, "provider": "zhipuai",
                                      "model": "glm-5",
                                      "auth_info": "API key from /ws/.jaato/zhipuai_auth.json"}}
    base.update(over)
    return base


class TestRouterVerb:
    def test_defaults_to_the_callers_session_and_names_the_credential_source(self):
        router, sm, sink = _make_router()
        sm.reload_session_env.return_value = _outcome()
        router._handle_session_reload_env("c1", "s1", [])
        sm.reload_session_env.assert_called_once_with("s1")
        msg = _sent(sink)[0]
        assert isinstance(msg, SystemMessageEvent)
        assert "zhipuai / glm-5 rebuilt" in msg.message
        assert "zhipuai_auth.json" in msg.message

    def test_an_explicit_id_targets_that_session(self):
        router, sm, sink = _make_router()
        sm.reload_session_env.return_value = _outcome(session_id="other")
        router._handle_session_reload_env("c1", "s1", ["other"])
        sm.reload_session_env.assert_called_once_with("other")

    def test_no_session_is_a_usage_error(self):
        router, sm, sink = _make_router()
        router._handle_session_reload_env("c1", None, [])
        assert isinstance(_sent(sink)[0], ErrorEvent)
        sm.reload_session_env.assert_not_called()

    def test_mid_turn_says_retry_when_idle(self):
        router, sm, sink = _make_router()
        sm.reload_session_env.return_value = _outcome(ok=False, was_processing=True, result=None)
        router._handle_session_reload_env("c1", "s1", [])
        assert "mid-turn" in _sent(sink)[0].message

    def test_not_loaded_says_so(self):
        router, sm, sink = _make_router()
        sm.reload_session_env.return_value = _outcome(found=False, ok=False, result=None)
        router._handle_session_reload_env("c1", "s1", [])
        assert "not loaded" in _sent(sink)[0].message

    def test_provider_failure_carries_the_reason(self):
        router, sm, sink = _make_router()
        sm.reload_session_env.return_value = _outcome(
            ok=False, error="provider rebuild failed: No Zhipu AI API key found", result=None)
        router._handle_session_reload_env("c1", "s1", [])
        assert "No Zhipu AI API key found" in _sent(sink)[0].message

    def test_rides_the_shared_admin_branch(self):
        router, sm, sink = _make_router()
        sm.reload_session_env.return_value = _outcome()
        router._dispatch_orphan_command("session.reload_env", "c1", [], session_id="s1")
        sm.reload_session_env.assert_called_once_with("s1")


class TestPostAuthHook:
    def _plugin(self, provider="zhipuai"):
        plugin = MagicMock()
        plugin.provider_name = provider
        return plugin

    def _live_session(self, provider="zhipuai"):
        session = MagicMock()
        session.session_id = "s1"
        session.server.model_provider = provider
        return session

    def test_key_on_the_live_sessions_provider_reloads_it(self):
        router, sm, sink = _make_router()
        sm.get_client_session.return_value = self._live_session()
        sm.reload_session_env.return_value = _outcome()
        router._maybe_reload_live_session_after_auth("c1", self._plugin(), ["key", "sk-x"])
        sm.reload_session_env.assert_called_once_with("s1")

    def test_status_never_reloads(self):
        router, sm, sink = _make_router()
        sm.get_client_session.return_value = self._live_session()
        router._maybe_reload_live_session_after_auth("c1", self._plugin(), ["status"])
        sm.reload_session_env.assert_not_called()

    def test_another_providers_session_is_left_alone(self):
        router, sm, sink = _make_router()
        sm.get_client_session.return_value = self._live_session(provider="anthropic")
        router._maybe_reload_live_session_after_auth("c1", self._plugin(), ["login"])
        sm.reload_session_env.assert_not_called()

    def test_no_live_session_is_a_no_op(self):
        router, sm, sink = _make_router()
        sm.get_client_session.return_value = None
        router._maybe_reload_live_session_after_auth("c1", self._plugin(), ["key", "sk-x"])
        sm.reload_session_env.assert_not_called()


class TestSessionManager:
    def _manager_with(self, server):
        manager = SessionManager.__new__(SessionManager)
        import threading
        manager._lock = threading.RLock()
        session = MagicMock()
        session.server = server
        manager._sessions = {"s1": session}
        return manager

    def test_refuses_a_mid_turn_session_before_the_rpc(self):
        server = MagicMock(); server.is_processing = True
        outcome = self._manager_with(server).reload_session_env("s1")
        assert outcome["found"] and outcome["was_processing"] and not outcome["ok"]
        server.reload_session_env.assert_not_called()

    def test_reports_the_runners_answer(self):
        server = MagicMock(); server.is_processing = False
        server.reload_session_env.return_value = {"applied": 2, "provider": "zhipuai",
                                                  "model": "glm-5", "auth_info": "x"}
        outcome = self._manager_with(server).reload_session_env("s1")
        assert outcome["ok"] and outcome["result"]["applied"] == 2

    def test_a_raise_is_reported_not_propagated(self):
        server = MagicMock(); server.is_processing = False
        server.reload_session_env.side_effect = RuntimeError("stage=busy")
        outcome = self._manager_with(server).reload_session_env("s1")
        assert not outcome["ok"] and "stage=busy" in outcome["error"]

    def test_unknown_session_is_not_found(self):
        outcome = self._manager_with(MagicMock()).reload_session_env("nope")
        assert outcome["found"] is False


class TestJaatoServerReload:
    def test_drops_the_once_only_flag_and_pushes_the_whole_dict(self):
        from server.core import JaatoServer
        server = JaatoServer.__new__(JaatoServer)
        server._session_env_resolved = True
        server._session_env = {"OLD": "1"}
        calls = []

        def _resolve():
            calls.append(server._session_env_resolved)
            server._session_env = {"JAATO_ZHIPUAI_API_KEY": "fresh", "MODEL_NAME": "glm-5"}
            server._session_env_resolved = True
        server._resolve_session_env = _resolve
        rpc = MagicMock()
        rpc.session_reload_env_threadsafe.return_value = {"applied": 2}
        server._runner_rpc = rpc

        result = server.reload_session_env()

        assert calls == [False], "the flag must be dropped before re-resolving"
        rpc.session_reload_env_threadsafe.assert_called_once_with(
            {"JAATO_ZHIPUAI_API_KEY": "fresh", "MODEL_NAME": "glm-5"}, timeout=90.0)
        assert result == {"applied": 2}

    def test_without_a_runner_it_only_re_resolves(self):
        from server.core import JaatoServer
        server = JaatoServer.__new__(JaatoServer)
        server._session_env_resolved = True
        server._session_env = {}
        server._resolve_session_env = lambda: server.__setattr__("_session_env", {"A": "b"})
        server._runner_rpc = None
        assert server.reload_session_env() == {"applied": 1, "runner": False}
