"""Tests for the SDK-parity WS verb handlers in
:meth:`SessionManager.handle_request`.

The handlers are thin adapters that map typed WS requests to public
:class:`JaatoSession` primitives or :class:`PermissionPlugin` /
:class:`PermissionPolicy` mutators.  These tests cover the dispatch
+ argument forwarding contract using fake ``JaatoServer`` /
``JaatoSession`` / ``PermissionPlugin`` doubles — the underlying
primitives have their own tests, so we only verify the routing layer.

See ``project_backlog_sdk_feature_parity.md`` for the full design
context.
"""

from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from jaato_sdk.events import (
    Event,
    InjectPromptRequest,
    PermissionAddBlacklistRequest,
    PermissionAddWhitelistRequest,
    PermissionClearRequest,
    PermissionPolicySnapshotEvent,
    PermissionPolicySnapshotRequest,
    PermissionRemoveRequest,
    PermissionSetDefaultRequest,
    ReplayMessagesRequest,
    ReplayMessagesResultEvent,
    ResolveForkPointRequest,
    ResolveForkPointResultEvent,
    SendMessageRequest,
)

from .session_manager import Session, SessionManager


class _FakeJaatoSession:
    """Captures primitive calls so tests can assert on them."""

    def __init__(self, history: Optional[List[Any]] = None) -> None:
        self.inject_calls: List[Tuple[str, Optional[str], Optional[Any]]] = []
        self.replay_calls: List[Tuple[List[Any], float]] = []
        self.resolve_calls: List[
            Tuple[List[Any], Optional[int], Optional[str], Optional[str]]
        ] = []
        self.replay_return_text = "replay-result"
        self.resolve_return_index = 7
        self.replay_should_raise: Optional[Exception] = None
        self.resolve_should_raise: Optional[Exception] = None
        self._history = history or []
        # Holder for the parallel_tools override the dispatcher sets.
        self._parallel_tools_override: Optional[bool] = None

    def inject_prompt(
        self,
        text: str,
        source_id: Optional[str] = None,
        source_type: Optional[Any] = None,
    ) -> None:
        self.inject_calls.append((text, source_id, source_type))

    def replay_messages(
        self, messages: List[Any], timeout: float = 120.0
    ) -> str:
        if self.replay_should_raise:
            raise self.replay_should_raise
        self.replay_calls.append((messages, timeout))
        return self.replay_return_text

    def resolve_fork_point(
        self,
        history: List[Any],
        after_message: Optional[int] = None,
        after_tool_call: Optional[str] = None,
        after_timestamp: Optional[str] = None,
    ) -> int:
        if self.resolve_should_raise:
            raise self.resolve_should_raise
        self.resolve_calls.append(
            (history, after_message, after_tool_call, after_timestamp)
        )
        return self.resolve_return_index

    def get_history(self) -> List[Any]:
        return self._history


class _FakePermissionPolicy:
    def __init__(self) -> None:
        self.whitelist_tools = set()
        self.whitelist_patterns: List[str] = []
        self.blacklist_tools = set()
        self.blacklist_patterns: List[str] = []
        self.session_whitelist = set()
        self.session_blacklist = set()
        self.default_policy = "ask"
        self.session_default_policy: Optional[str] = None
        self.calls: List[Tuple[str, Any]] = []

    def add_session_whitelist(self, pattern: str) -> None:
        self.calls.append(("add_session_whitelist", pattern))
        self.session_whitelist.add(pattern)

    def add_session_blacklist(self, pattern: str) -> None:
        self.calls.append(("add_session_blacklist", pattern))
        self.session_blacklist.add(pattern)

    def set_session_default_policy(self, policy: Optional[str]) -> None:
        if policy is not None and policy not in ("allow", "deny", "ask"):
            raise ValueError(f"Invalid policy: {policy}")
        self.calls.append(("set_session_default_policy", policy))
        self.session_default_policy = policy


class _FakePermissionPlugin:
    def __init__(self) -> None:
        self._policy = _FakePermissionPolicy()
        self.add_whitelist_calls: List[List[str]] = []

    def add_whitelist_tools(self, tools: List[str]) -> None:
        self.add_whitelist_calls.append(list(tools))
        for tool in tools:
            self._policy.whitelist_tools.add(tool)


class _FakeRegistry:
    def __init__(self, permission_plugin: Optional[_FakePermissionPlugin] = None) -> None:
        self._plugins = {}
        if permission_plugin is not None:
            self._plugins["permission"] = permission_plugin

    def get_plugin(self, name: str):
        return self._plugins.get(name)


class _FakeJaatoClient:
    def __init__(self, session: _FakeJaatoSession) -> None:
        self._session = session

    def get_session(self) -> _FakeJaatoSession:
        return self._session


class _FakeRunnerRPC:
    """Phase 3 §7c step 6.6.3.6: mirrors the RPC forwards onto
    the inner JaatoSession fake.  The daemon code now routes
    through ``server._runner_rpc.session_X_threadsafe(...)``
    rather than ``server._jaato.get_session().X(...)``, so the
    test fakes need to expose the threadsafe wrappers.  Each
    wrapper here just calls the corresponding method on the
    inner session — preserving the existing test assertions
    against inject_calls / replay_calls / resolve_calls / etc.
    """

    def __init__(self, session: _FakeJaatoSession, is_running=None) -> None:
        self._session = session
        # Lazy so a test can flip the server's flag AFTER construction.
        # Defaults to "a turn is running" so the source_type-mapping tests
        # below exercise the queue branch, which is what they are about.
        self._is_running = is_running or (lambda: True)

    def session_offer_message_threadsafe(
        self,
        text: str,
        *,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Stand-in for the runner's atomic queue-or-report.

        Mirrors the real contract: ``"queued"`` enqueues (a running turn's
        drain will collect it), ``"needs_turn"`` enqueues NOTHING and tells
        the daemon to start a turn.  The real one decides under a lock
        against the session's own ``_is_running``; this one reads the fake
        server's flag through a lazy callable so tests can set it.
        """
        if not self._is_running():
            return "needs_turn"
        from shared.message_queue import SourceType
        st_enum = SourceType(source_type) if source_type is not None else None
        self._session.inject_prompt(
            text, source_id=source_id, source_type=st_enum,
        )
        return "queued"

    def session_inject_prompt_threadsafe(
        self,
        text: str,
        *,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        # source_type arrives as a string (SourceType.value) — the
        # test session captures whatever the daemon sent.  Tests
        # asserting against the enum need to import SourceType and
        # compare via .value.
        from shared.message_queue import SourceType
        st_enum = SourceType(source_type) if source_type is not None else None
        self._session.inject_prompt(
            text, source_id=source_id, source_type=st_enum,
        )

    def session_replay_messages_threadsafe(
        self,
        messages: List[Any],
        *,
        replay_timeout: float = 120.0,
        timeout: Optional[float] = None,
    ) -> str:
        return self._session.replay_messages(messages, timeout=replay_timeout)

    def session_get_history_threadsafe(
        self, *, timeout: Optional[float] = None,
    ) -> List[Any]:
        return self._session.get_history()

    def session_resolve_fork_point_threadsafe(
        self,
        *,
        after_message: Optional[int] = None,
        after_tool_call: Optional[str] = None,
        after_timestamp: Optional[str] = None,
        history: Optional[List[Any]] = None,
        timeout: Optional[float] = None,
    ) -> int:
        # When history is None, fall back to session.get_history()
        # — matches the runner-side handler's default semantic.
        effective_history = history if history is not None else self._session.get_history()
        return self._session.resolve_fork_point(
            effective_history,
            after_message=after_message,
            after_tool_call=after_tool_call,
            after_timestamp=after_timestamp,
        )

    def session_set_parallel_tools_override_threadsafe(
        self,
        enabled: bool,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        self._session._parallel_tools_override = bool(enabled)


class _FakeJaatoServer:
    def __init__(
        self,
        jaato_session: _FakeJaatoSession,
        permission_plugin: Optional[_FakePermissionPlugin] = None,
    ) -> None:
        self._jaato = _FakeJaatoClient(jaato_session)
        self._runner_rpc = _FakeRunnerRPC(
            jaato_session, is_running=lambda: self._model_running,
        )
        self.registry = _FakeRegistry(permission_plugin)
        # BUSY by default.  The inject handler routes through the
        # queue-or-drive decision, and an IDLE target is DRIVEN rather than
        # injected -- so a fake that looks idle would exercise the drive path
        # instead of the source_type mapping these tests are about.
        self._model_running = True
        self._terminal_reason = None

    # The dispatch path also touches these — provide minimal stubs.
    def get_all_session_env(self):
        return {}


def _make_session(server: _FakeJaatoServer) -> Session:
    return Session(
        session_id="sess_1",
        name="test",
        server=server,  # type: ignore[arg-type]
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _setup(
    permission_plugin: Optional[_FakePermissionPlugin] = None,
    history: Optional[List[Any]] = None,
) -> Tuple[SessionManager, _FakeJaatoServer, _FakeJaatoSession, List[Tuple[str, Event]]]:
    """Build a manager + fake server + capture for emitted events."""
    jaato_session = _FakeJaatoSession(history=history)
    server = _FakeJaatoServer(jaato_session, permission_plugin)
    session = _make_session(server)

    manager = SessionManager()
    manager._sessions["sess_1"] = session

    captured: List[Tuple[str, Event]] = []
    manager._emit_to_client = lambda client_id, event: captured.append((client_id, event))  # type: ignore[assignment]

    return manager, server, jaato_session, captured


# ─────────────────────────────────────────────────────────────────────


class TestInjectPromptHandler:

    def test_string_source_type_maps_to_enum(self):
        from shared.message_queue import SourceType

        manager, _, jaato_session, captured = _setup()

        manager.handle_request(
            "client_1", "sess_1",
            InjectPromptRequest(text="steer me", source_type="user", source_id="ui"),
        )

        assert jaato_session.inject_calls == [
            ("steer me", "ui", SourceType.USER)
        ]
        assert captured == []  # success path is silent

    def test_child_source_type_for_followup(self):
        from shared.message_queue import SourceType

        manager, _, jaato_session, _ = _setup()

        manager.handle_request(
            "c", "sess_1",
            InjectPromptRequest(text="follow up", source_type="child"),
        )

        assert jaato_session.inject_calls == [("follow up", None, SourceType.CHILD)]

    def test_idle_session_is_driven_not_queued(self):
        """The handler must make the queue-or-drive decision at all.

        It used to call the runner's inject directly, bypassing
        ``inject_prompt_to_session`` and therefore ``message_delivery``, so
        it could not start a turn under ANY circumstances.  Since the
        runner-side inject only starts a turn while a ``send_message`` RPC is
        in flight, an inject into an idle session queued into a queue with no
        drainer -- permanently.  A watchdog's nudge-on-silence then landed in
        the same dead queue as the message it was sent to rescue.
        """
        manager, server, jaato_session, _ = _setup()
        server._model_running = False
        driven: List[str] = []
        manager.send_message_to_session = (  # type: ignore[method-assign]
            lambda sid, text: (driven.append(text), True)[1]
        )

        manager.handle_request(
            "c", "sess_1",
            InjectPromptRequest(text="wake up", source_type="user"),
        )

        assert driven == ["wake up"]
        assert jaato_session.inject_calls == []  # NOT queued into the void

    def test_request_id_is_answered_with_a_result_event(self):
        """A caller that asks to be told gets the status on one channel."""
        from jaato_sdk.events import InjectPromptResultEvent
        from shared.message_delivery import QUEUED

        manager, _, _, captured = _setup()

        manager.handle_request(
            "c", "sess_1",
            InjectPromptRequest(
                text="steer", source_type="user", request_id="req_abc",
            ),
        )

        assert len(captured) == 1
        _, event = captured[0]
        assert isinstance(event, InjectPromptResultEvent)
        assert event.request_id == "req_abc"
        assert event.status == QUEUED

    def test_failure_without_request_id_still_fails_loudly(self):
        """A pre-1.3 caller has no result channel, so a failure has to
        surface as an error or it is silent -- and silence is the expensive
        direction: the caller assumes delivery and stalls somewhere it
        cannot attribute."""
        manager, server, _, captured = _setup()
        server._terminal_reason = "error"       # target is dead

        manager.handle_request(
            "c", "sess_1",
            InjectPromptRequest(text="anyone there", source_type="user"),
        )

        assert len(captured) == 1
        _, event = captured[0]
        assert event.error_type == "SessionError"
        assert "terminated" in event.error

    def test_success_without_request_id_stays_silent(self):
        """Backward compatibility: the pre-1.3 fire-and-forget success path
        emits nothing, exactly as before."""
        manager, _, _, captured = _setup()

        manager.handle_request(
            "c", "sess_1",
            InjectPromptRequest(text="steer", source_type="user"),
        )

        assert captured == []

    def test_invalid_source_type_emits_error(self):
        manager, _, jaato_session, captured = _setup()

        manager.handle_request(
            "c", "sess_1",
            InjectPromptRequest(text="x", source_type="garbage"),
        )

        assert jaato_session.inject_calls == []  # never called
        assert len(captured) == 1
        _, event = captured[0]
        assert event.error_type == "ValidationError"
        assert "garbage" in event.error


class TestReplayMessagesHandler:

    def test_omitted_messages_uses_session_history(self):
        manager, _, jaato_session, captured = _setup(history=["msg1", "msg2"])

        manager.handle_request(
            "c", "sess_1",
            ReplayMessagesRequest(request_id="r1", messages=None),
        )

        # replay_messages is run in a worker thread — wait for it to settle.
        import time
        for _ in range(50):
            if jaato_session.replay_calls and captured:
                break
            time.sleep(0.02)

        assert jaato_session.replay_calls == [(["msg1", "msg2"], 120.0)]
        assert len(captured) == 1
        _, event = captured[0]
        assert isinstance(event, ReplayMessagesResultEvent)
        assert event.request_id == "r1"
        assert event.response_text == "replay-result"
        assert event.error == ""

    def test_exception_surfaces_via_result_event_error(self):
        manager, _, jaato_session, captured = _setup()
        jaato_session.replay_should_raise = RuntimeError("provider down")

        manager.handle_request(
            "c", "sess_1",
            ReplayMessagesRequest(request_id="r2"),
        )

        import time
        for _ in range(50):
            if captured:
                break
            time.sleep(0.02)

        assert len(captured) == 1
        _, event = captured[0]
        assert isinstance(event, ReplayMessagesResultEvent)
        assert event.request_id == "r2"
        assert event.response_text == ""
        assert "RuntimeError" in event.error
        assert "provider down" in event.error


class TestResolveForkPointHandler:

    def test_forwards_specifiers(self):
        manager, _, jaato_session, captured = _setup(history=["a", "b", "c"])

        manager.handle_request(
            "c", "sess_1",
            ResolveForkPointRequest(
                request_id="r3",
                after_tool_call="call_42",
            ),
        )

        assert jaato_session.resolve_calls == [
            (["a", "b", "c"], None, "call_42", None)
        ]
        assert len(captured) == 1
        _, event = captured[0]
        assert isinstance(event, ResolveForkPointResultEvent)
        assert event.request_id == "r3"
        assert event.fork_index == 7
        assert event.error == ""

    def test_exception_surfaces_with_negative_index(self):
        manager, _, jaato_session, captured = _setup()
        jaato_session.resolve_should_raise = ValueError("bad spec")

        manager.handle_request(
            "c", "sess_1",
            ResolveForkPointRequest(request_id="r4"),
        )

        assert len(captured) == 1
        _, event = captured[0]
        assert event.fork_index == -1
        assert "ValueError" in event.error
        assert "bad spec" in event.error


class TestPermissionHandlers:

    def test_add_whitelist_routes_to_plugin_and_policy(self):
        plugin = _FakePermissionPlugin()
        manager, _, _, _ = _setup(permission_plugin=plugin)

        manager.handle_request(
            "c", "sess_1",
            PermissionAddWhitelistRequest(
                tools=["read_file", "list_files"],
                patterns=["safe_*"],
            ),
        )

        # Tools go through plugin.add_whitelist_tools
        assert plugin.add_whitelist_calls == [["read_file", "list_files"]]
        # Patterns go through policy.add_session_whitelist
        assert ("add_session_whitelist", "safe_*") in plugin._policy.calls

    def test_add_blacklist_routes_through_policy(self):
        plugin = _FakePermissionPlugin()
        manager, _, _, _ = _setup(permission_plugin=plugin)

        manager.handle_request(
            "c", "sess_1",
            PermissionAddBlacklistRequest(
                tools=["dangerous_tool"],
                patterns=["evil_*"],
            ),
        )

        assert ("add_session_blacklist", "dangerous_tool") in plugin._policy.calls
        assert ("add_session_blacklist", "evil_*") in plugin._policy.calls

    def test_remove_whitelist_discards_from_session_sets(self):
        plugin = _FakePermissionPlugin()
        plugin._policy.whitelist_tools = {"keep_me", "remove_me"}
        plugin._policy.session_whitelist = {"keep_pattern", "remove_pattern"}

        manager, _, _, _ = _setup(permission_plugin=plugin)

        manager.handle_request(
            "c", "sess_1",
            PermissionRemoveRequest(
                target="whitelist",
                tools=["remove_me"],
                patterns=["remove_pattern"],
            ),
        )

        assert "remove_me" not in plugin._policy.whitelist_tools
        assert "keep_me" in plugin._policy.whitelist_tools
        assert "remove_pattern" not in plugin._policy.session_whitelist
        assert "keep_pattern" in plugin._policy.session_whitelist

    def test_remove_unknown_target_emits_error(self):
        plugin = _FakePermissionPlugin()
        manager, _, _, captured = _setup(permission_plugin=plugin)

        manager.handle_request(
            "c", "sess_1",
            PermissionRemoveRequest(target="garbage", tools=["x"]),
        )

        assert len(captured) == 1
        _, event = captured[0]
        assert event.error_type == "ValidationError"

    def test_clear_all_clears_session_state_only(self):
        plugin = _FakePermissionPlugin()
        plugin._policy.session_whitelist = {"a"}
        plugin._policy.session_blacklist = {"b"}
        plugin._policy.session_default_policy = "deny"
        # Base policy stays untouched.
        plugin._policy.whitelist_tools = {"base_white"}
        plugin._policy.blacklist_tools = {"base_black"}

        manager, _, _, _ = _setup(permission_plugin=plugin)

        manager.handle_request(
            "c", "sess_1",
            PermissionClearRequest(target="all"),
        )

        assert plugin._policy.session_whitelist == set()
        assert plugin._policy.session_blacklist == set()
        assert plugin._policy.session_default_policy is None
        # Base policy untouched
        assert plugin._policy.whitelist_tools == {"base_white"}
        assert plugin._policy.blacklist_tools == {"base_black"}

    def test_clear_only_whitelist_leaves_blacklist(self):
        plugin = _FakePermissionPlugin()
        plugin._policy.session_whitelist = {"w"}
        plugin._policy.session_blacklist = {"b"}

        manager, _, _, _ = _setup(permission_plugin=plugin)

        manager.handle_request(
            "c", "sess_1",
            PermissionClearRequest(target="whitelist"),
        )

        assert plugin._policy.session_whitelist == set()
        assert plugin._policy.session_blacklist == {"b"}

    def test_set_default_policy_routes_through_policy(self):
        plugin = _FakePermissionPlugin()
        manager, _, _, _ = _setup(permission_plugin=plugin)

        manager.handle_request(
            "c", "sess_1",
            PermissionSetDefaultRequest(policy="allow"),
        )

        assert ("set_session_default_policy", "allow") in plugin._policy.calls
        assert plugin._policy.session_default_policy == "allow"

    def test_set_invalid_policy_emits_error(self):
        plugin = _FakePermissionPlugin()
        manager, _, _, captured = _setup(permission_plugin=plugin)

        manager.handle_request(
            "c", "sess_1",
            PermissionSetDefaultRequest(policy="bogus"),
        )

        assert len(captured) == 1
        _, event = captured[0]
        assert event.error_type == "ValidationError"

    def test_policy_snapshot_returns_structured_view(self):
        plugin = _FakePermissionPlugin()
        plugin._policy.default_policy = "ask"
        plugin._policy.session_default_policy = "deny"
        plugin._policy.whitelist_tools = {"safe1", "safe2"}
        plugin._policy.whitelist_patterns = ["read_*"]
        plugin._policy.blacklist_tools = {"bad1"}
        plugin._policy.blacklist_patterns = ["delete_*"]
        plugin._policy.session_whitelist = {"sess_white"}
        plugin._policy.session_blacklist = {"sess_black"}

        manager, _, _, captured = _setup(permission_plugin=plugin)

        manager.handle_request(
            "c", "sess_1",
            PermissionPolicySnapshotRequest(request_id="snap1"),
        )

        assert len(captured) == 1
        _, event = captured[0]
        assert isinstance(event, PermissionPolicySnapshotEvent)
        assert event.request_id == "snap1"
        assert event.default_policy == "ask"
        assert event.session_default_policy == "deny"
        assert event.whitelist_tools == ["safe1", "safe2"]  # sorted
        assert event.whitelist_patterns == ["read_*"]
        assert event.blacklist_tools == ["bad1"]
        assert event.blacklist_patterns == ["delete_*"]
        assert event.session_whitelist == ["sess_white"]
        assert event.session_blacklist == ["sess_black"]

    def test_snapshot_with_no_plugin_returns_default_response(self):
        manager, _, _, captured = _setup()  # no permission plugin

        manager.handle_request(
            "c", "sess_1",
            PermissionPolicySnapshotRequest(request_id="snap2"),
        )

        # Returns a snapshot with only default_policy set, not an error
        # (caller can introspect that nothing else is configured).
        assert len(captured) == 1
        _, event = captured[0]
        assert isinstance(event, PermissionPolicySnapshotEvent)
        assert event.default_policy == "ask"
        assert event.whitelist_tools == []


class TestParallelToolsOverride:
    """SendMessageRequest.parallel_tools propagation onto the JaatoSession.

    Doesn't test the full send_message flow (that would need a real
    JaatoServer / runtime / provider), just that the dispatcher
    stashes the override on the session before the worker thread runs.
    """

    def test_parallel_tools_true_set_on_session(self):
        manager, _, jaato_session, _ = _setup()

        # Stub send_message so the worker thread doesn't blow up calling
        # methods on _FakeJaatoServer.
        manager._sessions["sess_1"].server.send_message = lambda *a, **kw: None  # type: ignore[attr-defined]

        manager.handle_request(
            "c", "sess_1",
            SendMessageRequest(text="hi", parallel_tools=True),
        )

        assert jaato_session._parallel_tools_override is True

    def test_parallel_tools_false_set_on_session(self):
        manager, _, jaato_session, _ = _setup()
        manager._sessions["sess_1"].server.send_message = lambda *a, **kw: None  # type: ignore[attr-defined]

        manager.handle_request(
            "c", "sess_1",
            SendMessageRequest(text="hi", parallel_tools=False),
        )

        assert jaato_session._parallel_tools_override is False

    def test_parallel_tools_none_leaves_override_unset(self):
        manager, _, jaato_session, _ = _setup()
        manager._sessions["sess_1"].server.send_message = lambda *a, **kw: None  # type: ignore[attr-defined]

        manager.handle_request(
            "c", "sess_1",
            SendMessageRequest(text="hi"),
        )

        assert jaato_session._parallel_tools_override is None
