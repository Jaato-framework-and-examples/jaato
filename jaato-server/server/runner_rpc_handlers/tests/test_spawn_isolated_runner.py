"""Tests for the daemon-side ``subagent.spawn_isolated_runner`` handler.

Phase 4 §4.3.2.

Pins three surfaces:

1. **Args validation** — every required key, every type check,
   confused-deputy parent_session_id echo check.
2. **Stub return shape** — when validation passes, the handler
   returns ``{"ok": False, "error": "...", "stage": "spawn"}``
   (the §4.3.2 contract until §4.3.3-§4.3.7 fill in real spawn
   logic).
3. **Lifecycle** — handler can be shut down idempotently;
   post-shutdown calls raise.

These tests are the load-bearing pin for the wire shape.  When
§4.3.3 lands the real spawn body, ``TestStubReturnShape`` will
update its assertions; the other suites stay valid.
"""

from __future__ import annotations

import pytest

from server.runner_rpc_handlers.spawn_isolated_runner import (
    SpawnIsolatedRunnerHandler,
    STAGE_SPAWN,
    register,
)


# ──────────────────────────────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────────────────────────────


def _valid_args(**overrides):
    """Build a valid request args dict, allowing per-test overrides."""
    base = {
        "parent_session_id": "sess-A",
        "subagent_id": "agent-1",
        "profile_payload": {
            "name": "researcher",
            "model": "claude-sonnet-4-5",
            "provider": "anthropic",
            "plugins": ["cli", "web_search"],
        },
        "task": "investigate X",
        "workspace_path": "/work/space",
    }
    base.update(overrides)
    return base


# ──────────────────────────────────────────────────────────────────────
# Constructor validation
# ──────────────────────────────────────────────────────────────────────


class TestConstructor:
    """Constructor refuses empty / non-str parent_session_id.  An
    empty parent id would break the sub-AppArmor profile name
    template (``jaato-ws-{parent}//{sub}``) so we fail-fast at
    construction."""

    def test_valid_parent_session_id_succeeds(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        assert handler._parent_session_id == "sess-A"
        assert handler._closed is False

    def test_empty_parent_session_id_raises(self):
        with pytest.raises(ValueError, match="parent_session_id must be"):
            SpawnIsolatedRunnerHandler(parent_session_id="")

    def test_non_str_parent_session_id_raises(self):
        with pytest.raises(ValueError, match="parent_session_id must be"):
            SpawnIsolatedRunnerHandler(parent_session_id=42)  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────
# Args validation — required keys
# ──────────────────────────────────────────────────────────────────────


class TestRequiredKeyValidation:
    """Every required key has a missing-key test pin so the wire
    contract can't silently drop a field."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("missing_key", [
        "parent_session_id",
        "subagent_id",
        "profile_payload",
        "task",
        "workspace_path",
    ])
    async def test_missing_required_key_raises(self, missing_key):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        args = _valid_args()
        del args[missing_key]
        with pytest.raises(
            ValueError, match=f"missing required arg '{missing_key}'",
        ):
            await handler.handle(args)


# ──────────────────────────────────────────────────────────────────────
# Args validation — type checks
# ──────────────────────────────────────────────────────────────────────


class TestTypeValidation:
    """Type checks on each arg.  Pins the wire contract against
    silent type-coercion regressions (e.g., a future refactor that
    accepts int subagent_ids would break the sub-profile name
    template)."""

    @pytest.mark.asyncio
    async def test_empty_subagent_id_raises(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        with pytest.raises(ValueError, match="subagent_id"):
            await handler.handle(_valid_args(subagent_id=""))

    @pytest.mark.asyncio
    async def test_non_str_subagent_id_raises(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        with pytest.raises(ValueError, match="subagent_id"):
            await handler.handle(_valid_args(subagent_id=42))

    @pytest.mark.asyncio
    async def test_non_dict_profile_payload_raises(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        with pytest.raises(ValueError, match="profile_payload must be a dict"):
            await handler.handle(_valid_args(profile_payload="not-a-dict"))

    # Phase 5 §5.8 — typed validator integration pins.
    @pytest.mark.asyncio
    async def test_profile_payload_unknown_key_rejected_at_handler(self):
        """§5.8 wire-up: an unknown top-level key in
        profile_payload surfaces as a ValueError with the
        handler-method-name prefix.  Pins the
        SpawnIsolatedRunnerHandler ↔ validate_profile_payload
        bridge added by §5.8."""
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        bad_payload = {
            "name": "researcher",
            "model": "claude-sonnet-4-5",
            "ssrf_target": "http://internal-mgmt-api",  # unknown
        }
        with pytest.raises(
            ValueError,
            match=(
                r"subagent\.spawn_isolated_runner: profile_payload "
                r"validation failed: .*ssrf_target"
            ),
        ):
            await handler.handle(_valid_args(profile_payload=bad_payload))

    @pytest.mark.asyncio
    async def test_profile_payload_validator_runs_before_routing(self):
        """§5.8 pin: validator runs at the handler boundary,
        BEFORE the routed dispatch to SessionManager helper.

        A handler with ``_session_manager`` left None would
        otherwise return the §4.3.2 "unwired stub" envelope.
        §5.8's validator fires first so an invalid payload never
        reaches the routing decision — important because the
        spawn-machinery path is what allocates AppArmor sub-
        profiles / sub-cgroups, and we want validation to gate
        BEFORE any resource provisioning."""
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        assert handler._session_manager is None  # unwired
        bad_payload = {"name": "x", "unexpected": "y"}
        with pytest.raises(
            ValueError, match="profile_payload validation failed",
        ):
            await handler.handle(_valid_args(profile_payload=bad_payload))

    @pytest.mark.asyncio
    async def test_non_str_task_raises(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        with pytest.raises(ValueError, match="task must be a str"):
            await handler.handle(_valid_args(task=42))

    @pytest.mark.asyncio
    async def test_empty_workspace_path_raises(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        with pytest.raises(ValueError, match="workspace_path"):
            await handler.handle(_valid_args(workspace_path=""))

    @pytest.mark.asyncio
    async def test_non_dict_agent_params_raises(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        with pytest.raises(ValueError, match="agent_params"):
            await handler.handle(_valid_args(agent_params="not-a-dict"))

    @pytest.mark.asyncio
    async def test_none_agent_params_passes_validation(self):
        """``None`` is explicitly allowed for agent_params — caller
        may pass it to signal "no template params" without forcing
        an empty dict."""
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        result = await handler.handle(_valid_args(agent_params=None))
        # Validation passed → stub return.
        assert result["ok"] is False
        assert result["stage"] == STAGE_SPAWN


# ──────────────────────────────────────────────────────────────────────
# Confused-deputy protection
# ──────────────────────────────────────────────────────────────────────


class TestConfusedDeputyProtection:
    """The handler is bound to a parent_session_id at construction
    time.  Request args' parent_session_id MUST echo the handler-
    bound value.  Prevents a runner under session A from spawning a
    sub-runner "under" session B."""

    @pytest.mark.asyncio
    async def test_matching_parent_session_id_succeeds(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        result = await handler.handle(
            _valid_args(parent_session_id="sess-A"),
        )
        # Validation passed → stub return.
        assert result["ok"] is False
        assert result["stage"] == STAGE_SPAWN

    @pytest.mark.asyncio
    async def test_mismatched_parent_session_id_raises(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        with pytest.raises(
            ValueError, match="parent_session_id 'sess-B' != "
                              "handler-bound parent_session_id 'sess-A'",
        ):
            await handler.handle(
                _valid_args(parent_session_id="sess-B"),
            )


# ──────────────────────────────────────────────────────────────────────
# Stub return shape
# ──────────────────────────────────────────────────────────────────────


class TestUnwiredStubReturnShape:
    """When ``set_spawn_dependencies()`` has NOT been called, the
    handler returns a "handler not yet wired" stub envelope with
    ``stage=spawn``.  This is the §4.3.2 surface contract — args
    validation passed, but the daemon-side SessionManager bridge
    isn't in place.

    Typical causes: transient state during session bootstrap,
    test harness that skips the SessionManager wire-up,
    misconfigured spawn callback.

    Phase 4 §4.3.3: this suite specifically tests the UNWIRED
    branch.  Once a SessionManager is wired (via
    ``set_spawn_dependencies``), the handler routes through the
    helper — that's covered in ``TestRoutedHelperBridge``."""

    @pytest.mark.asyncio
    async def test_valid_args_returns_handler_not_wired_envelope(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        result = await handler.handle(_valid_args())

        assert result["ok"] is False
        assert "handler not yet wired" in result["error"].lower()
        assert "§4.3.3" in result["error"]
        assert "set_spawn_dependencies" in result["error"]
        assert result["stage"] == STAGE_SPAWN
        # Workaround instruction must point to the default-share path.
        assert "default-share" in result["error"]
        assert "isolated" in result["error"]


# ──────────────────────────────────────────────────────────────────────
# Routed-helper bridge (§4.3.3)
# ──────────────────────────────────────────────────────────────────────


class _StubSessionManager:
    """Records calls to ``_spawn_isolated_runner`` and returns a
    canned envelope.  Stand-in for the daemon's SessionManager
    in handler-level tests.
    """

    def __init__(self, canned_response):
        self.canned_response = canned_response
        self.calls = []

    def _spawn_isolated_runner(self, **kwargs):
        self.calls.append(kwargs)
        return self.canned_response


class TestRoutedHelperBridge:
    """Once ``set_spawn_dependencies()`` is called, the handler
    routes valid requests through
    ``session_manager._spawn_isolated_runner(...)`` instead of
    returning the unwired stub.  Pins:

    1. The setter stores the reference.
    2. After the setter, ``handle()`` calls the helper with the
       expected kwargs.
    3. The helper's return value flows through the handler
       unchanged.
    4. The ``isolated`` control flag is stripped from
       ``agent_params`` before forwarding to the helper.
    5. The setter is idempotent.
    """

    def test_setter_stores_session_manager_reference(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        assert handler._session_manager is None
        sm = _StubSessionManager({"ok": False, "stage": "sub_profile"})
        handler.set_spawn_dependencies(session_manager=sm)
        assert handler._session_manager is sm

    def test_setter_is_idempotent(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        sm1 = _StubSessionManager({"ok": False})
        sm2 = _StubSessionManager({"ok": False})
        handler.set_spawn_dependencies(session_manager=sm1)
        handler.set_spawn_dependencies(session_manager=sm2)
        # Re-calling replaces (caller trusted — same as Phase 3
        # handler-setter patterns).
        assert handler._session_manager is sm2

    @pytest.mark.asyncio
    async def test_handle_routes_through_helper_when_wired(self):
        canned = {
            "ok": False,
            "error": "sub-profile not yet implemented",
            "stage": "sub_profile",
            "isolated_session_id": "sess-A__sub_agent-1",
            "profile_name": "researcher",
        }
        sm = _StubSessionManager(canned)
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        handler.set_spawn_dependencies(session_manager=sm)

        result = await handler.handle(_valid_args())

        # Helper invoked exactly once.
        assert len(sm.calls) == 1
        call = sm.calls[0]
        assert call["parent_session_id"] == "sess-A"
        assert call["subagent_id"] == "agent-1"
        assert call["profile_payload"] == {
            "name": "researcher",
            "model": "claude-sonnet-4-5",
            "provider": "anthropic",
            "plugins": ["cli", "web_search"],
        }
        assert call["task"] == "investigate X"
        assert call["workspace_path"] == "/work/space"
        # Helper's return value flows through unchanged.
        assert result == canned

    @pytest.mark.asyncio
    async def test_handle_strips_isolated_flag_from_agent_params(self):
        """The ``isolated`` key is a routing control flag, not
        template data; the handler must strip it before forwarding
        to the helper.  Otherwise the spawned subagent's dynamic-
        instruction templates would see ``{{isolated}}`` substitute
        to ``true``, which the subagent has no use for."""
        sm = _StubSessionManager({"ok": False, "stage": "sub_profile"})
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        handler.set_spawn_dependencies(session_manager=sm)

        await handler.handle(_valid_args(agent_params={
            "isolated": True,
            "case_id": "42",
            "username": "alice",
        }))

        call = sm.calls[0]
        # Stripped from forwarded agent_params.
        assert "isolated" not in call["agent_params"]
        # Other template keys preserved bit-exact.
        assert call["agent_params"] == {"case_id": "42", "username": "alice"}

    @pytest.mark.asyncio
    async def test_handle_forwards_none_agent_params_unchanged(self):
        """When the caller omits ``agent_params``, the handler must
        forward ``None`` (not an empty dict) so the helper can
        distinguish "no template params" from "empty template
        dict"."""
        sm = _StubSessionManager({"ok": False, "stage": "sub_profile"})
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        handler.set_spawn_dependencies(session_manager=sm)

        await handler.handle(_valid_args())  # no agent_params

        call = sm.calls[0]
        assert call["agent_params"] is None

    @pytest.mark.asyncio
    async def test_handle_forwards_optional_args_through(self):
        """Optional kwargs (display_name, parent_agent_id) flow
        through to the helper when present."""
        sm = _StubSessionManager({"ok": False, "stage": "sub_profile"})
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        handler.set_spawn_dependencies(session_manager=sm)

        await handler.handle(_valid_args(
            display_name="my-agent",
            parent_agent_id="parent-7",
        ))

        call = sm.calls[0]
        assert call["display_name"] == "my-agent"
        assert call["parent_agent_id"] == "parent-7"

    @pytest.mark.asyncio
    async def test_validation_errors_fire_before_routing(self):
        """Args validation happens BEFORE the routing decision;
        bad args raise ``ValueError`` whether or not the helper
        is wired (so misconfigured callers get the same diagnostic
        regardless of bridge state)."""
        sm = _StubSessionManager({"ok": True})
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        handler.set_spawn_dependencies(session_manager=sm)

        # Malformed profile_payload (string instead of dict).
        with pytest.raises(ValueError, match="profile_payload"):
            await handler.handle(_valid_args(profile_payload="not-a-dict"))

        # Helper was NOT invoked — validation rejected the request first.
        assert sm.calls == []


# ──────────────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────────────


class TestLifecycle:
    """``shutdown()`` is idempotent and post-shutdown calls raise.
    Pins the close-flag contract so §4.3.6's extended shutdown
    (in-flight spawn cleanup) can extend cleanly."""

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        handler.shutdown()
        handler.shutdown()  # second call is a no-op
        assert handler._closed is True

    @pytest.mark.asyncio
    async def test_handle_after_shutdown_raises(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        handler.shutdown()
        with pytest.raises(RuntimeError, match="is closed"):
            await handler.handle(_valid_args())


# ──────────────────────────────────────────────────────────────────────
# Register helper
# ──────────────────────────────────────────────────────────────────────


class _StubRPCServer:
    """Minimal stand-in capturing register() calls."""

    def __init__(self):
        self.registered = {}

    def register(self, method_name, handler_fn):
        self.registered[method_name] = handler_fn


class TestRegisterHelper:
    """The ``register(rpc_server, handler)`` convenience routes the
    handler's ``handle`` method under the canonical method name.
    Pins the method-name string so refactors can't silently change
    the wire identifier."""

    def test_register_binds_method_name(self):
        rpc_server = _StubRPCServer()
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        register(rpc_server, handler)
        assert "subagent.spawn_isolated_runner" in rpc_server.registered
        # Bound-method ``==`` checks self + underlying function; ``is``
        # would fail because Python creates a fresh bound-method
        # object on each attribute access.
        assert rpc_server.registered["subagent.spawn_isolated_runner"] == (
            handler.handle
        )
