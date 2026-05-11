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


class TestStubReturnShape:
    """The §4.3.2 stub returns a typed envelope so callers can
    branch on ``ok`` and surface ``stage`` for diagnostics.  These
    assertions pin the wire shape that §4.3.7's opt-in branch
    will read.

    When §4.3.3-§4.3.7 land actual spawn logic, this suite updates
    to assert ``ok: True`` + session_id etc.; the args-validation
    and lifecycle suites remain valid."""

    @pytest.mark.asyncio
    async def test_valid_args_returns_not_implemented_envelope(self):
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        result = await handler.handle(_valid_args())

        assert result["ok"] is False
        assert "not yet implemented" in result["error"].lower()
        assert "§4.3.3-§4.3.7" in result["error"]
        assert result["stage"] == STAGE_SPAWN
        # Workaround instruction must point to the default-share path.
        assert "default-share" in result["error"]
        assert "isolated=" in result["error"]

    @pytest.mark.asyncio
    async def test_stub_envelope_includes_audit_doc_pointer(self):
        """Caller needs a pointer to the tracking doc for the
        deferred work."""
        handler = SpawnIsolatedRunnerHandler(parent_session_id="sess-A")
        result = await handler.handle(_valid_args())

        assert "phase4_implementation_audits.md" in result["error"]


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
