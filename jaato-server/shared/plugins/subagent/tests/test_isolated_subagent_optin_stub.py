"""Phase 4 §4.3.1 tracer-bullet tests: isolated-subagent opt-in detection.

Pins three contracts for the ``agent_params.isolated`` opt-in path defined
in parent design §4.3 (``docs/design/per_session_confined_runner.md``):

1. **Helper truth table** — ``_is_isolated_optin`` returns True only for
   explicit truthy values; all other shapes (missing, None, False, empty
   string, unrelated keys) preserve the §4.3 default-share contract.

2. **Spawn-path detection + synchronous error** — ``spawn_subagent``
   returns a clear synchronous error response when the opt-in is set,
   without touching the in-runtime executor.  The error message must
   reference Phase 4 §4.3 / the audit doc so callers can find the
   tracking doc.

3. **No regression on default path** — when the opt-in is absent or
   False, the spawn flow reaches profile resolution (i.e., the
   detection check does NOT short-circuit the existing path).

These tests are the load-bearing pin for §4.3.1.  When §4.3.7 wires
the actual isolated-runner spawn machinery, ``test_2`` will be updated
to assert successful isolated spawn (not error response) and the stub
error case will move to a "feature-flag-off" branch.
"""

from unittest.mock import MagicMock

import pytest

from ..plugin import SubagentPlugin, _is_isolated_optin


# ──────────────────────────────────────────────────────────────────────
# 1. Helper truth table
# ──────────────────────────────────────────────────────────────────────


class TestIsIsolatedOptinHelper:
    """Direct unit tests for ``_is_isolated_optin``.

    The helper is the single source of truth for the opt-in decision;
    every other code path queries it.  These tests fence the helper's
    truth-table contract so future refactors can't silently flip the
    default-share semantics.
    """

    def test_none_agent_params_is_not_isolated(self):
        """``agent_params=None`` (caller passed nothing) → default-share."""
        assert _is_isolated_optin(None) is False

    def test_empty_dict_is_not_isolated(self):
        """``agent_params={}`` (caller passed empty dict) → default-share."""
        assert _is_isolated_optin({}) is False

    def test_explicit_false_is_not_isolated(self):
        """``isolated: False`` (explicit opt-out) → default-share."""
        assert _is_isolated_optin({"isolated": False}) is False

    def test_none_value_is_not_isolated(self):
        """``isolated: None`` (caller cleared the flag) → default-share."""
        assert _is_isolated_optin({"isolated": None}) is False

    def test_empty_string_is_not_isolated(self):
        """Empty string is falsy → default-share."""
        assert _is_isolated_optin({"isolated": ""}) is False

    def test_unrelated_keys_alone_are_not_isolated(self):
        """Template params unrelated to opt-in → default-share."""
        assert _is_isolated_optin({"username": "alice", "case": "X"}) is False

    def test_true_is_isolated(self):
        """``isolated: True`` (explicit opt-in) → isolated-runner path."""
        assert _is_isolated_optin({"isolated": True}) is True

    def test_true_alongside_other_keys_is_still_isolated(self):
        """Opt-in flag coexists with template params."""
        assert _is_isolated_optin(
            {"isolated": True, "case": "X", "username": "alice"}
        ) is True


# ──────────────────────────────────────────────────────────────────────
# 2. Spawn-path detection: synchronous error when opt-in is set
# ──────────────────────────────────────────────────────────────────────


def _make_initialized_plugin() -> SubagentPlugin:
    """Build a ``SubagentPlugin`` initialized to the minimum state needed
    to reach the §4.3.1 detection branch in ``spawn_subagent``.

    The detection branch sits AFTER the remote-spawn block but BEFORE
    workspace resolution / profile resolution, so we don't need profile
    or runtime fixtures here.  Setting ``_initialized = True`` is
    sufficient because:

    - The ``not self._initialized`` early-return runs first.
    - The empty-task early-return runs second.
    - The self-spawning loop check runs third (passes because no
      ``profile`` arg is set on the spawn call).
    - The remote-spawn block runs fourth (passes because no ``server``
      arg is set).
    - The §4.3.1 detection branch runs fifth — and that's the branch
      under test.
    """
    plugin = SubagentPlugin()
    plugin._initialized = True
    return plugin


class TestSpawnSubagentDetectsOptin:
    """Integration-level fence: ``_execute_spawn_subagent`` returns a
    synchronous error response when the opt-in is set, regardless of
    profile / agent args.  The error message must reference Phase 4
    §4.3 / the audit doc so callers can find the tracking work."""

    def test_isolated_true_returns_synchronous_error(self):
        """The opt-in fires a structured ``SubagentResult.error`` —
        not a ``NotImplementedError`` exception.  Synchronous response
        means no thread is spawned and the caller can react immediately."""
        plugin = _make_initialized_plugin()
        # Sentinel: assert the executor is never touched.
        plugin._executor = MagicMock()

        result = plugin._execute_spawn_subagent({
            "task": "do something",
            "agent_params": {"isolated": True},
        })

        assert result["success"] is False
        # Caller should be told this is Phase 4 §4.3 work-in-progress.
        error_msg = result["error"].lower()
        assert "isolated" in error_msg
        assert "§4.3" in result["error"] or "4.3" in error_msg
        assert "not yet implemented" in error_msg
        # Pointer to the audit doc so callers can find tracking.
        assert "phase4_implementation_audits" in result["error"]
        # Workaround instructions must mention the default-share path.
        assert "default-share" in error_msg or "share" in error_msg
        # Executor was NOT touched — no thread spawned for the stub.
        plugin._executor.submit.assert_not_called()

    def test_isolated_true_short_circuits_before_workspace_resolution(self):
        """The detection branch sits AFTER remote-spawn but BEFORE
        workspace resolution.  This means we can return an error even
        when no profile / runtime / workspace is configured — the stub
        gives a clean diagnostic instead of a NoneType crash."""
        plugin = _make_initialized_plugin()
        # Intentionally leave _workspace_path, _runtime, _config, etc. unset.

        result = plugin._execute_spawn_subagent({
            "task": "do something",
            "agent_params": {"isolated": True},
        })

        assert result["success"] is False
        assert "isolated" in result["error"].lower()
        # The error should be the §4.3.1 stub, not a downstream crash.
        assert "not yet implemented" in result["error"].lower()


# ──────────────────────────────────────────────────────────────────────
# 3. No regression on default-share path
# ──────────────────────────────────────────────────────────────────────


class TestSpawnSubagentDefaultShareUnchanged:
    """Fence: when the opt-in is absent or False, ``spawn_subagent``
    does NOT short-circuit at the detection branch — it falls through
    to the existing workspace/profile resolution flow.

    These tests don't drive a successful spawn (too much fixture
    overhead); they only assert that the detection branch's error
    response is NOT what comes back.  The downstream code will fail
    with a DIFFERENT error (missing profile / missing runtime / etc.),
    confirming the detection branch let the flow pass through.
    """

    def test_no_agent_params_falls_through_detection(self):
        """``agent_params`` absent → detection passes; downstream
        error (not the §4.3.1 stub) bubbles up."""
        plugin = _make_initialized_plugin()
        # Leave _config, _runtime, _workspace_path unset so the
        # downstream flow surfaces a different error.

        result = plugin._execute_spawn_subagent({"task": "do something"})

        # The detection branch did NOT fire; a different downstream
        # error path produced the result.
        assert result["success"] is False
        assert "not yet implemented" not in result["error"].lower()
        assert "isolated" not in result["error"].lower()

    def test_isolated_false_falls_through_detection(self):
        """Explicit ``isolated: False`` → detection passes; same
        downstream error as the no-flag case."""
        plugin = _make_initialized_plugin()

        result = plugin._execute_spawn_subagent({
            "task": "do something",
            "agent_params": {"isolated": False},
        })

        assert result["success"] is False
        assert "not yet implemented" not in result["error"].lower()
        assert "isolated" not in result["error"].lower()

    def test_unrelated_agent_params_fall_through_detection(self):
        """Template params with no ``isolated`` key → detection
        passes; same downstream error as the no-flag case."""
        plugin = _make_initialized_plugin()

        result = plugin._execute_spawn_subagent({
            "task": "do something",
            "agent_params": {"username": "alice", "case_id": "42"},
        })

        assert result["success"] is False
        assert "not yet implemented" not in result["error"].lower()
        assert "isolated" not in result["error"].lower()
