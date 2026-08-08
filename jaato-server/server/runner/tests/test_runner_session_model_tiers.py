"""Runner-side ``_build_session`` model-tiers wiring (envelope v3).

The regression these tests pin: pre-v3, ``profile.model_tiers``
never reached the runner because :class:`SessionInitEnvelope`
didn't carry the field.  Pool-served sessions therefore left
``JaatoSession._tier_config = None`` and
``LifecycleTools.get_tool_schemas`` skipped registering
``enter_tier`` — the model never saw the tool regardless of the
profile.  Discovery: 2026-05-14 in-pane TUI smoke test of
``tier-test`` profile against zhipuai (glm-5/glm-5-turbo/glm-4.7-flash).

These tests target the runner-side helper directly with a stub
runtime so we can read back the ``tier_config`` kwarg without
spinning up a real session.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from server.runner.session import _build_session
from shared.session_envelope import SessionInitEnvelope


class _CapturingRuntime:
    """Records the args ``_build_session`` passes to
    :meth:`JaatoRuntime.create_session`.

    The runner-side helper only ever calls ``create_session``; nothing
    else is touched.  Returning a sentinel session is fine — these
    tests don't exercise the session, only the construction kwargs.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def create_session(self, **kwargs: Any) -> object:
        self.calls.append(dict(kwargs))
        return object()


def _envelope_with_tiers(tiers: Optional[Dict[str, Any]]) -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-tier",
        workspace_path="/tmp/ws",
        profile_name="tier-test",
        provider_name="zhipuai",
        model_name="glm-5-turbo",
        model_tiers=tiers,
    )


def test_envelope_with_model_tiers_forwards_tier_config() -> None:
    """v3 happy path: profile-declared tiers reach the runner, the
    helper resolves a :class:`ModelTierConfig`, and ``create_session``
    receives it on its ``tier_config`` kwarg.  Without this wiring the
    ``enter_tier`` tool never registers."""
    runtime = _CapturingRuntime()
    env = _envelope_with_tiers({
        "planner": "glm-5",
        "dispatcher": "glm-5-turbo",
        "executor": "glm-4.7-flash",
        "initial": "dispatcher",
        "fallback": "dispatcher",
    })

    _build_session(runtime, env)  # type: ignore[arg-type]

    assert len(runtime.calls) == 1
    kwargs = runtime.calls[0]
    tier_config = kwargs.get("tier_config")
    assert tier_config is not None, (
        "envelope.model_tiers must produce a non-None tier_config; "
        "without it _tier_config stays None and enter_tier is suppressed"
    )
    # Resolved config exposes the three tiers + initial choice.
    assert tier_config.initial_tier == "dispatcher"
    assert set(tier_config.tiers.keys()) == {"planner", "dispatcher", "executor"}
    assert tier_config.tiers["planner"].model == "glm-5"
    assert tier_config.tiers["executor"].model == "glm-4.7-flash"


def test_envelope_without_model_tiers_passes_none() -> None:
    """Single-model sessions (envelope.model_tiers None) must keep
    ``tier_config=None`` so the legacy code path is preserved and the
    ``enter_tier`` tool stays unregistered."""
    runtime = _CapturingRuntime()
    env = _envelope_with_tiers(None)

    _build_session(runtime, env)  # type: ignore[arg-type]

    assert runtime.calls[0]["tier_config"] is None


def test_invalid_model_tiers_degrades_to_single_model_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed tier dict must not crash the runner — the helper
    catches the resolve failure, logs a warning, and falls through to
    single-model mode.  Operators get an enter_tier-less session
    instead of a hard ``BootstrapError``."""
    runtime = _CapturingRuntime()
    # ``initial`` references a tier not declared → ModelTierConfig
    # __post_init__ raises.
    env = _envelope_with_tiers({
        "planner": "glm-5",
        "initial": "executor",  # not in the declared tiers above
    })

    with caplog.at_level("WARNING", logger="server.runner.session"):
        _build_session(runtime, env)  # type: ignore[arg-type]

    assert runtime.calls[0]["tier_config"] is None
    assert any(
        "tier config rejected" in record.message for record in caplog.records
    ), "operator should see a one-line warning naming the resolve failure"


def test_envelope_model_tiers_empty_dict_treated_as_none() -> None:
    """``{}`` is the wire-equivalent of "no tiers declared" — the
    resolver short-circuits and ``tier_config`` stays None."""
    runtime = _CapturingRuntime()
    env = _envelope_with_tiers({})

    _build_session(runtime, env)  # type: ignore[arg-type]

    assert runtime.calls[0]["tier_config"] is None
