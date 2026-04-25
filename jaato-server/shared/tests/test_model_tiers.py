"""Tests for per-turn model-tier switching.

Covers:
  * ``ModelTierConfig`` validation, normalization (shorthand + rich
    forms), fallback routing, V1 same-provider enforcement.
  * Resolution order: profile ``model_tiers`` → env vars → None.
  * ``JaatoSession`` tier-mode initialization (model override, active
    tier set, single-model mode preserved when no tier_config supplied).
  * ``JaatoSession.switch_tier`` outcomes: switched, already_at_tier,
    fallback_used, and ``provider.connect`` side-effect.
  * ``LifecycleTools.enter_tier`` tool registration (only when tier
    mode active) and execution paths (valid switch, invalid arg,
    invalid tier name, programmer-error guard).
  * ``_get_effective_system_instruction`` composition: with/without
    tier mode, with/without base instructions.
"""

import pytest
from unittest.mock import MagicMock

from ..model_tiers import (
    DEFAULT_INITIAL_TIER,
    DEFAULT_TIER_FALLBACK,
    ENV_TIER_FALLBACK,
    ENV_TIER_INITIAL,
    ENV_TIER_MODEL_KEYS,
    ModelTierConfig,
    ModelTierConfigError,
    RESERVED_FALLBACK_KEY,
    RESERVED_INITIAL_KEY,
    TierEntry,
    TIER_DISPATCHER,
    TIER_EXECUTOR,
    TIER_PLANNER,
    VALID_TIER_NAMES,
)


# ---------------------------------------------------------------------------
# ModelTierConfig — schema validation, normalization, fallback routing
# ---------------------------------------------------------------------------


class TestModelTierConfigValidation:
    """Schema-level validation of the unified ``model_tiers`` dict."""

    def test_simple_shorthand_normalised_to_TierEntry(self):
        cfg = ModelTierConfig.from_unified_dict({"dispatcher": "claude-sonnet-4-6"})
        entry = cfg.tiers["dispatcher"]
        assert isinstance(entry, TierEntry)
        assert entry.model == "claude-sonnet-4-6"
        assert entry.provider is None

    def test_rich_form_with_provider(self):
        cfg = ModelTierConfig.from_unified_dict({
            "dispatcher": {"model": "x", "provider": "anthropic"},
        })
        assert cfg.tiers["dispatcher"] == TierEntry(model="x", provider="anthropic")

    def test_mixed_shorthand_and_rich_form(self):
        cfg = ModelTierConfig.from_unified_dict({
            "planner": {"model": "p", "provider": "anthropic"},
            "dispatcher": "d",
        })
        assert cfg.tiers["planner"].provider == "anthropic"
        assert cfg.tiers["dispatcher"].provider is None

    def test_defaults_when_initial_and_fallback_unset(self):
        cfg = ModelTierConfig.from_unified_dict({
            "planner": "p", "dispatcher": "d", "executor": "e",
        })
        assert cfg.initial_tier == DEFAULT_INITIAL_TIER
        assert cfg.tier_fallback == DEFAULT_TIER_FALLBACK

    def test_reserved_keys_consumed_not_treated_as_tiers(self):
        cfg = ModelTierConfig.from_unified_dict({
            "planner": "p",
            "dispatcher": "d",
            RESERVED_INITIAL_KEY: "planner",
            RESERVED_FALLBACK_KEY: "dispatcher",
        })
        assert cfg.initial_tier == "planner"
        assert cfg.tier_fallback == "dispatcher"
        assert set(cfg.tiers) == {"planner", "dispatcher"}

    def test_reserved_initial_must_be_string(self):
        with pytest.raises(ModelTierConfigError, match="must be a string"):
            ModelTierConfig.from_unified_dict({
                "dispatcher": "d", RESERVED_INITIAL_KEY: 42,
            })

    def test_reserved_fallback_must_be_string(self):
        with pytest.raises(ModelTierConfigError, match="must be a string"):
            ModelTierConfig.from_unified_dict({
                "dispatcher": "d", RESERVED_FALLBACK_KEY: ["dispatcher"],
            })

    def test_empty_tiers_rejected(self):
        with pytest.raises(ModelTierConfigError, match="at least one"):
            ModelTierConfig.from_unified_dict({})

    def test_only_reserved_keys_rejected(self):
        """``initial`` / ``fallback`` alone aren't enough — at least
        one tier→model mapping is required."""
        with pytest.raises(ModelTierConfigError, match="at least one"):
            ModelTierConfig.from_unified_dict({
                RESERVED_INITIAL_KEY: "dispatcher",
                RESERVED_FALLBACK_KEY: "dispatcher",
            })

    def test_unknown_tier_name_rejected(self):
        with pytest.raises(ModelTierConfigError, match="unknown tier names"):
            ModelTierConfig.from_unified_dict({"unknown_tier": "x"})

    def test_initial_must_be_in_tiers(self):
        with pytest.raises(ModelTierConfigError, match="initial_tier"):
            ModelTierConfig.from_unified_dict({
                "executor": "e", RESERVED_INITIAL_KEY: "planner",
            })

    def test_fallback_must_be_in_tiers(self):
        with pytest.raises(ModelTierConfigError, match="tier_fallback"):
            ModelTierConfig.from_unified_dict({
                "executor": "e",
                RESERVED_INITIAL_KEY: "executor",
                RESERVED_FALLBACK_KEY: "planner",
            })

    def test_empty_model_string_rejected(self):
        with pytest.raises(ModelTierConfigError, match="empty model"):
            ModelTierConfig.from_unified_dict({"dispatcher": ""})

    def test_invalid_value_type_rejected(self):
        with pytest.raises(ModelTierConfigError, match="expected str or dict"):
            ModelTierConfig.from_unified_dict({"dispatcher": 42})

    def test_v1_same_provider_enforcement(self):
        with pytest.raises(ModelTierConfigError, match="same-provider"):
            ModelTierConfig.from_unified_dict({
                "planner": {"model": "claude-opus", "provider": "anthropic"},
                "dispatcher": {"model": "qwen", "provider": "ollama"},
            })

    def test_unset_providers_treated_as_consistent(self):
        cfg = ModelTierConfig.from_unified_dict({
            "planner": "p", "dispatcher": "d",
        })
        assert cfg.tiers["planner"].provider is None
        assert cfg.tiers["dispatcher"].provider is None

    def test_mixed_explicit_and_unset_providers_allowed(self):
        cfg = ModelTierConfig.from_unified_dict({
            "planner": {"model": "p", "provider": "anthropic"},
            "dispatcher": "d",  # provider=None
        })
        assert cfg is not None  # validated successfully


class TestModelTierConfigModelFor:

    def _cfg(self, **overrides):
        unified = {
            "planner": "p", "dispatcher": "d", "executor": "e",
            RESERVED_INITIAL_KEY: "dispatcher",
            RESERVED_FALLBACK_KEY: "dispatcher",
        }
        unified.update(overrides)
        return ModelTierConfig.from_unified_dict(unified)

    def test_declared_tier_returns_itself(self):
        cfg = self._cfg()
        actual, entry = cfg.model_for("planner")
        assert actual == "planner"
        assert entry.model == "p"

    def test_undeclared_tier_routes_to_fallback(self):
        cfg = ModelTierConfig.from_unified_dict({
            "dispatcher": "d", RESERVED_FALLBACK_KEY: "dispatcher",
        })
        actual, entry = cfg.model_for("planner")
        assert actual == "dispatcher"
        assert entry.model == "d"

    def test_invalid_tier_name_raises(self):
        cfg = self._cfg()
        with pytest.raises(ModelTierConfigError, match="unknown tier"):
            cfg.model_for("not-a-tier")


# ---------------------------------------------------------------------------
# ModelTierConfig.resolve — profile / env / none
# ---------------------------------------------------------------------------


class TestModelTierConfigResolve:

    def test_profile_wins_over_env(self):
        cfg = ModelTierConfig.resolve(
            profile_model_tiers={"dispatcher": "from-profile"},
            env={"JAATO_TIER_DISPATCHER": "from-env"},
        )
        assert cfg.tiers["dispatcher"].model == "from-profile"

    def test_env_used_when_profile_lacks_tiers(self):
        cfg = ModelTierConfig.resolve(
            profile_model_tiers=None,
            env={"JAATO_TIER_DISPATCHER": "from-env"},
        )
        assert cfg.tiers["dispatcher"].model == "from-env"

    def test_no_profile_no_env_returns_none(self):
        assert ModelTierConfig.resolve(profile_model_tiers=None, env={}) is None

    def test_empty_profile_tiers_falls_back_to_env(self):
        cfg = ModelTierConfig.resolve(
            profile_model_tiers={},
            env={"JAATO_TIER_DISPATCHER": "from-env"},
        )
        assert cfg.tiers["dispatcher"].model == "from-env"

    def test_env_initial_and_fallback_picked_up(self):
        cfg = ModelTierConfig.from_env(env={
            "JAATO_TIER_PLANNER": "p",
            "JAATO_TIER_DISPATCHER": "d",
            ENV_TIER_INITIAL: "planner",
            ENV_TIER_FALLBACK: "planner",
        })
        assert cfg.initial_tier == "planner"
        assert cfg.tier_fallback == "planner"

    def test_env_with_blank_values_ignored(self):
        cfg = ModelTierConfig.from_env(env={
            "JAATO_TIER_DISPATCHER": "d",
            "JAATO_TIER_PLANNER": "   ",  # blank-only — should be skipped
        })
        assert "planner" not in cfg.tiers
        assert "dispatcher" in cfg.tiers


# ---------------------------------------------------------------------------
# JaatoSession tier-mode integration
# ---------------------------------------------------------------------------


class TestJaatoSessionTierMode:

    def _session(self, model="default-model"):
        from ..jaato_session import JaatoSession
        rt = MagicMock()
        rt.create_provider.return_value = MagicMock()
        rt.get_tool_schemas.return_value = []
        rt.get_executors.return_value = {}
        rt.get_system_instructions.return_value = "Base."
        rt.permission_plugin = None
        rt.registry = MagicMock()
        rt.registry._exposed = []
        rt.registry.get_plugin.return_value = None
        rt.registry.collect_prerequisite_policies.return_value = []
        rt.reliability_plugin = None
        return JaatoSession(rt, model)

    def _three_tier_cfg(self):
        return ModelTierConfig.from_unified_dict({
            "planner": "p", "dispatcher": "d", "executor": "e",
        })

    def test_no_tier_config_keeps_single_model_mode(self):
        s = self._session(model="my-model")
        s.configure()
        assert s._tier_config is None
        assert s._active_tier is None
        assert s._model_name == "my-model"

    def test_tier_config_overrides_initial_model(self):
        s = self._session(model="legacy-model")
        s.configure(tier_config=self._three_tier_cfg())
        assert s._active_tier == "dispatcher"
        # Initial model should be the dispatcher tier's model, not the legacy.
        assert s._model_name == "d"

    def test_switch_tier_changes_model_and_calls_provider_connect(self):
        s = self._session()
        s.configure(tier_config=self._three_tier_cfg())
        provider = s._provider
        provider.connect.reset_mock()

        result = s.switch_tier("planner")

        assert result["status"] == "switched"
        assert result["active_tier"] == "planner"
        assert result["model"] == "p"
        assert s._active_tier == "planner"
        assert s._model_name == "p"
        provider.connect.assert_called_once_with("p", skip_model_test=True)

    def test_switch_tier_idempotent(self):
        s = self._session()
        s.configure(tier_config=self._three_tier_cfg())
        provider = s._provider
        provider.connect.reset_mock()

        result = s.switch_tier("dispatcher")  # already at dispatcher

        assert result["status"] == "already_at_tier"
        assert result["active_tier"] == "dispatcher"
        provider.connect.assert_not_called()

    def test_switch_tier_fallback_used_when_target_undeclared(self):
        cfg = ModelTierConfig.from_unified_dict({
            "dispatcher": "d", RESERVED_FALLBACK_KEY: "dispatcher",
        })
        s = self._session()
        s.configure(tier_config=cfg)

        # Initial tier was dispatcher; ask for planner (not declared).
        # The fallback IS dispatcher — same as current → no change.
        result = s.switch_tier("planner")
        assert result["active_tier"] == "dispatcher"
        # status is "already_at_tier" because actual==active even after
        # fallback rerouting; the requested_tier field still tells the
        # model the original ask, so it knows fallback engaged.
        assert result["requested_tier"] == "planner"

    def test_switch_tier_fallback_when_routes_to_different_tier(self):
        # Initial tier is executor; fallback is dispatcher; planner not
        # declared.  Asking for planner → routes to dispatcher (different
        # from current executor) → status=fallback_used.
        cfg = ModelTierConfig.from_unified_dict({
            "dispatcher": "d", "executor": "e",
            RESERVED_INITIAL_KEY: "executor",
            RESERVED_FALLBACK_KEY: "dispatcher",
        })
        s = self._session()
        s.configure(tier_config=cfg)

        result = s.switch_tier("planner")

        assert result["status"] == "fallback_used"
        assert result["active_tier"] == "dispatcher"
        assert result["requested_tier"] == "planner"
        assert result["model"] == "d"

    def test_switch_tier_in_single_model_mode_raises(self):
        s = self._session()
        s.configure()  # no tier_config
        with pytest.raises(RuntimeError, match="single-model mode"):
            s.switch_tier("planner")

    def test_get_effective_system_instruction_appends_tier_line(self):
        s = self._session()
        s.configure(tier_config=self._three_tier_cfg())
        eff = s._get_effective_system_instruction()
        assert "Base." in eff
        assert "currently operating in the `dispatcher` tier" in eff

    def test_get_effective_system_instruction_no_tier_no_line(self):
        s = self._session()
        s.configure()  # single-model mode
        eff = s._get_effective_system_instruction()
        assert "currently operating" not in (eff or "")


# ---------------------------------------------------------------------------
# LifecycleTools.enter_tier tool
# ---------------------------------------------------------------------------


class TestEnterTierTool:

    def _session_with_tier(self, **override):
        """Build a session in tier mode with a default 3-tier config."""
        from ..jaato_session import JaatoSession
        rt = MagicMock()
        rt.create_provider.return_value = MagicMock()
        rt.get_tool_schemas.return_value = []
        rt.get_executors.return_value = {}
        rt.get_system_instructions.return_value = "Base."
        rt.permission_plugin = None
        rt.registry = MagicMock()
        rt.registry._exposed = []
        rt.registry.get_plugin.return_value = None
        rt.registry.collect_prerequisite_policies.return_value = []
        rt.reliability_plugin = None
        s = JaatoSession(rt, "ignored")
        cfg = override.get("cfg") or ModelTierConfig.from_unified_dict({
            "planner": "p", "dispatcher": "d", "executor": "e",
        })
        s.configure(tier_config=cfg)
        return s

    def _lifecycle_tools(self, session):
        from ..lifecycle_tools import LifecycleTools
        return LifecycleTools(session)

    def test_enter_tier_registered_when_tier_mode_active(self):
        s = self._session_with_tier()
        lt = self._lifecycle_tools(s)
        names = {sch.name for sch in lt.get_tool_schemas()}
        assert "enter_tier" in names
        assert "enter_tier" in lt.get_executors()
        assert "enter_tier" in lt.get_auto_approved_tools()

    def test_enter_tier_not_registered_in_single_model_mode(self):
        from ..jaato_session import JaatoSession
        rt = MagicMock()
        rt.create_provider.return_value = MagicMock()
        rt.get_tool_schemas.return_value = []
        rt.get_executors.return_value = {}
        rt.get_system_instructions.return_value = None
        rt.permission_plugin = None
        rt.registry = MagicMock()
        rt.registry._exposed = []
        rt.registry.get_plugin.return_value = None
        rt.registry.collect_prerequisite_policies.return_value = []
        rt.reliability_plugin = None
        s = JaatoSession(rt, "m")
        s.configure()  # no tier_config
        lt = self._lifecycle_tools(s)
        names = {sch.name for sch in lt.get_tool_schemas()}
        assert "enter_tier" not in names
        assert "enter_tier" not in lt.get_executors()
        assert "enter_tier" not in lt.get_auto_approved_tools()

    def test_enter_tier_schema_constrains_to_three_names(self):
        s = self._session_with_tier()
        lt = self._lifecycle_tools(s)
        schema = next(sch for sch in lt.get_tool_schemas() if sch.name == "enter_tier")
        enum = schema.parameters["properties"]["name"]["enum"]
        assert set(enum) == VALID_TIER_NAMES

    def test_enter_tier_valid_switch(self):
        s = self._session_with_tier()
        lt = self._lifecycle_tools(s)
        result = lt._execute_enter_tier({"name": "planner"})
        assert result["status"] == "switched"
        assert result["active_tier"] == "planner"
        assert s._active_tier == "planner"

    def test_enter_tier_missing_arg(self):
        s = self._session_with_tier()
        lt = self._lifecycle_tools(s)
        result = lt._execute_enter_tier({})
        assert result["error"] == "invalid_argument"
        assert "non-empty" in result["message"]

    def test_enter_tier_empty_string(self):
        s = self._session_with_tier()
        lt = self._lifecycle_tools(s)
        result = lt._execute_enter_tier({"name": "   "})
        assert result["error"] == "invalid_argument"

    def test_enter_tier_unknown_name(self):
        s = self._session_with_tier()
        lt = self._lifecycle_tools(s)
        result = lt._execute_enter_tier({"name": "supervisor"})
        assert result["error"] == "invalid_tier"
        assert "supervisor" in result["message"]

    def test_enter_tier_in_single_model_mode_returns_error(self):
        """If somehow enter_tier gets executed despite not being
        registered, the executor returns a structured error rather
        than crashing — defence in depth."""
        from ..jaato_session import JaatoSession
        rt = MagicMock()
        rt.create_provider.return_value = MagicMock()
        rt.get_tool_schemas.return_value = []
        rt.get_executors.return_value = {}
        rt.get_system_instructions.return_value = None
        rt.permission_plugin = None
        rt.registry = MagicMock()
        rt.registry._exposed = []
        rt.registry.get_plugin.return_value = None
        rt.registry.collect_prerequisite_policies.return_value = []
        rt.reliability_plugin = None
        s = JaatoSession(rt, "m")
        s.configure()  # no tier_config

        from ..lifecycle_tools import LifecycleTools
        lt = LifecycleTools(s)
        result = lt._execute_enter_tier({"name": "planner"})
        assert result["error"] == "tier_mode_inactive"
