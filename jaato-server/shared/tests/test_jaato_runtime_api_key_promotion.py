"""Tests for ``plugin_configs.<provider>.api_key`` → ``ProviderConfig.api_key``
promotion in :meth:`JaatoRuntime._create_provider_for_send`.

Server 0.6.132+ (PR-149) fix.  Pre-PR-149, the whole
``plugin_configs[provider]`` dict was merged into
``ProviderConfig.extra`` — but every provider's ``initialize()`` reads
the auth key from the top-level ``config.api_key`` field, NOT from
``config.extra["api_key"]``.  Result: profile-supplied
``plugin_configs.openrouter.api_key: pass://...`` was silently
ignored.

The merge now promotes ``api_key`` to the top-level
``ProviderConfig.api_key`` field before merging the rest of the
dict into ``extra``.  Universal: every provider whose
``ProviderConfig.api_key`` is the auth knob (openrouter, zhipuai,
anthropic, nim, github_models, ollama, lmstudio) benefits.

This file tests the promotion logic in isolation (no live provider).
End-to-end coverage lives in the per-provider integration tests.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

import pytest

from shared.plugins.model_provider.base import ProviderConfig


def _do_promotion(
    base_config: ProviderConfig,
    plugin_configs: Dict[str, Dict[str, Any]],
    provider_name: str,
) -> ProviderConfig:
    """Apply the same promotion logic JaatoRuntime uses.

    Mirrors ``jaato_runtime.py:1136-1158`` so test stays decoupled
    from the runtime constructor (which requires async + provider
    discovery to instantiate).
    """
    config = base_config
    provider_overrides = plugin_configs.get(provider_name)
    if provider_overrides:
        overrides = dict(provider_overrides)
        promoted_api_key = overrides.pop("api_key", None)
        merged_extra = {**config.extra, **overrides}
        replace_kwargs: Dict[str, Any] = {"extra": merged_extra}
        if promoted_api_key:
            replace_kwargs["api_key"] = promoted_api_key
        config = replace(config, **replace_kwargs)
    return config


class TestApiKeyPromotion:

    def test_api_key_promoted_to_top_level(self):
        """Pin: ``plugin_configs.openrouter.api_key`` lands on
        ``config.api_key`` (top-level), NOT ``config.extra``."""
        base = ProviderConfig(project="", location="")
        plugin_configs = {
            "openrouter": {"api_key": "sk-or-secret"},
        }
        result = _do_promotion(base, plugin_configs, "openrouter")
        assert result.api_key == "sk-or-secret"
        assert "api_key" not in result.extra

    def test_other_keys_stay_in_extra(self):
        """Pin: non-api_key fields still flow to ``extra`` per the
        existing namespacing contract (api_params, routing,
        framework_overrides, etc.)."""
        base = ProviderConfig(project="", location="")
        plugin_configs = {
            "openrouter": {
                "api_key": "sk-or-secret",
                "api_params": {"temperature": 0.5},
                "routing": {"order": ["anthropic"]},
            },
        }
        result = _do_promotion(base, plugin_configs, "openrouter")
        assert result.api_key == "sk-or-secret"
        assert result.extra == {
            "api_params": {"temperature": 0.5},
            "routing": {"order": ["anthropic"]},
        }

    def test_no_api_key_in_plugin_configs_leaves_config_unchanged(self):
        """Pin: when plugin_configs has no api_key, ``config.api_key``
        is unchanged (preserves pre-fix behavior for profiles that
        only tune api_params / routing)."""
        base = ProviderConfig(project="", location="", api_key="from-elsewhere")
        plugin_configs = {
            "openrouter": {"api_params": {"temperature": 0.5}},
        }
        result = _do_promotion(base, plugin_configs, "openrouter")
        assert result.api_key == "from-elsewhere"
        assert result.extra == {"api_params": {"temperature": 0.5}}

    def test_plugin_configs_api_key_wins_over_existing_top_level(self):
        """Pin: an api_key in plugin_configs OVERRIDES an api_key
        already on the base ProviderConfig.  Operator's profile
        intent wins."""
        base = ProviderConfig(api_key="from-env")
        plugin_configs = {
            "openrouter": {"api_key": "from-profile"},
        }
        result = _do_promotion(base, plugin_configs, "openrouter")
        assert result.api_key == "from-profile"

    def test_empty_provider_overrides_skip_replace(self):
        """Pin: when ``plugin_configs[provider]`` is empty/missing,
        no replace happens (no spurious config mutation)."""
        base = ProviderConfig(project="proj", location="loc", api_key="abc")
        # Different provider name → no overrides apply
        plugin_configs = {"openrouter": {"api_key": "secret"}}
        result = _do_promotion(base, plugin_configs, "zhipuai")
        # zhipuai entry doesn't exist → no change
        assert result is base
        assert result.api_key == "abc"

    def test_zhipuai_api_key_promoted_same_path(self):
        """Pin: zhipuai uses the same promotion path as openrouter.
        v135 retrospective: zhipuai's plugin_configs.api_key
        documented surface was silently ignored too; this PR fixes
        both with one change."""
        base = ProviderConfig()
        plugin_configs = {
            "zhipuai": {"api_key": "zhipu-secret", "api_params": {"top_p": 0.9}},
        }
        result = _do_promotion(base, plugin_configs, "zhipuai")
        assert result.api_key == "zhipu-secret"
        assert result.extra == {"api_params": {"top_p": 0.9}}

    def test_provider_overrides_dict_not_mutated(self):
        """Pin: caller's plugin_configs dict is NOT mutated by the
        promotion (defensive copy)."""
        original_pc = {"openrouter": {"api_key": "secret", "api_params": {"top_p": 0.5}}}
        # Capture original state
        original_pc_copy = {k: dict(v) for k, v in original_pc.items()}

        base = ProviderConfig()
        _do_promotion(base, original_pc, "openrouter")

        assert original_pc == original_pc_copy, (
            "plugin_configs dict was mutated by promotion logic"
        )
