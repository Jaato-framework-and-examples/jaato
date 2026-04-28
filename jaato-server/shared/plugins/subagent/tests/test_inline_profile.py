"""Tests for ``build_inline_profile`` — the SDK-side entry point that turns
a dict (carried in ``CommandRequest.payload['spec']``) into a
``SubagentProfile`` without going through disk.

The disk-load path is covered by ``test_profile_discovery.py``; this file
focuses on the inline-spec parser added for the SDK ``session.new``
inline-profile feature.
"""

import pytest

from ..config import SubagentProfile, build_inline_profile


class TestBuildInlineProfile:
    """Smoke + edge cases for build_inline_profile."""

    def test_minimal_spec_with_only_model(self):
        """A spec with only ``model`` produces a profile with safe defaults."""
        p = build_inline_profile({"model": "claude-sonnet-4-5"})
        assert isinstance(p, SubagentProfile)
        assert p.model == "claude-sonnet-4-5"
        assert p.plugins == []
        assert p.preloaded_plugins == set()
        assert p.system_instructions is None
        assert p.provider is None
        assert p.gc is None
        assert p.runtime_limits is None
        assert p.inherits is None
        assert p.name == "<inline>"
        assert p.description == "Inline session spec"

    def test_full_spec(self):
        """All recognized fields are forwarded onto the profile."""
        p = build_inline_profile({
            "model": "claude-sonnet-4-5",
            "provider": "anthropic",
            "plugins": ["cli", "web_search"],
            "plugin_configs": {"cli": {"timeout": 30}},
            "system_instructions": "You are a researcher.",
            "max_turns": 25,
            "env": {"FOO": "bar"},
        })
        assert p.model == "claude-sonnet-4-5"
        assert p.provider == "anthropic"
        assert p.plugins == ["cli", "web_search"]
        assert p.plugin_configs == {"cli": {"timeout": 30}}
        assert p.system_instructions == "You are a researcher."
        assert p.max_turns == 25
        assert p.env == {"FOO": "bar"}

    def test_preload_annotation_in_plugin_list(self):
        """``plugin(preload)`` syntax is split the same as on-disk profiles."""
        p = build_inline_profile({
            "model": "X",
            "plugins": ["cli", "todo(preload)", "web_search(preload)"],
        })
        assert p.plugins == ["cli", "todo", "web_search"]
        assert p.preloaded_plugins == {"todo", "web_search"}

    def test_inherits_is_silently_dropped(self):
        """Inline specs are atomic — ``inherits`` makes no sense and is dropped."""
        p = build_inline_profile({
            "model": "X",
            "inherits": ["readonly", "web_capable"],
        })
        assert p.inherits is None

    def test_invalid_runtime_limits_raises_value_error(self):
        """Garbage in ``runtime_limits`` surfaces as a clear ValueError."""
        with pytest.raises(ValueError, match="runtime_limits"):
            build_inline_profile({
                "model": "X",
                "runtime_limits": "not-a-dict",
            })

    def test_explicit_name_and_description(self):
        """Caller can override the placeholder name/description for traces."""
        p = build_inline_profile(
            {"model": "X"},
            name="ops-task",
            description="Ad-hoc operations session",
        )
        assert p.name == "ops-task"
        assert p.description == "Ad-hoc operations session"

    def test_does_not_enforce_model_field(self):
        """Model presence is enforced by the caller (SessionManager), not here.

        ``build_inline_profile`` is a pure parser — its job is shape, not
        policy. The empty-model case is validated upstream where the
        client-facing ``ErrorEvent`` lives.
        """
        p = build_inline_profile({})
        assert p.model is None

    def test_model_tiers_forwarded(self):
        """``model_tiers`` (per-turn tier config) is preserved verbatim."""
        tiers = {
            "initial": "fast",
            "fast": "claude-haiku",
            "deep": "claude-opus",
        }
        p = build_inline_profile({"model": "claude-haiku", "model_tiers": tiers})
        assert p.model_tiers == tiers
