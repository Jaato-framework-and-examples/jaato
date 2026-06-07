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
        p = build_inline_profile({"model": "claude-sonnet-4-5", "plugins": []})
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
            "plugins": [],
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
            {"model": "X", "plugins": []},
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
        p = build_inline_profile({"plugins": []})
        assert p.model is None

    def test_model_tiers_forwarded(self):
        """``model_tiers`` (per-turn tier config) is preserved verbatim."""
        tiers = {
            "initial": "fast",
            "fast": "claude-haiku",
            "deep": "claude-opus",
        }
        p = build_inline_profile({"model": "claude-haiku", "plugins": [], "model_tiers": tiers})
        assert p.model_tiers == tiers

    def test_apparmor_default_is_false(self):
        """PR-A (2026-05-14): ``apparmor`` absent → False (back-compat)."""
        p = build_inline_profile({"model": "m", "plugins": []})
        assert p.apparmor is False

    def test_apparmor_true_parsed(self):
        """PR-A: explicit ``apparmor: true`` reaches the dataclass.  The
        actual confinement still depends on the caller-level kwarg /
        client_config opt-in (see ``test_ipc_apparmor_provision.py``);
        this test just pins the parse → field path."""
        p = build_inline_profile({"model": "m", "plugins": [], "apparmor": True})
        assert p.apparmor is True

    def test_apparmor_false_explicit_parsed(self):
        """Explicit ``apparmor: false`` is preserved (vs. absent).  Useful
        for inline specs that want to lock the field at the spec layer
        before PR-B flips the profile default to True."""
        p = build_inline_profile({"model": "m", "plugins": [], "apparmor": False})
        assert p.apparmor is False

    def test_apparmor_truthy_value_coerced(self):
        """``data.get('apparmor', False)`` is wrapped in ``bool()``.  A
        truthy non-bool (1, non-empty list, etc.) coerces to True;
        falsy (0, empty) coerces to False.  Pins the coercion contract
        so YAML / JSON differences across loaders don't regress."""
        p_truthy = build_inline_profile({"model": "m", "plugins": [], "apparmor": 1})
        assert p_truthy.apparmor is True
        p_falsy = build_inline_profile({"model": "m", "plugins": [], "apparmor": 0})
        assert p_falsy.apparmor is False

    def test_apparmor_fragments_absent_resolves_none(self):
        """Piece 1 (2026-05-14): field absent → None on the dataclass.
        Distinct from explicit empty list.  Workspace-default
        "compose all fragments" applies at render time."""
        p = build_inline_profile({"model": "m", "plugins": []})
        assert p.apparmor_fragments is None

    def test_apparmor_fragments_explicit_empty_list_preserved(self):
        """Explicit ``apparmor_fragments: []`` is distinct from
        absent: the maximally locked-down stage in a cascade."""
        p = build_inline_profile({"model": "m", "plugins": [], "apparmor_fragments": []})
        assert p.apparmor_fragments == []

    def test_apparmor_fragments_non_empty_list_parsed(self):
        p = build_inline_profile({
            "model": "m",
            "plugins": [],
            "apparmor_fragments": ["host_validator", "kb-enablement-2"],
        })
        assert p.apparmor_fragments == ["host_validator", "kb-enablement-2"]

    def test_apparmor_fragments_strips_whitespace(self):
        """Entries get stripped — operator typos with surrounding
        whitespace don't silently break the basename match."""
        p = build_inline_profile({
            "model": "m",
            "plugins": [],
            "apparmor_fragments": ["  host_validator  ", "kb-fixtures"],
        })
        assert p.apparmor_fragments == ["host_validator", "kb-fixtures"]

    def test_apparmor_fragments_rejects_non_list(self):
        """Strings and ints are rejected — quiet coercion would mask
        operator typos (e.g. ``apparmor_fragments: host_validator``
        in YAML, which yields a string, not a single-item list)."""
        with pytest.raises(ValueError, match="must be a list of strings"):
            build_inline_profile({
                "model": "m", "plugins": [], "apparmor_fragments": "host_validator",
            })

    def test_apparmor_fragments_rejects_non_string_entry(self):
        with pytest.raises(ValueError, match=r"\[0\] must be a non-empty string"):
            build_inline_profile({
                "model": "m", "plugins": [], "apparmor_fragments": [42, "foo"],
            })

    def test_apparmor_fragments_rejects_empty_string_entry(self):
        """Empty-string entry would silently match nothing — reject
        loudly so the operator notices the typo."""
        with pytest.raises(ValueError, match=r"\[1\] must be a non-empty string"):
            build_inline_profile({
                "model": "m", "plugins": [], "apparmor_fragments": ["host_validator", ""],
            })
