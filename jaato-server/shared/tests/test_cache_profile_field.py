"""The common ``cache:`` profile field reaches every provider that caches.

WHY IT EXISTS.  Caching is delivered three ways -- Anthropic breakpoints,
Google ``CachedContent``, OpenRouter's gateway annotation -- with three
knob spellings, two config layers and different defaults.  Before this
field an author had to know which mechanism their provider used in order
to turn caching on at all, and §4 of
``docs/design/model-tier-prompt-cache.md`` records what that cost: a
documented knob that reached no ingress and was ignored for months.

The risk this guard exists for is the same one in a new shape -- a
provider that CAN cache but has no delivery mapping, so the common field
is silently inert for it.  ``test_every_caching_provider_has_a_mapping``
derives its expectation from the capability contract rather than a list,
so a new caching provider cannot land without one.
"""

import importlib
import pathlib

import pytest

from shared.jaato_runtime import (
    CACHE_FIELD_DELIVERY,
    cache_field_to_provider_extra,
    resolve_provider_extra,
)
from shared.plugins.subagent.config import CacheProfileConfig, parse_cache_block
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

#: The defect, put back: the common field stops being laid beneath the
#: per-provider knobs, so declaring `cache:` does nothing.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/jaato_runtime.py",
        find="    extra = _layer_onto(dict(base_extra), cache_extra or {})",
        replace="    extra = dict(base_extra)",
        test="test_the_common_field_reaches_the_provider_extras",
        because="a declared cache: field that changes nothing",
    ),
]

PROVIDER_DIR = (pathlib.Path(__file__).resolve().parents[1]
                / "plugins" / "model_provider")


def _caching_providers():
    """Providers whose capability contract declares prompt_caching=True."""
    found = []
    for entry in sorted(p.name for p in PROVIDER_DIR.iterdir() if p.is_dir()):
        if entry.startswith("_") or entry in {"tests", "bundle_common", "echo"}:
            continue
        try:
            mod = importlib.import_module(
                f"shared.plugins.model_provider.{entry}")
        except Exception:
            continue
        caps = getattr(mod, "PROVIDER_CAPABILITIES", None)
        if caps is not None and getattr(caps, "prompt_caching", False):
            found.append(entry)
    return found


class TestCoverage:
    def test_every_caching_provider_has_a_mapping(self):
        """Derived from the capability contract, not from a list.

        A provider declaring it can cache, with no delivery entry, would
        accept a `cache:` block and do nothing with it -- the silently
        inert knob this whole area exists to have stopped.
        """
        caching = set(_caching_providers())
        assert caching, "no provider declares prompt_caching — guard is blind"
        missing = caching - set(CACHE_FIELD_DELIVERY)
        assert not missing, (
            f"{sorted(missing)} declare prompt_caching=True but have no "
            f"CACHE_FIELD_DELIVERY entry, so the common cache: field is "
            f"silently inert for them"
        )

    def test_no_mapping_for_a_provider_that_cannot_cache(self):
        """The converse: a stale entry would emit knobs nobody reads."""
        caching = set(_caching_providers())
        extra = set(CACHE_FIELD_DELIVERY) - caching
        assert not extra, (
            f"{sorted(extra)} have a delivery mapping but do not declare "
            f"prompt_caching=True"
        )


class TestTranslation:
    CACHE = CacheProfileConfig(enabled=True, ttl="1h", history=False)

    def test_anthropic_gets_flat_plugin_knobs(self):
        assert cache_field_to_provider_extra(self.CACHE, "anthropic") == {
            "enable_caching": True, "cache_ttl": "1h", "cache_history": False}

    def test_google_gets_a_duration_in_seconds(self):
        """Its API takes a Google duration string, not 5m/1h."""
        out = cache_field_to_provider_extra(self.CACHE, "google_genai")
        assert out["cache_ttl"] == "3600s"
        assert "cache_history" not in out, (
            "CachedContent holds system+tools; there is no history "
            "breakpoint to switch on, so the key would be meaningless"
        )

    def test_openrouter_gets_its_api_params_layer(self):
        """It caches internally and reads from a sub-dict, not flat."""
        assert cache_field_to_provider_extra(self.CACHE, "openrouter") == {
            "api_params": {"cache_prompt": True, "cache_ttl": "1h"}}

    def test_auto_leaves_the_provider_default_alone(self):
        """`auto` means "do not decide" — writing a value would invert it."""
        out = cache_field_to_provider_extra(CacheProfileConfig(), "anthropic")
        assert "enable_caching" not in out

    def test_auto_is_a_real_value_for_openrouter(self):
        """The one exception: `cache_prompt: "auto"` exists in its API."""
        out = cache_field_to_provider_extra(CacheProfileConfig(), "openrouter")
        assert out["api_params"]["cache_prompt"] == "auto"

    def test_a_provider_that_cannot_cache_is_a_no_op(self):
        assert cache_field_to_provider_extra(
            self.CACHE, "nim", supports_caching=False) == {}

    def test_no_cache_block_is_a_no_op(self):
        assert cache_field_to_provider_extra(None, "anthropic") == {}


class TestPrecedence:
    def test_the_common_field_reaches_the_provider_extras(self):
        cache_extra = cache_field_to_provider_extra(
            CacheProfileConfig(enabled=True), "anthropic")
        extra, _ = resolve_provider_extra({}, None, "anthropic",
                                          cache_extra=cache_extra)
        assert extra["enable_caching"] is True

    def test_an_explicit_provider_knob_wins(self):
        """`plugin_configs.<provider>` is the escape hatch, so it escapes.

        An earlier draft of §7 had this backwards — the profile field
        beating the knobs it called "the escape hatch".
        """
        cache_extra = cache_field_to_provider_extra(
            CacheProfileConfig(enabled=True, ttl="1h"), "anthropic")
        extra, _ = resolve_provider_extra(
            {}, {"anthropic": {"cache_ttl": "5m"}}, "anthropic",
            cache_extra=cache_extra)
        assert extra["cache_ttl"] == "5m"      # the explicit knob
        assert extra["enable_caching"] is True  # the field, where no clash

    def test_a_sub_dict_is_merged_not_replaced(self):
        """The failure a flat update would cause.

        OpenRouter's cache knobs live inside `api_params`; a profile that
        also sets `api_params.temperature` must keep both, whichever
        layer lands last.
        """
        cache_extra = cache_field_to_provider_extra(
            CacheProfileConfig(enabled=True), "openrouter")
        extra, _ = resolve_provider_extra(
            {}, {"openrouter": {"api_params": {"temperature": 0.0}}},
            "openrouter", cache_extra=cache_extra)
        assert extra["api_params"]["temperature"] == 0.0
        assert extra["api_params"]["cache_prompt"] is True


class TestParsing:
    def test_absent_block_is_none(self):
        assert parse_cache_block({}) is None

    @pytest.mark.parametrize("block,why", [
        ({"enabled": "sometimes"}, "enabled"),
        ({"ttl": "7m"}, "ttl"),
        ({"history": "yes"}, "history"),
    ])
    def test_a_bad_value_raises_rather_than_defaulting(self, block, why):
        """Silently falling back to a default is how §4 happened."""
        with pytest.raises(ValueError, match=why):
            parse_cache_block({"cache": block})

    def test_string_booleans_are_accepted(self):
        """YAML authors write `enabled: "true"` — meet them there."""
        assert parse_cache_block({"cache": {"enabled": "true"}}).enabled is True
        assert parse_cache_block({"cache": {"enabled": "off"}}).enabled is False
