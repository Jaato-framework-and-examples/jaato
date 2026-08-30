"""Tests for GoogleGenAICachePlugin."""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from shared.plugins.cache_google_genai.plugin import (
    GoogleGenAICachePlugin,
    create_plugin,
    _MIN_CACHE_TOKENS,
    _CHARS_PER_TOKEN,
)


# ==================== Factory ====================


class TestCreatePlugin:
    def test_factory_returns_plugin(self):
        plugin = create_plugin()
        assert isinstance(plugin, GoogleGenAICachePlugin)

    def test_factory_returns_new_instance(self):
        p1 = create_plugin()
        p2 = create_plugin()
        assert p1 is not p2


# ==================== Properties ====================


class TestProperties:
    def test_name(self):
        plugin = GoogleGenAICachePlugin()
        assert plugin.name == "cache_google_genai"

    def test_provider_name(self):
        plugin = GoogleGenAICachePlugin()
        assert plugin.provider_name == "google_genai"

    def test_compatible_models(self):
        plugin = GoogleGenAICachePlugin()
        models = plugin.compatible_models
        assert any("gemini" in m for m in models)

    def test_compatible_models_cover_gemini_families(self):
        """Verify all major Gemini model families are covered."""
        plugin = GoogleGenAICachePlugin()
        models = plugin.compatible_models
        assert any("1.5" in m for m in models)
        assert any("2.0" in m for m in models)
        assert any("2.5" in m for m in models)
        assert any("3" in m for m in models)


# ==================== Initialize / Shutdown ====================


class TestInitializeShutdown:
    def test_initialize_with_caching_enabled(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize({"enable_caching": True, "model_name": "gemini-2.5-flash"})
        assert plugin._enabled is True
        assert plugin._model_name == "gemini-2.5-flash"

    def test_initialize_with_caching_disabled(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize({"enable_caching": False})
        assert plugin._enabled is False

    def test_initialize_default_disabled(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize({})
        assert plugin._enabled is False

    def test_initialize_none_config(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize(None)
        assert plugin._enabled is False

    def test_initialize_custom_ttl(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize({"enable_caching": True, "cache_ttl": "1800s"})
        assert plugin._cache_ttl == "1800s"

    def test_initialize_default_ttl(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize({"enable_caching": True})
        assert plugin._cache_ttl == "3600s"

    def test_shutdown_deletes_cached_content(self):
        plugin = GoogleGenAICachePlugin()
        plugin._client = MagicMock()
        plugin._cached_content_name = "cachedContents/abc123"
        plugin.shutdown()
        plugin._client.caches.delete.assert_called_once_with(
            name="cachedContents/abc123"
        )
        assert plugin._cached_content_name is None

    def test_shutdown_noop_when_no_cache(self):
        plugin = GoogleGenAICachePlugin()
        plugin._client = MagicMock()
        plugin.shutdown()
        plugin._client.caches.delete.assert_not_called()

    def test_shutdown_noop_when_no_client(self):
        plugin = GoogleGenAICachePlugin()
        plugin._cached_content_name = "cachedContents/abc123"
        plugin.shutdown()  # Should not raise


# ==================== Set Client ====================


class TestSetClient:
    def test_stores_client_reference(self):
        plugin = GoogleGenAICachePlugin()
        client = MagicMock()
        plugin.set_client(client)
        assert plugin._client is client


# ==================== Prepare Request — Disabled / Passthrough ====================


class TestPrepareRequestPassthrough:
    """When caching is disabled, prepare_request is a passthrough."""

    def test_passthrough_when_disabled(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize({"enable_caching": False})

        system = "System prompt"
        result = plugin.prepare_request(system, [], [])

        assert result["system"] is system
        assert "cached_content" not in result

    def test_passthrough_when_no_client(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize({"enable_caching": True, "model_name": "gemini-2.5-flash"})
        # No client set

        result = plugin.prepare_request("System", [], [])
        assert "cached_content" not in result

    def test_passthrough_when_no_model(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize({"enable_caching": True})
        plugin.set_client(MagicMock())
        # No model_name

        result = plugin.prepare_request("System", [], [])
        assert "cached_content" not in result

    def test_passthrough_when_content_below_threshold(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize(
            {"enable_caching": True, "model_name": "gemini-2.5-flash"}
        )
        plugin.set_client(MagicMock())

        # Small system prompt — below 32K token threshold
        result = plugin.prepare_request("Hello", [], [])
        assert "cached_content" not in result

    def test_passthrough_system_unchanged(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        system = [{"type": "text", "text": "System prompt"}]
        result = plugin.prepare_request(system, [], [])
        assert result["system"] is system

    def test_passthrough_tools_unchanged(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        tools = [{"name": "a", "input_schema": {}}]
        result = plugin.prepare_request([], tools, [])
        assert result["tools"] is tools

    def test_passthrough_messages_unchanged(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        messages = [{"role": "user", "content": "Hi"}]
        result = plugin.prepare_request([], [], messages)
        assert result["messages"] is messages

    def test_no_cache_breakpoint(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        result = plugin.prepare_request([], [], [])
        assert result["cache_breakpoint_index"] == -1

    def test_returns_all_required_keys(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        result = plugin.prepare_request("sys", [], [])
        assert "system" in result
        assert "tools" in result
        assert "messages" in result
        assert "cache_breakpoint_index" in result


# ==================== Prepare Request — Active Caching ====================


class TestPrepareRequestActive:
    """When caching is enabled with sufficient content, CachedContent is created."""

    def _make_plugin(self, model="gemini-2.5-flash", ttl="3600s"):
        """Create a configured plugin with a mock client.

        Patches ``_create_cached_content`` to avoid real API calls while
        still testing the prepare_request orchestration logic.
        """
        plugin = GoogleGenAICachePlugin()
        plugin.initialize({
            "enable_caching": True,
            "model_name": model,
            "cache_ttl": ttl,
        })
        client = MagicMock()
        plugin.set_client(client)

        # Patch _create_cached_content to simulate successful cache creation
        original_create = plugin._create_cached_content

        def fake_create(system, tools):
            plugin._cached_content_name = "cachedContents/test-cache-id"
            # The real ``_create_cached_content`` records WHICH model the
            # cache was bound to, and the mismatch guard in
            # ``prepare_request`` compares against it.  A double that sets
            # only the name is a shape production never has, and it reads
            # as a cache bound to nothing -- which the guard correctly
            # withholds.
            plugin._cached_content_model = plugin._model_name
            client.caches.create()  # Track that create was called

        plugin._create_cached_content = fake_create
        return plugin, client

    def _large_system(self):
        """System instruction large enough to meet token threshold."""
        return "x" * (_MIN_CACHE_TOKENS * _CHARS_PER_TOKEN + 100)

    def test_creates_cached_content(self):
        plugin, client = self._make_plugin()

        result = plugin.prepare_request(self._large_system(), [], [])

        assert result["cached_content"] == "cachedContents/test-cache-id"
        client.caches.create.assert_called_once()

    def test_reuses_cache_on_same_content(self):
        plugin, client = self._make_plugin()
        system = self._large_system()

        plugin.prepare_request(system, [], [])
        plugin.prepare_request(system, [], [])

        # Only one create call — cache is reused
        assert client.caches.create.call_count == 1

    def test_recreates_cache_on_content_change(self):
        plugin, client = self._make_plugin()

        plugin.prepare_request(self._large_system(), [], [])

        # Simulate content change — old cache should be deleted, new one created
        plugin.prepare_request(self._large_system() + " changed", [], [])

        # Create called twice (initial + recreate)
        assert client.caches.create.call_count == 2

    def test_cache_creation_failure_falls_through(self):
        """If cache creation fails, plugin falls through to non-cached mode."""
        plugin = GoogleGenAICachePlugin()
        plugin.initialize({
            "enable_caching": True,
            "model_name": "gemini-2.5-flash",
        })
        client = MagicMock()
        plugin.set_client(client)

        # Override _create_cached_content to simulate failure
        def failing_create(system, tools):
            plugin._cached_content_name = None

        plugin._create_cached_content = failing_create

        result = plugin.prepare_request(self._large_system(), [], [])

        assert "cached_content" not in result
        assert result["cache_breakpoint_index"] == -1

    def test_cached_content_name_property(self):
        plugin, _ = self._make_plugin()

        assert plugin.cached_content_name is None
        plugin.prepare_request(self._large_system(), [], [])
        assert plugin.cached_content_name == "cachedContents/test-cache-id"

    def test_result_includes_standard_keys(self):
        plugin, _ = self._make_plugin()
        system = self._large_system()

        result = plugin.prepare_request(system, [{"name": "t"}], [{"role": "user"}])

        assert result["system"] is system
        assert result["tools"] == [{"name": "t"}]
        assert result["messages"] == [{"role": "user"}]
        assert result["cache_breakpoint_index"] == -1
        assert "cached_content" in result


# ==================== Token Threshold ====================


class TestTokenThreshold:
    def test_below_threshold_returns_false(self):
        plugin = GoogleGenAICachePlugin()
        # Small content
        assert plugin._meets_token_threshold("short", []) is False

    def test_above_threshold_returns_true(self):
        plugin = GoogleGenAICachePlugin()
        large = "x" * (_MIN_CACHE_TOKENS * _CHARS_PER_TOKEN + 1)
        assert plugin._meets_token_threshold(large, []) is True

    def test_tools_contribute_to_threshold(self):
        plugin = GoogleGenAICachePlugin()
        # System alone below threshold, but tools push it over
        half = "x" * (_MIN_CACHE_TOKENS * _CHARS_PER_TOKEN // 2)
        tools = [{"name": "t", "description": half}]
        assert plugin._meets_token_threshold(half, tools) is True

    def test_none_system_handled(self):
        plugin = GoogleGenAICachePlugin()
        assert plugin._meets_token_threshold(None, []) is False

    def test_empty_tools_handled(self):
        plugin = GoogleGenAICachePlugin()
        assert plugin._meets_token_threshold("x" * 200_000, None) is True


# ==================== Content Hash ====================


class TestContentHash:
    def test_same_content_same_hash(self):
        plugin = GoogleGenAICachePlugin()
        h1 = plugin._compute_content_hash("sys", [{"name": "a"}])
        h2 = plugin._compute_content_hash("sys", [{"name": "a"}])
        assert h1 == h2

    def test_different_system_different_hash(self):
        plugin = GoogleGenAICachePlugin()
        h1 = plugin._compute_content_hash("sys1", [])
        h2 = plugin._compute_content_hash("sys2", [])
        assert h1 != h2

    def test_different_tools_different_hash(self):
        plugin = GoogleGenAICachePlugin()
        h1 = plugin._compute_content_hash("sys", [{"name": "a"}])
        h2 = plugin._compute_content_hash("sys", [{"name": "b"}])
        assert h1 != h2

    def test_hash_is_hex_string(self):
        plugin = GoogleGenAICachePlugin()
        h = plugin._compute_content_hash("sys", [])
        assert len(h) == 64  # SHA-256 hex digest


# ==================== Tool Conversion ====================


class TestToolConversion:
    def test_empty_tools_returns_none(self):
        result = GoogleGenAICachePlugin._convert_tools_to_sdk([])
        assert result is None

    def test_none_tools_returns_none(self):
        result = GoogleGenAICachePlugin._convert_tools_to_sdk(None)
        assert result is None

    def test_conversion_with_empty_list_returns_none(self):
        """Empty tool list returns None (graceful degradation)."""
        result = GoogleGenAICachePlugin._convert_tools_to_sdk([])
        assert result is None

    def test_dict_tools_converted(self):
        """Tools in dict format are converted to SDK format."""
        try:
            from google.genai import types as genai_types
        except ImportError:
            pytest.skip("google-genai not installed")

        tools = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        result = GoogleGenAICachePlugin._convert_tools_to_sdk(tools)
        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], genai_types.Tool)

    def test_toolschema_objects_converted(self):
        """ToolSchema-like objects are converted to SDK format."""
        try:
            from google.genai import types as genai_types
        except ImportError:
            pytest.skip("google-genai not installed")

        tool = MagicMock()
        tool.name = "my_tool"
        tool.description = "Does things"
        tool.parameters = {"type": "object", "properties": {"x": {"type": "string"}}}

        result = GoogleGenAICachePlugin._convert_tools_to_sdk([tool])
        assert result is not None
        assert len(result) == 1


# ==================== Extract Cache Usage ====================


class TestExtractCacheUsage:
    def test_tracks_cache_read_tokens(self):
        plugin = GoogleGenAICachePlugin()

        usage = MagicMock(spec=[])
        usage.cache_read_tokens = 250
        usage.cache_creation_tokens = 0

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_read_tokens == 250

    def test_tracks_cache_creation_tokens(self):
        plugin = GoogleGenAICachePlugin()

        usage = MagicMock(spec=[])
        usage.cache_read_tokens = 0
        usage.cache_creation_tokens = 500

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_creation_tokens == 500

    def test_accumulates_across_calls(self):
        plugin = GoogleGenAICachePlugin()

        for _ in range(4):
            usage = MagicMock(spec=[])
            usage.cache_read_tokens = 100
            usage.cache_creation_tokens = 50
            plugin.extract_cache_usage(usage)

        assert plugin.total_cache_read_tokens == 400
        assert plugin.total_cache_creation_tokens == 200

    def test_ignores_none(self):
        plugin = GoogleGenAICachePlugin()

        usage = MagicMock(spec=[])
        usage.cache_read_tokens = None
        usage.cache_creation_tokens = None

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_read_tokens == 0
        assert plugin.total_cache_creation_tokens == 0

    def test_ignores_zero(self):
        plugin = GoogleGenAICachePlugin()

        usage = MagicMock(spec=[])
        usage.cache_read_tokens = 0
        usage.cache_creation_tokens = 0

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_read_tokens == 0
        assert plugin.total_cache_creation_tokens == 0

    def test_ignores_negative(self):
        plugin = GoogleGenAICachePlugin()

        usage = MagicMock(spec=[])
        usage.cache_read_tokens = -1
        usage.cache_creation_tokens = -1

        plugin.extract_cache_usage(usage)
        assert plugin.total_cache_read_tokens == 0
        assert plugin.total_cache_creation_tokens == 0


# ==================== GC Coordination ====================


class TestOnGCResult:
    def test_gc_with_freed_tokens_increments_count(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        result = MagicMock()
        result.tokens_freed = 500

        plugin.on_gc_result(result)
        assert plugin.gc_invalidation_count == 1

    def test_gc_without_freed_tokens_no_increment(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        result = MagicMock()
        result.tokens_freed = 0

        plugin.on_gc_result(result)
        assert plugin.gc_invalidation_count == 0

    def test_count_accumulates(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        for _ in range(3):
            result = MagicMock()
            result.tokens_freed = 100
            plugin.on_gc_result(result)

        assert plugin.gc_invalidation_count == 3


# ==================== Budget ====================


class TestSetBudget:
    def test_stores_budget_reference(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        budget = MagicMock()
        plugin.set_budget(budget)
        assert plugin._budget is budget

    def test_replaces_previous_budget(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize()

        budget1 = MagicMock()
        budget2 = MagicMock()
        plugin.set_budget(budget1)
        plugin.set_budget(budget2)
        assert plugin._budget is budget2


# ==================== Protocol Compliance ====================


class TestProtocolCompliance:
    """Verify the plugin satisfies the CachePlugin protocol."""

    def test_has_all_protocol_methods(self):
        plugin = GoogleGenAICachePlugin()
        assert hasattr(plugin, "name")
        assert hasattr(plugin, "provider_name")
        assert hasattr(plugin, "compatible_models")
        assert hasattr(plugin, "initialize")
        assert hasattr(plugin, "shutdown")
        assert hasattr(plugin, "set_budget")
        assert hasattr(plugin, "prepare_request")
        assert hasattr(plugin, "extract_cache_usage")
        assert hasattr(plugin, "on_gc_result")

    def test_isinstance_check(self):
        """Verify runtime_checkable protocol compliance."""
        from shared.plugins.cache.base import CachePlugin

        plugin = GoogleGenAICachePlugin()
        assert isinstance(plugin, CachePlugin)

    def test_has_set_client_method(self):
        """Active caching requires set_client for API access."""
        plugin = GoogleGenAICachePlugin()
        assert hasattr(plugin, "set_client")
        assert callable(plugin.set_client)


# ==================== Delete Cached Content ====================


class TestDeleteCachedContent:
    def test_delete_calls_client(self):
        plugin = GoogleGenAICachePlugin()
        plugin._client = MagicMock()
        plugin._cached_content_name = "cachedContents/xyz"

        plugin._delete_cached_content()

        plugin._client.caches.delete.assert_called_once_with(
            name="cachedContents/xyz"
        )
        assert plugin._cached_content_name is None

    def test_delete_ignores_api_error(self):
        """Expired cache should not cause errors."""
        plugin = GoogleGenAICachePlugin()
        plugin._client = MagicMock()
        plugin._client.caches.delete.side_effect = Exception("Not found")
        plugin._cached_content_name = "cachedContents/expired"

        plugin._delete_cached_content()  # Should not raise
        assert plugin._cached_content_name is None

    def test_delete_noop_when_no_name(self):
        plugin = GoogleGenAICachePlugin()
        plugin._client = MagicMock()
        plugin._cached_content_name = None

        plugin._delete_cached_content()
        plugin._client.caches.delete.assert_not_called()

    def test_delete_noop_when_no_client(self):
        plugin = GoogleGenAICachePlugin()
        plugin._cached_content_name = "cachedContents/xyz"

        plugin._delete_cached_content()  # Should not raise


# ==================== Model rebinding (tier switches) ====================


class TestSetModelName:
    """A ``CachedContent`` belongs to ONE model.

    ``client.caches.create(model=...)`` binds it, and the reuse test in
    ``prepare_request`` hashes system+tools only — it has no notion of the
    model.  So rebinding the model has to discard the cache, or a tier
    switch that left system and tools untouched would hand the new model a
    cache name bound to the old one.  See
    ``docs/design/model-tier-prompt-cache.md`` §5.2/§5.3.
    """

    def _live_plugin(self):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize({"enable_caching": True,
                           "model_name": "gemini-3-flash"})
        plugin.set_client(MagicMock())
        plugin._cached_content_name = "cachedContents/abc"
        plugin._content_hash = "deadbeef"
        return plugin

    def test_a_new_model_discards_the_cache(self):
        plugin = self._live_plugin()

        plugin.set_model_name("gemini-3-pro")

        assert plugin._model_name == "gemini-3-pro"
        assert plugin._cached_content_name is None
        assert plugin._content_hash is None, (
            "a surviving hash lets the next prepare_request reuse a cache "
            "bound to the previous model"
        )

    def test_the_orphaned_cache_is_deleted_not_just_forgotten(self):
        """It is server-side and billed for its TTL."""
        plugin = self._live_plugin()

        plugin.set_model_name("gemini-3-pro")

        plugin._client.caches.delete.assert_called_once_with(
            name="cachedContents/abc")

    def test_the_same_model_is_a_no_op(self):
        """The session pushes the model on every wire, including re-wires
        that changed nothing; that must not throw away a live cache."""
        plugin = self._live_plugin()

        plugin.set_model_name("gemini-3-flash")

        assert plugin._cached_content_name == "cachedContents/abc"
        assert plugin._content_hash == "deadbeef"
        plugin._client.caches.delete.assert_not_called()


# ============ The mismatch guard (§5.3, second half) ============


class TestMismatchGuard:
    """A ``CachedContent`` name is never emitted to a different model.

    ``set_model_name`` already discards the cache when the model changes,
    so on the paths we know about this cannot fire.  It exists because
    that fix closes a PATH and this closes the CLASS: the reuse test in
    ``prepare_request`` hashes system+tools and has no notion of the
    model, so if anything ever sets ``_model_name`` without going through
    the setter, nothing else here would notice.

    Degrading to an uncached request is correct and merely slower;
    emitting the name would send a reference the API will not honour.
    """

    def _plugin_with_live_cache(self, bound_to="gemini-3-flash"):
        plugin = GoogleGenAICachePlugin()
        plugin.initialize({"enable_caching": True, "model_name": bound_to})
        plugin.set_client(MagicMock())
        plugin._cached_content_name = "cachedContents/abc"
        plugin._cached_content_model = bound_to
        plugin._content_hash = "matches-so-no-rebuild"
        return plugin

    def _prepare(self, plugin):
        # A system block big enough to clear the ~32k threshold, and a
        # hash that matches, so the reuse path is the one under test.
        system = "x" * (_MIN_CACHE_TOKENS * _CHARS_PER_TOKEN + 10)
        plugin._content_hash = plugin._compute_content_hash(system, [])
        return plugin.prepare_request(system, [], [])

    def test_a_matching_model_still_gets_the_cache(self):
        """The guard must not break the normal path."""
        plugin = self._plugin_with_live_cache()
        assert self._prepare(plugin).get("cached_content") == "cachedContents/abc"

    def test_a_mismatched_model_withholds_the_name(self):
        """The crux: the model moved without the cache being dropped."""
        plugin = self._plugin_with_live_cache(bound_to="gemini-3-flash")
        plugin._model_name = "gemini-3-pro"          # bypasses set_model_name

        result = self._prepare(plugin)

        assert "cached_content" not in result, (
            "handed a request on gemini-3-pro a CachedContent created for "
            "gemini-3-flash"
        )

    def test_the_withholding_is_counted_not_only_logged(self):
        """A log line nobody reads is not an observation."""
        plugin = self._plugin_with_live_cache(bound_to="gemini-3-flash")
        plugin._model_name = "gemini-3-pro"

        self._prepare(plugin)

        assert plugin._mismatched_content_withheld == 1
        assert plugin.get_telemetry_attributes()["cache.mismatched_withheld"] == 1

    def test_creation_records_the_model_it_bound_to(self):
        """The guard needs the binding recorded, or it compares nothing."""
        plugin = GoogleGenAICachePlugin()
        plugin.initialize({"enable_caching": True, "model_name": "gemini-3-pro"})
        client = MagicMock()
        client.caches.create.return_value = SimpleNamespace(
            name="cachedContents/new")
        plugin.set_client(client)

        plugin._create_cached_content("sys", [])

        assert plugin._cached_content_name == "cachedContents/new"
        assert plugin._cached_content_model == "gemini-3-pro"

    def test_rebinding_clears_the_recorded_model_too(self):
        """Otherwise a stale binding outlives the cache it described."""
        plugin = self._plugin_with_live_cache(bound_to="gemini-3-flash")

        plugin.set_model_name("gemini-3-pro")

        assert plugin._cached_content_name is None
        assert plugin._cached_content_model is None
