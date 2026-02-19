"""Tests for service connector session persistence."""

import pytest
from unittest.mock import MagicMock, patch

from ..plugin import ServiceConnectorPlugin
from ..types import (
    AuthConfig,
    DiscoveredService,
    EndpointSchema,
    ServiceConfig,
)


class TestServiceConnectorPersistence:
    """Tests for get_persistence_state / restore_persistence_state."""

    @pytest.fixture
    def plugin(self):
        """Create a plugin with mock schema store."""
        p = ServiceConnectorPlugin()
        p._schema_store = MagicMock()
        p._initialized = True
        return p

    def _make_discovered(self, name: str) -> DiscoveredService:
        """Create a minimal DiscoveredService for testing."""
        config = ServiceConfig(
            name=name,
            base_url=f"https://{name}.example.com",
            title=f"{name} API",
            version="1.0",
        )
        endpoints = [
            EndpointSchema(
                method="GET",
                path="/items/{id}",
                summary="Get an item",
            ),
        ]
        return DiscoveredService(
            config=config,
            endpoints=endpoints,
            source=f"https://{name}.example.com/openapi.json",
        )

    def test_empty_state_returns_empty_dict(self, plugin):
        """No discovered services -> empty state."""
        state = plugin.get_persistence_state()
        assert state == {}

    def test_state_captures_discovered_services(self, plugin):
        """Discovered services are captured as name list."""
        plugin._discovered_services["github"] = self._make_discovered("github")
        plugin._discovered_services["stripe"] = self._make_discovered("stripe")

        state = plugin.get_persistence_state()

        assert state["version"] == 1
        assert sorted(state["discovered_services"]) == ["github", "stripe"]

    def test_restore_prewarms_cache(self, plugin):
        """Restore loads services from SchemaStore into memory."""
        github_svc = self._make_discovered("github")

        # SchemaStore returns config + endpoints for "github"
        plugin._schema_store.load_discovered_service.return_value = (
            github_svc.config,
            github_svc.endpoints,
        )
        plugin._schema_store.get_discovered_source.return_value = github_svc.source

        state = {
            "discovered_services": ["github"],
            "version": 1,
        }
        plugin.restore_persistence_state(state)

        # Should have pre-warmed the cache
        assert "github" in plugin._discovered_services
        assert plugin._discovered_services["github"].config.name == "github"

    def test_restore_skips_missing_services(self, plugin):
        """Services not found in SchemaStore are silently skipped."""
        plugin._schema_store.load_discovered_service.return_value = None

        state = {
            "discovered_services": ["nonexistent"],
            "version": 1,
        }
        plugin.restore_persistence_state(state)

        assert "nonexistent" not in plugin._discovered_services

    def test_restore_does_not_overwrite_existing(self, plugin):
        """Already-cached services are not reloaded."""
        existing = self._make_discovered("github")
        plugin._discovered_services["github"] = existing

        state = {
            "discovered_services": ["github"],
            "version": 1,
        }
        plugin.restore_persistence_state(state)

        # SchemaStore should not have been called for "github"
        plugin._schema_store.load_discovered_service.assert_not_called()
        # Original object should still be there
        assert plugin._discovered_services["github"] is existing

    def test_roundtrip(self, plugin):
        """State survives a save/restore cycle."""
        plugin._discovered_services["github"] = self._make_discovered("github")
        plugin._discovered_services["stripe"] = self._make_discovered("stripe")

        state = plugin.get_persistence_state()

        # Simulate new plugin instance
        plugin2 = ServiceConnectorPlugin()
        plugin2._schema_store = MagicMock()
        plugin2._initialized = True

        # SchemaStore returns data for both services
        def mock_load(name):
            svc = self._make_discovered(name)
            return (svc.config, svc.endpoints)

        plugin2._schema_store.load_discovered_service.side_effect = mock_load
        plugin2._schema_store.get_discovered_source.return_value = "https://example.com"

        plugin2.restore_persistence_state(state)

        assert sorted(plugin2._discovered_services.keys()) == ["github", "stripe"]
