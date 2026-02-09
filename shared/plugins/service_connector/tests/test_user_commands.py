"""Tests for service connector user commands."""

import os
import pytest
from unittest.mock import patch

from ..plugin import ServiceConnectorPlugin
from ..types import (
    AuthConfig,
    AuthType,
    DiscoveredService,
    EndpointSchema,
    ParameterLocation,
    ServiceConfig,
)
from ...base import CommandCompletion, HelpLines


@pytest.fixture
def plugin(tmp_path):
    """Create an initialized plugin with a temp workspace."""
    p = ServiceConnectorPlugin()
    p.initialize({"workspace_path": str(tmp_path)})
    return p


@pytest.fixture
def plugin_with_services(plugin):
    """Plugin pre-loaded with two discovered services."""
    plugin._discovered_services["petstore"] = DiscoveredService(
        config=ServiceConfig(
            name="petstore",
            base_url="https://petstore.example.com/v1",
            title="Petstore API",
            version="1.0.0",
            auth=AuthConfig(type=AuthType.API_KEY, key_name="X-API-Key",
                            key_location=ParameterLocation.HEADER,
                            value_env="PETSTORE_KEY"),
        ),
        endpoints=[
            EndpointSchema(method="GET", path="/pets", summary="List all pets"),
            EndpointSchema(method="POST", path="/pets", summary="Create a pet"),
            EndpointSchema(method="GET", path="/pets/{petId}", summary="Get pet by ID"),
            EndpointSchema(method="DELETE", path="/pets/{petId}", summary="Delete a pet"),
        ],
        source="https://petstore.example.com/openapi.json",
    )
    plugin._discovered_services["github"] = DiscoveredService(
        config=ServiceConfig(
            name="github",
            base_url="https://api.github.com",
            title="GitHub API",
            version="2022-11-28",
            auth=AuthConfig(type=AuthType.BEARER, value_env="GITHUB_TOKEN"),
        ),
        endpoints=[
            EndpointSchema(method="GET", path="/user", summary="Get authenticated user"),
            EndpointSchema(method="GET", path="/repos/{owner}/{repo}", summary="Get a repo"),
        ],
        source="https://api.github.com/openapi.json",
    )
    return plugin


class TestGetUserCommands:
    """Tests for get_user_commands()."""

    def test_returns_services_command(self, plugin):
        commands = plugin.get_user_commands()
        assert len(commands) == 1
        assert commands[0].name == "services"
        assert commands[0].share_with_model is False
        assert commands[0].parameters is not None
        assert len(commands[0].parameters) == 2

    def test_services_in_auto_approved(self, plugin):
        approved = plugin.get_auto_approved_tools()
        assert "services" in approved


class TestGetCommandCompletions:
    """Tests for get_command_completions()."""

    def test_wrong_command_returns_empty(self, plugin):
        assert plugin.get_command_completions("other", []) == []

    def test_no_args_returns_all_subcommands(self, plugin):
        completions = plugin.get_command_completions("services", [])
        values = [c.value for c in completions]
        assert "list" in values
        assert "show" in values
        assert "endpoints" in values
        assert "auth" in values
        assert "remove" in values
        assert "help" in values

    def test_partial_subcommand_filters(self, plugin):
        completions = plugin.get_command_completions("services", ["sh"])
        values = [c.value for c in completions]
        assert values == ["show"]

    def test_partial_subcommand_multiple_matches(self, plugin):
        completions = plugin.get_command_completions("services", ["l"])
        values = [c.value for c in completions]
        assert "list" in values

    def test_show_completes_service_names(self, plugin_with_services):
        completions = plugin_with_services.get_command_completions(
            "services", ["show", ""]
        )
        values = [c.value for c in completions]
        assert "petstore" in values
        assert "github" in values

    def test_show_filters_service_names(self, plugin_with_services):
        completions = plugin_with_services.get_command_completions(
            "services", ["show", "pet"]
        )
        values = [c.value for c in completions]
        assert values == ["petstore"]

    def test_endpoints_completes_service_names(self, plugin_with_services):
        completions = plugin_with_services.get_command_completions(
            "services", ["endpoints", ""]
        )
        values = [c.value for c in completions]
        assert "petstore" in values
        assert "github" in values

    def test_endpoints_completes_methods(self, plugin_with_services):
        completions = plugin_with_services.get_command_completions(
            "services", ["endpoints", "petstore", "G"]
        )
        values = [c.value for c in completions]
        assert values == ["GET"]

    def test_endpoints_all_methods(self, plugin_with_services):
        completions = plugin_with_services.get_command_completions(
            "services", ["endpoints", "petstore", ""]
        )
        values = [c.value for c in completions]
        assert set(values) == {"GET", "POST", "PUT", "DELETE", "PATCH"}

    def test_auth_completes_service_names(self, plugin_with_services):
        completions = plugin_with_services.get_command_completions(
            "services", ["auth", "g"]
        )
        values = [c.value for c in completions]
        assert values == ["github"]

    def test_remove_completes_service_names(self, plugin_with_services):
        completions = plugin_with_services.get_command_completions(
            "services", ["remove", ""]
        )
        values = [c.value for c in completions]
        assert "petstore" in values

    def test_no_completions_past_depth(self, plugin_with_services):
        completions = plugin_with_services.get_command_completions(
            "services", ["show", "petstore", "extra"]
        )
        assert completions == []

    def test_list_no_further_completions(self, plugin_with_services):
        completions = plugin_with_services.get_command_completions(
            "services", ["list", "anything"]
        )
        assert completions == []


class TestExecuteUserCommand:
    """Tests for execute_user_command()."""

    def test_unknown_command(self, plugin):
        result = plugin.execute_user_command("other", {})
        assert "Unknown command" in result

    def test_unknown_subcommand(self, plugin):
        result = plugin.execute_user_command("services", {"subcommand": "bogus", "rest": ""})
        assert "Unknown subcommand" in result

    def test_default_is_list(self, plugin):
        result = plugin.execute_user_command("services", {"subcommand": "", "rest": ""})
        assert "No services discovered" in result


class TestCmdList:
    """Tests for 'services list'."""

    def test_empty_list(self, plugin):
        result = plugin.execute_user_command("services", {"subcommand": "list", "rest": ""})
        assert "No services discovered" in result

    def test_list_services(self, plugin_with_services):
        result = plugin_with_services.execute_user_command(
            "services", {"subcommand": "list", "rest": ""}
        )
        assert "petstore" in result
        assert "github" in result
        assert "2 service(s)" in result
        assert "endpoints" in result

    def test_list_shows_auth_type(self, plugin_with_services):
        result = plugin_with_services.execute_user_command(
            "services", {"subcommand": "list", "rest": ""}
        )
        assert "auth=apiKey" in result
        assert "auth=bearer" in result


class TestCmdShow:
    """Tests for 'services show <service>'."""

    def test_show_missing_arg(self, plugin):
        result = plugin.execute_user_command("services", {"subcommand": "show", "rest": ""})
        assert "Usage" in result

    def test_show_not_found(self, plugin):
        result = plugin.execute_user_command(
            "services", {"subcommand": "show", "rest": "nonexistent"}
        )
        assert "not found" in result

    def test_show_service(self, plugin_with_services):
        result = plugin_with_services.execute_user_command(
            "services", {"subcommand": "show", "rest": "petstore"}
        )
        assert "petstore" in result
        assert "Petstore API" in result
        assert "1.0.0" in result
        assert "https://petstore.example.com/v1" in result
        assert "apiKey" in result
        assert "4" in result  # endpoint count
        assert "openapi.json" in result  # source


class TestCmdEndpoints:
    """Tests for 'services endpoints <service> [method]'."""

    def test_endpoints_missing_arg(self, plugin):
        result = plugin.execute_user_command(
            "services", {"subcommand": "endpoints", "rest": ""}
        )
        assert "Usage" in result

    def test_endpoints_not_found(self, plugin):
        result = plugin.execute_user_command(
            "services", {"subcommand": "endpoints", "rest": "nonexistent"}
        )
        assert "not found" in result

    def test_endpoints_all(self, plugin_with_services):
        result = plugin_with_services.execute_user_command(
            "services", {"subcommand": "endpoints", "rest": "petstore"}
        )
        assert "GET" in result
        assert "POST" in result
        assert "DELETE" in result
        assert "/pets" in result
        assert "4 endpoint(s)" in result

    def test_endpoints_filter_by_method(self, plugin_with_services):
        result = plugin_with_services.execute_user_command(
            "services", {"subcommand": "endpoints", "rest": "petstore GET"}
        )
        assert "GET" in result
        assert "POST" not in result
        assert "DELETE" not in result
        assert "filtered: GET" in result
        assert "2 endpoint(s)" in result

    def test_endpoints_filter_no_match(self, plugin_with_services):
        result = plugin_with_services.execute_user_command(
            "services", {"subcommand": "endpoints", "rest": "petstore PATCH"}
        )
        assert "No endpoints" in result


class TestCmdAuth:
    """Tests for 'services auth <service>'."""

    def test_auth_missing_arg(self, plugin):
        result = plugin.execute_user_command(
            "services", {"subcommand": "auth", "rest": ""}
        )
        assert "Usage" in result

    def test_auth_not_found(self, plugin):
        result = plugin.execute_user_command(
            "services", {"subcommand": "auth", "rest": "nonexistent"}
        )
        assert "not found" in result

    def test_auth_api_key_set(self, plugin_with_services):
        with patch.dict(os.environ, {"PETSTORE_KEY": "secret123"}):
            result = plugin_with_services.execute_user_command(
                "services", {"subcommand": "auth", "rest": "petstore"}
            )
        assert "apiKey" in result
        assert "PETSTORE_KEY" in result
        assert "(set)" in result

    def test_auth_bearer_missing(self, plugin_with_services):
        # Ensure GITHUB_TOKEN is not set
        env = os.environ.copy()
        env.pop("GITHUB_TOKEN", None)
        with patch.dict(os.environ, env, clear=True):
            result = plugin_with_services.execute_user_command(
                "services", {"subcommand": "auth", "rest": "github"}
            )
        assert "bearer" in result
        assert "GITHUB_TOKEN" in result
        assert "MISSING" in result

    def test_auth_none(self, plugin):
        plugin._discovered_services["noauth"] = DiscoveredService(
            config=ServiceConfig(
                name="noauth",
                base_url="https://example.com",
                auth=AuthConfig(type=AuthType.NONE),
            ),
        )
        result = plugin.execute_user_command(
            "services", {"subcommand": "auth", "rest": "noauth"}
        )
        assert "none" in result
        assert "No authentication configured" in result


class TestCmdRemove:
    """Tests for 'services remove <service>'."""

    def test_remove_missing_arg(self, plugin):
        result = plugin.execute_user_command(
            "services", {"subcommand": "remove", "rest": ""}
        )
        assert "Usage" in result

    def test_remove_not_found(self, plugin):
        result = plugin.execute_user_command(
            "services", {"subcommand": "remove", "rest": "nonexistent"}
        )
        assert "not found" in result

    def test_remove_from_memory(self, plugin_with_services):
        assert "petstore" in plugin_with_services._discovered_services
        result = plugin_with_services.execute_user_command(
            "services", {"subcommand": "remove", "rest": "petstore"}
        )
        assert "Removed" in result
        assert "memory" in result
        assert "petstore" not in plugin_with_services._discovered_services

    def test_remove_from_disk(self, plugin_with_services):
        # Save to disk first
        svc = plugin_with_services._discovered_services["github"]
        plugin_with_services._schema_store.save_discovered_service(
            "github", svc.config, svc.endpoints, source=svc.source
        )
        # Remove from memory to test disk-only removal
        del plugin_with_services._discovered_services["github"]

        result = plugin_with_services.execute_user_command(
            "services", {"subcommand": "remove", "rest": "github"}
        )
        assert "Removed" in result
        assert "disk" in result


class TestCmdHelp:
    """Tests for 'services help'."""

    def test_help_returns_helplines(self, plugin):
        result = plugin.execute_user_command(
            "services", {"subcommand": "help", "rest": ""}
        )
        assert isinstance(result, HelpLines)
        assert len(result.lines) > 0

    def test_help_contains_subcommands(self, plugin):
        result = plugin.execute_user_command(
            "services", {"subcommand": "help", "rest": ""}
        )
        text = "\n".join(line[0] for line in result.lines)
        assert "list" in text
        assert "show" in text
        assert "endpoints" in text
        assert "auth" in text
        assert "remove" in text


class TestServiceNamesMerge:
    """Test that service names merge from memory and disk."""

    def test_merge_memory_and_disk(self, plugin_with_services):
        # petstore and github are in memory. Save a third one only to disk.
        plugin_with_services._schema_store.save_discovered_service(
            "stripe",
            ServiceConfig(name="stripe", base_url="https://api.stripe.com"),
            [],
        )
        names = plugin_with_services._get_all_service_names()
        assert "petstore" in names
        assert "github" in names
        assert "stripe" in names

    def test_deduplication(self, plugin_with_services):
        # Save petstore to disk too
        svc = plugin_with_services._discovered_services["petstore"]
        plugin_with_services._schema_store.save_discovered_service(
            "petstore", svc.config, svc.endpoints
        )
        names = plugin_with_services._get_all_service_names()
        # Should not have duplicates
        assert names.count("petstore") == 1
