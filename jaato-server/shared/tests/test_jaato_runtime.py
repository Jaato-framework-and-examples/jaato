"""Tests for JaatoRuntime - shared environment for the jaato framework."""

import pytest
from unittest.mock import MagicMock, patch

from ..jaato_runtime import JaatoRuntime


class TestJaatoRuntimeInitialization:
    """Tests for JaatoRuntime initialization."""

    def test_init_default_provider(self):
        """Test default provider name."""
        runtime = JaatoRuntime()
        assert runtime.provider_name == "google_genai"

    def test_init_custom_provider(self):
        """Test custom provider name."""
        runtime = JaatoRuntime(provider_name="anthropic")
        assert runtime.provider_name == "anthropic"

    def test_not_connected_initially(self):
        """Test that runtime is not connected initially."""
        runtime = JaatoRuntime()
        assert not runtime.is_connected

    def test_properties_none_initially(self):
        """Test that properties are None initially."""
        runtime = JaatoRuntime()
        assert runtime.project is None
        assert runtime.location is None
        assert runtime.registry is None
        assert runtime.permission_plugin is None
        assert runtime.ledger is None


class TestJaatoRuntimeConnect:
    """Tests for JaatoRuntime.connect()."""

    def test_connect_sets_project_and_location(self):
        """Test that connect sets project and location."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        assert runtime.project == "my-project"
        assert runtime.location == "us-central1"
        assert runtime.is_connected

    def test_connect_multiple_times(self):
        """Test that connect can be called multiple times."""
        runtime = JaatoRuntime()
        runtime.connect("project-1", "us-central1")
        runtime.connect("project-2", "eu-west1")

        assert runtime.project == "project-2"
        assert runtime.location == "eu-west1"


class TestJaatoRuntimeConfigurePlugins:
    """Tests for JaatoRuntime.configure_plugins()."""

    def test_configure_plugins_stores_registry(self):
        """Test that configure_plugins stores the registry."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_registry = MagicMock()
        mock_registry.get_enabled_tool_schemas.return_value = []
        mock_registry.get_enabled_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        runtime.configure_plugins(mock_registry)

        assert runtime.registry == mock_registry

    def test_configure_plugins_stores_permission_plugin(self):
        """Test that configure_plugins stores the permission plugin."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_registry = MagicMock()
        mock_registry.get_enabled_tool_schemas.return_value = []
        mock_registry.get_enabled_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        mock_permission = MagicMock()
        mock_permission.get_tool_schemas.return_value = []
        mock_permission.get_executors.return_value = {}
        mock_permission.get_system_instructions.return_value = None

        runtime.configure_plugins(mock_registry, permission_plugin=mock_permission)

        assert runtime.permission_plugin == mock_permission

    def test_configure_plugins_caches_tool_schemas(self):
        """Test that configure_plugins caches tool schemas."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_schema = MagicMock()
        mock_schema.name = "test_tool"
        mock_schema.discoverability = "core"  # Mark as core for deferred loading

        mock_registry = MagicMock()
        mock_registry.get_enabled_tool_schemas.return_value = [mock_schema]
        # get_core_tool_schemas is used when deferred loading is enabled
        mock_registry.get_core_tool_schemas.return_value = [mock_schema]
        mock_registry.get_enabled_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        runtime.configure_plugins(mock_registry)

        schemas = runtime.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "test_tool"


class TestJaatoRuntimeCreateSession:
    """Tests for JaatoRuntime.create_session()."""

    def test_create_session_requires_connection(self):
        """Test that create_session requires runtime to be connected."""
        runtime = JaatoRuntime()

        with pytest.raises(RuntimeError, match="not connected"):
            runtime.create_session("gemini-2.5-flash")

    def test_create_session_requires_plugins(self):
        """Test that create_session requires plugins to be configured."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        with pytest.raises(RuntimeError, match="not configured"):
            runtime.create_session("gemini-2.5-flash")

    @patch('shared.jaato_runtime.load_provider')
    def test_create_session_returns_session(self, mock_load_provider):
        """Test that create_session returns a JaatoSession."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        # Setup mock registry
        mock_registry = MagicMock()
        mock_registry.get_enabled_tool_schemas.return_value = []
        mock_registry.get_enabled_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        # Setup mock provider
        mock_provider = MagicMock()
        mock_load_provider.return_value = mock_provider

        runtime.configure_plugins(mock_registry)
        session = runtime.create_session("gemini-2.5-flash")

        assert session is not None
        assert session.model_name == "gemini-2.5-flash"
        assert session.runtime == runtime


    @patch('shared.jaato_runtime.load_provider')
    def test_create_session_threads_agent_id(self, mock_load_provider):
        """Pin: ``runtime.create_session(agent_id=...)`` threads the
        value into ``JaatoSession.__init__`` so the resulting
        session's ``_agent_id`` reflects the daemon-resolved agent
        identity (``--agent <name>``).

        Regression context: pre-thread (PR #79 + this PR), the
        runner-side ``bootstrap_session`` correctly built an
        envelope with ``envelope.agent_id == "discovery"`` but the
        kwarg never reached ``JaatoSession.__init__`` — the session
        kept the default ``"main"``.  Downstream
        ``AgentCompletedEvent.agent_id == "main"`` broke reactor
        rules keying on logical agent identity."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_registry = MagicMock()
        mock_registry.get_enabled_tool_schemas.return_value = []
        mock_registry.get_enabled_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None
        mock_load_provider.return_value = MagicMock()

        runtime.configure_plugins(mock_registry)
        session = runtime.create_session(
            "gemini-2.5-flash", agent_id="discovery",
        )

        assert session.agent_id == "discovery", (
            f"runtime.create_session must thread agent_id into the "
            f"resulting JaatoSession; got {session.agent_id!r}"
        )

    @patch('shared.jaato_runtime.load_provider')
    def test_create_session_defaults_agent_id_to_main(
        self, mock_load_provider,
    ):
        """Pin: omitting ``agent_id`` from
        ``runtime.create_session`` falls back to ``"main"`` —
        preserves backward compat with callers that don't yet
        thread the field (most daemon-side legacy paths)."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_registry = MagicMock()
        mock_registry.get_enabled_tool_schemas.return_value = []
        mock_registry.get_enabled_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None
        mock_load_provider.return_value = MagicMock()

        runtime.configure_plugins(mock_registry)
        session = runtime.create_session("gemini-2.5-flash")

        assert session.agent_id == "main"


class TestJaatoRuntimeCreateProvider:
    """Tests for JaatoRuntime.create_provider()."""

    def test_create_provider_requires_connection(self):
        """Test that create_provider requires runtime to be connected."""
        runtime = JaatoRuntime()

        with pytest.raises(RuntimeError, match="not connected"):
            runtime.create_provider("gemini-2.5-flash")

    @patch('shared.jaato_runtime.load_provider')
    def test_create_provider_returns_provider(self, mock_load_provider):
        """Test that create_provider returns a provider instance."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_provider = MagicMock()
        mock_load_provider.return_value = mock_provider

        provider = runtime.create_provider("gemini-2.5-flash")

        assert provider == mock_provider
        mock_provider.connect.assert_called_once_with("gemini-2.5-flash", skip_model_test=False)

    @patch('shared.jaato_runtime.load_provider')
    def test_create_provider_merges_plugin_configs_into_extra(self, mock_load_provider):
        """Profile's plugin_configs[provider_name] must flow into config.extra.

        This is the wiring that lets profiles tune provider-specific knobs
        (e.g. LM Studio's host / load params) without changing provider
        registration code.  The provider-agnostic default config is kept
        intact and per-session overrides are applied on top.
        """
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_provider = MagicMock()
        mock_load_provider.return_value = mock_provider

        plugin_configs = {
            "google_genai": {
                "host": "http://localhost:1234",
                "load": {"context_length": 16384, "flash_attention": True},
            },
            "other_plugin": {"irrelevant": "value"},
        }
        runtime.create_provider(
            "gemini-2.5-flash",
            plugin_configs=plugin_configs,
        )

        # Inspect the ProviderConfig handed to load_provider
        _, supplied_config = mock_load_provider.call_args[0]
        assert supplied_config.extra["host"] == "http://localhost:1234"
        assert supplied_config.extra["load"] == {
            "context_length": 16384,
            "flash_attention": True,
        }
        # Non-matching plugin_configs entries must not leak into extra
        assert "irrelevant" not in supplied_config.extra

    @patch('shared.jaato_runtime.load_provider')
    def test_create_provider_without_plugin_configs_leaves_extra_untouched(
        self, mock_load_provider,
    ):
        """The wiring is opt-in — omitting plugin_configs must not perturb extra."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")
        mock_load_provider.return_value = MagicMock()

        runtime.create_provider("gemini-2.5-flash")
        _, supplied_config = mock_load_provider.call_args[0]
        # Only framework-injected keys (like workspace_path) may appear; no
        # user/profile knobs should be present.
        assert "load" not in supplied_config.extra
        assert "host" not in supplied_config.extra


class TestJaatoRuntimeVerifyAuth:
    """Tests for JaatoRuntime.verify_auth() — provider-config plumbing.

    Regression: previously verify_auth ignored profile knobs entirely
    (passed config=None), so providers that resolve credentials from
    plugin_configs (e.g. LM Studio's optional bearer token, NIM custom
    base_url) saw an environment-only view that didn't match what
    initialize() would later see.
    """

    @patch('shared.jaato_runtime.load_provider')
    def test_verify_auth_passes_plugin_config_into_provider_config(
        self, mock_load_provider,
    ):
        """plugin_configs[provider_name] must reach provider.verify_auth(config=...)."""
        runtime = JaatoRuntime()
        runtime.connect("p", "loc")

        provider = MagicMock()
        provider.verify_auth.return_value = True
        mock_load_provider.return_value = provider

        plugin_configs = {
            "google_genai": {
                "host": "http://gpu-box.lan:1234",
                "api_token": "secret-abc",
            },
            "irrelevant_plugin": {"key": "value"},
        }
        runtime.verify_auth(plugin_configs=plugin_configs)

        # Provider was called with a ProviderConfig whose extra carries
        # the matching provider's overrides — and only those.
        kwargs = provider.verify_auth.call_args.kwargs
        config = kwargs["config"]
        assert config is not None
        assert config.extra["host"] == "http://gpu-box.lan:1234"
        assert config.extra["api_token"] == "secret-abc"
        assert "key" not in config.extra  # other plugin's keys don't leak

    @patch('shared.jaato_runtime.load_provider')
    def test_verify_auth_without_plugin_configs_passes_none_config(
        self, mock_load_provider,
    ):
        """Backwards-compat: callers who don't supply plugin_configs must still work,
        and providers receive config=None (the documented contract)."""
        runtime = JaatoRuntime()
        runtime.connect("p", "loc")
        provider = MagicMock()
        provider.verify_auth.return_value = True
        mock_load_provider.return_value = provider

        runtime.verify_auth()

        kwargs = provider.verify_auth.call_args.kwargs
        assert kwargs["config"] is None

    @patch('shared.jaato_runtime.load_provider')
    def test_verify_auth_no_match_in_plugin_configs(self, mock_load_provider):
        """plugin_configs without an entry for the active provider → config=None.

        This is the protective path: random plugin_configs entries can't
        accidentally inject knobs into a provider they don't apply to.
        """
        runtime = JaatoRuntime()
        runtime.connect("p", "loc")
        provider = MagicMock()
        provider.verify_auth.return_value = True
        mock_load_provider.return_value = provider

        runtime.verify_auth(plugin_configs={"some_other_provider": {"x": 1}})

        kwargs = provider.verify_auth.call_args.kwargs
        assert kwargs["config"] is None


class TestJaatoRuntimeGetToolSchemas:
    """Tests for JaatoRuntime.get_tool_schemas()."""

    def test_get_tool_schemas_empty_without_registry(self):
        """Test that get_tool_schemas returns empty list without registry."""
        runtime = JaatoRuntime()
        assert runtime.get_tool_schemas() == []

    def test_get_tool_schemas_returns_cached(self):
        """Test that get_tool_schemas returns cached schemas."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_schema = MagicMock()
        mock_schema.name = "cached_tool"
        mock_schema.discoverability = "core"

        mock_registry = MagicMock()
        mock_registry.get_enabled_tool_schemas.return_value = [mock_schema]
        # get_core_tool_schemas is used when deferred loading is enabled
        mock_registry.get_core_tool_schemas.return_value = [mock_schema]
        mock_registry.get_enabled_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        runtime.configure_plugins(mock_registry)

        schemas = runtime.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "cached_tool"

    def test_get_tool_schemas_filtered_by_plugin_names(self):
        """Test that get_tool_schemas can filter by plugin names."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        # Mark schemas as core so they pass the deferred loading filter
        mock_schema_cli = MagicMock()
        mock_schema_cli.name = "cli_tool"
        mock_schema_cli.discoverability = "core"
        mock_schema_mcp = MagicMock()
        mock_schema_mcp.name = "mcp_tool"
        mock_schema_mcp.discoverability = "core"

        mock_cli_plugin = MagicMock()
        mock_cli_plugin.get_tool_schemas.return_value = [mock_schema_cli]
        mock_mcp_plugin = MagicMock()
        mock_mcp_plugin.get_tool_schemas.return_value = [mock_schema_mcp]

        mock_registry = MagicMock()
        mock_registry.get_enabled_tool_schemas.return_value = [mock_schema_cli, mock_schema_mcp]
        mock_registry.get_core_tool_schemas.return_value = [mock_schema_cli, mock_schema_mcp]
        mock_registry.get_enabled_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.side_effect = lambda name: {
            'cli': mock_cli_plugin,
            'mcp': mock_mcp_plugin,
            'subagent': None,
            'background': None
        }.get(name)

        runtime.configure_plugins(mock_registry)

        # Filter by plugin names
        schemas = runtime.get_tool_schemas(plugin_names=['cli'])
        assert len(schemas) == 1
        assert schemas[0].name == "cli_tool"


class TestJaatoRuntimeGetExecutors:
    """Tests for JaatoRuntime.get_executors()."""

    def test_get_executors_empty_without_registry(self):
        """Test that get_executors returns empty dict without registry."""
        runtime = JaatoRuntime()
        assert runtime.get_executors() == {}

    def test_get_executors_returns_cached(self):
        """Test that get_executors returns cached executors."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        def mock_executor(args):
            return "result"

        mock_registry = MagicMock()
        mock_registry.get_enabled_tool_schemas.return_value = []
        mock_registry.get_enabled_executors.return_value = {"test_tool": mock_executor}
        mock_registry.get_system_instructions.return_value = None
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        runtime.configure_plugins(mock_registry)

        executors = runtime.get_executors()
        assert "test_tool" in executors
        assert executors["test_tool"] == mock_executor


class TestJaatoRuntimeGetSystemInstructions:
    """Tests for JaatoRuntime.get_system_instructions()."""

    def test_get_system_instructions_none_without_registry(self):
        """Test that get_system_instructions returns base instructions or None without registry."""
        runtime = JaatoRuntime()
        # May return base instructions if file exists, or None
        instructions = runtime.get_system_instructions()
        # Just verify it doesn't crash - base instructions depend on file presence
        assert instructions is None or isinstance(instructions, str)

    def test_get_system_instructions_returns_cached(self):
        """Test that get_system_instructions includes registry instructions."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_registry = MagicMock()
        mock_registry.get_enabled_tool_schemas.return_value = []
        mock_registry.get_core_tool_schemas.return_value = []
        mock_registry.get_enabled_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = "Be helpful."
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        runtime.configure_plugins(mock_registry)

        instructions = runtime.get_system_instructions()
        # Registry instructions should be included (may also include base instructions)
        assert "Be helpful." in instructions

    def test_get_system_instructions_with_additional(self):
        """Test that get_system_instructions can add additional instructions."""
        runtime = JaatoRuntime()
        runtime.connect("my-project", "us-central1")

        mock_registry = MagicMock()
        mock_registry.get_enabled_tool_schemas.return_value = []
        mock_registry.get_core_tool_schemas.return_value = []
        mock_registry.get_enabled_executors.return_value = {}
        mock_registry.get_system_instructions.return_value = "Be helpful."
        mock_registry.get_auto_approved_tools.return_value = []
        mock_registry.get_plugin.return_value = None

        runtime.configure_plugins(mock_registry)

        instructions = runtime.get_system_instructions(additional="Be concise.")
        assert "Be concise." in instructions
        assert "Be helpful." in instructions


class TestCreateProviderUnresolvedSecret:
    """create_provider fails loud on an unresolved secret-URI api_key.

    Multi-cause cascade-of-bugs sibling: the nebius regression was a
    ``pass://`` api_key passed through literally (no resolver registered — the
    providing plugin wasn't installed) and sent to the provider as the literal
    credential → a confusing upstream 401.  ``_resolve_secret_uri`` stays
    lenient (so service_connector keeps its graceful credential-missing
    reporting); the strict refusal lives at the provider boundary.
    """

    @patch('shared.jaato_runtime.load_provider')
    def test_rejects_unresolved_secret_uri_api_key(self, mock_load_provider):
        from ..plugins.subagent.config import SecretResolutionError

        runtime = JaatoRuntime(provider_name="nebius")
        runtime.connect("p", "us-central1")

        with pytest.raises(SecretResolutionError, match="unresolved secret"):
            runtime.create_provider(
                model="some-model",
                provider_name="nebius",
                plugin_configs={
                    "nebius": {"api_key": "pass://jaato/nebius/api-key"},
                },
            )
        # Refused BEFORE constructing/connecting the provider.
        mock_load_provider.assert_not_called()

    @patch('shared.jaato_runtime.load_provider')
    def test_allows_plain_string_api_key(self, mock_load_provider):
        """A normal (resolved) api_key is a plain string — not a secret URI —
        so the boundary check is a no-op and provider construction proceeds."""
        mock_load_provider.return_value = MagicMock()

        runtime = JaatoRuntime(provider_name="nebius")
        runtime.connect("p", "us-central1")

        runtime.create_provider(
            model="some-model",
            provider_name="nebius",
            plugin_configs={"nebius": {"api_key": "nbk-a-real-looking-key"}},
        )
        mock_load_provider.assert_called_once()
