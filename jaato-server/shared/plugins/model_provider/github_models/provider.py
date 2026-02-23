"""GitHub Models provider implementation.

This provider enables access to AI models through the GitHub Models API,
supporting multiple models from different providers (OpenAI, Anthropic, Google)
through a unified interface.

Authentication methods:
- Personal Access Token (PAT) with `models: read` scope
- Fine-grained PAT (recommended for enterprise SSO)
- GitHub App token with `models: read` permission

Enterprise features:
- Organization-attributed billing via org-scoped endpoint
- Enterprise policy compliance
- SSO support (fine-grained PATs auto-authorized)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Lazy imports - SDK is only loaded when actually used
from ._lazy import (
    get_chat_client_class,
    get_models,
    get_azure_key_credential,
    get_response_format_json,
)

if TYPE_CHECKING:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import (
        AssistantMessage,
        SystemMessage,
        UserMessage,
    )
    from azure.core.credentials import AzureKeyCredential
    from .copilot_client import CopilotClient, CopilotResponse, ResponsesAPIResponse

from ..base import (
    FunctionCallDetectedCallback,
    ModelProviderPlugin,
    ProviderConfig,
    StreamingCallback,
    ThinkingCallback,
    UsageUpdateCallback,
)
from jaato_sdk.plugins.model_provider.types import (
    CancelledException,
    CancelToken,
    FinishReason,
    Message,
    Part,
    ProviderResponse,
    Role,
    ToolResult,
    ToolSchema,
    TokenUsage,
)
from .converters import (
    clear_tool_name_mapping,
    extract_reasoning_from_response,
    extract_reasoning_from_stream_delta,
    get_original_tool_name,
    history_from_sdk,
    history_to_sdk,
    register_tool_name_mapping,
    response_from_sdk,
    sanitize_tool_name,
    serialize_history,
    deserialize_history,
    tool_results_to_sdk,
    tool_schemas_to_sdk,
)
from .env import (
    DEFAULT_ENDPOINT,
    resolve_auth_method,
    resolve_endpoint,
    resolve_enterprise,
    resolve_organization,
    resolve_token,
    get_checked_credential_locations,
)
from .errors import (
    ContextLimitError,
    InfrastructureError,
    ModelNotFoundError,
    ModelsDisabledError,
    PayloadTooLargeError,
    RateLimitError,
    TokenInvalidError,
    TokenNotFoundError,
    TokenPermissionError,
)

# GitHub Models catalog API endpoint (models.github.ai as of May 2025)
CATALOG_API_ENDPOINT = "https://models.github.ai/catalog/models"

# Models that expose reasoning/thinking content via `reasoning_content`.
# OpenAI o-series models do NOT expose reasoning text (hidden internally).
# DeepSeek-R1 and similar open reasoning models DO expose it.
REASONING_CAPABLE_MODELS = [
    "deepseek/deepseek-r1",
    "deepseek-r1",
]


@dataclass
class ModelInfo:
    """Model information from the GitHub Models catalog API.

    Attributes:
        id: Model identifier (e.g., 'openai/gpt-4o').
        max_input_tokens: Maximum input tokens allowed.
        max_output_tokens: Maximum output tokens allowed.
        context_window: Total context window size (max_input + max_output or API-provided).
    """
    id: str
    max_input_tokens: int = 0
    max_output_tokens: int = 0
    context_window: int = 0

    @property
    def effective_context_limit(self) -> int:
        """Get the effective context limit for GC calculations.

        Uses context_window if available, otherwise sums input + output limits.
        Falls back to DEFAULT_CONTEXT_LIMIT if no data available.
        """
        if self.context_window > 0:
            return self.context_window
        if self.max_input_tokens > 0:
            # Context window is typically max_input_tokens (output is separate budget)
            return self.max_input_tokens
        return DEFAULT_CONTEXT_LIMIT


# Fallback context window limits for known models (total tokens)
# Used when API doesn't return token limits or API call fails
# These are approximate - actual limits may vary
FALLBACK_CONTEXT_LIMITS: Dict[str, int] = {
    # OpenAI models
    "openai/gpt-4o": 128_000,
    "openai/gpt-4o-mini": 128_000,
    "openai/gpt-4-turbo": 128_000,
    "openai/gpt-4": 8_192,
    "openai/gpt-3.5-turbo": 16_385,
    "openai/o1-preview": 128_000,
    "openai/o1-mini": 128_000,
    # Anthropic models
    "anthropic/claude-3.5-sonnet": 200_000,
    "anthropic/claude-3-opus": 200_000,
    "anthropic/claude-3-sonnet": 200_000,
    "anthropic/claude-3-haiku": 200_000,
    # Google models
    "google/gemini-1.5-pro": 1_048_576,
    "google/gemini-1.5-flash": 1_048_576,
    # Meta models
    "meta/llama-3.1-405b-instruct": 128_000,
    "meta/llama-3.1-70b-instruct": 128_000,
    "meta/llama-3.1-8b-instruct": 128_000,
    # Mistral models
    "mistral/mistral-large": 32_000,
    "mistral/mistral-small": 32_000,
}

# Keep MODEL_CONTEXT_LIMITS as alias for backwards compatibility
MODEL_CONTEXT_LIMITS = FALLBACK_CONTEXT_LIMITS

DEFAULT_CONTEXT_LIMIT = 128_000


@dataclass
class GitHubModelsConfig:
    """Configuration for GitHub Models provider.

    Attributes:
        token: GitHub token (PAT or App token) with models:read permission.
        organization: Organization for billing attribution (optional).
        enterprise: Enterprise name for context (optional).
        endpoint: API endpoint URL (defaults to GitHub Models endpoint).
    """
    token: Optional[str] = None
    organization: Optional[str] = None
    enterprise: Optional[str] = None
    endpoint: str = DEFAULT_ENDPOINT


class GitHubModelsProvider:
    """GitHub Models provider.

    This provider supports:
    - Multiple AI models (GPT, Claude, Gemini, Llama, Mistral, etc.)
    - Organization-attributed billing
    - Enterprise policy compliance
    - Function calling with manual control
    - Token counting (estimated)

    Usage:
        provider = GitHubModelsProvider()
        provider.initialize(ProviderConfig(
            api_key='ghp_your_token',  # Or set GITHUB_TOKEN env var
            extra={
                'organization': 'your-org',  # Optional, for billing
            }
        ))
        provider.connect('openai/gpt-4o')
        response = provider.send_message("Hello!")

    Environment variables:
        GITHUB_TOKEN: Personal access token or app token
        JAATO_GITHUB_ORGANIZATION: Organization for billing attribution
        JAATO_GITHUB_ENTERPRISE: Enterprise name (for context/debugging)
        JAATO_GITHUB_ENDPOINT: Override the API endpoint URL
    """

    # Cache TTL in seconds (5 minutes)
    _MODELS_CACHE_TTL = 300

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        self._client: Optional[ChatCompletionsClient] = None
        self._model_name: Optional[str] = None

        # Configuration
        self._token: Optional[str] = None
        self._organization: Optional[str] = None
        self._enterprise: Optional[str] = None
        self._endpoint: str = DEFAULT_ENDPOINT

        # Copilot API support (for OAuth authentication)
        self._use_copilot_api: bool = False
        self._copilot_client: Optional["CopilotClient"] = None

        # Session state
        self._system_instruction: Optional[str] = None
        self._tools: Optional[List[ToolSchema]] = None
        self._history: List[Message] = []
        self._last_usage: TokenUsage = TokenUsage()

        # Thinking/reasoning configuration
        self._enable_thinking: bool = True  # Extract reasoning by default when available

        # Models cache: (timestamp, {model_id: ModelInfo})
        self._models_cache: Optional[Tuple[float, Dict[str, ModelInfo]]] = None

        # Agent context for trace identification
        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None
        self._agent_id: str = "main"

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main"
    ) -> None:
        """Set agent context for trace identification.

        Args:
            agent_type: Type of agent ("main" or "subagent").
            agent_name: Optional name for the agent (e.g., profile name).
            agent_id: Unique identifier for the agent instance.
        """
        self._agent_type = agent_type
        self._agent_name = agent_name
        self._agent_id = agent_id

    def _get_trace_prefix(self) -> str:
        """Get the trace prefix including agent context."""
        if self._agent_type == "main":
            return "github_models:main"
        elif self._agent_name:
            return f"github_models:subagent:{self._agent_name}"
        else:
            return f"github_models:subagent:{self._agent_id}"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        from shared.trace import provider_trace
        prefix = self._get_trace_prefix()
        provider_trace(prefix, msg)

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "github_models"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider with credentials.

        Supports GitHub PAT and App tokens. Uses fail-fast validation
        to catch configuration errors early.

        Args:
            config: Configuration with authentication details.
                - api_key: GitHub token (PAT or App token)
                - extra['organization']: Organization for billing (optional)
                - extra['enterprise']: Enterprise name (optional)
                - extra['endpoint']: Override API endpoint (optional)

        Raises:
            TokenNotFoundError: No token found.
            TokenInvalidError: Token is invalid or rejected.
            ModelsDisabledError: GitHub Models is disabled for the org/enterprise.
        """
        if config is None:
            config = ProviderConfig()

        # Set workspace path from config.extra if provided
        # This ensures token resolution can find workspace-specific OAuth tokens
        # even when JAATO_WORKSPACE_ROOT env var isn't set (e.g., subagent spawning)
        workspace_path = config.extra.get('workspace_path')
        if workspace_path and not os.environ.get('JAATO_WORKSPACE_ROOT'):
            os.environ['JAATO_WORKSPACE_ROOT'] = workspace_path
            self._trace(f"[INIT] Set JAATO_WORKSPACE_ROOT from config.extra: {workspace_path}")

        # Resolve configuration
        raw_token = config.api_key or resolve_token()
        self._organization = config.extra.get('organization') or resolve_organization()
        self._enterprise = config.extra.get('enterprise') or resolve_enterprise()
        self._endpoint = config.extra.get('endpoint') or resolve_endpoint()

        # Validate token
        if not raw_token:
            raise TokenNotFoundError(
                auth_method=resolve_auth_method(),
                checked_locations=get_checked_credential_locations(resolve_auth_method()),
            )

        # If using OAuth, exchange for Copilot token and use Copilot API
        from .env import resolve_token_source
        token_source = resolve_token_source()
        self._trace(f"[INIT] token_source={token_source}, has_api_key={bool(config.api_key)}")
        if token_source == "oauth" and not config.api_key:
            # Using device code OAuth - use Copilot API (api.githubcopilot.com)
            self._trace("[INIT] Using OAuth - getting Copilot token...")
            from .oauth import get_stored_access_token
            copilot_token = get_stored_access_token()
            self._trace(f"[INIT] get_stored_access_token returned: {bool(copilot_token)}")
            if copilot_token:
                self._token = copilot_token
                self._use_copilot_api = True
                self._trace("[INIT] Creating Copilot client...")
                from .copilot_client import CopilotClient
                self._copilot_client = CopilotClient(copilot_token)
                self._trace("[INIT] Copilot client created successfully")
            else:
                # Exchange failed - fall back to OAuth token (may not work)
                self._trace("[INIT] Copilot token exchange failed, using raw OAuth token")
                self._token = raw_token
                self._trace("[INIT] Creating Azure SDK client...")
                self._client = self._create_client()
                self._trace("[INIT] Client created successfully")
        else:
            # Using PAT from env var or explicit api_key - use GitHub Models API
            self._trace("[INIT] Using PAT or explicit api_key")
            self._token = raw_token
            self._trace("[INIT] Creating Azure SDK client...")
            self._client = self._create_client()
            self._trace("[INIT] Client created successfully")

        # Verify connectivity
        self._trace("[INIT] Verifying connectivity...")
        self._verify_connectivity()
        self._trace("[INIT] Initialize complete")

    def _create_client(self) -> "ChatCompletionsClient":
        """Create the ChatCompletionsClient.

        Returns:
            Initialized ChatCompletionsClient.
        """
        self._trace("[_create_client] Getting chat client class...")
        client_class = get_chat_client_class()
        self._trace("[_create_client] Getting credential class...")
        cred_class = get_azure_key_credential()
        self._trace(f"[_create_client] Creating client with endpoint={self._endpoint}")
        client = client_class(
            endpoint=self._endpoint,
            credential=cred_class(self._token),
        )
        self._trace("[_create_client] Client instance created")
        return client

    def _verify_connectivity(self) -> None:
        """Verify connectivity by making a lightweight API call.

        Note: The azure-ai-inference SDK doesn't have a list_models endpoint,
        so we skip connectivity verification at init time. Errors will be
        caught on first send_message call.
        """
        # The SDK doesn't provide a lightweight connectivity check
        # Verification happens on first actual API call
        pass

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None
    ) -> bool:
        """Verify that authentication is configured.

        For GitHub Models, this checks for:
        1. Device Code OAuth tokens (stored via github-auth login)
        2. GITHUB_TOKEN environment variable

        Args:
            allow_interactive: If True, starts device code flow for login.
            on_message: Optional callback for status messages.

        Returns:
            True if authentication is configured.
            False if no credentials found.

        Raises:
            TokenNotFoundError: If allow_interactive=False and no token found.
        """
        from .env import resolve_token_source

        token = resolve_token()
        if token:
            source = resolve_token_source()
            if on_message:
                if source == "oauth":
                    on_message("Found GitHub token (device code OAuth)")
                else:
                    on_message("Found GitHub token (environment variable)")
            return True

        # No token found
        if not allow_interactive:
            raise TokenNotFoundError(
                checked_locations=get_checked_credential_locations()
            )

        # Try interactive device code flow
        if on_message:
            on_message("No GitHub token found. Starting device code authentication...")

        try:
            from .oauth import login_interactive

            def emit_message(msg: str) -> None:
                if on_message:
                    on_message(msg)

            tokens, _ = login_interactive(on_message=emit_message, auto_poll=True)
            if tokens:
                if on_message:
                    on_message("Successfully authenticated with GitHub")
                return True
            else:
                if on_message:
                    on_message("Authentication failed or was cancelled")
                return False
        except Exception as e:
            if on_message:
                on_message(f"Interactive authentication failed: {e}")
            return False

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            self._client.close()
        self._client = None
        if self._copilot_client:
            self._copilot_client.close()
        self._copilot_client = None
        self._use_copilot_api = False
        self._model_name = None
        self._history = []

    # ==================== Connection ====================

    def connect(self, model: str) -> None:
        """Set the model to use.

        Model validation is deferred to the first API call to avoid
        unnecessary API usage during initialization.

        Args:
            model: Model ID (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet').
        """
        self._model_name = model

    def _verify_model_responds(self) -> None:
        """Verify the model can actually respond.

        Sends a minimal test message to catch issues like:
        - Invalid model name
        - Model access restrictions
        - Organization disabled GitHub Models
        """
        if self._use_copilot_api:
            if not self._copilot_client or not self._model_name:
                return  # Will fail later with clear error
            try:
                # Send minimal request via Copilot API
                messages = [{"role": "user", "content": "hi"}]
                self._copilot_client.complete(
                    model=self._copilot_model_name(),
                    messages=messages,
                    max_tokens=1,
                )
            except Exception as e:
                self._handle_api_error(e)
        else:
            if not self._client or not self._model_name:
                return  # Will fail later with clear error
            try:
                # Send minimal request via Azure SDK
                messages = [get_models().UserMessage(content="hi")]
                self._client.complete(
                    model=self._model_name,
                    messages=messages,
                    max_tokens=1,
                )
            except Exception as e:
                self._handle_api_error(e)

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected and ready."""
        if self._use_copilot_api:
            return self._copilot_client is not None and self._model_name is not None
        return self._client is not None and self._model_name is not None

    @property
    def model_name(self) -> Optional[str]:
        """Get the current model name."""
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List available models from the GitHub Models catalog API.

        Fetches models from the GitHub Models marketplace API and caches
        the results for _MODELS_CACHE_TTL seconds.

        Args:
            prefix: Optional filter prefix (e.g., 'openai/', 'anthropic/').

        Returns:
            List of model IDs (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet').
        """
        models_info = self._get_models_info()
        model_ids = list(models_info.keys())

        if prefix:
            model_ids = [m for m in model_ids if m.startswith(prefix)]
        return sorted(model_ids)

    def _get_models_info(self) -> Dict[str, ModelInfo]:
        """Get cached model info, fetching from API if needed.

        Returns:
            Dict mapping model IDs to ModelInfo objects.
        """
        if not self._token:
            # Fallback to static list if not initialized
            return {
                model_id: ModelInfo(
                    id=model_id,
                    context_window=limit,
                )
                for model_id, limit in FALLBACK_CONTEXT_LIMITS.items()
            }

        # Check cache validity
        now = time.time()
        if self._models_cache:
            cache_time, cached_models = self._models_cache
            if now - cache_time < self._MODELS_CACHE_TTL:
                return cached_models

        # Fetch models based on API type
        models_info: Dict[str, ModelInfo] = {}
        if self._use_copilot_api and self._copilot_client:
            # Copilot API - get models with token limits (returns dicts)
            raw_info = self._copilot_client.list_models_with_info()
            for model_id, info in raw_info.items():
                models_info[model_id] = ModelInfo(
                    id=info.get("id", model_id),
                    max_input_tokens=info.get("max_input_tokens", 0),
                    max_output_tokens=info.get("max_output_tokens", 0),
                    context_window=info.get("context_window", 0),
                )
        else:
            models_info = self._fetch_models_from_api()

        if models_info:
            # Update cache
            self._models_cache = (now, models_info)
        else:
            # API failed, fallback to static list
            models_info = {
                model_id: ModelInfo(
                    id=model_id,
                    context_window=limit,
                )
                for model_id, limit in FALLBACK_CONTEXT_LIMITS.items()
            }

        return models_info

    def get_model_info(self, model_id: Optional[str] = None) -> Optional[ModelInfo]:
        """Get detailed model info including token limits.

        Args:
            model_id: Model ID to look up. Uses current model if not specified.

        Returns:
            ModelInfo if found, None otherwise.
        """
        model_id = model_id or self._model_name
        if not model_id:
            return None

        models_info = self._get_models_info()
        return models_info.get(model_id)

    def _fetch_models_from_api(self) -> Dict[str, ModelInfo]:
        """Fetch available models from the GitHub Models catalog API.

        Uses the shared httpx client which handles proxy configuration,
        Kerberos/SPNEGO authentication, and JAATO_NO_PROXY exact host matching.

        Returns:
            Dict mapping model IDs to ModelInfo, or empty dict on failure.
        """
        import httpx
        from shared.http import get_httpx_client

        try:
            with get_httpx_client(timeout=10.0) as client:
                response = client.get(
                    CATALOG_API_ENDPOINT,
                    headers={
                        "Accept": "application/vnd.github+json",
                        "Authorization": f"Bearer {self._token}",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                )
                response.raise_for_status()
                data = response.json()

            # Extract model info from the response
            # Response format: list of model objects with 'id', 'max_input_tokens', 'max_output_tokens'
            models: Dict[str, ModelInfo] = {}

            def parse_model(model: Dict[str, Any]) -> Optional[ModelInfo]:
                """Parse a model object into ModelInfo."""
                model_id = model.get('id') or model.get('name') or model.get('model_id')
                if not model_id:
                    return None

                # Extract token limits - try various field names used by GitHub API
                max_input = (
                    model.get('max_input_tokens') or
                    model.get('maxInputTokens') or
                    model.get('input_token_limit') or
                    0
                )
                max_output = (
                    model.get('max_output_tokens') or
                    model.get('maxOutputTokens') or
                    model.get('output_token_limit') or
                    0
                )
                context_window = (
                    model.get('context_window') or
                    model.get('contextWindow') or
                    model.get('context_length') or
                    0
                )

                return ModelInfo(
                    id=model_id,
                    max_input_tokens=int(max_input) if max_input else 0,
                    max_output_tokens=int(max_output) if max_output else 0,
                    context_window=int(context_window) if context_window else 0,
                )

            if isinstance(data, list):
                for model in data:
                    info = parse_model(model)
                    if info:
                        models[info.id] = info
            elif isinstance(data, dict):
                # Response might be paginated with a 'models' key
                model_list = data.get('models', data.get('items', data.get('data', [])))
                for model in model_list:
                    info = parse_model(model)
                    if info:
                        models[info.id] = info

            return models
        except (httpx.HTTPError, Exception):
            # API or network error, return empty to trigger fallback
            return {}

    # ==================== Session Management ====================

    def create_session(
        self,
        system_instruction: Optional[str] = None,
        tools: Optional[List[ToolSchema]] = None,
        history: Optional[List[Message]] = None
    ) -> None:
        """Create or reset the chat session.

        Args:
            system_instruction: System prompt for the model.
            tools: List of available tools.
            history: Previous conversation history to restore.
        """
        if self._use_copilot_api:
            if not self._copilot_client or not self._model_name:
                raise RuntimeError("Provider not initialized. Call initialize() and connect() first.")
        else:
            if not self._client or not self._model_name:
                raise RuntimeError("Provider not initialized. Call initialize() and connect() first.")

        self._system_instruction = system_instruction
        self._tools = tools
        self._history = list(history) if history else []

        # Clear tool name mapping when tools change
        clear_tool_name_mapping()

    def get_history(self) -> List[Message]:
        """Get the current conversation history.

        Returns:
            List of messages in internal format.
        """
        return list(self._history)

    # ==================== Messaging ====================

    def generate(self, prompt: str) -> ProviderResponse:
        """Simple one-shot generation without session context.

        Args:
            prompt: The prompt text.

        Returns:
            ProviderResponse with the model's response.
        """
        if self._use_copilot_api:
            if not self._copilot_client or not self._model_name:
                raise RuntimeError("Provider not connected. Call connect() first.")

            messages = [{"role": "user", "content": prompt}]

            try:
                response = self._copilot_client.complete(
                    model=self._copilot_model_name(),
                    messages=messages,
                )
                provider_response = self._copilot_response_to_provider(response)
                self._last_usage = provider_response.usage
                return provider_response
            except Exception as e:
                self._handle_api_error(e)
                raise
        else:
            if not self._client or not self._model_name:
                raise RuntimeError("Provider not connected. Call connect() first.")

            messages = [get_models().UserMessage(content=prompt)]

            try:
                response = self._client.complete(
                    model=self._model_name,
                    messages=messages,
                )
                provider_response = response_from_sdk(response)
                self._last_usage = provider_response.usage
                return provider_response
            except Exception as e:
                self._handle_api_error(e)
                raise

    def send_message(
        self,
        message: str,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Send a user message and get a response.

        Args:
            message: The user's message text.
            response_schema: Optional JSON Schema to constrain the response.
                Note: GitHub Models has limited structured output support.

        Returns:
            ProviderResponse with text and/or function calls.
        """
        # Add user message to history first
        self._trace(f"[SEND_MESSAGE] history_before_append={len(self._history)}")
        self._history.append(Message.from_text(Role.USER, message))

        if self._use_copilot_api:
            if not self._copilot_client or not self._model_name:
                raise RuntimeError("No chat session. Call create_session() first.")

            # Build messages for Copilot API
            self._trace(f"[SEND_MESSAGE] Building messages, history_len={len(self._history)}")
            messages = self._build_copilot_messages()
            tools = self._build_copilot_tools()

            try:
                # Route to Responses API for codex models
                if self._is_responses_api_model():
                    self._trace(f"[SEND_MESSAGE] Using Responses API for {self._copilot_model_name()}")
                    response = self._copilot_client.complete_responses(
                        model=self._copilot_model_name(),
                        messages=messages,
                        system_instruction=self._system_instruction,
                        tools=tools,
                    )
                    provider_response = self._responses_api_response_to_provider(response)
                else:
                    response = self._copilot_client.complete(
                        model=self._copilot_model_name(),
                        messages=messages,
                        tools=tools,
                    )
                    provider_response = self._copilot_response_to_provider(response)

                self._last_usage = provider_response.usage

                # Add assistant response to history
                self._add_response_to_history(provider_response)

                # Parse structured output if schema was requested
                text = provider_response.get_text()
                if response_schema and text:
                    try:
                        provider_response.structured_output = json.loads(text)
                    except json.JSONDecodeError:
                        pass

                return provider_response
            except Exception as e:
                # Rollback the user message we added
                if self._history and self._history[-1].role == Role.USER:
                    self._history.pop()
                self._handle_api_error(e)
                raise
        else:
            if not self._client or not self._model_name:
                raise RuntimeError("No chat session. Call create_session() first.")

            # Build messages list (includes user message from history)
            messages = self._build_messages()

            # Build completion kwargs
            kwargs = self._build_completion_kwargs(response_schema)

            try:
                response = self._client.complete(
                    model=self._model_name,
                    messages=messages,
                    **kwargs,
                )
                provider_response = response_from_sdk(response)
                self._last_usage = provider_response.usage

                # Add assistant response to history
                self._add_response_to_history(provider_response)

                # Parse structured output if schema was requested
                text = provider_response.get_text()
                if response_schema and text:
                    try:
                        provider_response.structured_output = json.loads(text)
                    except json.JSONDecodeError:
                        pass

                return provider_response
            except Exception as e:
                # Rollback the user message we added
                if self._history and self._history[-1].role == Role.USER:
                    self._history.pop()
                self._handle_api_error(e)
                raise

    def send_message_with_parts(
        self,
        parts: List[Part],
        response_schema: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Send a message with multiple parts.

        Note: Multimodal support depends on the underlying model.

        Args:
            parts: List of Part objects forming the message.
            response_schema: Optional JSON Schema to constrain the response.

        Returns:
            ProviderResponse with text and/or function calls.
        """
        # For now, extract text content only
        # Full multimodal support would require model-specific handling
        text_parts = [p.text for p in parts if p.text]
        combined_text = "".join(text_parts)

        return self.send_message(combined_text, response_schema)

    def send_tool_results(
        self,
        results: List[ToolResult],
        response_schema: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Send tool execution results back to the model.

        Args:
            results: List of tool execution results.
            response_schema: Optional JSON Schema to constrain the response.

        Returns:
            ProviderResponse with the model's next response.
        """
        # Add tool results to history
        for result in results:
            self._history.append(Message(
                role=Role.TOOL,
                parts=[Part(function_response=result)],
            ))

        if self._use_copilot_api:
            if not self._copilot_client or not self._model_name:
                raise RuntimeError("No chat session. Call create_session() first.")

            # Build messages for Copilot API
            messages = self._build_copilot_messages()
            tools = self._build_copilot_tools()

            try:
                # Route to Responses API for codex models
                if self._is_responses_api_model():
                    self._trace(f"[SEND_TOOL_RESULTS] Using Responses API for {self._copilot_model_name()}")
                    response = self._copilot_client.complete_responses(
                        model=self._copilot_model_name(),
                        messages=messages,
                        system_instruction=self._system_instruction,
                        tools=tools,
                    )
                    provider_response = self._responses_api_response_to_provider(response)
                else:
                    response = self._copilot_client.complete(
                        model=self._copilot_model_name(),
                        messages=messages,
                        tools=tools,
                    )
                    provider_response = self._copilot_response_to_provider(response)

                self._last_usage = provider_response.usage

                # Add assistant response to history
                self._add_response_to_history(provider_response)

                # Parse structured output if schema was requested
                text = provider_response.get_text()
                if response_schema and text:
                    try:
                        provider_response.structured_output = json.loads(text)
                    except json.JSONDecodeError:
                        pass

                return provider_response
            except Exception as e:
                # Rollback the tool result messages we added
                self._rollback_tool_results(len(results))
                self._handle_api_error(e)
                raise
        else:
            if not self._client or not self._model_name:
                raise RuntimeError("No chat session. Call create_session() first.")

            # Build messages including tool results
            messages = self._build_messages()

            # Build completion kwargs
            kwargs = self._build_completion_kwargs(response_schema)

            try:
                response = self._client.complete(
                    model=self._model_name,
                    messages=messages,
                    **kwargs,
                )
                provider_response = response_from_sdk(response)
                self._last_usage = provider_response.usage

                # Add assistant response to history
                self._add_response_to_history(provider_response)

                # Parse structured output if schema was requested
                text = provider_response.get_text()
                if response_schema and text:
                    try:
                        provider_response.structured_output = json.loads(text)
                    except json.JSONDecodeError:
                        pass

                return provider_response
            except Exception as e:
                # Rollback the tool result messages we added
                self._rollback_tool_results(len(results))
                self._handle_api_error(e)
                raise

    def _build_messages(self) -> List:
        """Build the messages list for the API call."""
        messages = []

        # Add system instruction if present
        if self._system_instruction:
            messages.append(get_models().SystemMessage(content=self._system_instruction))

        # Convert history to SDK format
        messages.extend(history_to_sdk(self._history))

        return messages

    def _build_completion_kwargs(self, response_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build kwargs for the complete() call."""
        kwargs = {}

        # Add tools if configured
        if self._tools:
            sdk_tools = tool_schemas_to_sdk(self._tools)
            if sdk_tools:
                kwargs['tools'] = sdk_tools

        # Add response format for structured output
        response_format_json = get_response_format_json()
        if response_schema and response_format_json is not None:
            kwargs['response_format'] = response_format_json()

        return kwargs

    def _add_response_to_history(self, response: ProviderResponse) -> None:
        """Add the model's response to history."""
        if response.parts:
            self._history.append(Message(role=Role.MODEL, parts=response.parts))

    def _rollback_tool_results(self, count: int) -> None:
        """Remove the last `count` tool result messages from history.

        Called on API failure to prevent duplicate tool messages on retry.
        GitHub Models appends one history message per tool result (unlike
        Anthropic which batches them into a single message), so we need
        to pop multiple entries.
        """
        for _ in range(count):
            if self._history and self._history[-1].role == Role.TOOL:
                self._history.pop()

    # ==================== Copilot API Helpers ====================

    def _copilot_model_name(self) -> str:
        """Convert GitHub Models model name to Copilot API model name.

        GitHub Models uses 'openai/gpt-4o', Copilot API uses 'gpt-4o'.
        """
        if not self._model_name:
            return ""
        # Strip provider prefix if present (e.g., 'openai/gpt-4o' -> 'gpt-4o')
        if "/" in self._model_name:
            return self._model_name.split("/", 1)[1]
        return self._model_name

    def _is_responses_api_model(self) -> bool:
        """Check if the current model requires the Responses API.

        Codex models (gpt-5-codex, gpt-5.2-codex, etc.) use the Responses API
        instead of the Chat Completions API.

        Returns:
            True if model requires Responses API.
        """
        from .copilot_client import is_responses_api_model
        model_name = self._copilot_model_name()
        return is_responses_api_model(model_name)

    def _responses_api_response_to_provider(self, response: "ResponsesAPIResponse") -> ProviderResponse:
        """Convert Responses API response to ProviderResponse."""
        from .copilot_client import ResponsesAPIResponse
        from jaato_sdk.plugins.model_provider.types import FunctionCall

        parts = []
        finish_reason = FinishReason.STOP
        thinking = None

        for item in response.output:
            if item.type == "message":
                # Extract text from content array
                if item.content:
                    for content_item in item.content:
                        if content_item.get("type") == "output_text":
                            text = content_item.get("text", "")
                            if text:
                                parts.append(Part.from_text(text))

            elif item.type == "reasoning":
                # Reasoning/thinking block (Responses API format)
                if self._enable_thinking and hasattr(item, "summary"):
                    summaries = item.summary or []
                    reasoning_texts = []
                    for s in summaries:
                        if isinstance(s, dict):
                            reasoning_texts.append(s.get("text", ""))
                        elif hasattr(s, "text"):
                            reasoning_texts.append(s.text)
                    if reasoning_texts:
                        thinking = "\n".join(t for t in reasoning_texts if t)

            elif item.type == "function_call":
                # Convert function call
                try:
                    args = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                # Restore original tool name from sanitized version
                sanitized_name = item.name or ""
                original_name = get_original_tool_name(sanitized_name)
                fc = FunctionCall(
                    id=item.call_id,
                    name=original_name,
                    args=args,
                )
                parts.append(Part.from_function_call(fc))
                finish_reason = FinishReason.TOOL_USE

        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,
            thinking=thinking,
        )

    def _build_copilot_messages(self) -> List[Dict[str, Any]]:
        """Build messages list for Copilot API (OpenAI format).

        Tool names in history are sanitized to match OpenAI's pattern.
        """
        messages = []

        # Add system instruction if present
        if self._system_instruction:
            messages.append({"role": "system", "content": self._system_instruction})

        # Convert history to OpenAI format
        for msg in self._history:
            if msg.role == Role.USER:
                text = msg.text or ""
                messages.append({"role": "user", "content": text})
            elif msg.role == Role.MODEL:
                text = msg.text or ""
                # Check for tool calls
                tool_calls = []
                for part in msg.parts:
                    if part.function_call:
                        # Sanitize tool name for API compatibility
                        sanitized_name = sanitize_tool_name(part.function_call.name)
                        tool_calls.append({
                            "id": part.function_call.id or f"call_{sanitized_name}",
                            "type": "function",
                            "function": {
                                "name": sanitized_name,
                                "arguments": json.dumps(part.function_call.args or {}),
                            }
                        })
                msg_dict: Dict[str, Any] = {"role": "assistant", "content": text or None}
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                messages.append(msg_dict)
            elif msg.role == Role.TOOL:
                for part in msg.parts:
                    if part.function_response:
                        result = part.function_response.result
                        if isinstance(result, dict):
                            content = json.dumps(result)
                        else:
                            content = str(result) if result is not None else ""
                        # Sanitize tool name for tool_call_id generation
                        sanitized_name = sanitize_tool_name(part.function_response.name)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": part.function_response.call_id or f"call_{sanitized_name}",
                            "content": content,
                        })

        # Validate tool_call_id references match assistant tool_calls
        # Collect all tool_call IDs from assistant messages
        assistant_tool_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("id"):
                        assistant_tool_ids.add(tc["id"])

        # Check tool messages reference valid IDs
        for msg in messages:
            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id and assistant_tool_ids and tool_call_id not in assistant_tool_ids:
                    self._trace(f"WARNING: tool_call_id '{tool_call_id}' not found in assistant tool_calls: {assistant_tool_ids}")

        return messages

    def _build_copilot_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Build tools list for Copilot API (OpenAI format).

        Tool names are sanitized to match OpenAI's pattern ^[a-zA-Z0-9_-]{1,64}$.
        """
        if not self._tools:
            return None

        tools = []
        for tool in self._tools:
            # Sanitize tool name for OpenAI-compatible API
            sanitized_name = sanitize_tool_name(tool.name)
            register_tool_name_mapping(sanitized_name, tool.name)

            tool_dict = {
                "type": "function",
                "function": {
                    "name": sanitized_name,
                    "description": tool.description or "",
                }
            }
            if tool.parameters:
                tool_dict["function"]["parameters"] = tool.parameters
            tools.append(tool_dict)

        return tools if tools else None

    def _copilot_response_to_provider(self, response: "CopilotResponse") -> ProviderResponse:
        """Convert Copilot API response to ProviderResponse."""
        from .copilot_client import CopilotResponse
        from jaato_sdk.plugins.model_provider.types import FunctionCall

        parts = []
        finish_reason = FinishReason.STOP
        thinking = None

        if response.choices:
            choice = response.choices[0]
            # Extract reasoning/thinking content (e.g. DeepSeek-R1)
            if self._enable_thinking:
                reasoning = getattr(choice.message, "reasoning_content", None)
                if reasoning:
                    thinking = reasoning

            # Extract text
            if choice.message.content:
                parts.append(Part.from_text(choice.message.content))
            # Extract tool calls
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    # Restore original tool name from sanitized version
                    sanitized_name = tc.get("function", {}).get("name", "")
                    original_name = get_original_tool_name(sanitized_name)
                    fc = FunctionCall(
                        id=tc.get("id"),
                        name=original_name,
                        args=args,
                    )
                    parts.append(Part.from_function_call(fc))
                finish_reason = FinishReason.TOOL_USE

            # Map finish reason
            if choice.finish_reason:
                finish_reason = self._map_finish_reason(choice.finish_reason)

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,
            thinking=thinking,
        )

    def _handle_api_error(self, error: Exception) -> None:
        """Handle API errors and convert to appropriate exceptions."""
        import httpx
        import requests.exceptions

        # Check for httpx transport errors (Copilot client uses httpx).
        # TransportError covers both NetworkError (ConnectError, ReadError)
        # and ProtocolError (RemoteProtocolError) — all transient failures.
        if isinstance(error, httpx.TransportError):
            raise InfrastructureError(
                status_code=0,  # No HTTP status for network-level errors
                original_error=f"{type(error).__name__}: {error}",
            ) from error

        # Check for chunked encoding errors (response ended prematurely)
        # These are transient network errors that should be retriable
        if isinstance(error, requests.exceptions.ChunkedEncodingError):
            raise InfrastructureError(
                status_code=0,  # No HTTP status for network-level errors
                original_error=f"ChunkedEncodingError: {error}",
            ) from error

        # Check for connection errors (network issues)
        if isinstance(error, requests.exceptions.ConnectionError):
            raise InfrastructureError(
                status_code=0,
                original_error=f"ConnectionError: {error}",
            ) from error

        # Check for timeout errors
        if isinstance(error, requests.exceptions.Timeout):
            raise InfrastructureError(
                status_code=0,
                original_error=f"Timeout: {error}",
            ) from error

        error_str = str(error).lower()

        # Check for disabled error
        if "disabled" in error_str and "models" in error_str:
            raise ModelsDisabledError(
                organization=self._organization,
                enterprise=self._enterprise,
                original_error=str(error),
            ) from error

        # Check for auth errors
        if "401" in error_str or "unauthorized" in error_str:
            if "disabled" in error_str:
                raise ModelsDisabledError(
                    organization=self._organization,
                    enterprise=self._enterprise,
                    original_error=str(error),
                ) from error
            raise TokenInvalidError(
                reason="Token rejected by API",
                token_prefix=self._token[:10] if self._token else None,
                original_error=str(error),
            ) from error

        # Check for permission errors
        if "403" in error_str or "forbidden" in error_str:
            raise TokenPermissionError(
                organization=self._organization,
                enterprise=self._enterprise,
                original_error=str(error),
            ) from error

        # Check for rate limit errors
        if "429" in error_str or "rate limit" in error_str:
            raise RateLimitError(
                original_error=str(error),
            ) from error

        # Check for payload too large errors (HTTP 413)
        # This indicates the request body is too large for the API
        if "413" in error_str or "payload too large" in error_str:
            raise PayloadTooLargeError(
                original_error=str(error),
            ) from error

        # Check for context/token limit errors
        if any(x in error_str for x in ("tokens_limit_reached", "too large", "max size", "context_length")):
            # Try to extract max tokens from error message
            import re
            max_tokens = None
            match = re.search(r'max (?:size|tokens)[:\s]+(\d+)', error_str)
            if match:
                max_tokens = int(match.group(1))
            raise ContextLimitError(
                model=self._model_name or "unknown",
                max_tokens=max_tokens,
                original_error=str(error),
            ) from error

        # Check for model not found (various formats).
        # Exclude "tool_call_id ... not found" which is a different error
        # (orphaned tool results referencing non-existent tool calls).
        if (any(x in error_str for x in ("404", "not found", "unknown_model", "unknown model"))
                and "tool_call_id" not in error_str):
            raise ModelNotFoundError(
                model=self._model_name or "unknown",
                available_models=self.list_models(),
                original_error=str(error),
            ) from error

        # Check for infrastructure errors (5xx)
        import re
        status_match = re.search(r'http (\d{3})', error_str)
        if status_match:
            status_code = int(status_match.group(1))
            if 500 <= status_code < 600:
                raise InfrastructureError(
                    status_code=status_code,
                    original_error=str(error),
                ) from error

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Count tokens for the given content.

        Note: This is an estimate as GitHub Models API doesn't provide
        a token counting endpoint. Uses ~4 chars per token heuristic.

        Args:
            content: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        # Rough estimate: ~4 characters per token
        return len(content) // 4

    def get_context_limit(self) -> int:
        """Get the context window size for the current model.

        Attempts to use API-fetched token limits first, then falls back
        to hardcoded limits if API data is unavailable.

        Returns:
            Maximum tokens the model can handle.
        """
        if not self._model_name:
            return DEFAULT_CONTEXT_LIMIT

        # Try to get from API-fetched model info first
        model_info = self.get_model_info()
        if model_info:
            limit = model_info.effective_context_limit
            if limit > 0:
                return limit

        # Fallback to hardcoded limits
        # Try exact match
        if self._model_name in FALLBACK_CONTEXT_LIMITS:
            return FALLBACK_CONTEXT_LIMITS[self._model_name]

        # Try prefix match
        for model_prefix, limit in FALLBACK_CONTEXT_LIMITS.items():
            if self._model_name.startswith(model_prefix):
                return limit

        return DEFAULT_CONTEXT_LIMIT

    def get_token_usage(self) -> TokenUsage:
        """Get token usage from the last response.

        Returns:
            TokenUsage with prompt/output/total counts.
        """
        return self._last_usage

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """Check if structured output is supported.

        Note: Structured output support varies by model.
        OpenAI models generally support it, others may not.

        Returns:
            True for OpenAI models, False otherwise.
        """
        if not self._model_name:
            return False
        return self._model_name.startswith("openai/")

    def supports_streaming(self) -> bool:
        """Check if streaming is supported.

        Returns:
            True - GitHub Models supports streaming via Azure AI Inference SDK.
        """
        return True

    def supports_stop(self) -> bool:
        """Check if mid-turn cancellation (stop) is supported.

        Returns:
            True - GitHub Models supports stop via streaming cancellation.
        """
        return True

    def supports_thinking(self) -> bool:
        """Check if reasoning/thinking content is supported.

        Returns True for models known to expose ``reasoning_content``
        (e.g. DeepSeek-R1).  OpenAI o-series models use reasoning
        internally but never surface it through the Chat Completions API,
        so they return False here.

        Returns:
            True if the current model exposes reasoning content.
        """
        return self._is_reasoning_capable()

    def set_thinking_config(self, config: 'ThinkingConfig') -> None:
        """Set thinking configuration.

        For reasoning-capable models this enables/disables extraction of
        ``reasoning_content`` from responses.  Models that don't expose
        reasoning ignore this setting.

        Args:
            config: ThinkingConfig with enabled flag and budget.
        """
        self._enable_thinking = config.enabled

    def _is_reasoning_capable(self) -> bool:
        """Check if the current model exposes reasoning content."""
        if not self._model_name:
            return False
        name_lower = self._model_name.lower()
        for prefix in REASONING_CAPABLE_MODELS:
            if name_lower.startswith(prefix) or name_lower.endswith(prefix):
                return True
        return False

    # ==================== Streaming ====================

    def _copilot_streaming_response(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None,
        trace_prefix: str = "STREAM",
    ) -> ProviderResponse:
        """Handle streaming response from Copilot API.

        Args:
            messages: Messages in OpenAI format.
            tools: Tools in OpenAI format.
            on_chunk: Callback for each text chunk.
            cancel_token: Optional cancellation token.
            on_usage_update: Optional usage callback.
            on_thinking: Optional callback for reasoning/thinking chunks.
            trace_prefix: Prefix for trace logging.

        Returns:
            ProviderResponse with accumulated response.
        """
        from jaato_sdk.plugins.model_provider.types import FunctionCall

        accumulated_text = []
        accumulated_thinking: List[str] = []
        parts = []
        finish_reason = FinishReason.UNKNOWN
        function_calls = []
        usage = TokenUsage()
        was_cancelled = False

        # Track tool call accumulation (streaming sends tool calls in pieces)
        tool_call_accumulators: Dict[int, Dict[str, Any]] = {}

        def flush_text_block():
            nonlocal accumulated_text
            if accumulated_text:
                text = ''.join(accumulated_text)
                parts.append(Part.from_text(text))
                accumulated_text = []

        def flush_tool_calls():
            """Flush accumulated tool calls as Parts."""
            nonlocal tool_call_accumulators
            for idx in sorted(tool_call_accumulators.keys()):
                tc = tool_call_accumulators[idx]
                if tc.get("function", {}).get("name"):
                    try:
                        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    # OpenAI/Copilot API should always return an ID for tool calls.
                    # If missing, log for investigation - this indicates a parsing issue.
                    tool_id = tc.get("id")
                    # Restore original tool name from sanitized version
                    sanitized_name = tc["function"]["name"]
                    original_name = get_original_tool_name(sanitized_name)
                    if not tool_id:
                        self._trace(f"ERROR: Missing tool call ID from API for {sanitized_name} - this will cause 400 errors")
                    fc = FunctionCall(
                        id=tool_id,  # May be None - will cause API error, which is correct
                        name=original_name,
                        args=args,
                    )
                    parts.append(Part.from_function_call(fc))
                    function_calls.append(fc)
            tool_call_accumulators.clear()

        try:
            self._trace(f"{trace_prefix}_START")
            chunk_count = 0

            for choice in self._copilot_client.complete_stream(
                model=self._copilot_model_name(),
                messages=messages,
                tools=tools,
            ):
                if cancel_token and cancel_token.is_cancelled:
                    self._trace(f"{trace_prefix}_CANCELLED after {chunk_count} chunks")
                    was_cancelled = True
                    finish_reason = FinishReason.CANCELLED
                    break

                delta = choice.delta

                # Extract reasoning/thinking (e.g. DeepSeek-R1)
                if self._enable_thinking:
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        self._trace(f"{trace_prefix}_THINKING len={len(reasoning)}")
                        accumulated_thinking.append(reasoning)
                        if on_thinking:
                            on_thinking(reasoning)

                # Accumulate text
                if delta.content:
                    chunk_count += 1
                    self._trace(f"{trace_prefix}_CHUNK[{chunk_count}] len={len(delta.content)}")
                    accumulated_text.append(delta.content)
                    on_chunk(delta.content)

                # Accumulate tool calls (they come in pieces)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.get("index", 0)
                        if idx not in tool_call_accumulators:
                            # Log first occurrence of tool call
                            tc_id = tc.get("id")
                            tc_name = tc.get("function", {}).get("name", "")
                            self._trace(f"TOOL_CALL_START idx={idx} id={tc_id!r} name={tc_name!r}")
                            tool_call_accumulators[idx] = {
                                "id": tc_id,
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        acc = tool_call_accumulators[idx]
                        if tc.get("id"):
                            acc["id"] = tc["id"]
                        if tc.get("function", {}).get("name"):
                            acc["function"]["name"] = tc["function"]["name"]
                        if tc.get("function", {}).get("arguments"):
                            acc["function"]["arguments"] += tc["function"]["arguments"]

                # Extract finish reason
                if choice.finish_reason:
                    finish_reason = self._map_finish_reason(choice.finish_reason)

            self._trace(f"{trace_prefix}_END chunks={chunk_count} finish_reason={finish_reason}")

        except Exception as e:
            self._trace(f"{trace_prefix}_ERROR {type(e).__name__}: {e}")
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                self._handle_api_error(e)
                raise

        # Flush remaining text and tool calls
        flush_text_block()
        flush_tool_calls()

        if function_calls and not was_cancelled:
            finish_reason = FinishReason.TOOL_USE

        thinking = ''.join(accumulated_thinking) if accumulated_thinking else None

        return ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,
            thinking=thinking,
        )

    def _copilot_responses_streaming(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None,
        trace_prefix: str = "RESPONSES_STREAM",
    ) -> ProviderResponse:
        """Handle streaming response from Copilot Responses API.

        Args:
            messages: Messages in OpenAI format.
            tools: Tools in OpenAI format.
            on_chunk: Callback for each text chunk.
            cancel_token: Optional cancellation token.
            on_usage_update: Optional usage callback.
            on_thinking: Optional callback for reasoning/thinking chunks.
            trace_prefix: Prefix for trace logging.

        Returns:
            ProviderResponse with accumulated response.
        """
        from jaato_sdk.plugins.model_provider.types import FunctionCall

        accumulated_text = []
        accumulated_thinking: List[str] = []
        parts = []
        finish_reason = FinishReason.UNKNOWN
        function_calls = []
        usage = TokenUsage()
        was_cancelled = False

        def flush_text_block():
            nonlocal accumulated_text
            if accumulated_text:
                text = ''.join(accumulated_text)
                parts.append(Part.from_text(text))
                accumulated_text = []

        try:
            self._trace(f"{trace_prefix}_START")
            chunk_count = 0

            for event in self._copilot_client.complete_responses_stream(
                model=self._copilot_model_name(),
                messages=messages,
                system_instruction=self._system_instruction,
                tools=tools,
            ):
                if cancel_token and cancel_token.is_cancelled:
                    self._trace(f"{trace_prefix}_CANCELLED after {chunk_count} chunks")
                    was_cancelled = True
                    finish_reason = FinishReason.CANCELLED
                    break

                event_type = event.get("type", "")

                if event_type == "text":
                    # Text content
                    text = event.get("text", "")
                    if text:
                        chunk_count += 1
                        self._trace(f"{trace_prefix}_CHUNK[{chunk_count}] len={len(text)}")
                        accumulated_text.append(text)
                        on_chunk(text)

                elif event_type == "reasoning":
                    # Reasoning/thinking content (Responses API)
                    if self._enable_thinking:
                        reasoning_text = event.get("text", "")
                        if reasoning_text:
                            self._trace(f"{trace_prefix}_THINKING len={len(reasoning_text)}")
                            accumulated_thinking.append(reasoning_text)
                            if on_thinking:
                                on_thinking(reasoning_text)

                elif event_type == "function_call":
                    # Function call complete
                    flush_text_block()
                    try:
                        args = json.loads(event.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    # Restore original tool name from sanitized version
                    sanitized_name = event.get("name", "")
                    original_name = get_original_tool_name(sanitized_name)
                    fc = FunctionCall(
                        id=event.get("call_id"),
                        name=original_name,
                        args=args,
                    )
                    parts.append(Part.from_function_call(fc))
                    function_calls.append(fc)
                    self._trace(f"{trace_prefix}_FUNC_CALL name={fc.name}")

                elif event_type == "done":
                    # Completion with optional usage
                    usage_data = event.get("usage", {})
                    if usage_data:
                        usage = TokenUsage(
                            prompt_tokens=usage_data.get("input_tokens", 0),
                            output_tokens=usage_data.get("output_tokens", 0),
                            total_tokens=usage_data.get("total_tokens", 0),
                        )
                        self._trace(f"{trace_prefix}_USAGE prompt={usage.prompt_tokens} output={usage.output_tokens}")
                        if on_usage_update and usage.total_tokens > 0:
                            on_usage_update(usage)
                    finish_reason = FinishReason.STOP

            self._trace(f"{trace_prefix}_END chunks={chunk_count} finish_reason={finish_reason}")

        except Exception as e:
            self._trace(f"{trace_prefix}_ERROR {type(e).__name__}: {e}")
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                self._handle_api_error(e)
                raise

        # Flush remaining text
        flush_text_block()

        if function_calls and not was_cancelled:
            finish_reason = FinishReason.TOOL_USE

        thinking = ''.join(accumulated_thinking) if accumulated_thinking else None

        return ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,
            thinking=thinking,
        )

    def send_message_streaming(
        self,
        message: str,
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None
    ) -> ProviderResponse:
        """Send a message with streaming response and optional cancellation.

        Args:
            message: The user's message text.
            on_chunk: Callback invoked for each text chunk as it streams.
            cancel_token: Optional token to request cancellation mid-stream.
            response_schema: Optional JSON Schema to constrain the response.
            on_usage_update: Optional callback for real-time token usage updates.

        Returns:
            ProviderResponse with accumulated text and/or function calls.
        """
        # Add user message to history first
        self._history.append(Message.from_text(Role.USER, message))

        if self._use_copilot_api:
            if not self._copilot_client or not self._model_name:
                raise RuntimeError("No chat session. Call create_session() first.")

            messages = self._build_copilot_messages()
            tools = self._build_copilot_tools()

            try:
                # Route to Responses API for codex models
                if self._is_responses_api_model():
                    self._trace(f"[SEND_MESSAGE_STREAMING] Using Responses API for {self._copilot_model_name()}")
                    provider_response = self._copilot_responses_streaming(
                        messages=messages,
                        tools=tools,
                        on_chunk=on_chunk,
                        cancel_token=cancel_token,
                        on_usage_update=on_usage_update,
                        on_thinking=on_thinking,
                        trace_prefix="RESPONSES_STREAM",
                    )
                else:
                    provider_response = self._copilot_streaming_response(
                        messages=messages,
                        tools=tools,
                        on_chunk=on_chunk,
                        cancel_token=cancel_token,
                        on_usage_update=on_usage_update,
                        on_thinking=on_thinking,
                        trace_prefix="STREAM",
                    )

                self._last_usage = provider_response.usage
                self._add_response_to_history(provider_response)

                # Parse structured output if schema was requested
                final_text = provider_response.get_text()
                if response_schema and final_text and provider_response.finish_reason != FinishReason.CANCELLED:
                    try:
                        provider_response.structured_output = json.loads(final_text)
                    except json.JSONDecodeError:
                        pass

                return provider_response
            except Exception as e:
                # Rollback the user message we added
                if self._history and self._history[-1].role == Role.USER:
                    self._history.pop()
                self._handle_api_error(e)
                raise

        # Azure SDK path
        if not self._client or not self._model_name:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Build messages list (includes user message from history)
        messages = self._build_messages()

        # Build completion kwargs
        kwargs = self._build_completion_kwargs(response_schema)
        kwargs['stream'] = True

        # Accumulate response with parts preserving order
        accumulated_text = []  # Text chunks for current text block
        accumulated_thinking: List[str] = []  # Reasoning/thinking chunks
        parts = []  # Ordered parts preserving text/function_call interleaving
        finish_reason = FinishReason.UNKNOWN
        function_calls = []  # Also keep flat list for backwards compatibility
        usage = TokenUsage()
        was_cancelled = False

        def flush_text_block():
            """Flush accumulated text as a single Part."""
            nonlocal accumulated_text
            if accumulated_text:
                text = ''.join(accumulated_text)
                parts.append(Part.from_text(text))
                accumulated_text = []

        try:
            self._trace(f"STREAM_START message_len={len(message)}")
            self._trace(f"STREAM_INPUT>>>\n{message}\n<<<STREAM_INPUT")
            chunk_count = 0
            response_stream = self._client.complete(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )

            for chunk in response_stream:
                # Check for cancellation
                if cancel_token and cancel_token.is_cancelled:
                    self._trace(f"STREAM_CANCELLED after {chunk_count} chunks")
                    was_cancelled = True
                    finish_reason = FinishReason.CANCELLED
                    break

                # Extract content from choices
                if hasattr(chunk, 'choices') and chunk.choices:
                    for choice in chunk.choices:
                        # Extract text delta
                        if hasattr(choice, 'delta') and choice.delta:
                            delta = choice.delta

                            # Extract reasoning/thinking (e.g. DeepSeek-R1)
                            if self._enable_thinking:
                                reasoning_chunk = extract_reasoning_from_stream_delta(delta)
                                if reasoning_chunk:
                                    self._trace(f"STREAM_THINKING len={len(reasoning_chunk)}")
                                    accumulated_thinking.append(reasoning_chunk)
                                    if on_thinking:
                                        on_thinking(reasoning_chunk)

                            if hasattr(delta, 'content') and delta.content:
                                chunk_count += 1
                                self._trace(f"STREAM_CHUNK[{chunk_count}] len={len(delta.content)} text={repr(delta.content)}")
                                accumulated_text.append(delta.content)
                                on_chunk(delta.content)

                            # Extract tool calls
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                from .converters import extract_function_calls_from_stream_delta
                                new_calls = extract_function_calls_from_stream_delta(delta.tool_calls)
                                for fc in new_calls:
                                    self._trace(f"STREAM_FUNC_CALL name={fc.name}")
                                    # Flush any pending text before adding function call
                                    flush_text_block()
                                    # Add function call as a part
                                    parts.append(Part.from_function_call(fc))
                                function_calls.extend(new_calls)

                        # Extract finish reason
                        if hasattr(choice, 'finish_reason') and choice.finish_reason:
                            finish_reason = self._map_finish_reason(choice.finish_reason)

                # Extract usage from final chunk
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=getattr(chunk.usage, 'prompt_tokens', 0) or 0,
                        output_tokens=getattr(chunk.usage, 'completion_tokens', 0) or 0,
                        total_tokens=getattr(chunk.usage, 'total_tokens', 0) or 0
                    )
                    self._trace(f"STREAM_USAGE prompt={usage.prompt_tokens} output={usage.output_tokens} total={usage.total_tokens}")
                    # Notify about usage update for real-time accounting
                    if on_usage_update and usage.total_tokens > 0:
                        on_usage_update(usage)

            # Get accumulated output before flushing
            all_text = ''.join(accumulated_text)
            self._trace(f"STREAM_END chunks={chunk_count} finish_reason={finish_reason} output_len={len(all_text)}")
            self._trace(f"STREAM_OUTPUT>>>\n{all_text}\n<<<STREAM_OUTPUT")

        except Exception as e:
            self._trace(f"STREAM_ERROR {type(e).__name__}: {e}")
            # If cancelled during iteration, treat as cancellation
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                # Rollback the user message we added
                if self._history and self._history[-1].role == Role.USER:
                    self._history.pop()
                self._handle_api_error(e)
                raise

        # Flush any remaining text as final part
        flush_text_block()

        # If we have function calls, update finish reason
        if function_calls and not was_cancelled:
            finish_reason = FinishReason.TOOL_USE

        thinking = ''.join(accumulated_thinking) if accumulated_thinking else None

        provider_response = ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,
            thinking=thinking,
        )

        self._last_usage = usage

        # Add assistant response to history
        self._add_response_to_history(provider_response)

        # Parse structured output if schema was requested
        final_text = provider_response.get_text()
        if response_schema and final_text and not was_cancelled:
            try:
                provider_response.structured_output = json.loads(final_text)
            except json.JSONDecodeError:
                pass

        return provider_response

    def send_tool_results_streaming(
        self,
        results: List[ToolResult],
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None
    ) -> ProviderResponse:
        """Send tool results with streaming response and optional cancellation.

        Args:
            results: List of tool execution results.
            on_chunk: Callback invoked for each text chunk as it streams.
            cancel_token: Optional token to request cancellation mid-stream.
            response_schema: Optional JSON Schema to constrain the response.
            on_usage_update: Optional callback for real-time token usage updates.

        Returns:
            ProviderResponse with accumulated text and/or function calls.
        """
        # Add tool results to history
        for result in results:
            self._history.append(Message(
                role=Role.TOOL,
                parts=[Part(function_response=result)],
            ))

        if self._use_copilot_api:
            if not self._copilot_client or not self._model_name:
                raise RuntimeError("No chat session. Call create_session() first.")

            messages = self._build_copilot_messages()
            tools = self._build_copilot_tools()

            try:
                # Route to Responses API for codex models
                if self._is_responses_api_model():
                    self._trace(f"[SEND_TOOL_RESULTS_STREAMING] Using Responses API for {self._copilot_model_name()}")
                    provider_response = self._copilot_responses_streaming(
                        messages=messages,
                        tools=tools,
                        on_chunk=on_chunk,
                        cancel_token=cancel_token,
                        on_usage_update=on_usage_update,
                        on_thinking=on_thinking,
                        trace_prefix="RESPONSES_STREAM_TOOL_RESULTS",
                    )
                else:
                    provider_response = self._copilot_streaming_response(
                        messages=messages,
                        tools=tools,
                        on_chunk=on_chunk,
                        cancel_token=cancel_token,
                        on_usage_update=on_usage_update,
                        on_thinking=on_thinking,
                        trace_prefix="STREAM_TOOL_RESULTS",
                    )

                self._last_usage = provider_response.usage
                self._add_response_to_history(provider_response)

                # Parse structured output if schema was requested
                final_text = provider_response.get_text()
                if response_schema and final_text and provider_response.finish_reason != FinishReason.CANCELLED:
                    try:
                        provider_response.structured_output = json.loads(final_text)
                    except json.JSONDecodeError:
                        pass

                return provider_response
            except Exception as e:
                # Rollback the tool result messages we added
                self._rollback_tool_results(len(results))
                self._handle_api_error(e)
                raise

        # Azure SDK path
        if not self._client or not self._model_name:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Build messages including tool results
        messages = self._build_messages()

        # Build completion kwargs
        kwargs = self._build_completion_kwargs(response_schema)
        kwargs['stream'] = True

        # Accumulate response with parts preserving order
        accumulated_text = []  # Text chunks for current text block
        accumulated_thinking: List[str] = []  # Reasoning/thinking chunks
        parts = []  # Ordered parts preserving text/function_call interleaving
        finish_reason = FinishReason.UNKNOWN
        function_calls = []  # Also keep flat list for backwards compatibility
        usage = TokenUsage()
        was_cancelled = False

        def flush_text_block():
            """Flush accumulated text as a single Part."""
            nonlocal accumulated_text
            if accumulated_text:
                text = ''.join(accumulated_text)
                parts.append(Part.from_text(text))
                accumulated_text = []

        try:
            tool_names = [r.name for r in results]
            self._trace(f"STREAM_TOOL_RESULTS_START tools={tool_names}")
            # Log tool results as input
            tool_results_summary = []
            for r in results:
                result_str = str(r.result) if r.result is not None else "None"
                tool_results_summary.append(f"{r.name}: {result_str}")
            self._trace(f"STREAM_TOOL_INPUT>>>\n" + "\n".join(tool_results_summary) + "\n<<<STREAM_TOOL_INPUT")
            chunk_count = 0
            response_stream = self._client.complete(
                model=self._model_name,
                messages=messages,
                **kwargs,
            )

            for chunk in response_stream:
                # Check for cancellation
                if cancel_token and cancel_token.is_cancelled:
                    self._trace(f"STREAM_TOOL_RESULTS_CANCELLED after {chunk_count} chunks")
                    was_cancelled = True
                    finish_reason = FinishReason.CANCELLED
                    break

                # Extract content from choices
                if hasattr(chunk, 'choices') and chunk.choices:
                    for choice in chunk.choices:
                        # Extract text delta
                        if hasattr(choice, 'delta') and choice.delta:
                            delta = choice.delta

                            # Extract reasoning/thinking (e.g. DeepSeek-R1)
                            if self._enable_thinking:
                                reasoning_chunk = extract_reasoning_from_stream_delta(delta)
                                if reasoning_chunk:
                                    self._trace(f"STREAM_TOOL_THINKING len={len(reasoning_chunk)}")
                                    accumulated_thinking.append(reasoning_chunk)
                                    if on_thinking:
                                        on_thinking(reasoning_chunk)

                            if hasattr(delta, 'content') and delta.content:
                                chunk_count += 1
                                self._trace(f"STREAM_TOOL_CHUNK[{chunk_count}] len={len(delta.content)} text={repr(delta.content)}")
                                accumulated_text.append(delta.content)
                                on_chunk(delta.content)

                            # Extract tool calls
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                from .converters import extract_function_calls_from_stream_delta
                                new_calls = extract_function_calls_from_stream_delta(delta.tool_calls)
                                for fc in new_calls:
                                    self._trace(f"STREAM_TOOL_FUNC_CALL name={fc.name}")
                                    # Flush any pending text before adding function call
                                    flush_text_block()
                                    # Add function call as a part
                                    parts.append(Part.from_function_call(fc))
                                function_calls.extend(new_calls)

                        # Extract finish reason
                        if hasattr(choice, 'finish_reason') and choice.finish_reason:
                            finish_reason = self._map_finish_reason(choice.finish_reason)

                # Extract usage from final chunk
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=getattr(chunk.usage, 'prompt_tokens', 0) or 0,
                        output_tokens=getattr(chunk.usage, 'completion_tokens', 0) or 0,
                        total_tokens=getattr(chunk.usage, 'total_tokens', 0) or 0
                    )
                    self._trace(f"STREAM_TOOL_USAGE prompt={usage.prompt_tokens} output={usage.output_tokens} total={usage.total_tokens}")
                    # Notify about usage update for real-time accounting
                    if on_usage_update and usage.total_tokens > 0:
                        on_usage_update(usage)

            # Get accumulated output before flushing
            all_text = ''.join(accumulated_text)
            self._trace(f"STREAM_TOOL_RESULTS_END chunks={chunk_count} finish_reason={finish_reason} output_len={len(all_text)}")
            self._trace(f"STREAM_TOOL_OUTPUT>>>\n{all_text}\n<<<STREAM_TOOL_OUTPUT")

        except Exception as e:
            self._trace(f"STREAM_TOOL_RESULTS_ERROR {type(e).__name__}: {e}")
            # If cancelled during iteration, treat as cancellation
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                # Rollback the tool result messages we added
                self._rollback_tool_results(len(results))
                self._handle_api_error(e)
                raise

        # Flush any remaining text as final part
        flush_text_block()

        # If we have function calls, update finish reason
        if function_calls and not was_cancelled:
            finish_reason = FinishReason.TOOL_USE

        thinking = ''.join(accumulated_thinking) if accumulated_thinking else None

        provider_response = ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,
            thinking=thinking,
        )

        self._last_usage = usage

        # Add assistant response to history
        self._add_response_to_history(provider_response)

        # Parse structured output if schema was requested
        final_text = provider_response.get_text()
        if response_schema and final_text and not was_cancelled:
            try:
                provider_response.structured_output = json.loads(final_text)
            except json.JSONDecodeError:
                pass

        return provider_response

    def _map_finish_reason(self, reason: str) -> FinishReason:
        """Map SDK finish reason string to internal FinishReason."""
        if not reason:
            return FinishReason.UNKNOWN

        reason_lower = reason.lower()
        if reason_lower == 'stop':
            return FinishReason.STOP
        elif reason_lower in ('length', 'max_tokens'):
            return FinishReason.MAX_TOKENS
        elif reason_lower in ('tool_calls', 'function_call'):
            return FinishReason.TOOL_USE
        elif reason_lower == 'content_filter':
            return FinishReason.SAFETY

        return FinishReason.UNKNOWN

    # ==================== Serialization ====================

    def serialize_history(self, history: List[Message]) -> str:
        """Serialize conversation history to a JSON string.

        Args:
            history: List of messages to serialize.

        Returns:
            JSON string representation.
        """
        return serialize_history(history)

    def deserialize_history(self, data: str) -> List[Message]:
        """Deserialize conversation history from a JSON string.

        Args:
            data: Previously serialized history string.

        Returns:
            List of Message objects.
        """
        return deserialize_history(data)

    # ==================== Error Classification for Retry ====================

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Classify an exception for retry purposes.

        GitHub Models API uses specific error types for rate limits
        and infrastructure errors.

        Args:
            exc: The exception to classify.

        Returns:
            Classification dict or None to use fallback.
        """
        from .errors import RateLimitError, InfrastructureError

        if isinstance(exc, RateLimitError):
            return {"transient": True, "rate_limit": True, "infra": False}

        if isinstance(exc, InfrastructureError):
            return {"transient": True, "rate_limit": False, "infra": True}

        # Fall back to global classification
        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """Extract retry-after hint from an exception.

        GitHub's RateLimitError includes retry_after attribute.

        Args:
            exc: The exception to extract retry-after from.

        Returns:
            Suggested delay in seconds, or None if not available.
        """
        from .errors import RateLimitError

        if isinstance(exc, RateLimitError) and exc.retry_after:
            return float(exc.retry_after)

        return None


def create_provider() -> GitHubModelsProvider:
    """Factory function for plugin discovery."""
    return GitHubModelsProvider()
