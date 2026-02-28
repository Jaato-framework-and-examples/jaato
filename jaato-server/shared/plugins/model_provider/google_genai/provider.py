"""Google GenAI (Vertex AI / Gemini) model provider implementation.

This provider encapsulates all interactions with the Google GenAI SDK,
supporting both:
- Google AI Studio (api.generativelanguage.googleapis.com) - API key auth
- Vertex AI (vertexai.googleapis.com) - GCP authentication

Authentication methods:
- API Key: For AI Studio, simple development use
- ADC (Application Default Credentials): For Vertex AI, local development
- Service Account File: For Vertex AI, production/CI use
- Impersonation: For Vertex AI, act as another service account
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Lazy imports - SDK is only loaded when actually used
from ._lazy import get_genai, get_types

if TYPE_CHECKING:
    from google import genai
    from google.genai import types

from ..base import (
    FunctionCallDetectedCallback,
    GoogleAuthMethod,
    ModelProviderPlugin,
    ProviderConfig,
    StreamingCallback,
    ThinkingCallback,
    UsageUpdateCallback,
)
from jaato_sdk.plugins.model_provider.types import (
    CancelToken,
    FinishReason,
    Message,
    ProviderResponse,
    Role,
    ThinkingConfig,
    ToolSchema,
    TokenUsage,
    TurnResult,
    Part,
)
from .converters import (
    extract_text_from_chunk,
    history_to_sdk,
    response_from_sdk,
    tool_schemas_to_sdk_tool,
    serialize_history,
    deserialize_history,
)
from .errors import (
    CredentialsNotFoundError,
    CredentialsInvalidError,
    CredentialsPermissionError,
    ProjectConfigurationError,
    ImpersonationError,
)
from .env import (
    resolve_auth_method,
    resolve_use_vertex,
    resolve_api_key,
    resolve_credentials_path,
    resolve_project,
    resolve_location,
    resolve_target_service_account,
    get_checked_credential_locations,
)


# Context window limits for known Gemini models (total tokens)
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    # Gemini 2.5 models
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-pro-preview-05-06": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-flash-preview-04-17": 1_048_576,
    # Gemini 2.0 models
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.0-flash-exp": 1_048_576,
    "gemini-2.0-flash-lite": 1_048_576,
    # Gemini 1.5 models
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-pro-latest": 2_097_152,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.5-flash-latest": 1_048_576,
    # Gemini 1.0 models (legacy)
    "gemini-1.0-pro": 32_760,
    "gemini-pro": 32_760,
}

DEFAULT_CONTEXT_LIMIT = 1_048_576


class GoogleGenAIProvider:
    """Google GenAI / Vertex AI model provider (stateless design).

    This provider is **stateless** with respect to conversation history.
    It does not maintain an internal chat session or message list.  The
    session layer (``JaatoSession``) owns the conversation history and
    passes the full message list to ``complete()`` on every call.

    Supports:
    - Dual endpoints: Google AI Studio (API key) and Vertex AI (GCP auth)
    - Multiple auth methods: API key, ADC, service account file
    - Gemini model family (1.5, 2.0, 2.5)
    - Stateless completion via ``complete()`` (batch and streaming)
    - Function calling with manual control
    - Token counting and context management
    - Optional CachedContent integration via cache plugin

    Usage::

        provider = GoogleGenAIProvider()
        provider.initialize(ProviderConfig(
            project='my-project',
            location='us-central1',
            use_vertex_ai=True,
            auth_method='auto',
        ))
        provider.connect('gemini-2.5-flash')
        response = provider.complete(
            messages=[Message(role=Role.USER, parts=[Part.from_text("Hello!")])],
            system_instruction="You are a helpful assistant.",
        )

    Environment variables (auto-detected if config not provided):
        GOOGLE_GENAI_API_KEY: API key for AI Studio
        GOOGLE_APPLICATION_CREDENTIALS: Service account file for Vertex AI
        JAATO_GOOGLE_PROJECT / GOOGLE_CLOUD_PROJECT: GCP project ID
        JAATO_GOOGLE_LOCATION: GCP region (e.g., us-central1)
        JAATO_GOOGLE_AUTH_METHOD: Force specific auth method
        JAATO_GOOGLE_USE_VERTEX: Force Vertex AI (true) or AI Studio (false)
    """

    # Cache TTL in seconds (5 minutes)
    _MODELS_CACHE_TTL = 300

    def __init__(self):
        """Initialize the provider (not yet connected).

        The provider is stateless with respect to conversation history.
        It holds only connection/auth state and per-call accounting.
        """
        self._client: Optional[genai.Client] = None
        self._model_name: Optional[str] = None
        self._project: Optional[str] = None
        self._location: Optional[str] = None

        # Authentication state
        self._use_vertex_ai: bool = True
        self._auth_method: GoogleAuthMethod = "auto"

        # Per-call token accounting (updated after each complete() call)
        self._last_usage: TokenUsage = TokenUsage()

        # Models cache: (timestamp, models_list)
        self._models_cache: Optional[Tuple[float, List[str]]] = None

        # Cache plugin (optional, for explicit CachedContent management)
        self._cache_plugin: Optional[Any] = None

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
            return "google_genai:main"
        elif self._agent_name:
            return f"google_genai:subagent:{self._agent_name}"
        else:
            return f"google_genai:subagent:{self._agent_id}"

    def _trace(self, msg: str) -> None:
        """Write trace message to file for debugging streaming interactions."""
        from shared.trace import provider_trace
        prefix = self._get_trace_prefix()
        provider_trace(prefix, msg)

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "google_genai"

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider with credentials.

        Supports both Google AI Studio (API key) and Vertex AI (GCP auth).
        Uses fail-fast validation to catch configuration errors early.

        Args:
            config: Configuration with authentication details.
                If not provided, configuration is loaded from environment variables.

        Raises:
            CredentialsNotFoundError: No credentials found for the auth method.
            CredentialsInvalidError: Credentials are malformed or rejected.
            ProjectConfigurationError: Missing project/location for Vertex AI.
        """
        if config is None:
            config = ProviderConfig()

        # Resolve configuration from environment if not explicitly set
        resolved_config = self._resolve_config(config)

        # Store resolved values
        self._use_vertex_ai = resolved_config.use_vertex_ai
        self._auth_method = resolved_config.auth_method
        self._project = resolved_config.project
        self._location = resolved_config.location

        # Validate configuration before attempting connection
        self._validate_config(resolved_config)

        # Create the client based on endpoint type
        self._client = self._create_client(resolved_config)

        # Verify connectivity with a lightweight API call
        self._verify_connectivity()

    def _resolve_config(self, config: ProviderConfig) -> ProviderConfig:
        """Resolve configuration by merging explicit config with environment.

        Explicit config values take precedence over environment variables.

        Args:
            config: Explicitly provided configuration.

        Returns:
            Fully resolved configuration.
        """
        # Resolve auth method
        auth_method = config.auth_method
        if auth_method == "auto":
            auth_method = resolve_auth_method()

        # Resolve use_vertex_ai
        # If api_key is provided or auth_method is api_key, default to AI Studio
        use_vertex_ai = config.use_vertex_ai
        if config.api_key or auth_method == "api_key":
            use_vertex_ai = False
        elif config.use_vertex_ai is True:  # Explicit True or default
            use_vertex_ai = resolve_use_vertex()

        # Resolve credentials
        api_key = config.api_key or resolve_api_key()
        credentials_path = config.credentials_path or resolve_credentials_path()

        # Resolve project/location (only needed for Vertex AI)
        project = config.project or resolve_project()
        location = config.location or resolve_location()

        # Resolve target service account (for impersonation)
        target_service_account = config.target_service_account or resolve_target_service_account()

        return ProviderConfig(
            project=project,
            location=location,
            api_key=api_key,
            credentials_path=credentials_path,
            use_vertex_ai=use_vertex_ai,
            auth_method=auth_method,
            target_service_account=target_service_account,
            credentials=config.credentials,
            extra=config.extra,
        )

    def _validate_config(self, config: ProviderConfig) -> None:
        """Validate configuration before creating client.

        Args:
            config: Resolved configuration to validate.

        Raises:
            CredentialsNotFoundError: Missing required credentials.
            ProjectConfigurationError: Missing project/location for Vertex AI.
            ImpersonationError: Missing target service account for impersonation.
        """
        if config.use_vertex_ai:
            # Vertex AI requires project and location
            if not config.project:
                raise ProjectConfigurationError(
                    project=config.project,
                    location=config.location,
                    reason="Project ID is required for Vertex AI",
                )
            if not config.location:
                raise ProjectConfigurationError(
                    project=config.project,
                    location=config.location,
                    reason="Location is required for Vertex AI",
                )

            # For service_account_file method, verify file exists
            if config.auth_method == "service_account_file":
                creds_path = config.credentials_path
                if not creds_path:
                    raise CredentialsNotFoundError(
                        auth_method=config.auth_method,
                        checked_locations=get_checked_credential_locations(config.auth_method),
                    )
                if not os.path.exists(creds_path):
                    raise CredentialsNotFoundError(
                        auth_method=config.auth_method,
                        checked_locations=[f"{creds_path} (file not found)"],
                        suggestion=f"Verify the file exists: {creds_path}",
                    )

            # For impersonation, verify target service account is set
            if config.auth_method == "impersonation":
                if not config.target_service_account:
                    raise ImpersonationError(
                        target_service_account=None,
                        reason="Target service account is required for impersonation",
                    )
        else:
            # AI Studio requires API key
            if not config.api_key:
                raise CredentialsNotFoundError(
                    auth_method="api_key",
                    checked_locations=get_checked_credential_locations("api_key"),
                )

    def _create_client(self, config: ProviderConfig) -> genai.Client:
        """Create the GenAI client based on configuration.

        Args:
            config: Resolved and validated configuration.

        Returns:
            Initialized genai.Client.

        Raises:
            CredentialsInvalidError: If credentials are rejected.
            ImpersonationError: If impersonation fails.
        """
        try:
            if config.use_vertex_ai:
                # Vertex AI mode - check for impersonation
                if config.auth_method == "impersonation":
                    credentials = self._create_impersonated_credentials(config)
                    return get_genai().Client(
                        vertexai=True,
                        project=config.project,
                        location=config.location,
                        credentials=credentials,
                    )
                else:
                    # Standard Vertex AI auth (ADC or service account file)
                    return get_genai().Client(
                        vertexai=True,
                        project=config.project,
                        location=config.location,
                    )
            else:
                # AI Studio mode with API key
                return get_genai().Client(
                    api_key=config.api_key,
                )
        except ImpersonationError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "api key" in error_msg or "invalid" in error_msg:
                raise CredentialsInvalidError(
                    auth_method=config.auth_method,
                    reason=str(e),
                    credentials_source="api_key" if config.api_key else "environment",
                ) from e
            raise

    def _create_impersonated_credentials(self, config: ProviderConfig):
        """Create impersonated credentials for service account impersonation.

        Args:
            config: Configuration with target_service_account set.

        Returns:
            Impersonated credentials object.

        Raises:
            ImpersonationError: If impersonation fails.
        """
        try:
            import google.auth
            from google.auth import impersonated_credentials

            # Get source credentials (ADC or from service account file)
            if config.credentials_path:
                # Use service account file as source
                from google.oauth2 import service_account
                source_credentials = service_account.Credentials.from_service_account_file(
                    config.credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            else:
                # Use ADC as source
                source_credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )

            # Create impersonated credentials
            target_credentials = impersonated_credentials.Credentials(
                source_credentials=source_credentials,
                target_principal=config.target_service_account,
                target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

            return target_credentials

        except Exception as e:
            error_msg = str(e).lower()

            # Try to get source principal for better error messages
            source_principal = None
            try:
                import google.auth
                creds, _ = google.auth.default()
                if hasattr(creds, 'service_account_email'):
                    source_principal = f"serviceAccount:{creds.service_account_email}"
                elif hasattr(creds, '_service_account_email'):
                    source_principal = f"serviceAccount:{creds._service_account_email}"
            except Exception as exc:
                logger.debug(f"Could not determine source principal: {exc}")

            if "permission" in error_msg or "403" in error_msg or "token creator" in error_msg:
                raise ImpersonationError(
                    target_service_account=config.target_service_account,
                    source_principal=source_principal,
                    reason="Source principal lacks Service Account Token Creator role",
                    original_error=str(e),
                ) from e
            else:
                raise ImpersonationError(
                    target_service_account=config.target_service_account,
                    source_principal=source_principal,
                    reason="Failed to create impersonated credentials",
                    original_error=str(e),
                ) from e

    def _verify_connectivity(self) -> None:
        """Verify connectivity by making a lightweight API call.

        Raises:
            CredentialsPermissionError: If credentials lack required permissions.
            CredentialsInvalidError: If credentials are rejected.
        """
        if not self._client:
            return

        try:
            # List models is a lightweight call that verifies auth works
            # Just fetch one to minimize overhead
            models = list(self._client.models.list())
            # We don't need to check the result, just that it didn't error
        except Exception as e:
            error_msg = str(e).lower()

            if "permission" in error_msg or "forbidden" in error_msg or "403" in error_msg:
                raise CredentialsPermissionError(
                    project=self._project,
                    original_error=str(e),
                ) from e
            elif "unauthorized" in error_msg or "401" in error_msg or "invalid" in error_msg:
                raise CredentialsInvalidError(
                    auth_method=self._auth_method,
                    reason=str(e),
                ) from e
            elif "not found" in error_msg or "404" in error_msg:
                raise ProjectConfigurationError(
                    project=self._project,
                    location=self._location,
                    reason=f"Project or API not found: {e}",
                ) from e
            # For other errors, let them propagate
            raise

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None
    ) -> bool:
        """Verify that authentication is configured.

        For Google GenAI, this checks for API key (AI Studio) or GCP credentials
        (Vertex AI). Interactive login is not supported.

        Args:
            allow_interactive: Ignored (no interactive login available).
            on_message: Optional callback for status messages.

        Returns:
            True if authentication is configured.
            False if no credentials found.

        Raises:
            CredentialsNotFoundError: If allow_interactive=False and no credentials found.
        """
        # Check for API key (AI Studio mode)
        api_key = resolve_api_key()
        if api_key:
            if on_message:
                on_message("Found Google API key")
            return True

        # Check for service account file
        creds_path = resolve_credentials_path()
        if creds_path:
            if on_message:
                on_message(f"Found service account credentials: {creds_path}")
            return True

        # Check for ADC (Application Default Credentials)
        # Try to import and check if ADC is available
        try:
            import google.auth
            credentials, project = google.auth.default()
            if credentials:
                if on_message:
                    on_message("Found Application Default Credentials")
                return True
        except Exception:
            pass

        # No credentials found
        if not allow_interactive:
            raise CredentialsNotFoundError(
                checked_locations=get_checked_credential_locations()
            )

        if on_message:
            on_message("No Google credentials found. Please configure API key or GCP credentials.")
        return False

    def shutdown(self) -> None:
        """Clean up resources."""
        self._client = None
        self._model_name = None

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used."""
        method = self._auth_method
        if method == "api_key":
            return "API key"
        elif method == "adc":
            return "ADC"
        elif method == "service_account_file":
            return "service account"
        elif method == "impersonation":
            return "impersonation"
        return method or ""

    # ==================== Connection ====================

    def connect(self, model: str) -> None:
        """Set the model to use and verify it responds.

        Args:
            model: Model name (e.g., 'gemini-2.5-flash').

        Raises:
            RuntimeError: Model doesn't exist or cannot be used.
        """
        self._model_name = model

        # Verify model can actually respond
        self._verify_model_responds()

    def _verify_model_responds(self) -> None:
        """Verify the model can actually respond.

        Sends a minimal test message to catch issues like:
        - Invalid model name
        - Model access restrictions
        - Quota/billing issues
        """
        if not self._client or not self._model_name:
            return  # Will fail later with clear error

        try:
            # Use generate_content for a one-off test (no chat session needed)
            self._client.models.generate_content(
                model=self._model_name,
                contents="hi",
                config={"max_output_tokens": 1},
            )
        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str:
                raise RuntimeError(
                    f"Model '{self._model_name}' not found or not accessible.\n"
                    f"Original error: {e}"
                ) from e
            elif "permission" in error_str or "403" in error_str:
                raise RuntimeError(
                    f"Permission denied for model '{self._model_name}'.\n"
                    f"Check your project configuration and API access.\n"
                    f"Original error: {e}"
                ) from e
            elif "quota" in error_str or "429" in error_str:
                raise RuntimeError(
                    f"Quota exceeded for model '{self._model_name}'.\n"
                    f"Original error: {e}"
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to verify model '{self._model_name}'.\n"
                    f"Original error: {e}"
                ) from e

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected and ready."""
        return self._client is not None and self._model_name is not None

    @property
    def model_name(self) -> Optional[str]:
        """Get the current model name."""
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List available models from the API with caching.

        Fetches models from the API and caches them for _MODELS_CACHE_TTL seconds.

        Args:
            prefix: Optional filter prefix (e.g., 'gemini').

        Returns:
            List of model names.
        """
        if not self._client:
            return []

        # Check cache validity
        now = time.time()
        if self._models_cache:
            cache_time, cached_models = self._models_cache
            if now - cache_time < self._MODELS_CACHE_TTL:
                # Cache is valid
                if prefix:
                    return sorted([m for m in cached_models if m.startswith(prefix)])
                return sorted(cached_models)

        # Fetch from API
        models = []
        try:
            for model in self._client.models.list():
                # Extract model name, stripping 'models/' prefix if present
                name = model.name
                if name.startswith('models/'):
                    name = name[7:]
                models.append(name)

            # Update cache
            self._models_cache = (now, models)
        except Exception as exc:
            logger.warning(f"Failed to list models from API: {exc}")
            # If API fails, return empty list (no fallback to static)
            return []

        if prefix:
            models = [m for m in models if m.startswith(prefix)]
        return sorted(models)

    # ==================== (Legacy session/messaging methods removed — Phase 4) ====

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Count tokens for the given content.

        Args:
            content: Text to count tokens for.

        Returns:
            Token count.
        """
        if not self._client or not self._model_name:
            return 0

        try:
            result = self._client.models.count_tokens(
                model=self._model_name,
                contents=content
            )
            return result.total_tokens
        except Exception as exc:
            logger.debug(f"Failed to count tokens, using estimate: {exc}")
            # Fallback: rough estimate (4 chars per token)
            return len(content) // 4

    def get_context_limit(self) -> int:
        """Get the context window size for the current model.

        Returns:
            Maximum tokens the model can handle.
        """
        if not self._model_name:
            return DEFAULT_CONTEXT_LIMIT

        # Try exact match first
        if self._model_name in MODEL_CONTEXT_LIMITS:
            return MODEL_CONTEXT_LIMITS[self._model_name]

        # Try prefix match
        for model_prefix, limit in MODEL_CONTEXT_LIMITS.items():
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

        Gemini models support structured output via response_schema.

        Returns:
            True - Gemini supports structured output.
        """
        return True

    def supports_streaming(self) -> bool:
        """Check if streaming is supported.

        Returns:
            True - Google GenAI supports streaming.
        """
        return True

    def supports_stop(self) -> bool:
        """Check if mid-turn cancellation (stop) is supported.

        Returns:
            True - Google GenAI supports stop via streaming cancellation.
        """
        return True

    def supports_thinking(self) -> bool:
        """Check if the current model supports thinking mode.

        Gemini 2.0+ models support thinking mode, but this is currently
        not implemented in this provider.

        Returns:
            False - Thinking mode not yet implemented for Google GenAI.
        """
        # TODO: Implement thinking mode when Google GenAI SDK supports it
        return False

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """Set the thinking/reasoning mode configuration.

        Currently a no-op as thinking mode is not yet implemented
        for Google GenAI provider.

        Args:
            config: ThinkingConfig with enabled flag and budget.
        """
        # TODO: Implement when Google GenAI SDK supports thinking mode
        pass

    # ==================== Stateless Completion ====================

    def complete(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        tools: Optional[List[ToolSchema]] = None,
        *,
        response_schema: Optional[Dict[str, Any]] = None,
        cancel_token: Optional[CancelToken] = None,
        on_chunk: Optional[StreamingCallback] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None,
    ) -> TurnResult:
        """Stateless completion: convert messages to SDK format, call API, return response.

        This is the sole entry point for model inference.  The caller
        (``JaatoSession``) is responsible for maintaining the message list
        and passing it in full each call.  The provider does not maintain
        any internal conversation state.

        Uses ``client.models.generate_content()`` (batch) or
        ``client.models.generate_content_stream()`` (streaming) directly.

        Returns ``TurnResult.from_provider_response(r)`` on success and
        **raises** transient errors for ``with_retry``.

        Args:
            messages: Full conversation history in provider-agnostic Message
                format. Must already include the latest user message or tool
                results — the provider does not append anything.
            system_instruction: System prompt text.
            tools: Available tool schemas.
            response_schema: Optional JSON Schema for structured output.
            cancel_token: Optional cancellation signal.
            on_chunk: If provided, enables streaming mode.
            on_usage_update: Real-time token usage callback (streaming).
            on_function_call: Callback when function call detected mid-stream.
            on_thinking: Callback for extended thinking content.

        Returns:
            A ``TurnResult`` classifying the outcome.

        Raises:
            RuntimeError: If provider is not initialized/connected.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("Provider not connected. Call initialize() and connect() first.")

        # Convert messages to SDK format (bypasses self._chat)
        sdk_contents = history_to_sdk(list(messages))

        # Build config from explicit parameters (NOT instance state)
        # Check cache plugin first (parameterized with explicit args)
        config = None
        if self._cache_plugin:
            cache_result = self._cache_plugin.prepare_request(
                system=system_instruction,
                tools=tools or [],
                messages=[],
            )
            cached_content_name = cache_result.get("cached_content")
            if cached_content_name:
                self._trace(f"COMPLETE_CACHE using CachedContent: {cached_content_name}")
                config_kwargs: Dict[str, Any] = {
                    "cached_content": cached_content_name,
                    "automatic_function_calling": get_types().AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                }
                if response_schema:
                    config_kwargs["response_mime_type"] = "application/json"
                    config_kwargs["response_schema"] = response_schema
                config = get_types().GenerateContentConfig(**config_kwargs)

        if config is None:
            # No cache — build config with system instruction and tools
            sdk_tool = tool_schemas_to_sdk_tool(tools) if tools else None
            config_kwargs = {
                "system_instruction": system_instruction,
                "tools": [sdk_tool] if sdk_tool else None,
                "automatic_function_calling": get_types().AutomaticFunctionCallingConfig(
                    disable=True
                ),
            }
            if response_schema:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = response_schema
            config = get_types().GenerateContentConfig(**config_kwargs)

        if on_chunk:
            # Streaming mode — use generate_content_stream (bypasses chat session)
            provider_response = self._complete_streaming(
                sdk_contents, config,
                on_chunk=on_chunk,
                cancel_token=cancel_token,
                response_schema=response_schema,
                on_usage_update=on_usage_update,
                on_function_call=on_function_call,
                on_thinking=on_thinking,
            )
            return TurnResult.from_provider_response(provider_response)
        else:
            # Batch mode
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=sdk_contents,
                config=config,
            )
            provider_response = response_from_sdk(response)
            self._last_usage = provider_response.usage

            # Handle structured output
            if response_schema:
                text = provider_response.get_text()
                if text:
                    try:
                        provider_response.structured_output = json.loads(text)
                    except json.JSONDecodeError:
                        pass

            return TurnResult.from_provider_response(provider_response)

    def _complete_streaming(
        self,
        sdk_contents: Any,
        config: Any,
        *,
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None,
    ) -> ProviderResponse:
        """Process a streaming generate_content call for complete().

        Uses ``client.models.generate_content_stream()`` directly, operating
        on an explicit contents list passed in from ``complete()``.

        Args:
            sdk_contents: Contents in SDK format (from ``history_to_sdk``).
            config: GenerateContentConfig with system instruction, tools, etc.
            on_chunk: Callback for each text chunk.
            cancel_token: Optional cancellation signal.
            response_schema: Optional JSON Schema for structured output parsing.
            on_usage_update: Real-time token usage callback.
            on_function_call: Callback when function call detected mid-stream.
            on_thinking: Callback for extended thinking content.

        Returns:
            ProviderResponse with accumulated parts, usage, and finish reason.
        """
        accumulated_text: List[str] = []
        parts: List[Part] = []
        finish_reason = FinishReason.UNKNOWN
        function_calls: List = []
        usage = TokenUsage()
        was_cancelled = False

        def flush_text_block():
            nonlocal accumulated_text
            if accumulated_text:
                parts.append(Part.from_text("".join(accumulated_text)))
                accumulated_text = []

        try:
            chunk_count = 0
            for chunk in self._client.models.generate_content_stream(
                model=self._model_name,
                contents=sdk_contents,
                config=config,
            ):
                if cancel_token and cancel_token.is_cancelled:
                    self._trace(f"COMPLETE_STREAM_CANCELLED after {chunk_count} chunks")
                    was_cancelled = True
                    finish_reason = FinishReason.CANCELLED
                    break

                # Extract text
                chunk_text = extract_text_from_chunk(chunk)
                if chunk_text:
                    chunk_count += 1
                    accumulated_text.append(chunk_text)
                    on_chunk(chunk_text)

                # Extract function calls and special parts
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                from .converters import function_call_from_sdk
                                fc = function_call_from_sdk(part.function_call)
                                if fc and fc not in function_calls:
                                    flush_text_block()
                                    if on_function_call:
                                        on_function_call(fc)
                                    parts.append(Part.from_function_call(fc))
                                    function_calls.append(fc)
                            elif hasattr(part, 'thought') and part.thought:
                                flush_text_block()
                                parts.append(Part(thought=part.thought))
                            elif hasattr(part, 'executable_code') and part.executable_code:
                                code = part.executable_code
                                code_str = getattr(code, 'code', str(code)) if code else ""
                                flush_text_block()
                                parts.append(Part(executable_code=code_str))
                            elif hasattr(part, 'code_execution_result') and part.code_execution_result:
                                result = part.code_execution_result
                                output = getattr(result, 'output', str(result)) if result else ""
                                flush_text_block()
                                parts.append(Part(code_execution_result=output))

                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                        from .converters import finish_reason_from_sdk
                        finish_reason = finish_reason_from_sdk(candidate.finish_reason)

                # Extract usage
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    metadata = chunk.usage_metadata
                    usage = TokenUsage(
                        prompt_tokens=getattr(metadata, 'prompt_token_count', 0) or 0,
                        output_tokens=getattr(metadata, 'candidates_token_count', 0) or 0,
                        total_tokens=getattr(metadata, 'total_token_count', 0) or 0,
                    )
                    cached_tokens = getattr(metadata, 'cached_content_token_count', None)
                    if cached_tokens is not None and cached_tokens > 0:
                        usage.cache_read_tokens = cached_tokens
                    if on_usage_update and usage.total_tokens > 0:
                        try:
                            on_usage_update(usage)
                        except Exception:
                            pass

        except Exception as e:
            if cancel_token and cancel_token.is_cancelled:
                was_cancelled = True
                finish_reason = FinishReason.CANCELLED
            else:
                raise

        flush_text_block()

        if function_calls and not was_cancelled:
            finish_reason = FinishReason.TOOL_USE

        provider_response = ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,
        )

        # Update per-call accounting (NOT conversation state)
        self._last_usage = usage

        # Handle structured output
        if response_schema and not was_cancelled:
            text = provider_response.get_text()
            if text:
                try:
                    provider_response.structured_output = json.loads(text)
                except json.JSONDecodeError:
                    pass

        return provider_response

    # ==================== (Legacy streaming methods removed — Phase 4) ====

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

    # ==================== Cache Plugin Delegation ====================

    def set_cache_plugin(self, plugin: Any) -> None:
        """Attach a cache control plugin for explicit CachedContent management.

        When set, the provider delegates cache creation/reuse decisions
        to this plugin.  The plugin receives the ``genai.Client`` so it
        can call ``client.caches.create()`` / ``client.caches.delete()``.

        Args:
            plugin: A ``GoogleGenAICachePlugin`` instance (duck-typed).
        """
        self._cache_plugin = plugin
        if hasattr(plugin, "set_client") and self._client:
            plugin.set_client(self._client)

    # (_get_cached_content_config removed — Phase 4: complete() has its own
    # inline cache logic using explicit parameters instead of instance state.)

    # ==================== Error Classification for Retry ====================

    def classify_error(self, exc: Exception) -> Optional[Dict[str, bool]]:
        """Classify an exception for retry purposes.

        Google GenAI SDK (python-genai) uses google.genai.errors.ClientError
        for HTTP 4xx errors including rate limits (429).

        Args:
            exc: The exception to classify.

        Returns:
            Classification dict or None to use fallback.
        """
        # Try to import google.genai.errors for precise classification
        try:
            from google.genai import errors as genai_errors
            if isinstance(exc, genai_errors.ClientError):
                lower = str(exc).lower()
                # Rate limit / quota errors
                if any(p in lower for p in ["429", "resource exhausted", "resource_exhausted", "rate limit", "quota"]):
                    return {"transient": True, "rate_limit": True, "infra": False}
                # Server errors that may be retried
                if any(p in lower for p in ["503", "500", "service unavailable", "internal"]):
                    return {"transient": True, "rate_limit": False, "infra": True}
                # Other 4xx errors (auth, bad request) - don't retry
                return {"transient": False, "rate_limit": False, "infra": False}
        except ImportError:
            pass

        # Try google.api_core exceptions (older SDK / fallback)
        try:
            from google.api_core import exceptions as google_exceptions
            if isinstance(exc, (google_exceptions.TooManyRequests, google_exceptions.ResourceExhausted)):
                return {"transient": True, "rate_limit": True, "infra": False}
            if isinstance(exc, (google_exceptions.ServiceUnavailable, google_exceptions.InternalServerError,
                               google_exceptions.DeadlineExceeded, google_exceptions.Aborted)):
                return {"transient": True, "rate_limit": False, "infra": True}
        except ImportError:
            pass

        # Fall back to global classification
        return None

    def get_retry_after(self, exc: Exception) -> Optional[float]:
        """Extract retry-after hint from an exception.

        Google's APIs may include retry hints in error messages or headers.

        Args:
            exc: The exception to extract retry-after from.

        Returns:
            Suggested delay in seconds, or None if not available.
        """
        # Check for retry_after attribute (some wrappers add this)
        if hasattr(exc, 'retry_after') and exc.retry_after:
            return float(exc.retry_after)

        # Check for HTTP response with Retry-After header
        if hasattr(exc, 'response') and hasattr(exc.response, 'headers'):
            retry_after = exc.response.headers.get('Retry-After')
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass

        return None


def create_provider() -> GoogleGenAIProvider:
    """Factory function for plugin discovery."""
    return GoogleGenAIProvider()
