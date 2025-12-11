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

import json
import os
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from ..base import GoogleAuthMethod, ModelProviderPlugin, ProviderConfig
from ..types import (
    Message,
    ProviderResponse,
    Role,
    ToolResult,
    ToolSchema,
    TokenUsage,
    Part,
)
from .converters import (
    history_from_sdk,
    history_to_sdk,
    message_from_sdk,
    response_from_sdk,
    tool_results_to_sdk_parts,
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
    """Google GenAI / Vertex AI model provider.

    This provider supports:
    - Dual endpoints: Google AI Studio (API key) and Vertex AI (GCP auth)
    - Multiple auth methods: API key, ADC, service account file
    - Gemini model family (1.5, 2.0, 2.5)
    - Multi-turn chat with SDK-managed history
    - Function calling with manual control
    - Token counting and context management

    Usage (Vertex AI - organization):
        provider = GoogleGenAIProvider()
        provider.initialize(ProviderConfig(
            project='my-project',
            location='us-central1',
            use_vertex_ai=True,
            auth_method='auto'  # Uses ADC or GOOGLE_APPLICATION_CREDENTIALS
        ))
        provider.connect('gemini-2.5-flash')
        response = provider.send_message("Hello!")

    Usage (AI Studio - personal):
        provider = GoogleGenAIProvider()
        provider.initialize(ProviderConfig(
            api_key='your-api-key',
            use_vertex_ai=False,
            auth_method='api_key'
        ))
        provider.connect('gemini-2.5-flash')
        response = provider.send_message("Hello!")

    Environment variables (auto-detected if config not provided):
        GOOGLE_GENAI_API_KEY: API key for AI Studio
        GOOGLE_APPLICATION_CREDENTIALS: Service account file for Vertex AI
        JAATO_GOOGLE_PROJECT / GOOGLE_CLOUD_PROJECT: GCP project ID
        JAATO_GOOGLE_LOCATION: GCP region (e.g., us-central1)
        JAATO_GOOGLE_AUTH_METHOD: Force specific auth method
        JAATO_GOOGLE_USE_VERTEX: Force Vertex AI (true) or AI Studio (false)
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        self._client: Optional[genai.Client] = None
        self._model_name: Optional[str] = None
        self._project: Optional[str] = None
        self._location: Optional[str] = None
        self._chat = None  # genai Chat object

        # Authentication state
        self._use_vertex_ai: bool = True
        self._auth_method: GoogleAuthMethod = "auto"

        # Current session configuration
        self._system_instruction: Optional[str] = None
        self._tools: Optional[List[ToolSchema]] = None
        self._last_usage: TokenUsage = TokenUsage()

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
                    return genai.Client(
                        vertexai=True,
                        project=config.project,
                        location=config.location,
                        credentials=credentials,
                    )
                else:
                    # Standard Vertex AI auth (ADC or service account file)
                    return genai.Client(
                        vertexai=True,
                        project=config.project,
                        location=config.location,
                    )
            else:
                # AI Studio mode with API key
                return genai.Client(
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
            except Exception:
                pass

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

    def shutdown(self) -> None:
        """Clean up resources."""
        self._chat = None
        self._client = None
        self._model_name = None

    # ==================== Connection ====================

    def connect(self, model: str) -> None:
        """Set the model to use.

        Args:
            model: Model name (e.g., 'gemini-2.5-flash').
        """
        self._model_name = model

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected and ready."""
        return self._client is not None and self._model_name is not None

    @property
    def model_name(self) -> Optional[str]:
        """Get the current model name."""
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List available models.

        Args:
            prefix: Optional filter prefix (e.g., 'gemini').

        Returns:
            List of model names.
        """
        if not self._client:
            return []

        models = []
        for model in self._client.models.list():
            if prefix and not model.name.startswith(prefix):
                continue
            models.append(model.name)

        return models

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
        if not self._client or not self._model_name:
            raise RuntimeError("Provider not initialized. Call initialize() and connect() first.")

        self._system_instruction = system_instruction
        self._tools = tools

        # Convert tools to SDK format
        sdk_tool = tool_schemas_to_sdk_tool(tools) if tools else None

        # Build config
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[sdk_tool] if sdk_tool else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        # Convert history to SDK format
        sdk_history = history_to_sdk(history) if history else None

        # Create the chat session
        self._chat = self._client.chats.create(
            model=self._model_name,
            config=config,
            history=sdk_history
        )

    def get_history(self) -> List[Message]:
        """Get the current conversation history.

        Returns:
            List of messages in internal format.
        """
        if not self._chat:
            return []

        sdk_history = list(self._chat.get_history())
        return history_from_sdk(sdk_history)

    # ==================== Messaging ====================

    def generate(self, prompt: str) -> ProviderResponse:
        """Simple one-shot generation without session context.

        Use this for basic prompts that don't need conversation history
        or function calling.

        Args:
            prompt: The prompt text.

        Returns:
            ProviderResponse with the model's response.
        """
        if not self._client or not self._model_name:
            raise RuntimeError("Provider not connected. Call connect() first.")

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt
        )
        provider_response = response_from_sdk(response)
        self._last_usage = provider_response.usage
        return provider_response

    def send_message(
        self,
        message: str,
        response_schema: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Send a user message and get a response.

        Args:
            message: The user's message text.
            response_schema: Optional JSON Schema to constrain the response.
                When provided, the model returns JSON matching this schema.

        Returns:
            ProviderResponse with text and/or function calls.
        """
        if not self._chat:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Build config override for structured output
        config = None
        if response_schema:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )

        response = self._chat.send_message(message, config=config)
        provider_response = response_from_sdk(response)
        self._last_usage = provider_response.usage

        # Parse structured output if schema was requested
        if response_schema and provider_response.text:
            try:
                provider_response.structured_output = json.loads(provider_response.text)
            except json.JSONDecodeError:
                # Model returned invalid JSON despite schema constraint
                pass

        return provider_response

    def send_message_with_parts(
        self,
        parts: List[Part],
        response_schema: Optional[Dict[str, Any]] = None
    ) -> ProviderResponse:
        """Send a message with multiple parts (text, images, etc.).

        Use this for multimodal input where the user message contains
        more than just text.

        Args:
            parts: List of Part objects forming the message.
            response_schema: Optional JSON Schema to constrain the response.

        Returns:
            ProviderResponse with text and/or function calls.
        """
        if not self._chat:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Import converter here to avoid circular imports
        from .converters import part_to_sdk

        # Convert internal Parts to SDK Parts
        sdk_parts = [part_to_sdk(p) for p in parts]

        # Create user Content with the parts
        user_content = types.Content(role='user', parts=sdk_parts)

        # Build config override for structured output
        config = None
        if response_schema:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )

        response = self._chat.send_message(user_content, config=config)
        provider_response = response_from_sdk(response)
        self._last_usage = provider_response.usage

        # Parse structured output if schema was requested
        if response_schema and provider_response.text:
            try:
                provider_response.structured_output = json.loads(provider_response.text)
            except json.JSONDecodeError:
                pass

        return provider_response

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
        if not self._chat:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Convert results to SDK parts
        sdk_parts = tool_results_to_sdk_parts(results)

        # Build config override for structured output
        config = None
        if response_schema:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )

        # Send to model
        response = self._chat.send_message(sdk_parts, config=config)
        provider_response = response_from_sdk(response)
        self._last_usage = provider_response.usage

        # Parse structured output if schema was requested
        if response_schema and provider_response.text:
            try:
                provider_response.structured_output = json.loads(provider_response.text)
            except json.JSONDecodeError:
                pass

        return provider_response

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
        except Exception:
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


def create_provider() -> GoogleGenAIProvider:
    """Factory function for plugin discovery."""
    return GoogleGenAIProvider()
