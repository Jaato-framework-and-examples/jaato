"""Antigravity provider implementation.

This provider enables access to AI models (Gemini 3, Claude) through
Google's Antigravity backend using Google OAuth authentication.

Features:
- Google OAuth authentication with PKCE
- Multi-account rotation for load balancing
- Automatic token refresh
- Streaming support with SSE
- Extended thinking support for compatible models
- Dual quota systems (Antigravity and Gemini CLI)

Reference: https://github.com/NoeFabris/opencode-antigravity-auth
"""

import json
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from ..base import (
    FunctionCallDetectedCallback,
    ModelProviderPlugin,
    ProviderConfig,
    StreamingCallback,
    ThinkingCallback,
    UsageUpdateCallback,
)
from ..types import (
    CancelledException,
    CancelToken,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    ThinkingConfig,
    ToolResult,
    ToolSchema,
    TokenUsage,
)
from .constants import (
    ANTIGRAVITY_API_CLIENT,
    ANTIGRAVITY_CLIENT_METADATA,
    ANTIGRAVITY_ENDPOINTS,
    ANTIGRAVITY_MODELS,
    ANTIGRAVITY_PRIMARY_ENDPOINT,
    ANTIGRAVITY_USER_AGENT,
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_OUTPUT_LIMIT,
    GEMINI_CLI_API_CLIENT,
    GEMINI_CLI_CLIENT_METADATA,
    GEMINI_CLI_ENDPOINT,
    GEMINI_CLI_MODELS,
    GEMINI_CLI_USER_AGENT,
    PROVIDER_ID,
)
from .converters import (
    build_generate_request,
    build_generation_config,
    deserialize_history,
    extract_finish_reason_from_stream_chunk,
    extract_function_calls_from_stream_chunk,
    extract_text_from_stream_chunk,
    extract_thinking_from_stream_chunk,
    extract_usage_from_stream_chunk,
    messages_to_api,
    parse_sse_event,
    response_from_api,
    serialize_history,
    tool_schemas_to_api,
)
from .env import (
    get_checked_credential_locations,
    resolve_auto_rotate,
    resolve_endpoint,
    resolve_project_id,
    resolve_quota_type,
    resolve_retry_empty,
    resolve_session_recovery,
    resolve_thinking_budget,
    resolve_thinking_level,
)
from .errors import (
    APIError,
    AntigravityProviderError,
    AuthenticationError,
    ContextLimitError,
    EndpointError,
    ModelNotFoundError,
    QuotaExceededError,
    RateLimitError,
    StreamingError,
    TokenExpiredError,
    TokenRefreshError,
    ToolResultMissingError,
)
from .oauth import (
    Account,
    AccountManager,
    get_valid_access_token,
    load_accounts,
    login as oauth_login,
    refresh_tokens,
    save_accounts,
)


class AntigravityProvider:
    """Antigravity provider for Google's IDE AI backend.

    This provider supports:
    - Multiple model families (Gemini 3, Claude)
    - Google OAuth authentication
    - Multi-account rotation for quota management
    - Streaming responses with SSE
    - Extended thinking for compatible models

    Usage:
        provider = AntigravityProvider()
        provider.initialize()  # Uses stored OAuth tokens
        provider.connect('antigravity-gemini-3-flash')
        response = provider.send_message("Hello!")

    To authenticate:
        from shared.plugins.model_provider.antigravity import oauth_login
        oauth_login()  # Opens browser for Google auth
    """

    def __init__(self):
        """Initialize the provider (not yet connected)."""
        self._account_manager: Optional[AccountManager] = None
        self._current_account: Optional[Account] = None
        self._model_name: Optional[str] = None
        self._api_model: Optional[str] = None

        # Configuration
        self._endpoint: str = ANTIGRAVITY_PRIMARY_ENDPOINT
        self._quota_type: str = "antigravity"
        self._project_id: Optional[str] = None
        self._auto_rotate: bool = True
        self._retry_empty: bool = True
        self._session_recovery: bool = True

        # Thinking configuration
        self._thinking_level: Optional[str] = None  # For Gemini 3
        self._thinking_budget: int = 8192  # For Claude thinking models

        # Session state
        self._system_instruction: Optional[str] = None
        self._tools: Optional[List[ToolSchema]] = None
        self._history: List[Message] = []
        self._last_usage: TokenUsage = TokenUsage()

        # Agent context for tracing
        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None
        self._agent_id: str = "main"

    @property
    def name(self) -> str:
        """Provider identifier."""
        return PROVIDER_ID

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider with OAuth tokens.

        Args:
            config: Configuration (optional). Supports:
                - extra['endpoint']: Override API endpoint
                - extra['quota_type']: 'antigravity' or 'gemini-cli'
                - extra['project_id']: Override project ID
                - extra['auto_rotate']: Enable account rotation (default: True)
                - extra['thinking_level']: Gemini 3 thinking level
                - extra['thinking_budget']: Claude thinking budget

        Raises:
            AuthenticationError: No OAuth tokens found.
        """
        if config is None:
            config = ProviderConfig()

        # Load account manager
        self._account_manager = load_accounts()

        # Get active account
        self._current_account = self._account_manager.get_active_account()

        if not self._current_account:
            raise AuthenticationError(
                message="No Antigravity accounts found. Run 'antigravity-auth login' to authenticate.",
                checked_locations=get_checked_credential_locations(),
            )

        # Parse configuration
        self._endpoint = config.extra.get("endpoint") or resolve_endpoint() or ANTIGRAVITY_PRIMARY_ENDPOINT
        self._quota_type = config.extra.get("quota_type") or resolve_quota_type()
        self._project_id = config.extra.get("project_id") or resolve_project_id() or self._current_account.project_id
        self._auto_rotate = config.extra.get("auto_rotate", resolve_auto_rotate())
        self._retry_empty = config.extra.get("retry_empty", resolve_retry_empty())
        self._session_recovery = config.extra.get("session_recovery", resolve_session_recovery())
        self._thinking_level = config.extra.get("thinking_level") or resolve_thinking_level()
        self._thinking_budget = config.extra.get("thinking_budget", resolve_thinking_budget())

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Verify authentication and optionally trigger interactive login.

        Args:
            allow_interactive: If True, allow interactive OAuth flow.
            on_message: Callback for status messages.

        Returns:
            True if authenticated.
        """
        # Check for existing tokens
        result = get_valid_access_token()
        if result:
            return True

        # No tokens - attempt interactive login if allowed
        if allow_interactive:
            try:
                oauth_login(on_message=on_message)
                return True
            except Exception as e:
                if on_message:
                    on_message(f"Authentication failed: {e}")
                return False

        return False

    @staticmethod
    def login(on_message: Optional[Callable[[str], None]] = None) -> None:
        """Run interactive OAuth login.

        Opens browser for Google authentication.

        Args:
            on_message: Callback for status messages.
        """
        oauth_login(on_message=on_message)

    def shutdown(self) -> None:
        """Clean up resources."""
        # Save any account state changes
        if self._account_manager:
            save_accounts(self._account_manager)

    # ==================== Connection ====================

    def connect(self, model: str) -> None:
        """Connect to a specific model.

        Args:
            model: Model ID (e.g., 'antigravity-gemini-3-flash', 'gemini-2.5-flash').

        Raises:
            ModelNotFoundError: Model not found.
        """
        # Determine model info
        if model in ANTIGRAVITY_MODELS:
            model_info = ANTIGRAVITY_MODELS[model]
            self._quota_type = "antigravity"
        elif model in GEMINI_CLI_MODELS:
            model_info = GEMINI_CLI_MODELS[model]
            self._quota_type = "gemini-cli"
        else:
            raise ModelNotFoundError(model)

        self._model_name = model
        self._api_model = model_info.get("api_model", model)

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected and ready."""
        return (
            self._current_account is not None
            and self._model_name is not None
        )

    @property
    def model_name(self) -> Optional[str]:
        """Get the current model name."""
        return self._model_name

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List available models.

        Args:
            prefix: Optional filter prefix.

        Returns:
            List of model IDs.
        """
        models = list(ANTIGRAVITY_MODELS.keys()) + list(GEMINI_CLI_MODELS.keys())

        if prefix:
            models = [m for m in models if m.startswith(prefix)]

        return sorted(models)

    # ==================== Session Management ====================

    def create_session(
        self,
        system_instruction: Optional[str] = None,
        tools: Optional[List[ToolSchema]] = None,
        history: Optional[List[Message]] = None,
    ) -> None:
        """Create or reset the chat session.

        Args:
            system_instruction: System prompt for the model.
            tools: List of available tools.
            history: Previous conversation history to restore.
        """
        if not self.is_connected:
            raise RuntimeError("Provider not connected. Call initialize() and connect() first.")

        self._system_instruction = system_instruction
        self._tools = tools
        self._history = list(history) if history else []

    def get_history(self) -> List[Message]:
        """Get the current conversation history.

        Returns:
            List of messages in internal format.
        """
        return list(self._history)

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Estimate token count for content.

        Note: This is an approximation. The API doesn't provide
        a dedicated token counting endpoint.

        Args:
            content: Text content to count.

        Returns:
            Estimated token count.
        """
        # Rough approximation: ~4 characters per token
        return len(content) // 4

    def get_context_limit(self) -> int:
        """Get the context window limit for the current model.

        Returns:
            Maximum context tokens.
        """
        if self._model_name:
            if self._model_name in ANTIGRAVITY_MODELS:
                return ANTIGRAVITY_MODELS[self._model_name].get("context_limit", DEFAULT_CONTEXT_LIMIT)
            if self._model_name in GEMINI_CLI_MODELS:
                return GEMINI_CLI_MODELS[self._model_name].get("context_limit", DEFAULT_CONTEXT_LIMIT)
        return DEFAULT_CONTEXT_LIMIT

    def get_token_usage(self) -> TokenUsage:
        """Get token usage from the last response.

        Returns:
            TokenUsage with prompt/output/total counts.
        """
        return self._last_usage

    # ==================== Capabilities ====================

    def supports_streaming(self) -> bool:
        """Check if streaming is supported.

        Returns:
            True (Antigravity supports streaming via SSE).
        """
        return True

    def supports_stop(self) -> bool:
        """Check if mid-turn cancellation is supported.

        Returns:
            True (can cancel streaming requests).
        """
        return True

    def supports_structured_output(self) -> bool:
        """Check if structured output is supported.

        Returns:
            True for Gemini models, False for Claude.
        """
        if self._model_name:
            return "gemini" in self._model_name.lower()
        return False

    def supports_thinking(self) -> bool:
        """Check if extended thinking is supported.

        Returns:
            True for Gemini 3 and Claude thinking models.
        """
        if not self._model_name:
            return False
        if "thinking" in self._model_name:
            return True
        if "gemini-3" in self._model_name:
            return True
        return False

    def set_thinking_config(self, config: ThinkingConfig) -> None:
        """Set thinking configuration dynamically.

        Updates the thinking settings for subsequent API calls.
        For Gemini 3, maps budget to thinking level:
        - budget <= 0: disabled
        - budget <= 8192: low
        - budget <= 16384: medium
        - budget > 16384: high

        For Claude thinking models, uses budget directly.

        Args:
            config: ThinkingConfig with enabled and budget settings.
        """
        if not config.enabled:
            self._thinking_level = None
            self._thinking_budget = 0
            return

        # Map budget to Gemini thinking level
        if self._model_name and "gemini-3" in self._model_name:
            if config.budget <= 8192:
                self._thinking_level = "low"
            elif config.budget <= 16384:
                self._thinking_level = "medium"
            else:
                self._thinking_level = "high"
        else:
            # Claude thinking models use budget directly
            self._thinking_budget = config.budget

    # ==================== Agent Context ====================

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main",
    ) -> None:
        """Set agent context for trace identification.

        Args:
            agent_type: Type of agent ("main" or "subagent").
            agent_name: Optional name for the agent.
            agent_id: Unique identifier for the agent instance.
        """
        self._agent_type = agent_type
        self._agent_name = agent_name
        self._agent_id = agent_id

    # ==================== Serialization ====================

    def serialize_history(self, history: List[Message]) -> str:
        """Serialize history to JSON string.

        Args:
            history: List of messages.

        Returns:
            JSON string.
        """
        return serialize_history(history)

    def deserialize_history(self, data: str) -> List[Message]:
        """Deserialize history from JSON string.

        Args:
            data: JSON string.

        Returns:
            List of messages.
        """
        return deserialize_history(data)

    # ==================== API Communication ====================

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers for the current quota type."""
        if not self._current_account:
            raise AuthenticationError("No active account")

        headers = {
            "Authorization": f"Bearer {self._current_account.tokens.access_token}",
            "Content-Type": "application/json",
        }

        if self._quota_type == "antigravity":
            headers["User-Agent"] = ANTIGRAVITY_USER_AGENT
            headers["X-Goog-Api-Client"] = ANTIGRAVITY_API_CLIENT
            headers["Client-Metadata"] = ANTIGRAVITY_CLIENT_METADATA
        else:
            headers["User-Agent"] = GEMINI_CLI_USER_AGENT
            headers["X-Goog-Api-Client"] = GEMINI_CLI_API_CLIENT
            headers["Client-Metadata"] = GEMINI_CLI_CLIENT_METADATA

        return headers

    def _get_endpoint(self) -> str:
        """Get the API endpoint for the current quota type."""
        if self._quota_type == "gemini-cli":
            return GEMINI_CLI_ENDPOINT
        return self._endpoint

    def _build_url(self, streaming: bool = False) -> str:
        """Build the API URL for generateContent.

        Args:
            streaming: Whether to use streaming endpoint.

        Returns:
            Full URL for the API call.
        """
        endpoint = self._get_endpoint()
        model = self._api_model

        if streaming:
            return f"{endpoint}/v1beta/models/{model}:streamGenerateContent?alt=sse"
        else:
            return f"{endpoint}/v1beta/models/{model}:generateContent"

    def _build_generation_config(
        self,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build generation configuration for the request."""
        # Get output limit for current model
        output_limit = DEFAULT_OUTPUT_LIMIT
        if self._model_name:
            if self._model_name in ANTIGRAVITY_MODELS:
                output_limit = ANTIGRAVITY_MODELS[self._model_name].get("output_limit", DEFAULT_OUTPUT_LIMIT)
            elif self._model_name in GEMINI_CLI_MODELS:
                output_limit = GEMINI_CLI_MODELS[self._model_name].get("output_limit", DEFAULT_OUTPUT_LIMIT)

        # Build thinking config
        thinking_config = None
        if self.supports_thinking():
            if "gemini-3" in (self._model_name or ""):
                # Gemini 3 uses thinkingLevel
                level = self._thinking_level or "low"
                thinking_config = {"thinkingLevel": level}
            elif "thinking" in (self._model_name or ""):
                # Claude thinking models use thinkingBudget
                thinking_config = {"thinkingBudget": self._thinking_budget}

        return build_generation_config(
            max_output_tokens=output_limit,
            thinking_config=thinking_config,
            response_schema=response_schema,
        )

    def _refresh_token_if_needed(self) -> None:
        """Refresh the OAuth token if expired."""
        if not self._current_account:
            return

        if self._current_account.tokens.is_expired:
            try:
                new_tokens = refresh_tokens(self._current_account.tokens.refresh_token)
                new_tokens.email = self._current_account.email
                new_tokens.project_id = self._current_account.project_id
                self._current_account.tokens = new_tokens

                if self._account_manager:
                    save_accounts(self._account_manager)
            except Exception as e:
                raise TokenRefreshError(f"Failed to refresh token: {e}")

    def _rotate_account_on_rate_limit(self) -> bool:
        """Rotate to next account on rate limit.

        Returns:
            True if successfully rotated, False if no other accounts.
        """
        if not self._auto_rotate or not self._account_manager or not self._current_account:
            return False

        next_account = self._account_manager.rotate_on_rate_limit(self._current_account.email)
        if next_account and next_account.email != self._current_account.email:
            self._current_account = next_account
            save_accounts(self._account_manager)
            return True

        return False

    def _handle_api_error(self, response: requests.Response) -> None:
        """Handle API error responses.

        Args:
            response: The HTTP response.

        Raises:
            Various AntigravityProviderError subclasses.
        """
        status = response.status_code

        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text

        if status == 401:
            raise TokenExpiredError("OAuth token expired or invalid")

        if status == 429:
            # Rate limit - try to rotate account
            if self._rotate_account_on_rate_limit():
                raise RateLimitError(
                    message="Rate limit hit, rotated to next account",
                    email=self._current_account.email if self._current_account else None,
                )
            raise RateLimitError(
                message=f"Rate limit exceeded: {error_msg}",
                email=self._current_account.email if self._current_account else None,
            )

        if status == 400 and "tool_result_missing" in error_msg.lower():
            raise ToolResultMissingError(error_msg)

        if status == 403:
            raise QuotaExceededError(f"Quota exceeded: {error_msg}")

        if status == 404:
            raise ModelNotFoundError(self._model_name or "unknown")

        raise APIError(
            message=f"API error ({status}): {error_msg}",
            status_code=status,
            response_body=response.text,
        )

    def _make_request(
        self,
        request_body: Dict[str, Any],
        streaming: bool = False,
        cancel_token: Optional[CancelToken] = None,
    ) -> requests.Response:
        """Make an API request with retry logic.

        Args:
            request_body: The request body.
            streaming: Whether to use streaming.
            cancel_token: Optional cancellation token.

        Returns:
            The HTTP response.

        Raises:
            Various AntigravityProviderError subclasses.
        """
        # Refresh token if needed
        self._refresh_token_if_needed()

        url = self._build_url(streaming=streaming)
        headers = self._get_headers()

        # Try multiple endpoints on failure
        endpoints_to_try = [self._endpoint]
        if self._quota_type == "antigravity":
            endpoints_to_try.extend([e for e in ANTIGRAVITY_ENDPOINTS if e != self._endpoint])

        last_error = None
        for endpoint in endpoints_to_try:
            if cancel_token and cancel_token.is_cancelled:
                raise CancelledException()

            try:
                if streaming:
                    url = f"{endpoint}/v1beta/models/{self._api_model}:streamGenerateContent?alt=sse"
                else:
                    url = f"{endpoint}/v1beta/models/{self._api_model}:generateContent"

                response = requests.post(
                    url,
                    json=request_body,
                    headers=headers,
                    timeout=120,
                    stream=streaming,
                )

                if response.status_code == 200:
                    return response

                # Handle specific errors that shouldn't trigger endpoint rotation
                if response.status_code in (401, 403, 429):
                    self._handle_api_error(response)

                last_error = APIError(
                    message=f"Request failed ({response.status_code})",
                    status_code=response.status_code,
                    response_body=response.text,
                )

            except requests.exceptions.RequestException as e:
                last_error = EndpointError(f"Request failed: {e}", tried_endpoints=[endpoint])

        if last_error:
            raise last_error

        raise EndpointError("All endpoints failed", tried_endpoints=endpoints_to_try)

    # ==================== Messaging ====================

    def generate(self, prompt: str) -> ProviderResponse:
        """Simple one-shot generation without session context.

        Args:
            prompt: The prompt text.

        Returns:
            ProviderResponse with the model's response.
        """
        if not self.is_connected:
            raise RuntimeError("Provider not connected. Call connect() first.")

        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        request_body = build_generate_request(
            contents=contents,
            generation_config=self._build_generation_config(),
        )

        response = self._make_request(request_body)
        provider_response = response_from_api(response.json())
        self._last_usage = provider_response.usage
        return provider_response

    def send_message(
        self,
        message: str,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        """Send a user message and get a response.

        Args:
            message: The user's message text.
            response_schema: Optional JSON Schema for structured output.

        Returns:
            ProviderResponse with text and/or function calls.
        """
        if not self.is_connected:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Add user message to history
        self._history.append(Message.from_text(Role.USER, message))

        # Build request
        contents = messages_to_api(self._history)
        tools = tool_schemas_to_api(self._tools) if self._tools else None

        request_body = build_generate_request(
            contents=contents,
            system_instruction=self._system_instruction,
            tools=tools,
            generation_config=self._build_generation_config(response_schema),
        )

        try:
            response = self._make_request(request_body)
            provider_response = response_from_api(response.json())
            self._last_usage = provider_response.usage

            # Add assistant response to history
            self._add_response_to_history(provider_response)

            # Handle structured output
            if response_schema:
                text = provider_response.get_text()
                if text:
                    try:
                        provider_response.structured_output = json.loads(text)
                    except json.JSONDecodeError:
                        pass

            return provider_response

        except Exception as e:
            # Remove user message on failure
            if self._history and self._history[-1].role == Role.USER:
                self._history.pop()
            raise

    def send_message_with_parts(
        self,
        parts: List[Part],
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        """Send a message with multiple parts (text, images, etc.).

        Args:
            parts: List of Part objects forming the message.
            response_schema: Optional JSON Schema for structured output.

        Returns:
            ProviderResponse with text and/or function calls.
        """
        if not self.is_connected:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Add multipart message to history
        self._history.append(Message(role=Role.USER, parts=parts))

        # Build request
        contents = messages_to_api(self._history)
        tools = tool_schemas_to_api(self._tools) if self._tools else None

        request_body = build_generate_request(
            contents=contents,
            system_instruction=self._system_instruction,
            tools=tools,
            generation_config=self._build_generation_config(response_schema),
        )

        try:
            response = self._make_request(request_body)
            provider_response = response_from_api(response.json())
            self._last_usage = provider_response.usage

            # Add assistant response to history
            self._add_response_to_history(provider_response)

            return provider_response

        except Exception as e:
            if self._history and self._history[-1].role == Role.USER:
                self._history.pop()
            raise

    def send_tool_results(
        self,
        results: List[ToolResult],
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        """Send tool execution results back to the model.

        Args:
            results: List of tool execution results.
            response_schema: Optional JSON Schema for structured output.

        Returns:
            ProviderResponse with the model's next response.
        """
        if not self.is_connected:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Add tool results to history
        tool_result_parts = [Part(function_response=r) for r in results]
        self._history.append(Message(role=Role.TOOL, parts=tool_result_parts))

        # Build request
        contents = messages_to_api(self._history)
        tools = tool_schemas_to_api(self._tools) if self._tools else None

        request_body = build_generate_request(
            contents=contents,
            system_instruction=self._system_instruction,
            tools=tools,
            generation_config=self._build_generation_config(response_schema),
        )

        try:
            response = self._make_request(request_body)
            provider_response = response_from_api(response.json())
            self._last_usage = provider_response.usage

            # Add assistant response to history
            self._add_response_to_history(provider_response)

            return provider_response

        except Exception as e:
            if self._history and self._history[-1].role == Role.TOOL:
                self._history.pop()
            raise

    # ==================== Streaming ====================

    def send_message_streaming(
        self,
        message: str,
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None,
    ) -> ProviderResponse:
        """Send a message with streaming response.

        Args:
            message: The user's message text.
            on_chunk: Callback for each text chunk.
            cancel_token: Optional cancellation token.
            response_schema: Optional JSON Schema for structured output.
            on_usage_update: Callback for usage updates.
            on_function_call: Callback when function call is detected.

        Returns:
            ProviderResponse with complete response.
        """
        if not self.is_connected:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Add user message to history
        self._history.append(Message.from_text(Role.USER, message))

        # Build request
        contents = messages_to_api(self._history)
        tools = tool_schemas_to_api(self._tools) if self._tools else None

        request_body = build_generate_request(
            contents=contents,
            system_instruction=self._system_instruction,
            tools=tools,
            generation_config=self._build_generation_config(response_schema),
        )

        try:
            response = self._make_request(request_body, streaming=True, cancel_token=cancel_token)
            provider_response = self._process_stream(
                response,
                on_chunk=on_chunk,
                cancel_token=cancel_token,
                on_usage_update=on_usage_update,
                on_function_call=on_function_call,
            )
            self._last_usage = provider_response.usage

            # Add assistant response to history
            self._add_response_to_history(provider_response)

            # Handle structured output
            if response_schema:
                text = provider_response.get_text()
                if text:
                    try:
                        provider_response.structured_output = json.loads(text)
                    except json.JSONDecodeError:
                        pass

            return provider_response

        except CancelledException:
            # Remove user message on cancellation
            if self._history and self._history[-1].role == Role.USER:
                self._history.pop()
            raise
        except Exception as e:
            if self._history and self._history[-1].role == Role.USER:
                self._history.pop()
            raise

    def send_tool_results_streaming(
        self,
        results: List[ToolResult],
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
        on_thinking: Optional[ThinkingCallback] = None,
    ) -> ProviderResponse:
        """Send tool results with streaming response.

        Args:
            results: List of tool execution results.
            on_chunk: Callback for each text chunk.
            cancel_token: Optional cancellation token.
            response_schema: Optional JSON Schema for structured output.
            on_usage_update: Callback for usage updates.
            on_function_call: Callback when function call is detected.

        Returns:
            ProviderResponse with complete response.
        """
        if not self.is_connected:
            raise RuntimeError("No chat session. Call create_session() first.")

        # Add tool results to history
        tool_result_parts = [Part(function_response=r) for r in results]
        self._history.append(Message(role=Role.TOOL, parts=tool_result_parts))

        # Build request
        contents = messages_to_api(self._history)
        tools = tool_schemas_to_api(self._tools) if self._tools else None

        request_body = build_generate_request(
            contents=contents,
            system_instruction=self._system_instruction,
            tools=tools,
            generation_config=self._build_generation_config(response_schema),
        )

        try:
            response = self._make_request(request_body, streaming=True, cancel_token=cancel_token)
            provider_response = self._process_stream(
                response,
                on_chunk=on_chunk,
                cancel_token=cancel_token,
                on_usage_update=on_usage_update,
                on_function_call=on_function_call,
            )
            self._last_usage = provider_response.usage

            # Add assistant response to history
            self._add_response_to_history(provider_response)

            return provider_response

        except CancelledException:
            if self._history and self._history[-1].role == Role.TOOL:
                self._history.pop()
            raise
        except Exception as e:
            if self._history and self._history[-1].role == Role.TOOL:
                self._history.pop()
            raise

    def _process_stream(
        self,
        response: requests.Response,
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
    ) -> ProviderResponse:
        """Process a streaming SSE response.

        Args:
            response: The HTTP response object.
            on_chunk: Callback for text chunks.
            cancel_token: Optional cancellation token.
            on_usage_update: Callback for usage updates.
            on_function_call: Callback for function calls.

        Returns:
            Complete ProviderResponse.
        """
        parts: List[Part] = []
        accumulated_text: List[str] = []
        thinking_text: List[str] = []
        usage = TokenUsage()
        finish_reason = FinishReason.UNKNOWN
        function_calls: List[FunctionCall] = []

        try:
            for line in response.iter_lines(decode_unicode=True):
                if cancel_token and cancel_token.is_cancelled:
                    # Flush accumulated text before cancelling
                    if accumulated_text:
                        parts.append(Part.from_text("".join(accumulated_text)))
                    raise CancelledException()

                if not line:
                    continue

                chunk_data = parse_sse_event(line)
                if not chunk_data:
                    continue

                if chunk_data.get("done"):
                    break

                # Extract text
                text = extract_text_from_stream_chunk(chunk_data)
                if text:
                    accumulated_text.append(text)
                    on_chunk(text)

                # Extract thinking
                thinking = extract_thinking_from_stream_chunk(chunk_data)
                if thinking:
                    thinking_text.append(thinking)

                # Extract function calls
                calls = extract_function_calls_from_stream_chunk(chunk_data)
                if calls:
                    # Flush text before function call
                    if accumulated_text:
                        parts.append(Part.from_text("".join(accumulated_text)))
                        accumulated_text = []

                    for call in calls:
                        function_calls.append(call)
                        parts.append(Part.from_function_call(call))
                        if on_function_call:
                            on_function_call(call)

                # Extract usage
                chunk_usage = extract_usage_from_stream_chunk(chunk_data)
                if chunk_usage:
                    usage = chunk_usage
                    if on_usage_update:
                        on_usage_update(usage.prompt_tokens, usage.output_tokens)

                # Extract finish reason
                chunk_finish = extract_finish_reason_from_stream_chunk(chunk_data)
                if chunk_finish:
                    finish_reason = chunk_finish

        except requests.exceptions.RequestException as e:
            raise StreamingError(f"Stream error: {e}")

        # Flush remaining text
        if accumulated_text:
            parts.append(Part.from_text("".join(accumulated_text)))

        # Override finish reason if we have function calls
        if function_calls:
            finish_reason = FinishReason.TOOL_USE

        # Build thinking string
        thinking = "\n".join(thinking_text) if thinking_text else None

        return ProviderResponse(
            parts=parts,
            usage=usage,
            finish_reason=finish_reason,
            thinking=thinking,
        )

    # ==================== Helper Methods ====================

    def _add_response_to_history(self, response: ProviderResponse) -> None:
        """Add assistant response to history.

        Args:
            response: The provider response.
        """
        # Filter out function_response parts (those are from tool results)
        history_parts = [
            p for p in response.parts
            if p.text is not None or p.function_call is not None
        ]

        if history_parts:
            self._history.append(Message(role=Role.MODEL, parts=history_parts))


def create_provider() -> AntigravityProvider:
    """Factory function for plugin discovery.

    Returns:
        New AntigravityProvider instance.
    """
    return AntigravityProvider()
