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
import os
from shared.session_context import get_workspace_root, get_config_root
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set

import httpx

from ..base import (
    MODALITY_TEXT,
    ModalityCapabilityMixin,
    FunctionCallDetectedCallback,
    ModelProviderPlugin,
    ProviderConfig,
    StreamingCallback,
    ThinkingCallback,
    UsageUpdateCallback,
    resolve_context_window,
    resolve_modalities,
)
from jaato_sdk.plugins.model_provider.types import (
    CancelledException,
    CancelToken,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    ThinkingConfig,
    ToolSchema,
    TokenUsage,
    TurnResult,
    require_terminated_stream,
    resolve_tool_use_finish,
)
from .constants import (
    ANTIGRAVITY_API_CLIENT,
    ANTIGRAVITY_CLIENT_METADATA,
    ANTIGRAVITY_ENDPOINTS,
    ANTIGRAVITY_MODELS,
    ANTIGRAVITY_PRIMARY_ENDPOINT,
    ANTIGRAVITY_USER_AGENT,
    DEFAULT_OUTPUT_LIMIT,
    GEMINI_CLI_API_CLIENT,
    GEMINI_CLI_CLIENT_METADATA,
    GEMINI_CLI_ENDPOINT,
    GEMINI_CLI_MODELS,
    MODEL_INPUT_MODALITIES,
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


class AntigravityProvider(ModalityCapabilityMixin):
    """Antigravity provider for Google's IDE AI backend.

    This provider supports:
    - Multiple model families (Gemini 3, Claude)
    - Google OAuth authentication
    - Multi-account rotation for quota management
    - Streaming responses with SSE
    - Extended thinking for compatible models

    The provider is **stateless** with respect to conversation history.
    All conversation state is managed by the session layer which calls
    ``complete()`` with the full message list on every turn.

    Usage:
        provider = AntigravityProvider()
        provider.initialize()  # Uses stored OAuth tokens
        provider.connect('antigravity-gemini-3-flash')
        response = provider.complete(messages, system_instruction="...")

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
        # Per-profile context-window override (plugin_configs.antigravity.
        # context_length).  Wins over the model tables' context_limit in
        # get_context_limit() — escape hatch for an unlisted model.  None = use
        # the tables; unknown model + no override → raise (no hardcoded fallback).
        self._context_length_knob: Optional[int] = None
        # INPUT-modality assertion (plugin_configs.antigravity.modalities)
        # layered over MODEL_INPUT_MODALITIES in modalities().
        self._modalities_knob: Optional[List[str]] = None
        # Sampling parameters (plugin_configs.antigravity.api_params.*).
        # ``None`` = omit from the generationConfig and let the backend apply
        # its server-side default.  A profile wanting determinism sets
        # ``api_params.temperature: 0.0`` (falsy → ``is not None`` guards).
        # Threaded through _build_generation_config() → build_generation_config().
        self._temperature: Optional[float] = None
        self._top_p: Optional[float] = None
        self._top_k: Optional[int] = None
        self._seed: Optional[int] = None

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

        # Per-call accounting (updated by complete())
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

        # Set workspace path from config.extra if provided
        # This ensures token resolution can find workspace-specific OAuth tokens
        # even when JAATO_WORKSPACE_ROOT env var isn't set (e.g., subagent spawning)
        workspace_path = config.extra.get('workspace_path')
        if workspace_path and not get_workspace_root():
            os.environ['JAATO_WORKSPACE_ROOT'] = workspace_path

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

        # Context-window override (plugin_configs.antigravity.context_length) via
        # the shared precedence helper.  Antigravity's model windows are
        # documented in the model tables, so there is no live auto-detect tier
        # and no env var — the knob is the only override.
        self._context_length_knob = resolve_context_window(
            detect_capacity=None,
            profile_value=config.extra.get("context_length"),
            env_value=None,
        )

        # INPUT-modality assertion (plugin_configs.antigravity.modalities)
        # — escape hatch for a model not in MODEL_INPUT_MODALITIES.
        modalities_override = config.extra.get("modalities")
        if modalities_override is not None:
            if not isinstance(modalities_override, (list, tuple)) or not all(
                isinstance(m, str) for m in modalities_override
            ):
                raise TypeError(
                    "Antigravity 'modalities' config must be a list of "
                    f"strings (e.g. [\"text\", \"image\"]), got "
                    f"{type(modalities_override).__name__}"
                )
            self._modalities_knob = list(modalities_override)

        # Sampling parameters (plugin_configs.antigravity.api_params).  They
        # live NESTED at config.extra["api_params"][...] — the canonical
        # namespaced layer mirroring anthropic / openrouter.  ``None`` means
        # "omit from the generationConfig and let the backend apply its
        # server-side default"; a profile wanting determinism sets
        # ``api_params.temperature: 0.0`` (falsy → ``is not None`` guards).
        api_params = config.extra.get("api_params") or {}
        if not isinstance(api_params, dict):
            raise TypeError(
                "Antigravity 'api_params' config must be a dict of "
                f"generationConfig fields, got {type(api_params).__name__}"
            )
        temp_extra = api_params.get("temperature")
        if temp_extra is not None:
            self._temperature = float(temp_extra)
        top_p_extra = api_params.get("top_p")
        if top_p_extra is not None:
            self._top_p = float(top_p_extra)
        top_k_extra = api_params.get("top_k")
        if top_k_extra is not None:
            self._top_k = int(top_k_extra)
        seed_extra = api_params.get("seed")
        if seed_extra is not None:
            self._seed = int(seed_extra)

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message: Optional[Callable[[str], None]] = None,
        config: Optional["ProviderConfig"] = None,
    ) -> bool:
        """Verify authentication and optionally trigger interactive login.

        ``config`` is accepted for protocol compatibility but unused — Antigravity
        credentials live in OAuth storage, not in the profile.

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

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used."""
        if self._current_account:
            return f"OAuth ({self._current_account.email})"
        return ""

    # ==================== Connection ====================

    def connect(self, model: str, *, skip_model_test: bool = False) -> None:
        """Connect to a specific model.

        Args:
            model: Model ID (e.g., 'antigravity-gemini-3-flash', 'gemini-2.5-flash').
            skip_model_test: Accepted for protocol compatibility; this provider
                does not perform network validation during connect.

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

        Resolution precedence (no hardcoded fallback, per project rule):
        1. ``context_length`` override knob (``_context_length_knob``).
        2. the model's ``context_limit`` from the ANTIGRAVITY_MODELS /
           GEMINI_CLI_MODELS tables.
        3. else raise — unknown model with no override is a configuration error.

        Returns:
            Maximum context tokens.

        Raises:
            ValueError: when neither the knob nor a table entry yields a value.
        """
        if self._context_length_knob:
            return self._context_length_knob
        if self._model_name:
            for table in (ANTIGRAVITY_MODELS, GEMINI_CLI_MODELS):
                if self._model_name in table:
                    limit = table[self._model_name].get("context_limit")
                    if limit:
                        return limit
        raise ValueError(
            f"Antigravity provider: no known context window for model "
            f"{self._model_name!r}, and no override is set.  Set "
            f"plugin_configs.antigravity.context_length in the profile.  No "
            f"hardcoded fallback exists per the project's no-fallback rule."
        )

    def modalities(self, model: Optional[str] = None) -> Set[str]:
        """INPUT modalities the active Antigravity model accepts.

        modalities knob -> MODEL_INPUT_MODALITIES (exact) -> text-only
        floor.  All current served models are multimodal; the floor is
        the safe answer for any future text-only addition.
        """
        model = model or self._model_name
        resolved = resolve_modalities(
            profile_value=self._modalities_knob,
            table_value=(
                MODEL_INPUT_MODALITIES.get(model) if model else None
            ),
        )
        return resolved if resolved is not None else {MODALITY_TEXT}

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

        # Profile sampling knobs (plugin_configs.antigravity.api_params).
        # ``None`` knobs are passed through and dropped by the helper's
        # ``is not None`` guards, so ``temperature: 0.0`` (determinism) reaches
        # the wire while unset knobs let the backend apply its defaults.
        return build_generation_config(
            max_output_tokens=output_limit,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            seed=self._seed,
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

    def _handle_api_error(self, response: httpx.Response) -> None:
        """Handle API error responses.

        Args:
            response: The HTTP response.

        Raises:
            Various AntigravityProviderError subclasses.
        """
        # ``complete()``'s except-clause calls this with the EXCEPTION,
        # not a response.  Everything below reads HTTP fields, so
        # ``status = response.status_code`` used to raise AttributeError
        # and mask whatever actually went wrong -- including the
        # interrupted-stream error that has to reach ``with_retry``
        # (#687).  Hand a non-response back for the caller to re-raise.
        if not isinstance(response, httpx.Response):
            return

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
    ) -> httpx.Response:
        """Make an API request with retry logic.

        Uses ``get_httpx_client()`` for unified proxy/SSL/Kerberos
        configuration.  For streaming requests the response body is not
        eagerly consumed — the caller iterates and closes.

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

                from shared.http import get_httpx_client

                # For streaming, we must NOT close the client before the
                # caller finishes iterating — so avoid the context manager.
                client = get_httpx_client(timeout=120.0)
                try:
                    if streaming:
                        req = client.build_request(
                            "POST", url, json=request_body, headers=headers,
                        )
                        response = client.send(req, stream=True)
                    else:
                        response = client.post(
                            url, json=request_body, headers=headers,
                        )

                    if response.status_code == 200:
                        return response

                    # Eagerly read the body for non-200 so we can inspect it
                    if streaming:
                        response.read()

                    # Handle specific errors that shouldn't trigger endpoint rotation
                    if response.status_code in (401, 403, 429):
                        self._handle_api_error(response)

                    last_error = APIError(
                        message=f"Request failed ({response.status_code})",
                        status_code=response.status_code,
                        response_body=response.text,
                    )
                except httpx.HTTPError:
                    raise
                except AntigravityProviderError:
                    raise
                except Exception:
                    client.close()
                    raise

            except httpx.HTTPError as e:
                last_error = EndpointError(f"Request failed: {e}", tried_endpoints=[endpoint])

        if last_error:
            raise last_error

        raise EndpointError("All endpoints failed", tried_endpoints=endpoints_to_try)

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
        """Stateless completion: convert messages to API format, call API, return response.

        The caller (session) is responsible for maintaining the message
        list and passing it in full each call.  This method does not
        hold or modify any conversation state.

        When ``on_chunk`` is provided, the response is streamed via SSE.
        When ``on_chunk`` is None, the response is returned in batch mode.

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
        if not self.is_connected:
            raise RuntimeError("Provider not connected. Call initialize() and connect() first.")

        # Build request from explicit parameters (NOT instance state)
        contents = messages_to_api(list(messages))
        api_tools = tool_schemas_to_api(tools) if tools else None

        request_body = build_generate_request(
            contents=contents,
            system_instruction=system_instruction,
            tools=api_tools,
            generation_config=self._build_generation_config(response_schema),
        )

        try:
            if on_chunk:
                # Streaming mode
                response = self._make_request(
                    request_body, streaming=True, cancel_token=cancel_token
                )
                provider_response = self._process_stream(
                    response,
                    on_chunk=on_chunk,
                    cancel_token=cancel_token,
                    on_usage_update=on_usage_update,
                    on_function_call=on_function_call,
                )
            else:
                # Batch mode
                response = self._make_request(request_body)
                provider_response = response_from_api(response.json())

            # Update per-call accounting (NOT conversation state)
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
        except Exception as e:
            self._handle_api_error(e)
            raise

    def _process_stream(
        self,
        response: httpx.Response,
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
    ) -> ProviderResponse:
        """Process a streaming SSE response.

        Args:
            response: The HTTP response object (streamed, not yet consumed).
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
        # Whether the SSE stream declared its own end -- either a chunk
        # carrying a finish reason, or the protocol's ``done`` sentinel.
        # Without one of those the connection simply stopped, and what
        # arrived is a fragment (#687).
        terminal_seen = False
        function_calls: List[FunctionCall] = []
        chunk_count = 0

        try:
            for line in response.iter_lines():
                if cancel_token and cancel_token.is_cancelled:
                    # Flush accumulated text before returning
                    if accumulated_text:
                        parts.append(Part.from_text("".join(accumulated_text)))
                    # Return response with CANCELLED finish reason (matches other providers)
                    # This allows mid-turn prompt handling to work at the session level
                    return ProviderResponse(
                        parts=parts,
                        usage=usage,
                        finish_reason=FinishReason.CANCELLED,
                        thinking="\n".join(thinking_text) if thinking_text else None,
                    )

                if not line:
                    continue

                chunk_data = parse_sse_event(line)
                if not chunk_data:
                    continue

                if chunk_data.get("done"):
                    terminal_seen = True
                    break

                # Extract text
                text = extract_text_from_stream_chunk(chunk_data)
                if text:
                    chunk_count += 1
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
                    terminal_seen = True
                    finish_reason = chunk_finish

        except httpx.HTTPError as e:
            raise StreamingError(f"Stream error: {e}")
        finally:
            # Close the underlying HTTP connection so Antigravity stops
            # generating immediately on cancel. ``httpx.Response`` from
            # ``client.send(req, stream=True)`` only sends TCP-close at
            # garbage-collection time, which on a cancelled turn can
            # leave the upstream model generating to completion.
            try:
                response.close()
            except Exception:  # pragma: no cover - best effort
                pass

        # Flush remaining text
        if accumulated_text:
            parts.append(Part.from_text("".join(accumulated_text)))

        # TOOL_USE fills in an unreported or merely-``stop`` finish; it
        # must not displace a terminal one.  A turn that hit the output
        # cap mid-``arguments`` carries fragments, not a request — see
        # ``resolve_tool_use_finish`` and issue #745.
        finish_reason = resolve_tool_use_finish(
            finish_reason,
            has_function_calls=bool(function_calls),
        )

        # Build thinking string
        thinking = "\n".join(thinking_text) if thinking_text else None

        # A stream that stopped arriving is not a turn that finished
        # (#687).  Raises rather than returning the fragment.  A
        # cancelled turn never reaches here -- it returns from inside the
        # loop above -- so ``was_cancelled`` is False by construction.
        return require_terminated_stream(
            ProviderResponse(
                parts=parts,
                usage=usage,
                finish_reason=finish_reason,
                thinking=thinking,
            ),
            terminal_seen=terminal_seen,
            was_cancelled=False,
            provider=self.name,
            model=self._model_name,
            chunks=chunk_count,
        )

def create_provider() -> AntigravityProvider:
    """Factory function for plugin discovery.

    Returns:
        New AntigravityProvider instance.
    """
    return AntigravityProvider()
