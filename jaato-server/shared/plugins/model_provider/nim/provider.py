"""NVIDIA NIM model provider implementation.

NIM exposes an OpenAI-compatible chat completions API, so this provider is a
thin subclass of :class:`OpenAICompatProvider` (the shared OpenAI-compat
machinery — streaming loop, completion skeleton, error mapping, capability
boilerplate).  Only NIM-specific concerns live here: identity, the error-class
parameterization, credential/context resolution, and auth introspection.

Supports both NVIDIA's hosted API (build.nvidia.com) and self-hosted NIM
containers (no API key required).

Environment variables:
    JAATO_NIM_API_KEY: API key for hosted NIM (nvapi-...)
    JAATO_NIM_BASE_URL: Endpoint (default: https://integrate.api.nvidia.com/v1)
    JAATO_NIM_MODEL: Default model name
    JAATO_NIM_CONTEXT_LENGTH: Override context window size
"""

from __future__ import annotations

from typing import Optional

from .._openai_compat.base import OpenAICompatProvider
from ..base import ProviderConfig, resolve_context_window
from .env import (
    ENV_NIM_CONTEXT_LENGTH,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
    is_self_hosted,
    get_checked_credential_locations,
)
from .errors import (
    APIKeyNotFoundError,
    AuthenticationError,
    ContextLimitError,
    InfrastructureError,
    ModelNotFoundError,
    RateLimitError,
)

# Models known to expose reasoning/thinking content via ``reasoning_content``.
REASONING_CAPABLE_MODELS = [
    "deepseek/deepseek-r1",
    "deepseek-r1",
]


class NIMProvider(OpenAICompatProvider):
    """NVIDIA NIM model provider.

    Provides access to NIM's model catalog via the OpenAI-compatible chat
    completions API, for both NVIDIA's hosted API and self-hosted NIM
    containers.  All transport machinery is inherited from
    :class:`OpenAICompatProvider`; see its docstring for the lifecycle.

    Usage:
        provider = NIMProvider()
        provider.initialize(ProviderConfig(api_key='nvapi-...'))
        provider.connect('meta/llama-3.1-70b-instruct')
        result = provider.complete(messages, system_instruction="You are helpful.")
    """

    # Parameterize the base's shared error mapping with NIM's taxonomy.
    _ERR_AUTHENTICATION = AuthenticationError
    _ERR_RATE_LIMIT = RateLimitError
    _ERR_MODEL_NOT_FOUND = ModelNotFoundError
    _ERR_CONTEXT_LIMIT = ContextLimitError
    _ERR_INFRASTRUCTURE = InfrastructureError

    REASONING_CAPABLE_MODELS = REASONING_CAPABLE_MODELS

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "nim"

    # ==================== Credential / context hooks ====================

    def _resolve_credentials(self, config: ProviderConfig) -> None:
        """Resolve the API key + base URL; validate (required for hosted).

        Pulls workspace_path / config_root from ``config.extra`` (injected by
        ``JaatoRuntime.create_provider``) so credential lookup resolves under
        the session's explicit config_root rather than the unreliable
        ``JAATO_CONFIG_ROOT`` env var for headless reactor-spawned sessions.
        """
        _ws_path = config.extra.get('workspace_path') if config.extra else None
        _config_root = config.extra.get('config_root') if config.extra else None

        self._api_key = config.api_key or resolve_api_key(
            workspace_path=_ws_path, config_root=_config_root,
        )
        self._base_url = config.extra.get("base_url") or resolve_base_url()

        # Required for hosted, optional for self-hosted NIM containers.
        if not self._api_key and not is_self_hosted(self._base_url):
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(),
            )

    def _detect_context(self, config: ProviderConfig) -> Optional[int]:
        """Resolve the context window.

        NIM's OpenAI-compatible ``/v1/models`` does not surface a per-model
        context window, so there is no auto-detect tier — resolution is
        profile-knob then env, else the fail-loud ``_context_error_message``.
        """
        return resolve_context_window(
            detect_capacity=None,
            profile_value=config.extra.get("context_length"),
            env_value=resolve_context_length(),
        )

    def _context_error_message(self) -> str:
        return (
            "NIM provider: context_length could not be resolved.  NIM's "
            "/v1/models does not report a per-model context window, and no "
            "manual override is set.  Set plugin_configs.nim.context_length "
            f"in the profile, or {ENV_NIM_CONTEXT_LENGTH} in the "
            "environment.  No hardcoded fallback exists per the project's "
            "no-fallback rule."
        )

    # ==================== Auth introspection ====================

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional["ProviderConfig"] = None,
    ) -> bool:
        """Verify that authentication is configured.

        Must work before ``initialize()`` — checks for the API key without ever
        touching ``self._client``.  A profile-supplied ``api_key`` (the daemon
        expands ``pass://`` secrets into the verify-time ``ProviderConfig``)
        takes effect during this pre-init gate; without reading it here a valid
        profile credential would be rejected before ``initialize()`` runs.

        A stored credential file that exists but cannot be loaded surfaces its
        load error via ``on_message`` instead of being swallowed into "No
        credentials found".

        Raises:
            APIKeyNotFoundError: If no key found and the endpoint is not
                self-hosted (unless ``allow_interactive``).
        """
        import os
        from .auth import try_load_credentials_with_reason
        from .env import ENV_NIM_API_KEY

        base_url = resolve_base_url()

        # Profile-supplied key wins (daemon expands pass:// into config.extra).
        profile_key: Optional[str] = None
        if config is not None:
            profile_key = config.api_key or (
                config.extra.get("api_key") if config.extra else None
            )
        if profile_key:
            if on_message:
                on_message("Found NIM API key (profile config)")
            return True

        # Env var (highest priority among the stored sources)
        env_key = os.environ.get(ENV_NIM_API_KEY)
        if env_key:
            if on_message:
                on_message("Found NIM API key (environment variable)")
            return True

        # Stored credentials, differentiating missing-file from broken-file.
        creds, load_error = try_load_credentials_with_reason()
        if creds and creds.api_key:
            if on_message:
                on_message("Found NIM API key (stored credentials)")
            return True

        if load_error:
            if on_message:
                on_message(
                    f"NIM credentials file found but could not be loaded: "
                    f"{load_error}"
                )
                on_message(
                    "Run 'nim-auth key <your_api_key>' to re-authenticate, "
                    f"or set {ENV_NIM_API_KEY}."
                )
            if not allow_interactive:
                raise APIKeyNotFoundError(
                    checked_locations=get_checked_credential_locations()
                )
            return False

        if is_self_hosted(base_url):
            if on_message:
                on_message(f"Self-hosted NIM endpoint ({base_url}), no API key required")
            return True

        if not allow_interactive:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations()
            )

        return False

    def get_auth_info(self) -> str:
        """Return a short description of the credential source used."""
        import os
        from .env import ENV_NIM_API_KEY

        if is_self_hosted(self._base_url):
            return f"Self-hosted NIM ({self._base_url})"

        if os.environ.get(ENV_NIM_API_KEY):
            return f"NIM API key ({ENV_NIM_API_KEY})"

        try:
            from .auth import get_credential_file_path
            extra = getattr(self._config, 'extra', None) or {}
            cred_path = get_credential_file_path(
                workspace_path=extra.get('workspace_path'),
                config_root=extra.get('config_root'),
            )
            if cred_path:
                return f"NIM API key ({cred_path})"
        except ImportError:
            pass

        return "NIM API key"


def create_provider() -> NIMProvider:
    """Factory function for plugin discovery.

    Returns:
        A new NIMProvider instance.
    """
    return NIMProvider()
