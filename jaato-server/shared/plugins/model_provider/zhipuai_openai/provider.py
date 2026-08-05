"""Zhipu AI (Z.AI) GLM provider via the OpenAI-compatible endpoint.

Z.AI exposes an OpenAI-compatible chat API, so this is a thin subclass of
:class:`OpenAICompatProvider` (hosted nim-family) — the shared transport,
streaming loop, error mapping, and capability boilerplate.

GLM specifics live here: API-key auth, a per-model context-window table (Z.AI's
catalog doesn't surface context length, so it's looked up from
``MODEL_CONTEXT_LIMITS``), extended-thinking capability (GLM-5 / GLM-4.7
non-flash), and the shared ``zhipuai`` auth helpers (the API key is the same
for both endpoints).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .._openai_compat.base import OpenAICompatProvider
from ..base import ProviderConfig
from .env import (
    DEFAULT_CONTEXT_LENGTH,
    resolve_api_key,
    resolve_base_url,
    resolve_context_length,
    resolve_enable_thinking,
    resolve_thinking_budget,
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

logger = logging.getLogger(__name__)


# Known GLM models + context windows.  Z.AI's catalog doesn't report context
# length, so this table is the source of truth for the window.
MODEL_CONTEXT_LIMITS = {
    "glm-5": 204800,
    "glm-5-turbo": 204800,
    "glm-4.7": 204800,
    "glm-4.7-flash": 204800,
    "glm-4.7-flashx": 204800,
    "glm-4.6": 204800,
    "glm-4.5": 131072,
    "glm-4.5-air": 131072,
    "glm-4.5-airx": 131072,
    "glm-4.5-flash": 131072,
    "glm-4.5-x": 131072,
}

KNOWN_MODELS = sorted(MODEL_CONTEXT_LIMITS.keys())


class ZhipuAIOpenAIProvider(OpenAICompatProvider):
    """Zhipu AI GLM provider over the OpenAI-compatible API.

    All transport machinery is inherited from :class:`OpenAICompatProvider`.
    """

    # Parameterize the base's shared error mapping with Z.AI's (nim-family) taxonomy.
    _ERR_AUTHENTICATION = AuthenticationError
    _ERR_RATE_LIMIT = RateLimitError
    _ERR_MODEL_NOT_FOUND = ModelNotFoundError
    _ERR_CONTEXT_LIMIT = ContextLimitError
    _ERR_INFRASTRUCTURE = InfrastructureError

    def __init__(self):
        super().__init__()
        self._context_length_override: Optional[int] = None
        self._thinking_budget: Optional[int] = None

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "zhipuai_openai"

    # ==================== Credential / context hooks ====================

    def _resolve_credentials(self, config: ProviderConfig) -> None:
        """Resolve the API key + base URL + extended-thinking config."""
        if config.api_key:
            self._api_key = config.api_key
            self._auth_info = "API key (config)"
        else:
            self._api_key = resolve_api_key()
            if self._api_key:
                self._auth_info = "API key (env/stored)"
        if not self._api_key:
            raise APIKeyNotFoundError(
                checked_locations=get_checked_credential_locations(),
            )
        self._base_url = (config.extra.get("base_url") or resolve_base_url()).rstrip("/")

        # Extended-thinking config (GLM): controls reasoning_content extraction.
        self._enable_thinking = config.extra.get(
            "enable_thinking", resolve_enable_thinking())
        self._thinking_budget = config.extra.get(
            "thinking_budget", resolve_thinking_budget())

    def _resolve_context(self, config: ProviderConfig) -> None:
        """Stash the manual override — Z.AI has no dynamic context endpoint, so
        the window is looked up per-model in ``get_context_limit`` (no fail-loud).
        """
        ctx = config.extra.get("context_length")
        self._context_length_override = int(ctx) if ctx else resolve_context_length()

    def get_context_limit(self) -> int:
        """Explicit override → per-model ``MODEL_CONTEXT_LIMITS`` → default.

        (Z.AI's OpenAI surface doesn't report context length; the table is the
        source of truth.  The ``DEFAULT_CONTEXT_LENGTH`` tail is pre-existing
        provider behavior, preserved by this migration.)
        """
        if self._context_length_override:
            return self._context_length_override
        return MODEL_CONTEXT_LIMITS.get(self._model_name, DEFAULT_CONTEXT_LENGTH)

    # ==================== Thinking ====================

    def supports_thinking(self) -> bool:
        """GLM-5 and GLM-4.7 (non-flash) have native chain-of-thought."""
        return self._is_thinking_capable()

    def set_thinking_config(self, config) -> None:
        """Set extended-thinking enable flag + budget."""
        self._enable_thinking = config.enabled
        if config.budget_tokens is not None:
            self._thinking_budget = config.budget_tokens

    def _is_thinking_capable(self) -> bool:
        """GLM-5 and GLM-4.7 (non-flash) support extended thinking."""
        if not self._model_name:
            return False
        name_lower = self._model_name.lower()
        if name_lower.startswith("glm-5"):
            return True
        return name_lower.startswith("glm-4.7") and "flash" not in name_lower

    # ==================== Auth introspection ====================

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message=None,
        config: Optional["ProviderConfig"] = None,
    ) -> bool:
        """Verify an API key is available (env / stored), without touching
        ``self._client``."""
        api_key = resolve_api_key()
        if api_key:
            if on_message:
                on_message("Found Zhipu AI API key")
            return True
        if on_message:
            on_message("No Zhipu AI credentials found")
        return False

    def get_auth_info(self) -> str:
        """Short description of the credential source used."""
        return self._auth_info or "Z.AI API key"

    # ==================== Catalog ====================

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List GLM models via Z.AI's ``GET /models``; fall back to KNOWN_MODELS."""
        models = self._fetch_remote_models() or KNOWN_MODELS.copy()
        if prefix:
            models = [m for m in models if m.startswith(prefix)]
        return sorted(models)

    def _fetch_remote_models(self) -> List[str]:
        """Fetch the model list from Z.AI; empty on failure / no key."""
        api_key = self._api_key or resolve_api_key()
        if not api_key:
            return []
        from ..zhipuai.provider import fetch_zhipuai_models
        return fetch_zhipuai_models(api_key)

    # ==================== Static Auth Helpers ====================

    @staticmethod
    def login(on_message=None, on_input=None) -> bool:
        """Interactive login (delegates to the shared zhipuai auth module)."""
        from ..zhipuai.auth import login_interactive
        return login_interactive(on_message=on_message, on_input=on_input) is not None

    @staticmethod
    def logout(on_message=None) -> None:
        """Clear stored credentials."""
        from ..zhipuai.auth import logout
        logout(on_message=on_message)

    @staticmethod
    def auth_status(on_message=None) -> bool:
        """Check authentication status."""
        from ..zhipuai.auth import status as _status
        return _status(on_message=on_message)


def create_provider() -> ZhipuAIOpenAIProvider:
    """Factory function for plugin discovery."""
    return ZhipuAIOpenAIProvider()
