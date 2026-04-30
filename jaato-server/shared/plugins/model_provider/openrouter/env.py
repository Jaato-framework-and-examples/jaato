"""Environment variable resolution for the OpenRouter provider.

Resolution priority:
1. Explicit ``ProviderConfig`` values supplied in code.
2. ``JAATO_OPENROUTER_*`` environment variables.
3. Stored credentials saved by ``openrouter-auth``.
4. Sensible defaults.
"""

import os
from typing import List, Optional

# ============================================================
# Environment Variable Names
# ============================================================

ENV_OPENROUTER_API_KEY = "JAATO_OPENROUTER_API_KEY"
ENV_OPENROUTER_BASE_URL = "JAATO_OPENROUTER_BASE_URL"
ENV_OPENROUTER_MODEL = "JAATO_OPENROUTER_MODEL"
ENV_OPENROUTER_CONTEXT_LENGTH = "JAATO_OPENROUTER_CONTEXT_LENGTH"
ENV_OPENROUTER_HTTP_REFERER = "JAATO_OPENROUTER_HTTP_REFERER"
ENV_OPENROUTER_APP_TITLE = "JAATO_OPENROUTER_APP_TITLE"

# OpenRouter uses these header names for app attribution / rankings.
HEADER_HTTP_REFERER = "HTTP-Referer"
HEADER_APP_TITLE = "X-OpenRouter-Title"

# Default OpenRouter endpoint.  The same key works for all upstream models.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Default context window when the model catalog doesn't report one.
DEFAULT_CONTEXT_LENGTH = 32768

# Default attribution headers — OpenRouter uses these for app rankings
# when an integrator opts in.  Users can override via env vars.
DEFAULT_HTTP_REFERER = "https://github.com/Jaato-framework-and-examples/jaato"
DEFAULT_APP_TITLE = "jaato"


def resolve_api_key() -> Optional[str]:
    """Resolve the OpenRouter API key from env or stored credentials."""
    env_key = os.environ.get(ENV_OPENROUTER_API_KEY)
    if env_key:
        return env_key
    try:
        from .auth import get_stored_api_key
        return get_stored_api_key()
    except ImportError:
        return None


def resolve_base_url() -> str:
    """Resolve the OpenRouter base URL from env."""
    return os.environ.get(ENV_OPENROUTER_BASE_URL, DEFAULT_BASE_URL)


def resolve_model() -> Optional[str]:
    """Resolve a default model name from env, if set."""
    return os.environ.get(ENV_OPENROUTER_MODEL)


def resolve_context_length() -> int:
    """Resolve the context window override from env, falling back to default."""
    value = os.environ.get(ENV_OPENROUTER_CONTEXT_LENGTH)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return DEFAULT_CONTEXT_LENGTH


def resolve_http_referer() -> str:
    """Resolve the HTTP-Referer header for OpenRouter app rankings."""
    return os.environ.get(ENV_OPENROUTER_HTTP_REFERER, DEFAULT_HTTP_REFERER)


def resolve_app_title() -> str:
    """Resolve the X-Title header for OpenRouter app rankings."""
    return os.environ.get(ENV_OPENROUTER_APP_TITLE, DEFAULT_APP_TITLE)


def get_checked_credential_locations() -> List[str]:
    """Describe which credential locations were checked.

    Returned by errors so users see exactly what the provider looked at
    before giving up on credentials.
    """
    locations: List[str] = []

    api_key = os.environ.get(ENV_OPENROUTER_API_KEY)
    if api_key:
        masked = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        locations.append(f"{ENV_OPENROUTER_API_KEY}: set ({masked})")
    else:
        locations.append(f"{ENV_OPENROUTER_API_KEY}: not set")

    try:
        from .auth import get_stored_api_key, get_credential_file_path
        stored_key = get_stored_api_key()
        if stored_key:
            cred_path = get_credential_file_path() or "openrouter_auth.json"
            masked = f"{stored_key[:8]}...{stored_key[-4:]}" if len(stored_key) > 12 else "***"
            locations.append(f"Stored credentials ({cred_path}): set ({masked})")
        else:
            locations.append(
                "Stored credentials: not configured (use 'openrouter-auth login')"
            )
    except ImportError:
        locations.append("Stored credentials: auth module not available")

    base_url = os.environ.get(ENV_OPENROUTER_BASE_URL)
    if base_url:
        locations.append(f"{ENV_OPENROUTER_BASE_URL}: {base_url}")
    else:
        locations.append(f"Endpoint: {DEFAULT_BASE_URL} (default)")

    model = os.environ.get(ENV_OPENROUTER_MODEL)
    if model:
        locations.append(f"{ENV_OPENROUTER_MODEL}: {model}")

    return locations
