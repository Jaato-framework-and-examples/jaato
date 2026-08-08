"""Environment variable resolution for the LM Studio provider.

LM Studio runs a local inference server that exposes three surfaces:
- OpenAI-compatible chat at ``/v1/chat/completions``
- Native REST for model listing at ``/api/v0/models``
- Native REST for model load-control at ``/api/v1/models/load``

The provider reads all three from the same host.  No credential is
required by default; an optional bearer token can be set when LM Studio
is configured with ``Require API Token`` enabled.

Environment variables:
    LMSTUDIO_HOST: Server URL (default: http://localhost:1234)
    LMSTUDIO_MODEL: Default model name
    LMSTUDIO_CONTEXT_LENGTH: Override context window size
    LMSTUDIO_API_TOKEN: Optional bearer token (only if LM Studio requires it)
"""

import os
from typing import Optional


ENV_HOST = "LMSTUDIO_HOST"
ENV_MODEL = "LMSTUDIO_MODEL"
ENV_CONTEXT_LENGTH = "LMSTUDIO_CONTEXT_LENGTH"
ENV_API_TOKEN = "LMSTUDIO_API_TOKEN"

DEFAULT_HOST = "http://localhost:1234"

# Context window to report when the loaded model's metadata isn't known.
# LM Studio's /api/v0/models surfaces each model's real max_context_length,
# so this value is only used as a conservative pre-load fallback.
DEFAULT_CONTEXT_LENGTH = 8192


def resolve_host() -> str:
    """Return the LM Studio server URL."""
    return os.environ.get(ENV_HOST, DEFAULT_HOST)  # env: LM Studio server URL (default http://localhost:1234)


def resolve_model() -> Optional[str]:
    """Return the default model name, if configured."""
    return os.environ.get(ENV_MODEL)  # env: default model name (unset = whatever LM Studio already has loaded)


def resolve_context_length() -> Optional[int]:
    """Return the context length override, if configured."""
    val = os.environ.get(ENV_CONTEXT_LENGTH)  # env: override context window (normally discovered from /api/v0/models)
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return None


def resolve_api_token() -> Optional[str]:
    """Return the optional bearer token.

    Returns None when LM Studio is running without ``Require API Token``
    (the common local-dev case).
    """
    return os.environ.get(ENV_API_TOKEN)  # env: bearer token, only when LM Studio requires auth
