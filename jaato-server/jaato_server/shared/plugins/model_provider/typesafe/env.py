"""Environment variable resolution for the TypeSafe decision provider.

Every value here is a FALLBACK: the profile's ``plugin_configs.typesafe``
knob is read first, by the provider (docs/design/decision-models.md §6.2).

    JAATO_TYPESAFE_API_KEY: API key (jaato namespace).
    TYPESAFE_API_KEY: TypeSafe's own variable (its SDKs read it), honoured
        beneath the jaato one so a host set up for ``typesafe-sdk`` needs
        nothing else.
    JAATO_TYPESAFE_BASE_URL: API root (default ``https://api.typesafe.ai``).
    JAATO_TYPESAFE_CONTEXT_LENGTH: the window the pre-send estimate checks.

Key resolution, highest first: the knob, ``JAATO_TYPESAFE_API_KEY``,
``TYPESAFE_API_KEY``, then the key stored by ``typesafe-auth key``.
"""

import os
from typing import Optional, Tuple

ENV_API_KEY = "JAATO_TYPESAFE_API_KEY"
ENV_VENDOR_API_KEY = "TYPESAFE_API_KEY"
ENV_BASE_URL = "JAATO_TYPESAFE_BASE_URL"
ENV_CONTEXT_LENGTH = "JAATO_TYPESAFE_CONTEXT_LENGTH"

#: TypeSafe's API root; the decision endpoint is ``/v1/systemone`` under it.
DEFAULT_BASE_URL = "https://api.typesafe.ai"


def resolve_env_api_key_with_source() -> Tuple[Optional[str], Optional[str]]:
    """``(key, variable name)`` from the environment: the jaato variable,
    then the vendor's.  ``(None, None)`` when neither is set."""
    jaato = os.environ.get(ENV_API_KEY)  # env: TypeSafe API key (fallback beneath plugin_configs.typesafe.api_key)
    if jaato:
        return jaato, ENV_API_KEY
    vendor = os.environ.get(ENV_VENDOR_API_KEY)  # env: TypeSafe's own key variable, read by its SDKs
    if vendor:
        return vendor, ENV_VENDOR_API_KEY
    return None, None


def resolve_api_key(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """The key from the environment, else the stored ``typesafe-auth`` key.

    The profile knob is not visible here; the provider reads it first.
    """
    env_key, _source = resolve_env_api_key_with_source()
    if env_key:
        return env_key
    try:
        from .auth import get_stored_api_key
    except ImportError:
        return None
    return get_stored_api_key(workspace_path=workspace_path, config_root=config_root)


def resolve_base_url() -> str:
    """The API root from the environment, else TypeSafe's."""
    return os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL  # env: TypeSafe API root (default https://api.typesafe.ai)


def resolve_context_length() -> Optional[int]:
    """The context-window override from the environment, if a number."""
    val = os.environ.get(ENV_CONTEXT_LENGTH)  # env: window the pre-send estimate checks (unchecked when unset)
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return None
