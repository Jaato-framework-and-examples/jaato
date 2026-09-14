"""Environment variable resolution for the native OpenAI provider.

Configuration is resolved from environment variables:
- ``JAATO_OPENAI_API_KEY`` (or the vendor's own ``OPENAI_API_KEY``) for the key
- ``JAATO_OPENAI_BASE_URL`` (or ``OPENAI_BASE_URL``) for the endpoint
- ``JAATO_OPENAI_MODEL`` for the default model name
- ``JAATO_OPENAI_CONTEXT_LENGTH`` for the context window (required in
  practice — see :func:`resolve_context_length`)
- ``JAATO_OPENAI_ORG_ID`` / ``JAATO_OPENAI_PROJECT_ID`` for the billing
  headers (the vendor's ``OPENAI_ORG_ID`` / ``OPENAI_PROJECT_ID`` are
  honored too)
- ``JAATO_OPENAI_API`` to select the wire: ``chat`` (Chat Completions,
  the default) or ``responses`` (the Responses API)

**Why the vendor's own names are honored.**  ``OPENAI_API_KEY`` is the
variable every OpenAI SDK, example and CI recipe already sets, and a
provider that ignored it would make the flagship integration the one that
needs extra configuration.  The ``JAATO_``-namespaced name still wins, so
a workspace that deliberately points jaato at a different key than the
ambient one is not overridden by the environment it happens to run in.
The same two-name pattern is already in the tree for ``nebius``
(``NEBIUS_API_KEY``) and ``ovhcloud`` (``OVH_AI_ENDPOINTS_ACCESS_TOKEN``).

Resolution priority:
1. Explicit config passed in code (``ProviderConfig`` / profile knob)
2. ``JAATO_OPENAI_*`` environment variables
3. The vendor's own ``OPENAI_*`` environment variables
4. The stored credential file (``openai_auth.json``; see ``auth.py``)
"""

import os
from typing import List, Optional

# ============================================================
# Environment Variable Names
# ============================================================

ENV_OPENAI_API_KEY = "JAATO_OPENAI_API_KEY"
#: The vendor's own documented variable, honored at lower priority.
ENV_VENDOR_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_BASE_URL = "JAATO_OPENAI_BASE_URL"
ENV_VENDOR_BASE_URL = "OPENAI_BASE_URL"
ENV_OPENAI_MODEL = "JAATO_OPENAI_MODEL"
ENV_OPENAI_CONTEXT_LENGTH = "JAATO_OPENAI_CONTEXT_LENGTH"
ENV_OPENAI_ORG_ID = "JAATO_OPENAI_ORG_ID"
ENV_VENDOR_ORG_ID = "OPENAI_ORG_ID"
ENV_OPENAI_PROJECT_ID = "JAATO_OPENAI_PROJECT_ID"
ENV_VENDOR_PROJECT_ID = "OPENAI_PROJECT_ID"
ENV_OPENAI_API = "JAATO_OPENAI_API"

#: OpenAI's own OpenAI-compatible endpoint.
DEFAULT_BASE_URL = "https://api.openai.com/v1"

#: The two wires this provider speaks.  ``chat`` is Chat Completions (the
#: shared ``_openai_compat`` transport); ``responses`` is the Responses
#: API, which is a different request and event shape and lives in
#: ``responses.py``.
API_CHAT = "chat"
API_RESPONSES = "responses"
VALID_APIS = (API_CHAT, API_RESPONSES)


def resolve_api_key(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Resolve the OpenAI API key from environment or stored credentials.

    Resolution priority:
    1. ``JAATO_OPENAI_API_KEY``.
    2. ``OPENAI_API_KEY`` (the vendor's own variable).
    3. The stored credential file ``openai_auth.json`` (resolves under
       ``config_root`` then the workspace then ``~/.jaato/`` per
       :func:`shared.config_resolver.resolve_config_search_path`).

    Args:
        workspace_path: Optional explicit workspace path passed through to
            the credential lookup.
        config_root: Optional read-only-config root override.

    Returns:
        API key if found, None otherwise.
    """
    env_key = (
        os.environ.get(ENV_OPENAI_API_KEY)  # env: OpenAI API key
        or os.environ.get(ENV_VENDOR_API_KEY)  # env: OpenAI API key (vendor's own name)
    )
    if env_key:
        return env_key
    try:
        from .auth import get_stored_api_key
        return get_stored_api_key(
            workspace_path=workspace_path, config_root=config_root,
        )
    except ImportError:
        return None


def resolve_base_url() -> str:
    """Resolve the OpenAI base URL from environment.

    Returns:
        The API base URL (``https://api.openai.com/v1`` unless overridden).
    """
    return (
        os.environ.get(ENV_OPENAI_BASE_URL)  # env: endpoint (default https://api.openai.com/v1)
        or os.environ.get(ENV_VENDOR_BASE_URL)  # env: endpoint (vendor's own name)
        or DEFAULT_BASE_URL
    )


def resolve_model() -> Optional[str]:
    """Resolve the default model name from environment.

    Returns:
        Model name if found, None otherwise.
    """
    return os.environ.get(ENV_OPENAI_MODEL)  # env: default model name (e.g. gpt-5.1)


def resolve_context_length() -> Optional[int]:
    """Resolve the context-window override from the environment.

    Unlike the gateways with a capacity-reporting catalog, OpenAI's
    ``GET /v1/models`` serves bare entries (``{id, object, created,
    owned_by}``) — it reports no context window for any model — so this
    override, or the ``plugin_configs.openai.context_length`` knob, is in
    practice **required**.  No hardcoded per-model table is substituted:
    a wrong window silently truncates or over-fills, and the project's
    no-fallback rule exists for exactly that failure.

    Returns:
        The override as an int, or ``None`` when unset/invalid.
    """
    value = os.environ.get(ENV_OPENAI_CONTEXT_LENGTH)  # env: context window (required — the catalog reports none)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return None


def resolve_organization() -> Optional[str]:
    """Resolve the organization id for the ``OpenAI-Organization`` header.

    Returns:
        The organization id, or ``None`` when unset.
    """
    return (
        os.environ.get(ENV_OPENAI_ORG_ID)  # env: organization for billing attribution
        or os.environ.get(ENV_VENDOR_ORG_ID)  # env: organization (vendor's own name)
    ) or None


def resolve_project() -> Optional[str]:
    """Resolve the project id for the ``OpenAI-Project`` header.

    Returns:
        The project id, or ``None`` when unset.
    """
    return (
        os.environ.get(ENV_OPENAI_PROJECT_ID)  # env: project for billing attribution
        or os.environ.get(ENV_VENDOR_PROJECT_ID)  # env: project (vendor's own name)
    ) or None


def resolve_api() -> Optional[str]:
    """Resolve the wire selector (``chat`` / ``responses``) from the env.

    The profile knob ``plugin_configs.openai.api`` takes precedence; this
    is the profile-free way to flip a session onto the Responses API.

    Returns:
        The lower-cased selector, or ``None`` when unset/blank.  Validation
        (is it one of :data:`VALID_APIS`?) belongs to the provider, which
        can name the profile key in its error.
    """
    value = os.environ.get(ENV_OPENAI_API, "")  # env: wire selector ("chat" or "responses")
    return value.strip().lower() or None


def is_self_hosted(base_url: str) -> bool:
    """Check whether the base URL points at a local proxy.

    OpenAI is a hosted service, so this is normally False; it stays for
    the case of a user fronting the API through a local gateway
    (``JAATO_OPENAI_BASE_URL=http://localhost:...``), where a key may not
    be required.

    Args:
        base_url: The configured endpoint URL.

    Returns:
        True if the URL appears to be a local instance.
    """
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    return (
        host in ("localhost", "127.0.0.1", "0.0.0.0")
        or host.startswith("192.168.")
        or host.startswith("10.")
    )


def get_checked_credential_locations(config=None) -> List[str]:
    """Get the list of locations checked for credentials.

    Used for error messages to help users understand what was checked.

    ``config`` (optional ``ProviderConfig``) surfaces the highest-precedence
    source — the profile ``plugin_configs.openai.api_key`` knob — which the
    env-only checks below cannot see.

    Returns:
        List of location descriptions.
    """
    from ..base import profile_api_key_location

    locations = [profile_api_key_location(config, "openai")]

    for var in (ENV_OPENAI_API_KEY, ENV_VENDOR_API_KEY):
        key = os.environ.get(var)  # env: OpenAI API key
        if key:
            masked = f"{key[:6]}...{key[-4:]}" if len(key) > 12 else "***"
            locations.append(f"{var}: set ({masked})")
        else:
            locations.append(f"{var}: not set")

    try:
        from .auth import get_stored_api_key, get_credential_file_path
        stored_key = get_stored_api_key()
        if stored_key:
            cred_path = get_credential_file_path() or "openai_auth.json"
            masked = (
                f"{stored_key[:6]}...{stored_key[-4:]}"
                if len(stored_key) > 12 else "***"
            )
            locations.append(f"Stored credentials ({cred_path}): set ({masked})")
        else:
            locations.append(
                "Stored credentials: not configured "
                "(<config_root>/openai_auth.json, else "
                "<workspace>/.jaato/, else ~/.jaato/)"
            )
    except ImportError:
        locations.append("Stored credentials: auth module not available")

    base_url = os.environ.get(ENV_OPENAI_BASE_URL) or os.environ.get(
        ENV_VENDOR_BASE_URL)
    if base_url:
        locations.append(f"Endpoint: {base_url}")
    else:
        locations.append(f"Endpoint: {DEFAULT_BASE_URL} (default)")

    model = os.environ.get(ENV_OPENAI_MODEL)
    if model:
        locations.append(f"{ENV_OPENAI_MODEL}: {model}")

    return locations
