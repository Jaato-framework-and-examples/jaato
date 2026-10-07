"""Environment variable resolution for the Azure OpenAI provider.

Azure needs three things OpenAI does not, and all three are resolved here:

``endpoint``
    The resource URL, ``https://<resource>.openai.azure.com`` (or a
    private/sovereign-cloud equivalent).  There is no default: an Azure
    resource is a thing you provisioned, and guessing one would be
    guessing at somebody's tenant.

``api_version``
    Azure pins its API surface per **date**, and the date decides which
    request fields exist.  It is a required, explicit value — a default
    that drifted with the SDK would silently change what a deployment
    accepts.

``deployment``
    On Azure the model is addressed by the *deployment name* the
    subscription chose, not the model id.  jaato's ``model:`` field
    carries that name (see the provider docstring), and this variable is
    only the default when a profile names none.

The vendor's own ``AZURE_OPENAI_*`` variables are honored at lower
priority than the ``JAATO_``-namespaced ones, for the same reason
``OPENAI_API_KEY`` is on the native provider: they are what every Azure
SDK sample and CI recipe already sets.

Resolution priority:
1. Explicit config passed in code (``ProviderConfig`` / profile knob)
2. ``JAATO_AZURE_OPENAI_*`` environment variables
3. The vendor's own ``AZURE_OPENAI_*`` environment variables
4. The stored credential file (``azure_openai_auth.json``; see ``auth.py``)
"""

import os
from typing import List, Optional

# ============================================================
# Environment Variable Names
# ============================================================

ENV_AZURE_API_KEY = "JAATO_AZURE_OPENAI_API_KEY"
ENV_VENDOR_API_KEY = "AZURE_OPENAI_API_KEY"
ENV_AZURE_ENDPOINT = "JAATO_AZURE_OPENAI_ENDPOINT"
ENV_VENDOR_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
ENV_AZURE_API_VERSION = "JAATO_AZURE_OPENAI_API_VERSION"
ENV_VENDOR_API_VERSION = "AZURE_OPENAI_API_VERSION"
ENV_AZURE_DEPLOYMENT = "JAATO_AZURE_OPENAI_DEPLOYMENT"
ENV_VENDOR_DEPLOYMENT = "AZURE_OPENAI_DEPLOYMENT"
ENV_AZURE_CONTEXT_LENGTH = "JAATO_AZURE_OPENAI_CONTEXT_LENGTH"
ENV_AZURE_AUTH = "JAATO_AZURE_OPENAI_AUTH"

#: The two credential kinds.  ``key`` is the resource's own API key;
#: ``aad`` is Microsoft Entra ID (formerly Azure AD) — a bearer token
#: minted per request from whatever credential the host already has
#: (managed identity, a workload identity, ``az login``), which is what a
#: corporate deployment normally mandates because it leaves no long-lived
#: secret in the environment at all.
AUTH_KEY = "key"
AUTH_AAD = "aad"
VALID_AUTH = (AUTH_KEY, AUTH_AAD)

#: The scope an Entra token is requested for.  Fixed by Azure, not a knob.
AAD_SCOPE = "https://cognitiveservices.azure.com/.default"


def resolve_api_key(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Resolve the Azure OpenAI resource key.

    Resolution priority: ``JAATO_AZURE_OPENAI_API_KEY`` → the vendor's
    ``AZURE_OPENAI_API_KEY`` → stored credentials.
    """
    env_key = (
        os.environ.get(ENV_AZURE_API_KEY)  # env: Azure OpenAI resource key
        or os.environ.get(ENV_VENDOR_API_KEY)  # env: Azure OpenAI resource key (vendor's own name)
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


def resolve_endpoint() -> Optional[str]:
    """Resolve the resource endpoint, or ``None`` when unset.

    No default: the endpoint names a resource in somebody's subscription.
    """
    return (
        os.environ.get(ENV_AZURE_ENDPOINT)  # env: resource URL (https://<resource>.openai.azure.com)
        or os.environ.get(ENV_VENDOR_ENDPOINT)  # env: resource URL (vendor's own name)
    ) or None


def resolve_api_version() -> Optional[str]:
    """Resolve the ``api-version`` query parameter, or ``None`` when unset.

    Deliberately has no default.  The date selects the API surface, so a
    framework-chosen default would decide, invisibly, which request
    fields a user's deployment accepts — and would change under them the
    day the default moved.
    """
    return (
        os.environ.get(ENV_AZURE_API_VERSION)  # env: Azure api-version date (e.g. 2024-10-21)
        or os.environ.get(ENV_VENDOR_API_VERSION)  # env: Azure api-version date (vendor's own name)
    ) or None


def resolve_deployment() -> Optional[str]:
    """Resolve the default deployment name, or ``None`` when unset."""
    return (
        os.environ.get(ENV_AZURE_DEPLOYMENT)  # env: default deployment name
        or os.environ.get(ENV_VENDOR_DEPLOYMENT)  # env: default deployment name (vendor's own name)
    ) or None


def resolve_context_length() -> Optional[int]:
    """Resolve the context-window override, or ``None`` when unset/invalid.

    Azure's data-plane listing reports deployments, not capacities, so
    this override — or ``plugin_configs.azure_openai.context_length`` —
    is required in practice.  No hardcoded per-model table is substituted
    (project no-fallback rule): the same deployment name can be pointed
    at a different model version by whoever owns the resource, so a table
    keyed on it would be wrong by construction.
    """
    value = os.environ.get(ENV_AZURE_CONTEXT_LENGTH)  # env: context window (required — Azure reports none)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return None


def resolve_auth_method() -> Optional[str]:
    """Resolve the credential kind (``key`` / ``aad``) from the environment.

    Returns the lower-cased value, or ``None`` when unset.  Validation
    belongs to the provider, which can name the profile key in its error.
    """
    value = os.environ.get(ENV_AZURE_AUTH, "")  # env: credential kind ("key" or "aad")
    return value.strip().lower() or None


def get_checked_credential_locations(config=None) -> List[str]:
    """Get the list of locations checked for credentials.

    ``config`` (optional ``ProviderConfig``) surfaces the highest-precedence
    source — the profile ``plugin_configs.azure_openai.api_key`` knob —
    which the env-only checks below cannot see.
    """
    from ..base import profile_api_key_location

    locations = [profile_api_key_location(config, "azure_openai")]

    for var in (ENV_AZURE_API_KEY, ENV_VENDOR_API_KEY):
        key = os.environ.get(var)  # env: Azure OpenAI resource key
        if key:
            masked = f"{key[:6]}...{key[-4:]}" if len(key) > 12 else "***"
            locations.append(f"{var}: set ({masked})")
        else:
            locations.append(f"{var}: not set")

    try:
        from .auth import get_stored_api_key, get_credential_file_path
        stored_key = get_stored_api_key()
        if stored_key:
            cred_path = get_credential_file_path() or "azure_openai_auth.json"
            masked = (
                f"{stored_key[:6]}...{stored_key[-4:]}"
                if len(stored_key) > 12 else "***"
            )
            locations.append(f"Stored credentials ({cred_path}): set ({masked})")
        else:
            locations.append(
                "Stored credentials: not configured "
                "(<config_root>/azure_openai_auth.json, else "
                "<workspace>/.jaato/, else ~/.jaato/)"
            )
    except ImportError:
        locations.append("Stored credentials: auth module not available")

    auth = resolve_auth_method()
    locations.append(
        f"{ENV_AZURE_AUTH}: {auth}" if auth
        else f"{ENV_AZURE_AUTH}: not set (default: {AUTH_KEY})"
    )
    endpoint = resolve_endpoint()
    locations.append(
        f"Endpoint: {endpoint}" if endpoint
        else f"{ENV_AZURE_ENDPOINT}: not set (required)"
    )
    version = resolve_api_version()
    locations.append(
        f"api-version: {version}" if version
        else f"{ENV_AZURE_API_VERSION}: not set (required)"
    )
    return locations
