"""Authentication for Azure OpenAI: resource key **or** Microsoft Entra ID.

Two credential kinds, and they are not variants of one thing:

**Resource key** — a long-lived secret from the resource's *Keys and
Endpoint* blade, sent as the ``api-key`` header.  Stored by this module,
using the shared :class:`ApiKeyStore` mechanics.

**Microsoft Entra ID** (formerly Azure AD) — no stored secret at all.  A
bearer token is minted per request from whatever identity the host
already has: a managed identity on an Azure VM or container, a workload
identity in AKS, a service principal from environment variables, or the
developer's own ``az login``.  This is what corporate deployments
normally mandate, and it is why "no key found" must not be the end of the
credential search here as it is for the other providers.

``azure-identity`` is an optional dependency (the ``azure-openai`` extra).
It is imported lazily and only on the Entra path, so a key-authenticated
deployment never needs it installed.

Validation probes ``GET /openai/models?api-version=...`` — the data-plane
listing, which is authenticated and free.  A key that reaches it is a key
that will reach a deployment.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Tuple

from .._api_key_store import (
    ApiKeyStore,
    StoredCredential,
    body_snippet,
    create_validation_client,
)
from .env import AAD_SCOPE

_STORE = ApiKeyStore("azure_openai_auth.json", "Azure OpenAI")

AzureOpenAICredentials = StoredCredential


def save_credentials(
    credentials: StoredCredential,
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> None:
    """Save credentials to persistent storage (mode 0600 on POSIX)."""
    _STORE.save(
        credentials, workspace_path=workspace_path, config_root=config_root,
    )


def try_load_credentials_with_reason(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Tuple[Optional[StoredCredential], Optional[str]]:
    """Load credentials, returning a reason string when the load fails."""
    return _STORE.try_load_with_reason(
        workspace_path=workspace_path, config_root=config_root,
    )


def load_credentials(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[StoredCredential]:
    """Load credentials, or ``None`` when absent or unreadable."""
    return _STORE.load(workspace_path=workspace_path, config_root=config_root)


def clear_credentials(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> None:
    """Clear stored credentials."""
    _STORE.clear(workspace_path=workspace_path, config_root=config_root)


def get_stored_api_key(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Get the stored resource key if available."""
    return _STORE.api_key(
        workspace_path=workspace_path, config_root=config_root,
    )


def get_stored_endpoint(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Get the resource endpoint stored alongside the key, if any.

    Worth storing because on Azure the key and the endpoint are one
    credential: a key from resource A authenticates against nothing at
    resource B, so the two travel together or neither is useful.
    """
    credentials = load_credentials(
        workspace_path=workspace_path, config_root=config_root,
    )
    return credentials.get("endpoint") if credentials else None


def get_credential_file_path(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """The path of the credential file that would be loaded, or ``None``."""
    return _STORE.display_path(
        workspace_path=workspace_path, config_root=config_root,
    )


# ==================== Microsoft Entra ID ====================

def azure_identity_available() -> bool:
    """Is ``azure-identity`` importable?

    Checked before the Entra path is chosen so the failure reads as "the
    optional dependency is missing" rather than as an authentication
    error thirty lines later.
    """
    try:
        import azure.identity  # noqa: F401
    except ImportError:
        return False
    return True


def build_token_provider(credential: Any = None) -> Callable[[], str]:
    """A callable the OpenAI SDK invokes per request for a bearer token.

    ``DefaultAzureCredential`` is the deliberate choice over naming one
    mechanism: it is the chain every Azure SDK uses, so the same profile
    works under a managed identity in production, a workload identity in
    AKS, and ``az login`` on a laptop, with nothing in the profile to
    change between them.

    Args:
        credential: An explicit ``azure.identity`` credential.  Mostly a
            test seam; production passes ``None`` and gets the chain.

    Returns:
        The zero-argument token provider the SDK's
        ``azure_ad_token_provider`` parameter expects.

    Raises:
        ImportError: If ``azure-identity`` is not installed.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    return get_bearer_token_provider(
        credential or DefaultAzureCredential(), AAD_SCOPE,
    )


# ==================== Validation ====================

def _classify_validation_status(status: int, snippet: str) -> Tuple[bool, str]:
    """Map a probe response status onto the ``(valid, detail)`` contract.

    ``403`` counts as VALID: the credential authenticated and merely
    lacks the role assignment for the listing.  ``404`` counts as valid
    too — an endpoint whose api-version predates the models listing still
    proves the key was accepted, because Azure answers an unknown key
    with 401 before it routes.
    """
    if 200 <= status < 300 or status in (403, 404):
        return (True, "")
    if status == 401:
        return (False, f"authentication_error: {status}: {snippet}")
    if status == 429:
        return (False, f"rate_limit: {status}: {snippet}")
    if 500 <= status < 600:
        return (False, f"server_error: {status}: {snippet}")
    return (False, f"http_error: {status}: {snippet}")


def validate_api_key(
    api_key: str,
    endpoint: str,
    api_version: str,
) -> tuple:
    """Validate a resource key against the data-plane models listing.

    Args:
        api_key: The Azure OpenAI resource key.
        endpoint: ``https://<resource>.openai.azure.com``.
        api_version: The ``api-version`` date to probe with.

    Returns:
        ``(valid, detail)`` — see the native provider's counterpart for
        the ``detail`` vocabulary.
    """
    import httpx

    url = f"{endpoint.rstrip('/')}/openai/models"
    try:
        response = create_validation_client().get(
            url,
            headers={"api-key": api_key},
            params={"api-version": api_version},
            timeout=30,
        )
    except httpx.HTTPStatusError as exc:
        status = getattr(exc.response, "status_code", None)
        snippet = body_snippet(exc.response) if exc.response is not None else ""
        if status is None:
            return (False, f"http_error: None: {snippet}")
        return _classify_validation_status(status, snippet)
    except Exception as exc:
        return (False, f"network_error: {exc}")

    return _classify_validation_status(
        response.status_code, body_snippet(response),
    )


def login_with_key(
    api_key: str,
    endpoint: str,
    api_version: str,
    on_message: Optional[Callable[[str], None]] = None,
    workspace_path: Optional[str] = None,
) -> Optional[StoredCredential]:
    """Validate a resource key and store it with its endpoint.

    Returns:
        The stored credential, or ``None`` when validation failed — an
        unvalidated key is never written.
    """
    if on_message:
        on_message("Validating Azure OpenAI key...")

    valid, detail = validate_api_key(api_key, endpoint, api_version)
    if not valid:
        if on_message:
            if detail.startswith("network_error"):
                on_message(
                    f"Could not reach {endpoint} to validate your key. "
                    f"({detail})"
                )
            else:
                on_message("Azure OpenAI key validation failed.")
        return None

    credentials = StoredCredential(
        api_key=api_key,
        created_at=time.time(),
        fields={"endpoint": endpoint, "api_version": api_version},
    )
    save_credentials(credentials, workspace_path=workspace_path)
    if on_message:
        on_message("Azure OpenAI key validated and saved.")
    return credentials
