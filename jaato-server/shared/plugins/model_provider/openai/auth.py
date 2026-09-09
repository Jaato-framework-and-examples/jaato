"""Authentication for the native OpenAI API.

OpenAI uses API key (Bearer token) authentication.  The filesystem
mechanics — where the credential file lives, 0600 on write, absent versus
corrupt on read — are the shared :class:`ApiKeyStore`; what is OpenAI's
own is here: the validation probe and the login flow.

API keys are created at https://platform.openai.com/api-keys.  Storage
follows the jaato convention: ``<config_root>/openai_auth.json``, else
``<workspace>/.jaato/openai_auth.json``, else ``~/.jaato/openai_auth.json``.
There is no ``openai-auth`` command plugin yet — as for ``nebius``,
``ovhcloud`` and ``doubleword``, :func:`login_with_key` is the entry point
an embedder or a later command plugin calls, and the file it writes is
read by the provider either way.

**Validation probes ``GET /v1/models``, not a completion.**  It is
authenticated, free and one round trip, so proving a key is live costs
nothing and bills nothing — unlike the gateways whose ``/models`` is
unauthenticated and whose keys can only be proven by attempting a
generation.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

from .._api_key_store import (
    ApiKeyStore,
    StoredCredential,
    body_snippet,
    create_validation_client,
)
from .env import DEFAULT_BASE_URL

#: The one instance; module-level functions delegate to it so the public
#: shape matches every other provider's ``auth`` module.
_STORE = ApiKeyStore("openai_auth.json", "OpenAI")

# Re-exported so a caller can name the credential type without reaching
# into the shared module.
OpenAICredentials = StoredCredential


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
    """Load credentials, returning a reason string when the load fails.

    ``(None, None)`` means no file; ``(None, reason)`` means the file is
    there and unusable, which is what lets ``verify_auth`` say so instead
    of reporting "no key found" for a corrupt file.
    """
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
    """Get the stored API key if available."""
    return _STORE.api_key(
        workspace_path=workspace_path, config_root=config_root,
    )


def get_stored_base_url(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Get the stored custom base URL if one was saved with the key."""
    credentials = load_credentials(
        workspace_path=workspace_path, config_root=config_root,
    )
    return credentials.get("base_url") if credentials else None


def get_credential_file_path(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """The path of the credential file that would be loaded, or ``None``."""
    return _STORE.display_path(
        workspace_path=workspace_path, config_root=config_root,
    )


def _classify_validation_status(status: int, snippet: str) -> Tuple[bool, str]:
    """Map a probe response status onto the ``(valid, detail)`` contract.

    ``403`` counts as VALID: the key authenticated and the organization
    is merely not permitted to list models.  Reporting that as a bad key
    would send a user to rotate a credential that works.
    """
    if 200 <= status < 300 or status == 403:
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
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    project: Optional[str] = None,
) -> tuple:
    """Validate an API key with an authenticated ``GET /v1/models``.

    Args:
        api_key: OpenAI API key to validate.
        base_url: Optional custom base URL.
        organization: Optional organization id (``OpenAI-Organization``).
        project: Optional project id (``OpenAI-Project``).

    Returns:
        ``(valid, detail)``.  ``detail`` carries a structured code:
        ``""`` (valid), ``"authentication_error: ..."``,
        ``"rate_limit: ..."``, ``"server_error: ..."``,
        ``"http_error: ..."`` or ``"network_error: ..."``.
    """
    import httpx

    url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    if organization:
        headers["OpenAI-Organization"] = organization
    if project:
        headers["OpenAI-Project"] = project

    try:
        response = create_validation_client().get(
            f"{url}/models", headers=headers, timeout=30,
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
    base_url: Optional[str] = None,
    on_message: Optional[Callable[[str], None]] = None,
    workspace_path: Optional[str] = None,
    organization: Optional[str] = None,
    project: Optional[str] = None,
) -> Optional[StoredCredential]:
    """Validate a provided API key and store it (non-interactive).

    Args:
        api_key: OpenAI API key.
        base_url: Optional custom base URL, saved with the key.
        on_message: Callback for status messages.
        workspace_path: Explicit workspace path for the write.
        organization: Optional organization id, saved with the key.
        project: Optional project id, saved with the key.

    Returns:
        The stored credential, or ``None`` when validation failed — an
        unvalidated key is never written, so a typo does not become the
        session's credential.
    """
    if on_message:
        on_message("Validating API key...")

    valid, detail = validate_api_key(
        api_key, base_url, organization=organization, project=project,
    )
    if not valid:
        if on_message:
            if detail.startswith("network_error"):
                on_message(
                    "Could not reach the OpenAI API to validate your key. "
                    f"({detail})"
                )
            else:
                on_message("API key validation failed.")
        return None

    credentials = StoredCredential(
        api_key=api_key,
        created_at=time.time(),
        fields={
            "base_url": base_url,
            "organization": organization,
            "project": project,
        },
    )
    save_credentials(credentials, workspace_path=workspace_path)
    if on_message:
        on_message("API key validated and saved.")
    return credentials
