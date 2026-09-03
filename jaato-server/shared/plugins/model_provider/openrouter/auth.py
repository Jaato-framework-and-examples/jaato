"""Authentication module for OpenRouter.

OpenRouter uses simple bearer-token authentication with API keys that
look like ``sk-or-v1-...`` (issued from https://openrouter.ai/settings/keys).
This module provides:

- Secure local storage of API keys (mode 0600, project / home fallback)
- Key validation against the public ``GET /api/v1/key`` endpoint
- Status / introspection helpers used by ``openrouter-auth`` and the
  provider's ``verify_auth``.

Storage follows the jaato convention used by the other auth plugins:

1. Project ``.jaato/openrouter_auth.json`` (project-specific)
2. Home ``~/.jaato/openrouter_auth.json`` (user-level default)
"""

import json
import logging
import os
from shared.session_context import get_workspace_root, get_config_root
from shared.secret_repr import secret_safe_repr
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from .env import DEFAULT_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterCredentials:
    """Stored OpenRouter credentials."""
    api_key: str
    created_at: float  # Unix timestamp
    base_url: Optional[str] = None  # Optional custom base URL

    # Never print the key: a bare dataclass repr put a live
    # ``sk-…`` into a pytest failure message, and from there into
    # scrollback and CI logs (#721).  ``to_dict`` below still
    # returns the real value — this guards display, not storage.
    __repr__ = secret_safe_repr("api_key")

    def to_dict(self) -> dict:
        data = {
            "api_key": self.api_key,
            "created_at": self.created_at,
        }
        if self.base_url:
            data["base_url"] = self.base_url
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "OpenRouterCredentials":
        return cls(
            api_key=data["api_key"],
            created_at=data.get("created_at", time.time()),
            base_url=data.get("base_url"),
        )


def _get_token_storage_path(
    for_write: bool = False,
    workspace_path: Optional[str] = None,
) -> Path:
    """Resolve the credential storage path.

    Mirrors the NIM auth helper: prefers project ``.jaato/`` when it
    exists, falls back to ``~/.jaato/``.  Honors ``JAATO_WORKSPACE_ROOT``
    so subagents and remote daemons resolve to the right workspace.
    """
    workspace = (
        workspace_path
        or get_workspace_root()
        or os.getcwd()
    )
    project_path = Path(workspace) / ".jaato" / "openrouter_auth.json"
    home_path = Path.home() / ".jaato" / "openrouter_auth.json"

    if for_write:
        if project_path.parent.exists():
            return project_path
        return home_path
    if project_path.exists():
        return project_path
    if home_path.exists():
        return home_path
    return home_path


def save_credentials(
    credentials: OpenRouterCredentials,
    workspace_path: Optional[str] = None,
) -> None:
    """Save credentials to persistent storage with secure permissions."""
    path = _get_token_storage_path(for_write=True, workspace_path=workspace_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(credentials.to_dict(), f, indent=2)

    if os.name == "posix":
        os.chmod(path, 0o600)


def load_credentials(
    workspace_path: Optional[str] = None,
) -> Optional[OpenRouterCredentials]:
    """Load credentials, returning ``None`` if absent or unreadable.

    Broken files log a warning (and the failure reason is available
    through :func:`try_load_credentials_with_reason` for surface-level
    diagnostics).
    """
    creds, _ = try_load_credentials_with_reason(workspace_path=workspace_path)
    return creds


def try_load_credentials_with_reason(
    workspace_path: Optional[str] = None,
) -> Tuple[Optional[OpenRouterCredentials], Optional[str]]:
    """Load credentials and return a reason string when the load fails.

    Returns ``(credentials, reason)``:

    - ``(OpenRouterCredentials, None)`` — file loaded successfully.
    - ``(None, None)`` — no credential file exists.
    - ``(None, "<reason>")`` — file exists but could not be loaded.
    """
    path = _get_token_storage_path(workspace_path=workspace_path)

    if not path.exists():
        return None, None

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, PermissionError) as exc:
        reason = f"cannot read {path}: {exc}"
        logger.warning("Failed to read OpenRouter credentials: %s", reason)
        return None, reason
    except json.JSONDecodeError as exc:
        reason = (
            f"invalid JSON at {path}: {exc.msg} "
            f"(line {exc.lineno}, col {exc.colno})"
        )
        logger.warning("Failed to parse OpenRouter credentials: %s", reason)
        return None, reason

    try:
        return OpenRouterCredentials.from_dict(data), None
    except (KeyError, TypeError) as exc:
        reason = (
            f"malformed credentials at {path}: missing or invalid field ({exc})"
        )
        logger.warning("Malformed OpenRouter credentials: %s", reason)
        return None, reason
    except Exception as exc:  # defensive — never mask unexpected failures
        reason = (
            f"unexpected error loading {path}: "
            f"{exc.__class__.__name__}: {exc}"
        )
        logger.warning("Unexpected error loading OpenRouter credentials: %s", reason)
        return None, reason


def clear_credentials(workspace_path: Optional[str] = None) -> None:
    """Delete the credential file if present."""
    path = _get_token_storage_path(workspace_path=workspace_path)
    if path.exists():
        path.unlink()


def get_stored_api_key(workspace_path: Optional[str] = None) -> Optional[str]:
    """Return the stored API key if available."""
    creds = load_credentials(workspace_path=workspace_path)
    if creds:
        return creds.api_key
    return None


def get_credential_file_path(
    workspace_path: Optional[str] = None,
) -> Optional[str]:
    """Return a human-friendly path of the credential file in use, if any."""
    path = _get_token_storage_path(workspace_path=workspace_path)
    if not path.exists():
        return None
    home = Path.home()
    if path.is_relative_to(home):
        return "~/" + str(path.relative_to(home))
    return str(path)


def get_stored_base_url() -> Optional[str]:
    """Return the stored custom base URL if available."""
    creds = load_credentials()
    if creds:
        return creds.base_url
    return None


def _create_validation_client():
    """Create an httpx client with proxy / Kerberos / corporate CA support.

    Mirrors the helper used by the other providers so validation
    requests honor the same network policy as real chat traffic.
    """
    from shared.ssl_helper import active_cert_bundle
    from shared.http.proxy import get_httpx_client

    kwargs = {}
    ca_bundle = active_cert_bundle()
    if ca_bundle:
        kwargs["verify"] = ca_bundle

    return get_httpx_client(**kwargs)


def _extract_body_snippet(response, limit: int = 300) -> str:
    """Return a short, safe snippet of a response body for error detail."""
    try:
        text = response.text or ""
    except Exception:
        return ""
    text = text.strip().replace("\n", " ")
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def validate_api_key(
    api_key: str,
    base_url: Optional[str] = None,
) -> tuple:
    """Validate an API key against OpenRouter's ``GET /key`` endpoint.

    OpenRouter exposes ``/api/v1/key`` which returns the rate-limit /
    credit information for the bearer token.  A 200 means the key is
    accepted; 401/403 mean it isn't.  Other status codes are reported
    as structured detail strings instead of being silently treated as
    success.

    Returns:
        A ``(valid, detail)`` tuple.  ``detail`` carries one of:

        - ``""`` — key is valid.
        - ``"authentication_error: <status>: <body>"`` — key rejected.
        - ``"rate_limit: <status>: <body>"`` — quota / 429.
        - ``"payment_required: <status>: <body>"`` — 402, billing inactive.
        - ``"server_error: <status>: <body>"`` — 5xx, transient outage.
        - ``"http_error: <status>: <body>"`` — any other unexpected status.
        - ``"network_error: <details>"`` — request never reached OpenRouter.
    """
    import httpx

    url = base_url or DEFAULT_BASE_URL
    test_url = f"{url.rstrip('/')}/key"

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    try:
        client = _create_validation_client()
        response = client.get(test_url, headers=headers, timeout=30)
    except httpx.HTTPStatusError as e:
        status = getattr(e.response, "status_code", None)
        snippet = _extract_body_snippet(e.response) if e.response is not None else ""
        if status in (401, 403):
            return (False, f"authentication_error: {status}: {snippet}")
        if status == 429:
            return (False, f"rate_limit: {status}: {snippet}")
        if status == 402:
            return (False, f"payment_required: {status}: {snippet}")
        if status is not None and 500 <= status < 600:
            return (False, f"server_error: {status}: {snippet}")
        return (False, f"http_error: {status}: {snippet}")
    except Exception as e:
        return (False, f"network_error: {e}")

    status = response.status_code
    if 200 <= status < 300:
        return (True, "")
    snippet = _extract_body_snippet(response)
    if status in (401, 403):
        return (False, f"authentication_error: {status}: {snippet}")
    if status == 429:
        return (False, f"rate_limit: {status}: {snippet}")
    if status == 402:
        return (False, f"payment_required: {status}: {snippet}")
    if 500 <= status < 600:
        return (False, f"server_error: {status}: {snippet}")
    return (False, f"http_error: {status}: {snippet}")


def login_with_key(
    api_key: str,
    base_url: Optional[str] = None,
    on_message: Optional[Callable[[str], None]] = None,
    workspace_path: Optional[str] = None,
) -> Optional[OpenRouterCredentials]:
    """Validate an API key and persist it on success.

    Args:
        api_key: OpenRouter API key (``sk-or-...``).
        base_url: Optional custom endpoint.
        on_message: Callback for status messages.
        workspace_path: Workspace root for project-local storage.

    Returns:
        Saved ``OpenRouterCredentials`` on success, or ``None`` if
        validation failed.
    """
    if on_message:
        on_message("Validating API key...")

    valid, detail = validate_api_key(api_key, base_url)
    if valid:
        credentials = OpenRouterCredentials(
            api_key=api_key,
            created_at=time.time(),
            base_url=base_url,
        )
        save_credentials(credentials, workspace_path=workspace_path)

        if on_message:
            on_message("API key validated and saved.")

        return credentials

    if on_message:
        if detail.startswith("network_error"):
            on_message(
                "Could not reach OpenRouter to validate your key. "
                f"({detail})"
            )
        else:
            on_message("API key validation failed.")
    return None
