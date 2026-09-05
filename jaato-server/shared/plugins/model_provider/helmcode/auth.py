"""Authentication module for the Helmcode API.

Helmcode uses API key (Bearer token) authentication.
This module provides:
- Secure local storage of API keys
- Key validation against the inference API
- Status checking

API keys are issued **per workspace** from the Helmcode dashboard
(API Keys -> Create key).  They belong to the organisation rather than to
an individual, and how many can exist at once depends on the plan.

Storage follows jaato convention:
1. Project .jaato/helmcode_auth.json (project-specific)
2. Home ~/.jaato/helmcode_auth.json (user-level default)
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
class HelmcodeCredentials:
    """Stored Helmcode credentials."""
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
    def from_dict(cls, data: dict) -> "HelmcodeCredentials":
        return cls(
            api_key=data["api_key"],
            created_at=data.get("created_at", time.time()),
            base_url=data.get("base_url"),
        )


def _get_token_storage_path(
    for_write: bool = False,
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Path:
    """Get path to credentials storage file.

    Follows jaato convention:
    1. Project tier — ``<config_root>/helmcode_auth.json`` when
       ``config_root`` is set, else ``<workspace>/.jaato/helmcode_auth.json``.
    2. Home tier — ``~/.jaato/helmcode_auth.json``.

    Uses JAATO_WORKSPACE_ROOT env var if set (for subagents), otherwise Path.cwd().
    Uses JAATO_CONFIG_ROOT env var when ``config_root`` is unset, so
    sessions with a session-level config-root override (exported by
    :meth:`server.core.JaatoServer._in_workspace`) route credential
    reads to the same out-of-tree path as the rest of the framework
    config.

    Args:
        for_write: If True, returns the path to write to.
        workspace_path: Optional explicit workspace path override.
        config_root: Optional explicit read-only-config root override.

    Returns:
        Path to credentials storage file.
    """
    workspace = workspace_path or get_workspace_root() or os.getcwd()
    effective_config_root = config_root or get_config_root()
    if effective_config_root:
        project_path = Path(effective_config_root).expanduser().resolve() / "helmcode_auth.json"
    else:
        project_path = Path(workspace) / ".jaato" / "helmcode_auth.json"
    home_path = Path.home() / ".jaato" / "helmcode_auth.json"

    if for_write:
        if project_path.parent.exists():
            return project_path
        return home_path
    else:
        if project_path.exists():
            return project_path
        if home_path.exists():
            return home_path
        return home_path


def save_credentials(credentials: HelmcodeCredentials, workspace_path: Optional[str] = None) -> None:
    """Save credentials to persistent storage."""
    path = _get_token_storage_path(for_write=True, workspace_path=workspace_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(credentials.to_dict(), f, indent=2)

    # Secure permissions on Unix
    if os.name == "posix":
        os.chmod(path, 0o600)


def load_credentials(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[HelmcodeCredentials]:
    """Load credentials from persistent storage.

    Returns None if the file is absent **or** fails to parse.  A broken
    file (corrupt JSON, missing ``api_key``, permission error) is logged
    at WARNING so it is visible in the provider trace log instead of
    being silently swallowed.  Callers that need to surface the actual
    reason should use :func:`try_load_credentials_with_reason`.

    See :func:`_get_token_storage_path` for the resolver chain;
    ``config_root`` overrides the workspace tier when set.
    """
    creds, _ = try_load_credentials_with_reason(
        workspace_path=workspace_path, config_root=config_root,
    )
    return creds


def try_load_credentials_with_reason(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Tuple[Optional[HelmcodeCredentials], Optional[str]]:
    """Load credentials and return a reason string when the load fails.

    Returns ``(credentials, reason)``:

    - ``(HelmcodeCredentials, None)`` — file loaded successfully.
    - ``(None, None)`` — no credential file exists.
    - ``(None, "<reason>")`` — file exists but could not be loaded.

    Lets ``verify_auth`` distinguish "not configured" from "configured
    but broken" and surface the specific failure (e.g. invalid JSON,
    missing field, permission error) instead of reporting "No API key
    found" for both.
    """
    path = _get_token_storage_path(
        workspace_path=workspace_path, config_root=config_root,
    )

    if not path.exists():
        return None, None

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, PermissionError) as exc:
        reason = f"cannot read {path}: {exc}"
        logger.warning("Failed to read Helmcode credentials: %s", reason)
        return None, reason
    except json.JSONDecodeError as exc:
        reason = f"invalid JSON at {path}: {exc.msg} (line {exc.lineno}, col {exc.colno})"
        logger.warning("Failed to parse Helmcode credentials: %s", reason)
        return None, reason

    try:
        return HelmcodeCredentials.from_dict(data), None
    except (KeyError, TypeError) as exc:
        reason = f"malformed credentials at {path}: missing or invalid field ({exc})"
        logger.warning("Malformed Helmcode credentials: %s", reason)
        return None, reason
    except Exception as exc:  # defensive — don't mask unexpected failures
        reason = f"unexpected error loading {path}: {exc.__class__.__name__}: {exc}"
        logger.warning("Unexpected error loading Helmcode credentials: %s", reason)
        return None, reason


def clear_credentials(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> None:
    """Clear stored credentials."""
    path = _get_token_storage_path(
        workspace_path=workspace_path, config_root=config_root,
    )
    if path.exists():
        path.unlink()


def get_stored_api_key(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Get stored API key if available.

    Returns:
        API key string, or None if not stored.
    """
    creds = load_credentials(
        workspace_path=workspace_path, config_root=config_root,
    )
    if creds:
        return creds.api_key
    return None


def get_credential_file_path(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Return the path of the credential file that would be loaded.

    Used by the provider to report which credential source was used
    in the "Connected to" message. Returns the resolved path of the
    first existing credential file, or None if no file exists.

    Returns:
        String path like ``"~/.jaato/helmcode_auth.json"`` or
        ``".jaato/helmcode_auth.json"``, or None.
    """
    path = _get_token_storage_path(
        workspace_path=workspace_path, config_root=config_root,
    )
    if not path.exists():
        return None
    home = Path.home()
    if path.is_relative_to(home):
        return "~/" + str(path.relative_to(home))
    return str(path)


def get_stored_base_url(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Get stored custom base URL if available.

    Returns:
        Base URL string, or None if not stored.
    """
    creds = load_credentials(
        workspace_path=workspace_path, config_root=config_root,
    )
    if creds:
        return creds.base_url
    return None


def _create_validation_client():
    """Create an httpx client with proxy, Kerberos, and CA bundle support.

    Uses the same pattern as other providers so validation requests
    go through corporate proxies correctly.
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
    """Validate an API key by making a test request.

    Sends a ``GET`` to the OpenAI-compatible ``/models`` endpoint rather
    than a chat completion.  Helmcode requires auth on the catalog —
    verified live: an unkeyed ``GET /v1/models`` answers ``401``
    ``{"error": {"type": "auth_error", ...}}`` — so reaching a ``200``
    proves the key, while costing no tokens and naming no probe model
    that could later rotate out of the catalogue.  Uses the project's
    httpx client with full proxy, Kerberos, and corporate CA bundle
    support.

    Args:
        api_key: Helmcode API key to validate.
        base_url: Optional custom base URL (default: the hosted gateway).

    Returns:
        A ``(valid, detail)`` tuple.  ``valid`` is True only when the
        request authenticates: 2xx, or 403 (the key is genuine but not
        entitled to list the catalogue — Helmcode rejects a bad key with
        401 before any entitlement check, so a 403 still proves it).
        Other non-auth errors do not masquerade as success.  ``detail``
        carries a structured code:

        - ``""`` — key is valid.
        - ``"authentication_error: <status>: <body>"`` — key rejected
          (401).
        - ``"rate_limit: <status>: <body>"`` — rate limit exceeded
          (429).  Key was NOT saved.
        - ``"payment_required: <status>: <body>"`` — prepaid credit
          exhausted (402).  Key was NOT saved.
        - ``"server_error: <status>: <body>"`` — the API is
          temporarily unavailable (5xx).
        - ``"http_error: <status>: <body>"`` — any other unexpected
          status.
        - ``"network_error: <details>"`` — request never reached
          Helmcode.
    """
    import httpx

    url = base_url or DEFAULT_BASE_URL
    test_url = f"{url.rstrip('/')}/models"

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    try:
        client = _create_validation_client()
        response = client.get(test_url, headers=headers, timeout=30)
    except httpx.HTTPStatusError as e:
        status = getattr(e.response, "status_code", None)
        snippet = _extract_body_snippet(e.response) if e.response is not None else ""
        if status == 401:
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
    if 200 <= status < 300 or status == 403:
        return (True, "")
    snippet = _extract_body_snippet(response)
    if status == 401:
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
) -> Optional[HelmcodeCredentials]:
    """Login with a provided API key (non-interactive).

    Args:
        api_key: Helmcode API key.
        base_url: Optional custom base URL.
        on_message: Callback for status messages.

    Returns:
        HelmcodeCredentials if successful, None if validation failed.
    """
    if on_message:
        on_message("Validating API key...")

    valid, detail = validate_api_key(api_key, base_url)
    if valid:
        credentials = HelmcodeCredentials(
            api_key=api_key,
            created_at=time.time(),
            base_url=base_url,
        )
        save_credentials(credentials, workspace_path=workspace_path)

        if on_message:
            on_message("API key validated and saved.")

        return credentials
    else:
        if on_message:
            if detail.startswith("network_error"):
                on_message(
                    "Could not reach the Helmcode API to "
                    f"validate your key. ({detail})"
                )
            else:
                on_message("API key validation failed.")
        return None
