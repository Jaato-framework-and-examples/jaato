"""Authentication module for the Xiaomi MiMo API.

Xiaomi MiMo uses API key (Bearer token) authentication.  This module provides:
- Secure local storage of API keys
- Key validation against the authenticated ``GET /v1/models`` catalog
- Status checking

API keys are generated at https://platform.xiaomimimo.com/#/console/api-keys

Storage follows jaato convention:
1. Project .jaato/mimo_auth.json (project-specific)
2. Home ~/.jaato/mimo_auth.json (user-level default)
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from shared.session_context import get_workspace_root, get_config_root
from shared.secret_repr import secret_safe_repr

from .env import DEFAULT_BASE_URL

logger = logging.getLogger(__name__)

_FILENAME = "mimo_auth.json"


@dataclass
class MiMoCredentials:
    """Stored Xiaomi MiMo credentials."""

    api_key: str
    created_at: float
    base_url: Optional[str] = None

    __repr__ = secret_safe_repr("api_key")

    def to_dict(self) -> dict:
        data = {"api_key": self.api_key, "created_at": self.created_at}
        if self.base_url:
            data["base_url"] = self.base_url
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MiMoCredentials":
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
    1. Project tier — ``<config_root>/mimo_auth.json`` when
       ``config_root`` is set, else ``<workspace>/.jaato/mimo_auth.json``.
    2. Home tier — ``~/.jaato/mimo_auth.json``.

    Uses JAATO_WORKSPACE_ROOT env var if set (for subagents), otherwise
    Path.cwd().  Uses JAATO_CONFIG_ROOT when ``config_root`` is unset, so
    sessions with a session-level config-root override route credential
    reads to the same out-of-tree path as the rest of the framework config.
    """
    workspace = workspace_path or get_workspace_root() or os.getcwd()
    effective_config_root = config_root or get_config_root()
    if effective_config_root:
        project_path = Path(effective_config_root).expanduser().resolve() / _FILENAME
    else:
        project_path = Path(workspace) / ".jaato" / _FILENAME
    home_path = Path.home() / ".jaato" / _FILENAME

    if for_write:
        return project_path if project_path.parent.exists() else home_path
    if project_path.exists():
        return project_path
    return home_path


def save_credentials(credentials: MiMoCredentials, workspace_path: Optional[str] = None) -> None:
    """Save credentials to persistent storage (0600 on Unix)."""
    path = _get_token_storage_path(for_write=True, workspace_path=workspace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(credentials.to_dict(), f, indent=2)
    if os.name == "posix":
        os.chmod(path, 0o600)


def load_credentials(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[MiMoCredentials]:
    """Load credentials; None when absent OR unreadable (reason logged)."""
    creds, _ = try_load_credentials_with_reason(
        workspace_path=workspace_path, config_root=config_root,
    )
    return creds


def try_load_credentials_with_reason(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Tuple[Optional[MiMoCredentials], Optional[str]]:
    """Load credentials and return a reason string when the load fails.

    Returns ``(credentials, reason)``:

    - ``(MiMoCredentials, None)`` — file loaded successfully.
    - ``(None, None)`` — no credential file exists.
    - ``(None, "<reason>")`` — file exists but could not be loaded.

    Lets ``verify_auth`` distinguish "not configured" from "configured
    but broken".
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
        logger.warning("Failed to read Xiaomi MiMo credentials: %s", reason)
        return None, reason
    except json.JSONDecodeError as exc:
        reason = f"invalid JSON at {path}: {exc.msg} (line {exc.lineno}, col {exc.colno})"
        logger.warning("Failed to parse Xiaomi MiMo credentials: %s", reason)
        return None, reason
    try:
        return MiMoCredentials.from_dict(data), None
    except (KeyError, TypeError) as exc:
        reason = f"malformed credentials at {path}: missing or invalid field ({exc})"
        logger.warning("Malformed Xiaomi MiMo credentials: %s", reason)
        return None, reason
    except Exception as exc:  # defensive — don't mask unexpected failures
        reason = f"unexpected error loading {path}: {exc.__class__.__name__}: {exc}"
        logger.warning("Unexpected error loading Xiaomi MiMo credentials: %s", reason)
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
    """Stored API key, or None."""
    creds = load_credentials(workspace_path=workspace_path, config_root=config_root)
    return creds.api_key if creds else None


def get_credential_file_path(
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Optional[str]:
    """Path of the credential file that would be loaded, or None.

    Used by the provider to report which credential source was used.
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
    """Stored custom base URL, or None."""
    creds = load_credentials(workspace_path=workspace_path, config_root=config_root)
    return creds.base_url if creds else None


def _create_validation_client():
    """httpx client with proxy, Kerberos and CA-bundle support."""
    from shared.ssl_helper import active_cert_bundle
    from shared.http.proxy import get_httpx_client

    kwargs = {}
    ca_bundle = active_cert_bundle()
    if ca_bundle:
        kwargs["verify"] = ca_bundle
    return get_httpx_client(**kwargs)


def _extract_body_snippet(response, limit: int = 300) -> str:
    """Short, safe snippet of a response body for error detail."""
    try:
        text = response.text or ""
    except Exception:
        return ""
    text = text.strip().replace("\n", " ")
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def _classify_status(status: int, snippet: str) -> tuple:
    """``(valid, detail)`` for one HTTP status of the validation probe."""
    if 200 <= status < 300:
        return (True, "")
    if status == 401:
        return (False, f"authentication_error: {status}: {snippet}")
    if status == 403:
        return (False, f"forbidden: {status}: {snippet}")
    if status == 429:
        return (False, f"rate_limit: {status}: {snippet}")
    if status == 402:
        return (False, f"payment_required: {status}: {snippet}")
    if 500 <= status < 600:
        return (False, f"server_error: {status}: {snippet}")
    return (False, f"http_error: {status}: {snippet}")


def validate_api_key(
    api_key: str,
    base_url: Optional[str] = None,
) -> tuple:
    """Validate an API key with a cheap authenticated GET.

    Probes the authenticated ``GET /v1/models`` catalog — no tokens are billed and the answer proves the
    key authenticates.  Uses the project's httpx client with proxy,
    Kerberos and corporate CA-bundle support.

    Returns:
        ``(valid, detail)``.  ``valid`` is True only on a 2xx.  ``detail``
        carries a structured code:

        - ``""`` — key is valid.
        - ``"authentication_error: <status>: <body>"`` — key rejected (401).
        - ``"forbidden: <status>: <body>"`` — key or region refused (403).
        - ``"rate_limit: <status>: <body>"`` — rate limited (429).
        - ``"payment_required: <status>: <body>"`` — balance exhausted (402).
        - ``"server_error: <status>: <body>"`` — 5xx.
        - ``"http_error: <status>: <body>"`` — any other status.
        - ``"network_error: <details>"`` — the request never arrived.
    """
    import httpx

    url = base_url or DEFAULT_BASE_URL
    test_url = f"{url.rstrip('/')}/models"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        client = _create_validation_client()
        response = client.get(test_url, headers=headers, timeout=30)
    except httpx.HTTPStatusError as e:
        status = getattr(e.response, "status_code", None) or 0
        snippet = _extract_body_snippet(e.response) if e.response is not None else ""
        return _classify_status(status, snippet)
    except Exception as e:
        return (False, f"network_error: {e}")
    return _classify_status(response.status_code, _extract_body_snippet(response))


def login_with_key(
    api_key: str,
    base_url: Optional[str] = None,
    on_message: Optional[Callable[[str], None]] = None,
    workspace_path: Optional[str] = None,
) -> Optional[MiMoCredentials]:
    """Validate ``api_key`` and store it (non-interactive).

    Returns:
        MiMoCredentials if successful, None if validation failed.
    """
    if on_message:
        on_message("Validating API key...")
    valid, detail = validate_api_key(api_key, base_url)
    if not valid:
        if on_message:
            if detail.startswith("network_error"):
                on_message(f"Could not reach the Xiaomi MiMo API to validate your key. ({detail})")
            else:
                on_message("API key validation failed.")
        return None
    credentials = MiMoCredentials(
        api_key=api_key, created_at=time.time(), base_url=base_url,
    )
    save_credentials(credentials, workspace_path=workspace_path)
    if on_message:
        on_message("API key validated and saved.")
    return credentials
