"""Authentication module for Zhipu AI (Z.AI).

Z.AI uses API key authentication. This module provides:
- Interactive API key collection and secure storage
- Token validation
- Status checking

API keys are obtained from:
- International: https://z.ai/model-api
- China: https://open.bigmodel.cn/

Storage follows jaato convention:
1. Project .jaato/zhipuai_auth.json (project-specific)
2. Home ~/.jaato/zhipuai_auth.json (user-level default)
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

from .env import DEFAULT_ZHIPUAI_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class ZhipuAICredentials:
    """Stored Z.AI credentials."""
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
    def from_dict(cls, data: dict) -> "ZhipuAICredentials":
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
    1. Project tier — ``<config_root>/zhipuai_auth.json`` when
       ``config_root`` is set, else
       ``<workspace>/.jaato/zhipuai_auth.json``.  Project-specific auth.
    2. Home tier — ``~/.jaato/zhipuai_auth.json``.  User-level default.

    Uses JAATO_WORKSPACE_ROOT env var if set (for subagents), otherwise Path.cwd().
    Uses JAATO_CONFIG_ROOT env var as a fallback for ``config_root``,
    so subagents and long-lived plugins inherit the override that
    ``JaatoServer`` exported on session activation.

    Args:
        for_write: If True, returns the path to write to (prefers project dir
                   if it exists, otherwise home). If False, returns the first
                   existing file or the default write location.
        workspace_path: Optional explicit workspace path override.
        config_root: Optional explicit read-only-config root override.
            When set, replaces the workspace-anchored project tier.
            See :mod:`shared.config_resolver`.

    Returns:
        Path to credentials storage file.
    """
    # Use explicit workspace path if set (thread-safe for subagents)
    workspace = workspace_path or get_workspace_root() or os.getcwd()
    effective_config_root = config_root or get_config_root()
    if effective_config_root:
        project_path = Path(effective_config_root).expanduser().resolve() / "zhipuai_auth.json"
    else:
        project_path = Path(workspace) / ".jaato" / "zhipuai_auth.json"
    home_path = Path.home() / ".jaato" / "zhipuai_auth.json"

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


def save_credentials(credentials: ZhipuAICredentials, workspace_path: Optional[str] = None) -> None:
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
) -> Optional[ZhipuAICredentials]:
    """Load credentials from persistent storage.

    Returns None if the file is absent **or** if it cannot be parsed.  When
    the file exists but fails to load (corrupt JSON, missing ``api_key``
    field, permission error, etc.) the failure is logged at WARNING so it
    is visible in the provider trace log instead of being silently
    swallowed.  Callers that need to surface the specific failure reason
    should use :func:`try_load_credentials_with_reason` instead.

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
) -> Tuple[Optional[ZhipuAICredentials], Optional[str]]:
    """Load credentials and return a reason string when the load fails.

    Unlike :func:`load_credentials`, this does not collapse errors to
    ``None``.  It returns a ``(credentials, reason)`` tuple:

    - ``(ZhipuAICredentials, None)`` — file loaded successfully.
    - ``(None, None)`` — no credential file exists.
    - ``(None, "<reason>")`` — file exists but could not be loaded.
      ``reason`` is a short human-readable string such as
      ``"invalid JSON at <path>: ..."`` or ``"missing 'api_key' field in <path>"``.

    This helper lets ``verify_auth`` and the status command distinguish
    "not configured" from "configured but broken" and surface the actual
    error instead of reporting "No credentials found".
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
        logger.warning("Failed to read Zhipu AI credentials: %s", reason)
        return None, reason
    except json.JSONDecodeError as exc:
        reason = f"invalid JSON at {path}: {exc.msg} (line {exc.lineno}, col {exc.colno})"
        logger.warning("Failed to parse Zhipu AI credentials: %s", reason)
        return None, reason

    try:
        return ZhipuAICredentials.from_dict(data), None
    except (KeyError, TypeError) as exc:
        reason = f"malformed credentials at {path}: missing or invalid field ({exc})"
        logger.warning("Malformed Zhipu AI credentials: %s", reason)
        return None, reason
    except Exception as exc:  # defensive — don't mask unexpected failures
        reason = f"unexpected error loading {path}: {exc.__class__.__name__}: {exc}"
        logger.warning("Unexpected error loading Zhipu AI credentials: %s", reason)
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
        String path like ``"~/.jaato/zhipuai_auth.json"`` or
        ``".jaato/zhipuai_auth.json"``, or None.
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

    Uses the same pattern as the Anthropic provider's ``_create_http_client()``
    so validation requests go through corporate proxies correctly.
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

    Sends a minimal POST to the Anthropic-compatible ``/v1/messages``
    endpoint.  Uses the project's httpx client with full proxy, Kerberos,
    and corporate CA bundle support.

    Args:
        api_key: Z.AI API key to validate.
        base_url: Optional custom base URL (same format as
            ``DEFAULT_ZHIPUAI_BASE_URL``, e.g.
            ``https://api.z.ai/api/anthropic``).

    Returns:
        A ``(valid, detail)`` tuple.  ``valid`` is True when the test
        request authenticates successfully (2xx or a 400 "bad request"
        that still went through auth).  ``detail`` carries a structured
        hint so callers can report the real reason for failure or
        partial success instead of silently treating provider errors as
        "key accepted":

        - ``""`` — key is valid (success path).
        - ``"authentication_error: <HTTP status>: <body>"`` — key rejected
          by the provider (401/403).
        - ``"rate_limit: <HTTP status>: <body>"`` — quota or rate limit
          exceeded (429).  **Key was not saved**; the provider refused the
          request before authenticating.
        - ``"payment_required: <HTTP status>: <body>"`` — billing/quota
          exhausted (402).  Key was not saved.
        - ``"server_error: <HTTP status>: <body>"`` — provider is
          temporarily unavailable (5xx).  Caller should decide whether
          to retry.
        - ``"http_error: <HTTP status>: <body>"`` — any other unexpected
          status the caller should surface verbatim.
        - ``"network_error: <details>"`` — request never reached the
          provider (connection refused, timeout, SSL handshake failure).
    """
    import httpx

    url = base_url or DEFAULT_ZHIPUAI_BASE_URL
    test_url = f"{url.rstrip('/')}/v1/messages"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    body = {
        "model": "glm-4.7",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }

    try:
        client = _create_validation_client()
        response = client.post(test_url, headers=headers, json=body, timeout=30)
    except httpx.HTTPStatusError as e:
        # Rare for direct .post() (httpx does not raise_for_status
        # automatically), but keep the branch for defensive parity.
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
    if 200 <= status < 300 or status == 400:
        # 2xx succeeded outright; 400 means the request reached the model
        # after auth (e.g. "model not found" for the test payload), so
        # the key itself is accepted.
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


def login_interactive(
    on_message: Optional[Callable[[str], None]] = None,
    on_input: Optional[Callable[[str], str]] = None,
) -> Optional[ZhipuAICredentials]:
    """Run interactive login flow.

    Prompts user for API key and validates it.

    Args:
        on_message: Callback for status messages.
        on_input: Callback to get user input. If None, uses builtin input().

    Returns:
        ZhipuAICredentials if successful, None if cancelled or failed.
    """
    if on_message:
        on_message("Zhipu AI (Z.AI) Authentication")
        on_message("")
        on_message("Get your API key from:")
        on_message("  International: https://z.ai/model-api")
        on_message("  China: https://open.bigmodel.cn/")
        on_message("")

    # Get API key from user
    prompt = "Enter your Z.AI API key: "
    if on_input:
        api_key = on_input(prompt)
    else:
        api_key = input(prompt)

    if not api_key or not api_key.strip():
        if on_message:
            on_message("No API key provided. Login cancelled.")
        return None

    api_key = api_key.strip()

    # Optional: ask for custom base URL
    if on_message:
        on_message("")
        on_message("Base URL (press Enter for default):")
        on_message(f"  Default: {DEFAULT_ZHIPUAI_BASE_URL}")

    base_url_prompt = "Custom base URL (or Enter to skip): "
    if on_input:
        base_url = on_input(base_url_prompt)
    else:
        base_url = input(base_url_prompt)

    base_url = base_url.strip() if base_url else None

    # Validate the API key
    if on_message:
        on_message("")
        on_message("Validating API key...")

    valid, detail = validate_api_key(api_key, base_url)
    if valid:
        credentials = ZhipuAICredentials(
            api_key=api_key,
            created_at=time.time(),
            base_url=base_url,
        )
        save_credentials(credentials)

        if on_message:
            on_message("API key validated and saved.")
            on_message(f"Credentials stored at: {_get_token_storage_path(for_write=True)}")

        return credentials
    else:
        if on_message:
            if detail.startswith("network_error"):
                on_message(
                    "Could not reach the Z.AI API to validate your key. "
                    f"({detail})"
                )
            else:
                on_message("API key validation failed. Please check your key and try again.")
        return None


def login_with_key(
    api_key: str,
    base_url: Optional[str] = None,
    on_message: Optional[Callable[[str], None]] = None,
    workspace_path: Optional[str] = None,
) -> Optional[ZhipuAICredentials]:
    """Login with a provided API key (non-interactive).

    Args:
        api_key: Z.AI API key.
        base_url: Optional custom base URL.
        on_message: Callback for status messages.

    Returns:
        ZhipuAICredentials if successful, None if validation failed.
    """
    if on_message:
        on_message("Validating API key...")

    valid, detail = validate_api_key(api_key, base_url)
    if valid:
        credentials = ZhipuAICredentials(
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
                    "Could not reach the Z.AI API to validate your key. "
                    f"({detail})"
                )
            else:
                on_message("API key validation failed.")
        return None


def logout(on_message: Optional[Callable[[str], None]] = None) -> None:
    """Clear stored credentials.

    Args:
        on_message: Callback for status messages.
    """
    path = _get_token_storage_path()
    if path.exists():
        clear_credentials()
        if on_message:
            on_message("Z.AI credentials cleared.")
    else:
        if on_message:
            on_message("No stored credentials found.")


def status(on_message: Optional[Callable[[str], None]] = None) -> bool:
    """Check authentication status.

    Args:
        on_message: Callback for status messages.

    Returns:
        True if valid credentials are stored.
    """
    creds = load_credentials()

    if not creds:
        if on_message:
            on_message("Not logged in to Z.AI.")
            on_message("Run 'zhipuai-auth login' to authenticate.")
        return False

    # Mask API key for display
    masked_key = creds.api_key[:8] + "..." + creds.api_key[-4:] if len(creds.api_key) > 12 else "***"

    if on_message:
        on_message("Z.AI Authentication Status:")
        on_message(f"  API Key: {masked_key}")
        if creds.base_url:
            on_message(f"  Base URL: {creds.base_url}")
        else:
            on_message(f"  Base URL: {DEFAULT_ZHIPUAI_BASE_URL} (default)")

        # Show when credentials were saved
        from datetime import datetime
        saved_at = datetime.fromtimestamp(creds.created_at).strftime("%Y-%m-%d %H:%M:%S")
        on_message(f"  Saved: {saved_at}")

    return True
