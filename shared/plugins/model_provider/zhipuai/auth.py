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
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .env import DEFAULT_ZHIPUAI_BASE_URL


@dataclass
class ZhipuAICredentials:
    """Stored Z.AI credentials."""
    api_key: str
    created_at: float  # Unix timestamp
    base_url: Optional[str] = None  # Optional custom base URL

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


def _get_token_storage_path(for_write: bool = False) -> Path:
    """Get path to credentials storage file.

    Follows jaato convention:
    1. Project .jaato/ first (project-specific auth)
    2. Home ~/.jaato/ second (user-level default)

    Uses JAATO_WORKSPACE_ROOT env var if set (for subagents), otherwise Path.cwd().

    Args:
        for_write: If True, returns the path to write to (prefers project dir
                   if it exists, otherwise home). If False, returns the first
                   existing file or the default write location.

    Returns:
        Path to credentials storage file.
    """
    # Use explicit workspace path if set (thread-safe for subagents)
    workspace = os.environ.get("JAATO_WORKSPACE_ROOT") or os.getcwd()
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


def save_credentials(credentials: ZhipuAICredentials) -> None:
    """Save credentials to persistent storage."""
    path = _get_token_storage_path(for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(credentials.to_dict(), f, indent=2)

    # Secure permissions on Unix
    if os.name == "posix":
        os.chmod(path, 0o600)


def load_credentials() -> Optional[ZhipuAICredentials]:
    """Load credentials from persistent storage."""
    path = _get_token_storage_path()

    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        return ZhipuAICredentials.from_dict(data)
    except Exception:
        return None


def clear_credentials() -> None:
    """Clear stored credentials."""
    path = _get_token_storage_path()
    if path.exists():
        path.unlink()


def get_stored_api_key() -> Optional[str]:
    """Get stored API key if available.

    Returns:
        API key string, or None if not stored.
    """
    creds = load_credentials()
    if creds:
        return creds.api_key
    return None


def get_stored_base_url() -> Optional[str]:
    """Get stored custom base URL if available.

    Returns:
        Base URL string, or None if not stored.
    """
    creds = load_credentials()
    if creds:
        return creds.base_url
    return None


def validate_api_key(
    api_key: str,
    base_url: Optional[str] = None,
) -> tuple:
    """Validate an API key by making a test request.

    Sends a minimal POST to the Anthropic-compatible ``/v1/messages``
    endpoint.  The Anthropic SDK appends ``/v1/messages`` to the base URL
    internally, so this function must do the same when using raw urllib.

    Args:
        api_key: Z.AI API key to validate.
        base_url: Optional custom base URL (same format as
            ``DEFAULT_ZHIPUAI_BASE_URL``, e.g.
            ``https://api.z.ai/api/anthropic``).

    Returns:
        A ``(valid, error_detail)`` tuple.  ``valid`` is True when the key
        is accepted.  ``error_detail`` is a human-readable hint when
        ``valid`` is False (empty string on success).
    """
    url = base_url or DEFAULT_ZHIPUAI_BASE_URL
    # The Anthropic SDK appends /v1/messages to the base URL, so we
    # must do the same when validating with raw urllib.
    test_url = f"{url.rstrip('/')}/v1/messages"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    # Minimal request body
    body = json.dumps({
        "model": "glm-4.7",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }).encode("utf-8")

    req = urllib.request.Request(
        test_url,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return (True, "")
    except urllib.error.HTTPError as e:
        # 401/403 means invalid key
        if e.code in (401, 403):
            return (False, "authentication_error")
        # Other HTTP errors (400 for bad request, etc.) indicate the key
        # was accepted but the minimal test request was rejected — that's
        # fine, the key itself is valid.
        return (True, "")
    except urllib.error.URLError as e:
        # DNS / connection-refused / SSL errors
        reason = str(e.reason) if hasattr(e, "reason") else str(e)
        return (False, f"network_error: {reason}")
    except Exception as e:
        # Unexpected errors (timeout, etc.)
        return (False, f"network_error: {e}")


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
        save_credentials(credentials)

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
