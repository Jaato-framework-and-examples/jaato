"""Authentication module for NVIDIA NIM.

NIM uses API key authentication (nvapi-... keys from build.nvidia.com).
This module provides:
- Secure local storage of API keys
- Key validation against the NIM endpoint
- Status checking

API keys are obtained from:
  https://build.nvidia.com/ → Settings → API Keys

Storage follows jaato convention:
1. Project .jaato/nim_auth.json (project-specific)
2. Home ~/.jaato/nim_auth.json (user-level default)
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .env import DEFAULT_BASE_URL


@dataclass
class NIMCredentials:
    """Stored NIM credentials."""
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
    def from_dict(cls, data: dict) -> "NIMCredentials":
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
    workspace = os.environ.get("JAATO_WORKSPACE_ROOT") or os.getcwd()
    project_path = Path(workspace) / ".jaato" / "nim_auth.json"
    home_path = Path.home() / ".jaato" / "nim_auth.json"

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


def save_credentials(credentials: NIMCredentials) -> None:
    """Save credentials to persistent storage."""
    path = _get_token_storage_path(for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(credentials.to_dict(), f, indent=2)

    # Secure permissions on Unix
    if os.name == "posix":
        os.chmod(path, 0o600)


def load_credentials() -> Optional[NIMCredentials]:
    """Load credentials from persistent storage."""
    path = _get_token_storage_path()

    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        return NIMCredentials.from_dict(data)
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


def get_credential_file_path() -> Optional[str]:
    """Return the path of the credential file that would be loaded.

    Used by the provider to report which credential source was used
    in the "Connected to" message. Returns the resolved path of the
    first existing credential file, or None if no file exists.

    Returns:
        String path like ``"~/.jaato/nim_auth.json"`` or
        ``".jaato/nim_auth.json"``, or None.
    """
    path = _get_token_storage_path()
    if not path.exists():
        return None
    home = Path.home()
    if path.is_relative_to(home):
        return "~/" + str(path.relative_to(home))
    return str(path)


def get_stored_base_url() -> Optional[str]:
    """Get stored custom base URL if available.

    Returns:
        Base URL string, or None if not stored.
    """
    creds = load_credentials()
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


def validate_api_key(
    api_key: str,
    base_url: Optional[str] = None,
) -> tuple:
    """Validate an API key by making a test request.

    Sends a minimal POST to the OpenAI-compatible ``/chat/completions``
    endpoint. Uses the project's httpx client with full proxy, Kerberos,
    and corporate CA bundle support.

    Args:
        api_key: NIM API key to validate (nvapi-...).
        base_url: Optional custom base URL (default: NVIDIA hosted API).

    Returns:
        A ``(valid, error_detail)`` tuple. ``valid`` is True when the key
        is accepted. ``error_detail`` is a human-readable hint when
        ``valid`` is False (empty string on success).
    """
    import httpx

    url = base_url or DEFAULT_BASE_URL
    test_url = f"{url.rstrip('/')}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Minimal payload — we only care about whether the key is accepted
    body = {
        "model": "meta/llama-3.1-8b-instruct",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }

    try:
        client = _create_validation_client()
        response = client.post(test_url, headers=headers, json=body, timeout=30)
        if response.status_code in (401, 403):
            return (False, "authentication_error")
        # Any other status (200, 400, etc.) means the key was accepted
        return (True, "")
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (401, 403):
            return (False, "authentication_error")
        return (True, "")
    except Exception as e:
        return (False, f"network_error: {e}")


def login_with_key(
    api_key: str,
    base_url: Optional[str] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> Optional[NIMCredentials]:
    """Login with a provided API key (non-interactive).

    Args:
        api_key: NIM API key (nvapi-...).
        base_url: Optional custom base URL.
        on_message: Callback for status messages.

    Returns:
        NIMCredentials if successful, None if validation failed.
    """
    if on_message:
        on_message("Validating API key...")

    valid, detail = validate_api_key(api_key, base_url)
    if valid:
        credentials = NIMCredentials(
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
                    "Could not reach the NIM API to validate your key. "
                    f"({detail})"
                )
            else:
                on_message("API key validation failed.")
        return None
