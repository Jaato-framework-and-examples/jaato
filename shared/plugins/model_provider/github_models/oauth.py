"""GitHub Device Code OAuth flow for GitHub Copilot/Models.

Implements the OAuth 2.0 Device Authorization Grant (RFC 8628) for GitHub.
This flow is designed for CLI tools and devices that cannot open a browser
callback URL directly.

Flow:
1. Request device and user codes from GitHub
2. Display user code and verification URL to user
3. Poll token endpoint until user completes authorization
4. Store and refresh tokens as needed

Reference: https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps
"""

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

# OAuth configuration for GitHub Copilot
# This is the public client ID used by GitHub Copilot extensions
OAUTH_CLIENT_ID = "Iv1.b507a08c87ecfe98"

# GitHub OAuth endpoints
DEVICE_CODE_URL = "https://github.com/login/device/code"
TOKEN_URL = "https://github.com/login/oauth/access_token"
VERIFICATION_URL = "https://github.com/login/device"

# Scopes for GitHub Models access
# read:user is minimal for authentication
OAUTH_SCOPES = "read:user"

# Default polling interval (seconds)
DEFAULT_POLL_INTERVAL = 5

# Device code expiration (typically 15 minutes)
DEVICE_CODE_EXPIRES_IN = 900


@dataclass
class OAuthTokens:
    """OAuth token set with metadata."""
    access_token: str
    token_type: str = "bearer"
    scope: str = ""

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OAuthTokens":
        return cls(
            access_token=data["access_token"],
            token_type=data.get("token_type", "bearer"),
            scope=data.get("scope", ""),
        )


@dataclass
class DeviceCodeResponse:
    """Response from device code request."""
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int

    @classmethod
    def from_dict(cls, data: dict) -> "DeviceCodeResponse":
        return cls(
            device_code=data["device_code"],
            user_code=data["user_code"],
            verification_uri=data.get("verification_uri", VERIFICATION_URL),
            expires_in=data.get("expires_in", DEVICE_CODE_EXPIRES_IN),
            interval=data.get("interval", DEFAULT_POLL_INTERVAL),
        )


def _make_request(
    url: str,
    data: dict,
    headers: Optional[dict] = None,
) -> dict:
    """Make HTTP POST request and return JSON response.

    Args:
        url: Request URL.
        data: Form data to POST.
        headers: Optional additional headers.

    Returns:
        Parsed JSON response.

    Raises:
        RuntimeError: If request fails.
    """
    default_headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "jaato/1.0",
    }
    if headers:
        default_headers.update(headers)

    encoded_data = urllib.parse.urlencode(data).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=encoded_data,
        headers=default_headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else str(e)
        try:
            error_data = json.loads(error_body)
            error_msg = error_data.get("error_description") or error_data.get("error") or error_body
        except json.JSONDecodeError:
            error_msg = error_body
        raise RuntimeError(f"HTTP {e.code}: {error_msg}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Request failed: {e.reason}") from e


def request_device_code(
    client_id: str = OAUTH_CLIENT_ID,
    scope: str = OAUTH_SCOPES,
) -> DeviceCodeResponse:
    """Request device and user codes from GitHub.

    Args:
        client_id: OAuth client ID.
        scope: OAuth scopes to request.

    Returns:
        DeviceCodeResponse with codes and verification URL.

    Raises:
        RuntimeError: If request fails.
    """
    data = {
        "client_id": client_id,
        "scope": scope,
    }

    response = _make_request(DEVICE_CODE_URL, data)

    if "error" in response:
        error_msg = response.get("error_description") or response.get("error")
        raise RuntimeError(f"Device code request failed: {error_msg}")

    return DeviceCodeResponse.from_dict(response)


def poll_for_token(
    device_code: str,
    interval: int = DEFAULT_POLL_INTERVAL,
    expires_in: int = DEVICE_CODE_EXPIRES_IN,
    client_id: str = OAUTH_CLIENT_ID,
    on_message: Optional[Callable[[str], None]] = None,
) -> OAuthTokens:
    """Poll token endpoint until user completes authorization.

    Args:
        device_code: Device code from device code request.
        interval: Polling interval in seconds.
        expires_in: Time until device code expires.
        client_id: OAuth client ID.
        on_message: Optional callback for status messages.

    Returns:
        OAuthTokens on successful authorization.

    Raises:
        TimeoutError: If device code expires before authorization.
        RuntimeError: If authorization is denied or fails.
    """
    data = {
        "client_id": client_id,
        "device_code": device_code,
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
    }

    start_time = time.time()
    current_interval = interval

    while True:
        # Check if device code has expired
        elapsed = time.time() - start_time
        if elapsed > expires_in:
            raise TimeoutError(
                "Device code expired. Please run 'github-auth login' again."
            )

        try:
            response = _make_request(TOKEN_URL, data)
        except RuntimeError:
            # Network error - wait and retry
            time.sleep(current_interval)
            continue

        # Check for error responses
        if "error" in response:
            error = response["error"]

            if error == "authorization_pending":
                # User hasn't completed authorization yet - keep polling
                if on_message:
                    remaining = int(expires_in - elapsed)
                    on_message(f"Waiting for authorization... ({remaining}s remaining)")
                time.sleep(current_interval)
                continue

            elif error == "slow_down":
                # Rate limited - increase interval
                current_interval += 5
                if on_message:
                    on_message(f"Rate limited, slowing down (interval: {current_interval}s)")
                time.sleep(current_interval)
                continue

            elif error == "expired_token":
                raise TimeoutError(
                    "Device code expired. Please run 'github-auth login' again."
                )

            elif error == "access_denied":
                raise RuntimeError(
                    "Authorization denied. User cancelled the authorization."
                )

            else:
                # Unknown error
                error_msg = response.get("error_description") or error
                raise RuntimeError(f"Authorization failed: {error_msg}")

        # Success - we have an access token
        if "access_token" in response:
            return OAuthTokens(
                access_token=response["access_token"],
                token_type=response.get("token_type", "bearer"),
                scope=response.get("scope", ""),
            )

        # Unexpected response
        time.sleep(current_interval)


# Token storage location
def _get_token_storage_path() -> Path:
    """Get path to token storage file."""
    # Use XDG config dir on Linux, AppData on Windows, ~/Library on macOS
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif os.name == "posix" and os.uname().sysname == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

    return base / "jaato" / "github_oauth.json"


def save_tokens(tokens: OAuthTokens) -> None:
    """Save tokens to persistent storage."""
    path = _get_token_storage_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(tokens.to_dict(), f)

    # Secure permissions on Unix
    if os.name == "posix":
        os.chmod(path, 0o600)


def load_tokens() -> Optional[OAuthTokens]:
    """Load tokens from persistent storage."""
    path = _get_token_storage_path()

    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        return OAuthTokens.from_dict(data)
    except Exception:
        return None


def clear_tokens() -> None:
    """Clear stored tokens."""
    path = _get_token_storage_path()
    if path.exists():
        path.unlink()


def get_stored_access_token() -> Optional[str]:
    """Get access token from storage.

    Returns:
        Access token if found, None otherwise.
    """
    tokens = load_tokens()
    if tokens:
        return tokens.access_token
    return None


# Pending device code state for two-step flow
def _get_pending_auth_path() -> Path:
    """Get path to pending auth state file."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif os.name == "posix" and os.uname().sysname == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

    return base / "jaato" / "github_pending_auth.json"


def save_pending_auth(device_code_response: DeviceCodeResponse) -> None:
    """Save pending device code for polling."""
    path = _get_pending_auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "device_code": device_code_response.device_code,
        "user_code": device_code_response.user_code,
        "verification_uri": device_code_response.verification_uri,
        "expires_in": device_code_response.expires_in,
        "interval": device_code_response.interval,
        "created_at": time.time(),
    }

    with open(path, "w") as f:
        json.dump(data, f)

    if os.name == "posix":
        os.chmod(path, 0o600)


def load_pending_auth() -> Optional[Tuple[DeviceCodeResponse, float]]:
    """Load pending device code state.

    Returns:
        Tuple of (DeviceCodeResponse, created_at) or None if not found.
    """
    path = _get_pending_auth_path()
    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        response = DeviceCodeResponse(
            device_code=data["device_code"],
            user_code=data["user_code"],
            verification_uri=data["verification_uri"],
            expires_in=data["expires_in"],
            interval=data["interval"],
        )
        return response, data["created_at"]
    except Exception:
        return None


def clear_pending_auth() -> None:
    """Clear pending auth state."""
    path = _get_pending_auth_path()
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass


def start_device_flow(
    on_message: Optional[Callable[[str], None]] = None,
) -> DeviceCodeResponse:
    """Start the device code flow.

    Requests device code and saves state for polling.

    Args:
        on_message: Optional callback for status messages.

    Returns:
        DeviceCodeResponse with user code and verification URL.
    """
    if on_message:
        on_message("Requesting device code from GitHub...")

    response = request_device_code()
    save_pending_auth(response)

    return response


def complete_device_flow(
    on_message: Optional[Callable[[str], None]] = None,
) -> Optional[OAuthTokens]:
    """Complete the device code flow by polling for token.

    Must be called after user has authorized at verification URL.

    Args:
        on_message: Optional callback for status messages.

    Returns:
        OAuthTokens if successful, None if no pending auth.
    """
    pending = load_pending_auth()
    if not pending:
        if on_message:
            on_message("No pending login found. Run 'github-auth login' first.")
        return None

    device_response, created_at = pending

    # Calculate remaining time
    elapsed = time.time() - created_at
    remaining = device_response.expires_in - elapsed

    if remaining <= 0:
        if on_message:
            on_message("Device code expired. Please run 'github-auth login' again.")
        clear_pending_auth()
        return None

    if on_message:
        on_message("Polling for authorization...")

    try:
        tokens = poll_for_token(
            device_code=device_response.device_code,
            interval=device_response.interval,
            expires_in=int(remaining),
            on_message=on_message,
        )
        save_tokens(tokens)
        clear_pending_auth()
        return tokens
    except (TimeoutError, RuntimeError) as e:
        if on_message:
            on_message(f"Authorization failed: {e}")
        return None


def login_interactive(
    on_message: Optional[Callable[[str], None]] = None,
    auto_poll: bool = True,
) -> Tuple[Optional[OAuthTokens], DeviceCodeResponse]:
    """Run full interactive device code login flow.

    Args:
        on_message: Optional callback for status messages.
        auto_poll: If True, automatically poll for token after displaying code.

    Returns:
        Tuple of (OAuthTokens or None, DeviceCodeResponse).
        Tokens are None if auto_poll is False or if authorization fails.
    """
    # Start device flow
    device_response = start_device_flow(on_message)

    if on_message:
        on_message("")
        on_message(f"Please visit: {device_response.verification_uri}")
        on_message(f"Enter code: {device_response.user_code}")
        on_message("")

    if not auto_poll:
        return None, device_response

    # Poll for token
    tokens = complete_device_flow(on_message)
    return tokens, device_response
