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

# Copilot token exchange endpoint
COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"

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
class CopilotToken:
    """Copilot-specific token obtained by exchanging OAuth token.

    This token is used for actual API calls to GitHub Copilot/Models.
    It has an expiration time and needs to be refreshed periodically.
    """
    token: str
    expires_at: int  # Unix timestamp
    refresh_in: int = 0  # Seconds until refresh recommended

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "expires_at": self.expires_at,
            "refresh_in": self.refresh_in,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CopilotToken":
        return cls(
            token=data["token"],
            expires_at=data["expires_at"],
            refresh_in=data.get("refresh_in", 0),
        )

    def is_expired(self) -> bool:
        """Check if token has expired."""
        return time.time() >= self.expires_at

    def needs_refresh(self) -> bool:
        """Check if token should be refreshed soon."""
        if self.refresh_in > 0:
            return time.time() >= (self.expires_at - self.refresh_in)
        # Default: refresh if less than 5 minutes remaining
        return time.time() >= (self.expires_at - 300)


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

    Respects JAATO_NO_PROXY for exact host matching (unlike standard NO_PROXY
    which does suffix matching). If JAATO_NO_PROXY is not set, falls back to
    standard proxy environment variables (HTTP_PROXY, HTTPS_PROXY, NO_PROXY).

    Args:
        url: Request URL.
        data: Form data to POST.
        headers: Optional additional headers.

    Returns:
        Parsed JSON response.

    Raises:
        RuntimeError: If request fails.
    """
    from shared.http import get_url_opener

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
        opener = get_url_opener(url)
        with opener.open(req, timeout=30) as response:
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


def _make_get_request(
    url: str,
    headers: Optional[dict] = None,
) -> dict:
    """Make HTTP GET request and return JSON response.

    Respects JAATO_NO_PROXY for exact host matching (unlike standard NO_PROXY
    which does suffix matching). If JAATO_NO_PROXY is not set, falls back to
    standard proxy environment variables (HTTP_PROXY, HTTPS_PROXY, NO_PROXY).

    Args:
        url: Request URL.
        headers: Optional additional headers.

    Returns:
        Parsed JSON response.

    Raises:
        RuntimeError: If request fails.
    """
    from shared.http import get_url_opener

    default_headers = {
        "Accept": "application/json",
        "User-Agent": "jaato/1.0",
    }
    if headers:
        default_headers.update(headers)

    req = urllib.request.Request(
        url,
        headers=default_headers,
        method="GET",
    )

    try:
        opener = get_url_opener(url)
        with opener.open(req, timeout=30) as response:
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


def exchange_oauth_for_copilot_token(oauth_token: str) -> CopilotToken:
    """Exchange OAuth token for Copilot-specific token.

    This is the third stage of authentication - after getting the OAuth token,
    we need to exchange it for a Copilot token that can be used for API calls.

    Args:
        oauth_token: GitHub OAuth access token.

    Returns:
        CopilotToken with the API token and expiration info.

    Raises:
        RuntimeError: If token exchange fails.
    """
    headers = {
        "Authorization": f"token {oauth_token}",
    }

    response = _make_get_request(COPILOT_TOKEN_URL, headers)

    if "token" not in response:
        raise RuntimeError(
            f"Token exchange failed: {response.get('message', 'No token in response')}"
        )

    return CopilotToken(
        token=response["token"],
        expires_at=response.get("expires_at", int(time.time()) + 3600),
        refresh_in=response.get("refresh_in", 0),
    )


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
    on_progress: Optional[Callable[[str], None]] = None,
) -> OAuthTokens:
    """Poll token endpoint until user completes authorization.

    Args:
        device_code: Device code from device code request.
        interval: Polling interval in seconds.
        expires_in: Time until device code expires.
        client_id: OAuth client ID.
        on_message: Optional callback for status messages.
        on_progress: Optional callback for progress updates (updates same line).

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
                remaining = int(expires_in - elapsed)
                progress_msg = f"Waiting for authorization... ({remaining}s remaining)"
                if on_progress:
                    on_progress(progress_msg)
                elif on_message:
                    on_message(progress_msg)
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
def _get_token_storage_path(for_write: bool = False) -> Path:
    """Get path to token storage file.

    Follows jaato convention:
    1. Project .jaato/ first (project-specific auth)
    2. Home ~/.jaato/ second (user-level default)

    Uses JAATO_WORKSPACE_ROOT env var if set (for subagents), otherwise Path.cwd().
    This avoids race conditions when subagents run in thread pools with
    process-wide CWD changes.

    Args:
        for_write: If True, returns the path to write to (prefers project dir
                   if it exists, otherwise home). If False, returns the first
                   existing file or the default write location.

    Returns:
        Path to token storage file.
    """
    # Use explicit workspace path if set (thread-safe for subagents)
    # Falls back to CWD for main agent
    workspace = os.environ.get("JAATO_WORKSPACE_ROOT") or os.getcwd()

    # Project-level path
    project_path = Path(workspace) / ".jaato" / "github_oauth.json"

    # User-level path (home directory)
    home_path = Path.home() / ".jaato" / "github_oauth.json"

    if for_write:
        # For writing: prefer project .jaato/ if directory exists
        if project_path.parent.exists():
            return project_path
        # Otherwise use home directory
        return home_path
    else:
        # For reading: check project first, then home
        if project_path.exists():
            return project_path
        if home_path.exists():
            return home_path
        # Default to home for new files
        return home_path


def save_tokens(
    oauth_tokens: OAuthTokens,
    copilot_token: Optional[CopilotToken] = None,
) -> None:
    """Save tokens to persistent storage.

    Args:
        oauth_tokens: OAuth tokens from device code flow.
        copilot_token: Optional Copilot token from exchange.
    """
    path = _get_token_storage_path(for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "oauth": oauth_tokens.to_dict(),
    }
    if copilot_token:
        data["copilot"] = copilot_token.to_dict()

    with open(path, "w") as f:
        json.dump(data, f)

    # Secure permissions on Unix
    if os.name == "posix":
        os.chmod(path, 0o600)


def load_tokens() -> Optional[OAuthTokens]:
    """Load OAuth tokens from persistent storage."""
    path = _get_token_storage_path()

    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        # Support both old format (direct) and new format (nested under "oauth")
        if "oauth" in data:
            return OAuthTokens.from_dict(data["oauth"])
        elif "access_token" in data:
            # Old format - direct OAuth token
            return OAuthTokens.from_dict(data)
        return None
    except Exception:
        return None


def load_copilot_token() -> Optional[CopilotToken]:
    """Load Copilot token from persistent storage."""
    path = _get_token_storage_path()

    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        if "copilot" in data:
            return CopilotToken.from_dict(data["copilot"])
        return None
    except Exception:
        return None


def save_copilot_token(copilot_token: CopilotToken) -> None:
    """Update stored Copilot token without modifying OAuth tokens."""
    path = _get_token_storage_path()

    if not path.exists():
        return

    try:
        with open(path) as f:
            data = json.load(f)
        data["copilot"] = copilot_token.to_dict()
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def clear_tokens() -> None:
    """Clear stored tokens."""
    path = _get_token_storage_path()
    if path.exists():
        path.unlink()


def clear_copilot_token() -> None:
    """Clear only the Copilot token, keeping OAuth tokens intact.

    This forces re-exchange of the OAuth token for a new Copilot token
    on the next API call. Useful when a 401 is received even though the
    token appeared valid.
    """
    path = _get_token_storage_path()
    if not path.exists():
        return

    try:
        with open(path) as f:
            data = json.load(f)
        if "copilot" in data:
            del data["copilot"]
            with open(path, "w") as f:
                json.dump(data, f)
    except Exception:
        pass


def _oauth_trace(msg: str) -> None:
    """Write trace message for debugging OAuth operations."""
    import datetime
    import tempfile
    trace_path = os.environ.get(
        "JAATO_PROVIDER_TRACE",
        os.path.join(tempfile.gettempdir(), "provider_trace.log")
    )
    try:
        with open(trace_path, "a") as f:
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[{ts}] [oauth] {msg}\n")
            f.flush()
    except Exception:
        pass


def get_stored_access_token() -> Optional[str]:
    """Get Copilot API token from storage, refreshing if needed.

    This function handles the full token lifecycle:
    1. Check for existing Copilot token
    2. If valid, return it
    3. If expired/missing, exchange OAuth token for new Copilot token
    4. Save and return the new token

    Returns:
        Copilot API token if available, None otherwise.
    """
    _oauth_trace("get_stored_access_token: start")

    # Try to get existing Copilot token
    _oauth_trace("get_stored_access_token: loading copilot token...")
    copilot_token = load_copilot_token()
    _oauth_trace(f"get_stored_access_token: copilot_token={bool(copilot_token)}")

    if copilot_token and not copilot_token.is_expired():
        _oauth_trace("get_stored_access_token: returning existing valid token")
        return copilot_token.token

    # Need to refresh - get OAuth token
    _oauth_trace("get_stored_access_token: token expired/missing, loading OAuth...")
    oauth_tokens = load_tokens()
    _oauth_trace(f"get_stored_access_token: oauth_tokens={bool(oauth_tokens)}")
    if not oauth_tokens:
        _oauth_trace("get_stored_access_token: no OAuth tokens, returning None")
        return None

    # Exchange OAuth token for Copilot token
    _oauth_trace("get_stored_access_token: exchanging for Copilot token...")
    try:
        copilot_token = exchange_oauth_for_copilot_token(oauth_tokens.access_token)
        _oauth_trace("get_stored_access_token: exchange successful, saving...")
        save_copilot_token(copilot_token)
        _oauth_trace("get_stored_access_token: returning new token")
        return copilot_token.token
    except RuntimeError as e:
        # Token exchange failed - OAuth token may be invalid
        _oauth_trace(f"get_stored_access_token: exchange failed: {e}")
        return None


def get_stored_oauth_token() -> Optional[str]:
    """Get raw OAuth token from storage (for debugging/status).

    Returns:
        OAuth access token if found, None otherwise.
    """
    tokens = load_tokens()
    if tokens:
        return tokens.access_token
    return None


# Pending device code state for two-step flow
def _get_pending_auth_path(for_write: bool = False) -> Path:
    """Get path to pending auth state file.

    Follows same convention as token storage:
    1. Project .jaato/ first
    2. Home ~/.jaato/ second
    """
    project_path = Path.cwd() / ".jaato" / "github_pending_auth.json"
    home_path = Path.home() / ".jaato" / "github_pending_auth.json"

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


def save_pending_auth(device_code_response: DeviceCodeResponse) -> None:
    """Save pending device code for polling."""
    path = _get_pending_auth_path(for_write=True)
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
    on_progress: Optional[Callable[[str], None]] = None,
) -> Optional[OAuthTokens]:
    """Complete the device code flow by polling for token.

    Must be called after user has authorized at verification URL.

    Args:
        on_message: Optional callback for status messages.
        on_progress: Optional callback for progress updates (updates same line).

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
            on_progress=on_progress,
        )

        # Exchange OAuth token for Copilot token
        copilot_token = None
        try:
            if on_message:
                on_message("Exchanging for Copilot API token...")
            copilot_token = exchange_oauth_for_copilot_token(tokens.access_token)
        except RuntimeError as e:
            if on_message:
                on_message(f"Warning: Could not get Copilot token: {e}")
            # Continue anyway - OAuth token is still valid

        save_tokens(tokens, copilot_token)
        clear_pending_auth()
        return tokens
    except (TimeoutError, RuntimeError) as e:
        if on_message:
            on_message(f"Authorization failed: {e}")
        return None


def login_interactive(
    on_message: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    auto_poll: bool = True,
) -> Tuple[Optional[OAuthTokens], DeviceCodeResponse]:
    """Run full interactive device code login flow.

    Args:
        on_message: Optional callback for status messages.
        on_progress: Optional callback for progress updates (updates same line).
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
    tokens = complete_device_flow(on_message, on_progress)
    return tokens, device_response
