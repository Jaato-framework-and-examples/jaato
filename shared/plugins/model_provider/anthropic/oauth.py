"""OAuth 2.0 PKCE flow for Anthropic Claude Pro/Max subscriptions.

Implements the same OAuth flow used by Claude Code CLI and OpenCode.

Reference: https://github.com/sst/opencode-anthropic-auth
"""

import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple
import urllib.request

# OAuth configuration (same as Claude Code CLI)
OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
OAUTH_AUTH_URL = "https://claude.ai/oauth/authorize"
OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
OAUTH_SCOPES = "org:create_api_key user:profile user:inference"

# Callback configuration - using Anthropic's hosted callback page
# (Local callback servers are not supported for this client_id)
CALLBACK_URL = "https://console.anthropic.com/oauth/code/callback"

# Legacy local callback settings (kept for reference)
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 19275
CALLBACK_PATH = "/oauth/callback"


@dataclass
class OAuthTokens:
    """OAuth token set with expiration tracking."""
    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp

    @property
    def is_expired(self) -> bool:
        """Check if access token is expired (with 5 min buffer)."""
        return time.time() > (self.expires_at - 300)

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OAuthTokens":
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
        )


def _generate_pkce_pair() -> Tuple[str, str]:
    """Generate PKCE code verifier and challenge.

    Claude Code uses a non-standard encoding for the verifier:
    - Standard base64, but with: + → ~, = → _, / → -

    The challenge uses standard base64url encoding.

    Returns:
        Tuple of (code_verifier, code_challenge)
    """
    # Generate random 32-byte verifier (same entropy as Claude Code)
    verifier_bytes = secrets.token_bytes(32)

    # Claude Code encoding: standard base64 with custom replacements
    # + → ~, / → -, and strip padding (despite JS code saying replace with _)
    # MITM capture shows 43-char verifier, not 44, so padding is stripped
    verifier_b64 = base64.b64encode(verifier_bytes).decode("ascii")
    code_verifier = (
        verifier_b64
        .rstrip("=")  # Strip padding (matches MITM observation)
        .replace("+", "~")
        .replace("/", "-")
    )

    # Challenge uses standard base64url (+ → -, / → _, strip =)
    challenge_bytes = hashlib.sha256(code_verifier.encode("ascii")).digest()
    challenge_b64 = base64.b64encode(challenge_bytes).decode("ascii")
    code_challenge = (
        challenge_b64
        .rstrip("=")
        .replace("+", "-")
        .replace("/", "_")
    )

    return code_verifier, code_challenge


def _generate_state() -> str:
    """Generate random state parameter for CSRF protection."""
    return secrets.token_urlsafe(32)


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    def log_message(self, format, *args):
        """Suppress HTTP server logging."""
        pass

    def do_GET(self):
        """Handle OAuth callback GET request."""
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path != CALLBACK_PATH:
            self.send_error(404)
            return

        query = urllib.parse.parse_qs(parsed.query)

        # Store result in server
        if "code" in query:
            self.server.oauth_code = query["code"][0]
            self.server.oauth_state = query.get("state", [None])[0]
            self.server.oauth_error = None
        elif "error" in query:
            self.server.oauth_code = None
            self.server.oauth_state = None
            self.server.oauth_error = query.get("error_description", query["error"])[0]

        # Send success page
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()

        if self.server.oauth_code:
            html = """
            <html>
            <head><title>Authentication Successful</title></head>
            <body style="font-family: system-ui; text-align: center; padding: 50px;">
                <h1>✓ Authentication Successful</h1>
                <p>You can close this window and return to jaato.</p>
            </body>
            </html>
            """
        else:
            html = f"""
            <html>
            <head><title>Authentication Failed</title></head>
            <body style="font-family: system-ui; text-align: center; padding: 50px;">
                <h1>✗ Authentication Failed</h1>
                <p>{self.server.oauth_error or 'Unknown error'}</p>
            </body>
            </html>
            """

        self.wfile.write(html.encode())

        # Signal completion
        self.server.oauth_complete = True


def build_auth_url() -> Tuple[str, str, str]:
    """Build OAuth authorization URL.

    Uses Anthropic's hosted callback page since local callbacks are not
    supported for this client_id. User will need to copy the code manually.

    Returns:
        Tuple of (auth_url, code_verifier, state).
    """
    code_verifier, code_challenge = _generate_pkce_pair()
    state = _generate_state()

    # Match exact parameter order used by Claude Code CLI
    auth_params = [
        ("code", "true"),  # Required by Anthropic's OAuth
        ("client_id", OAUTH_CLIENT_ID),
        ("response_type", "code"),
        ("redirect_uri", CALLBACK_URL),  # Use Anthropic's hosted callback
        ("scope", OAUTH_SCOPES),
        ("code_challenge", code_challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
    ]
    auth_url = f"{OAUTH_AUTH_URL}?{urllib.parse.urlencode(auth_params)}"

    return auth_url, code_verifier, state


def authorize_with_params(
    auth_url: str,
    code_verifier: str,
    state: str,
    timeout: int = 120,
) -> str:
    """Wait for OAuth callback after browser auth.

    Args:
        auth_url: Pre-built authorization URL.
        code_verifier: PKCE code verifier.
        state: OAuth state for CSRF protection.
        timeout: Seconds to wait for user to complete auth.

    Returns:
        Authorization code from callback.

    Raises:
        TimeoutError: If user doesn't complete auth in time.
        RuntimeError: If OAuth fails.
    """
    # Start callback server
    server = http.server.HTTPServer((CALLBACK_HOST, CALLBACK_PORT), _OAuthCallbackHandler)
    server.oauth_code = None
    server.oauth_state = None
    server.oauth_error = None
    server.oauth_complete = False
    server.timeout = 1

    # Open browser
    threading.Thread(target=webbrowser.open, args=(auth_url,), daemon=True).start()

    # Wait for callback
    start_time = time.time()
    try:
        while not server.oauth_complete:
            server.handle_request()
            if time.time() - start_time > timeout:
                raise TimeoutError("OAuth authentication timed out")
    finally:
        server.server_close()

    # Validate response
    if server.oauth_error:
        raise RuntimeError(f"OAuth error: {server.oauth_error}")

    if not server.oauth_code:
        raise RuntimeError("No authorization code received")

    if server.oauth_state != state:
        raise RuntimeError("OAuth state mismatch (possible CSRF attack)")

    return server.oauth_code


def authorize_interactive(
    timeout: int = 120,
    on_message: Optional[Callable[[str], None]] = None
) -> Tuple[str, str, str, str]:
    """Run interactive OAuth flow in browser.

    Opens browser to Claude's OAuth page and waits for callback.

    Args:
        timeout: Seconds to wait for user to complete auth.
        on_message: Optional callback for emitting status messages.

    Returns:
        Tuple of (authorization_code, code_verifier, state, auth_url)

    Raises:
        TimeoutError: If user doesn't complete auth in time.
        RuntimeError: If OAuth fails.
    """
    # Generate PKCE parameters
    code_verifier, code_challenge = _generate_pkce_pair()
    state = _generate_state()

    # Build callback URL
    redirect_uri = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"

    # Build authorization URL - match exact parameter order used by Claude Code CLI
    auth_params = [
        ("code", "true"),  # Required by Anthropic's OAuth
        ("client_id", OAUTH_CLIENT_ID),
        ("response_type", "code"),
        ("redirect_uri", redirect_uri),
        ("scope", OAUTH_SCOPES),
        ("code_challenge", code_challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
    ]
    auth_url = f"{OAUTH_AUTH_URL}?{urllib.parse.urlencode(auth_params)}"

    # Start callback server
    server = http.server.HTTPServer((CALLBACK_HOST, CALLBACK_PORT), _OAuthCallbackHandler)
    server.oauth_code = None
    server.oauth_state = None
    server.oauth_error = None
    server.oauth_complete = False
    server.timeout = 1  # Check every second

    # Emit status messages via callback
    msg1 = "Opening browser for authentication..."
    msg2 = f"If browser doesn't open, visit: {auth_url}"

    if on_message:
        on_message(msg1)
        on_message(msg2)

    # Open browser in background thread (some environments block on webbrowser.open)
    threading.Thread(target=webbrowser.open, args=(auth_url,), daemon=True).start()

    # Wait for callback
    start_time = time.time()
    try:
        while not server.oauth_complete:
            server.handle_request()
            if time.time() - start_time > timeout:
                raise TimeoutError("OAuth authentication timed out")
    finally:
        server.server_close()

    # Validate response
    if server.oauth_error:
        raise RuntimeError(f"OAuth error: {server.oauth_error}")

    if not server.oauth_code:
        raise RuntimeError("No authorization code received")

    if server.oauth_state != state:
        raise RuntimeError("OAuth state mismatch (possible CSRF attack)")

    return server.oauth_code, code_verifier, state, auth_url


def exchange_code_for_tokens(code: str, code_verifier: str, state: str) -> OAuthTokens:
    """Exchange authorization code for access/refresh tokens.

    Args:
        code: Authorization code from OAuth callback.
        code_verifier: PKCE code verifier.
        state: OAuth state from the auth request.

    Returns:
        OAuthTokens with access and refresh tokens.

    Raises:
        RuntimeError: If token exchange fails.
    """
    import requests

    # Build token request as JSON - matching Claude Code CLI exactly
    # Must include state and use JSON content-type
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": CALLBACK_URL,
        "client_id": OAUTH_CLIENT_ID,
        "code_verifier": code_verifier,
        "state": state,
        "expires_in": 31536000,  # Request 1 year token (same as Claude Code)
    }

    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "User-Agent": "axios/1.8.4",  # Match Claude Code's user agent
    }

    try:
        resp = requests.post(
            OAUTH_TOKEN_URL,
            json=token_data,  # Use json= for JSON body
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            # Get detailed error from response
            try:
                error_data = resp.json()
                error_msg = error_data.get("error_description") or error_data.get("error") or resp.text
            except Exception:
                error_msg = resp.text or f"HTTP {resp.status_code}"
            raise RuntimeError(f"Token exchange failed ({resp.status_code}): {error_msg}")
        data = resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Token exchange failed: {e}")

    # Parse response
    access_token = data.get("access_token")
    refresh_token = data.get("refresh_token")
    expires_in = data.get("expires_in", 3600)

    if not access_token or not refresh_token:
        raise RuntimeError(f"Invalid token response: {data}")

    return OAuthTokens(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=time.time() + expires_in,
    )


def refresh_tokens(refresh_token: str) -> OAuthTokens:
    """Refresh expired access token.

    Args:
        refresh_token: Current refresh token.

    Returns:
        New OAuthTokens (note: refresh token may also be rotated).

    Raises:
        RuntimeError: If refresh fails.
    """
    import requests

    # Build refresh request as JSON - matching Claude Code CLI
    token_data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": OAUTH_CLIENT_ID,
        "scope": OAUTH_SCOPES,
    }

    headers = {
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            OAUTH_TOKEN_URL,
            json=token_data,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        error_body = e.response.text if e.response else str(e)
        raise RuntimeError(f"Token refresh failed: {error_body}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Token refresh failed: {e}")

    access_token = data.get("access_token")
    # Anthropic rotates refresh tokens - use new one if provided
    new_refresh_token = data.get("refresh_token", refresh_token)
    expires_in = data.get("expires_in", 3600)

    if not access_token:
        raise RuntimeError(f"Invalid refresh response: {data}")

    return OAuthTokens(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_at=time.time() + expires_in,
    )


# Token storage location
def _get_token_storage_path(for_write: bool = False) -> Path:
    """Get path to token storage file.

    Follows jaato convention:
    1. Project .jaato/ first (project-specific auth)
    2. Home ~/.jaato/ second (user-level default)

    Uses JAATO_WORKSPACE_PATH env var if set (for subagents), otherwise Path.cwd().
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
    workspace = os.environ.get("JAATO_WORKSPACE_PATH") or os.getcwd()
    project_path = Path(workspace) / ".jaato" / "anthropic_oauth.json"
    home_path = Path.home() / ".jaato" / "anthropic_oauth.json"

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


def save_tokens(tokens: OAuthTokens) -> None:
    """Save tokens to persistent storage."""
    path = _get_token_storage_path(for_write=True)
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


def get_valid_access_token() -> Optional[str]:
    """Get a valid access token, refreshing if needed.

    Returns:
        Valid access token, or None if no tokens stored.

    Raises:
        RuntimeError: If refresh fails.
    """
    tokens = load_tokens()
    if not tokens:
        return None

    # Refresh if expired
    if tokens.is_expired:
        tokens = refresh_tokens(tokens.refresh_token)
        save_tokens(tokens)

    return tokens.access_token


def login(
    on_message: Optional[Callable[[str], None]] = None
) -> Tuple[OAuthTokens, str]:
    """Run full OAuth login flow.

    Opens browser, waits for auth, exchanges code for tokens.

    Args:
        on_message: Optional callback for emitting status messages in real-time.

    Returns:
        Tuple of (OAuthTokens, auth_url) after successful authentication.
        The auth_url is provided so callers can display it to users.
    """
    code, verifier, state, auth_url = authorize_interactive(on_message=on_message)
    tokens = exchange_code_for_tokens(code, verifier, state)
    save_tokens(tokens)
    return tokens, auth_url


# ==================== Pending Auth State ====================
# For two-step login flow where user must manually copy the auth code

def _get_pending_auth_path(for_write: bool = False) -> Path:
    """Get path to pending auth state file.

    Follows same convention as token storage:
    1. Project .jaato/ first
    2. Home ~/.jaato/ second
    """
    project_path = Path.cwd() / ".jaato" / "anthropic_pending_auth.json"
    home_path = Path.home() / ".jaato" / "anthropic_pending_auth.json"

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


def save_pending_auth(code_verifier: str, state: str) -> None:
    """Save pending auth state for two-step login flow."""
    path = _get_pending_auth_path(for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump({"code_verifier": code_verifier, "state": state}, f)

    if os.name == "posix":
        os.chmod(path, 0o600)


def load_pending_auth() -> Tuple[Optional[str], Optional[str]]:
    """Load pending auth state.

    Returns:
        Tuple of (code_verifier, state), or (None, None) if not found.
    """
    path = _get_pending_auth_path()
    if not path.exists():
        return None, None

    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("code_verifier"), data.get("state")
    except Exception:
        return None, None


def clear_pending_auth() -> None:
    """Clear pending auth state."""
    path = _get_pending_auth_path()
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass


def start_interactive_login(
    on_message: Optional[Callable[[str], None]] = None
) -> str:
    """Start two-step interactive login flow.

    Builds auth URL, saves pending state, emits messages, and opens browser.
    User must complete login by calling complete_interactive_login() with
    the authorization code.

    Args:
        on_message: Optional callback for status messages.

    Returns:
        The authorization URL.
    """
    auth_url, code_verifier, state = build_auth_url()

    # Save state for step 2
    save_pending_auth(code_verifier, state)

    if on_message:
        on_message("Opening browser for authentication...")
        on_message(f"If browser doesn't open, visit:\n{auth_url}")
        on_message("")
        on_message("After authenticating, copy the authorization code and run:")
        on_message("  anthropic-auth code <paste_code_here>")

    # Open browser in background
    threading.Thread(target=webbrowser.open, args=(auth_url,), daemon=True).start()

    return auth_url


def complete_interactive_login(
    auth_code: str,
    on_message: Optional[Callable[[str], None]] = None
) -> Optional[OAuthTokens]:
    """Complete two-step interactive login with authorization code.

    Args:
        auth_code: Authorization code from browser callback.
        on_message: Optional callback for status messages.

    Returns:
        OAuthTokens if successful, None if no pending auth or exchange failed.
    """
    code_verifier, state = load_pending_auth()

    if not code_verifier or not state:
        if on_message:
            on_message("No pending login found. Run 'anthropic-auth login' first.")
        return None

    # Strip URL fragment if present (callback shows code#state)
    if "#" in auth_code:
        auth_code = auth_code.split("#")[0]

    try:
        tokens = exchange_code_for_tokens(auth_code, code_verifier, state)
        save_tokens(tokens)
        clear_pending_auth()
        return tokens
    except Exception as e:
        if on_message:
            on_message(f"Token exchange failed: {e}")
        return None
