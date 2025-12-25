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
OAUTH_AUTH_URL = "https://console.anthropic.com/oauth/authorize"
OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
OAUTH_SCOPES = "org:create_api_key user:profile user:inference"

# Local callback server
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 19275  # Same port as Claude Code CLI uses
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

    Returns:
        Tuple of (code_verifier, code_challenge)
    """
    # Generate random 32-byte verifier
    verifier_bytes = secrets.token_bytes(32)
    code_verifier = base64.urlsafe_b64encode(verifier_bytes).rstrip(b"=").decode("ascii")

    # Generate SHA256 challenge
    challenge_bytes = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode("ascii")

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
    """Build OAuth authorization URL without starting server.

    Returns:
        Tuple of (auth_url, code_verifier, state) for use in authorize_with_params.
    """
    code_verifier, code_challenge = _generate_pkce_pair()
    state = _generate_state()

    redirect_uri = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"

    # Match exact parameter order used by Claude Code CLI
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
) -> Tuple[str, str, str]:
    """Run interactive OAuth flow in browser.

    Opens browser to Claude's OAuth page and waits for callback.

    Args:
        timeout: Seconds to wait for user to complete auth.
        on_message: Optional callback for emitting status messages.

    Returns:
        Tuple of (authorization_code, code_verifier, auth_url)

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

    return server.oauth_code, code_verifier, auth_url


def exchange_code_for_tokens(code: str, code_verifier: str) -> OAuthTokens:
    """Exchange authorization code for access/refresh tokens.

    Args:
        code: Authorization code from OAuth callback.
        code_verifier: PKCE code verifier.

    Returns:
        OAuthTokens with access and refresh tokens.

    Raises:
        RuntimeError: If token exchange fails.
    """
    redirect_uri = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"

    # Build token request
    token_data = {
        "grant_type": "authorization_code",
        "client_id": OAUTH_CLIENT_ID,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }

    # Make token request
    req = urllib.request.Request(
        OAUTH_TOKEN_URL,
        data=urllib.parse.urlencode(token_data).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else str(e)
        raise RuntimeError(f"Token exchange failed: {error_body}")
    except Exception as e:
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
    token_data = {
        "grant_type": "refresh_token",
        "client_id": OAUTH_CLIENT_ID,
        "refresh_token": refresh_token,
    }

    req = urllib.request.Request(
        OAUTH_TOKEN_URL,
        data=urllib.parse.urlencode(token_data).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else str(e)
        raise RuntimeError(f"Token refresh failed: {error_body}")
    except Exception as e:
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


# Token storage location (compatible with Claude Code CLI)
def _get_token_storage_path() -> Path:
    """Get path to token storage file."""
    # Use XDG config dir on Linux, AppData on Windows, ~/Library on macOS
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif os.name == "posix" and os.uname().sysname == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

    return base / "jaato" / "anthropic_oauth.json"


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
    code, verifier, auth_url = authorize_interactive(on_message=on_message)
    tokens = exchange_code_for_tokens(code, verifier)
    save_tokens(tokens)
    return tokens, auth_url
