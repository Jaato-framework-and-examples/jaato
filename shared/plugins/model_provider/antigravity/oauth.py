"""Google OAuth 2.0 PKCE flow for Antigravity authentication.

Implements Google OAuth to access Antigravity's AI model endpoints.
Users authenticate with their Google account to access models through
Google's Antigravity quota system.

Reference: https://github.com/NoeFabris/opencode-antigravity-auth
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import urllib.request

from .constants import (
    CALLBACK_HOST,
    CALLBACK_PATH,
    CALLBACK_PORT,
    CODE_ASSIST_ENDPOINTS,
    DEFAULT_PROJECT_ID,
    OAUTH_AUTH_URL,
    OAUTH_CLIENT_ID,
    OAUTH_CLIENT_SECRET,
    OAUTH_SCOPES,
    OAUTH_TOKEN_URL,
    OAUTH_USERINFO_URL,
)


@dataclass
class OAuthTokens:
    """OAuth token set with expiration tracking."""

    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp
    email: Optional[str] = None
    project_id: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Check if access token is expired (with 5 min buffer)."""
        return time.time() > (self.expires_at - 300)

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "email": self.email,
            "project_id": self.project_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OAuthTokens":
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
            email=data.get("email"),
            project_id=data.get("project_id"),
        )


@dataclass
class Account:
    """Represents a Google account for Antigravity authentication."""

    email: str
    tokens: OAuthTokens
    project_id: Optional[str] = None
    is_active: bool = True
    rate_limited_until: Optional[float] = None

    def is_rate_limited(self) -> bool:
        """Check if this account is currently rate limited."""
        if self.rate_limited_until is None:
            return False
        return time.time() < self.rate_limited_until

    def mark_rate_limited(self, duration: float = 60.0) -> None:
        """Mark this account as rate limited for the given duration."""
        self.rate_limited_until = time.time() + duration

    def clear_rate_limit(self) -> None:
        """Clear the rate limit for this account."""
        self.rate_limited_until = None

    def to_dict(self) -> dict:
        return {
            "email": self.email,
            "tokens": self.tokens.to_dict(),
            "project_id": self.project_id,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Account":
        return cls(
            email=data["email"],
            tokens=OAuthTokens.from_dict(data["tokens"]),
            project_id=data.get("project_id"),
            is_active=data.get("is_active", True),
        )


@dataclass
class AccountManager:
    """Manages multiple Google accounts for load balancing and quota rotation."""

    accounts: List[Account] = field(default_factory=list)
    current_index: int = 0

    def add_account(self, account: Account) -> None:
        """Add a new account."""
        # Check if account with same email already exists
        for i, existing in enumerate(self.accounts):
            if existing.email == account.email:
                self.accounts[i] = account  # Replace existing
                return
        self.accounts.append(account)

    def get_active_account(self) -> Optional[Account]:
        """Get the next available (non-rate-limited) account."""
        if not self.accounts:
            return None

        # Try to find an account that isn't rate limited
        for _ in range(len(self.accounts)):
            account = self.accounts[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.accounts)

            if account.is_active and not account.is_rate_limited():
                return account

        # All accounts are rate limited - return the first active one anyway
        for account in self.accounts:
            if account.is_active:
                return account

        return None

    def rotate_on_rate_limit(self, current_email: str) -> Optional[Account]:
        """Mark current account as rate limited and rotate to next."""
        for account in self.accounts:
            if account.email == current_email:
                account.mark_rate_limited()
                break
        return self.get_active_account()

    def to_dict(self) -> dict:
        return {
            "accounts": [a.to_dict() for a in self.accounts],
            "current_index": self.current_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AccountManager":
        return cls(
            accounts=[Account.from_dict(a) for a in data.get("accounts", [])],
            current_index=data.get("current_index", 0),
        )


def _generate_pkce_pair() -> Tuple[str, str]:
    """Generate PKCE code verifier and challenge.

    Uses standard PKCE with S256 method.

    Returns:
        Tuple of (code_verifier, code_challenge)
    """
    # Generate random 32-byte verifier
    verifier_bytes = secrets.token_bytes(32)

    # Standard base64url encoding for verifier
    code_verifier = (
        base64.urlsafe_b64encode(verifier_bytes)
        .decode("ascii")
        .rstrip("=")
    )

    # Challenge is SHA256 hash of verifier, base64url encoded
    challenge_bytes = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = (
        base64.urlsafe_b64encode(challenge_bytes)
        .decode("ascii")
        .rstrip("=")
    )

    return code_verifier, code_challenge


def _generate_state(project_id: Optional[str] = None) -> str:
    """Generate state parameter for CSRF protection.

    Optionally encodes the project_id in the state.
    """
    random_state = secrets.token_urlsafe(16)
    if project_id:
        # Encode project_id in state as base64url
        encoded = base64.urlsafe_b64encode(project_id.encode()).decode().rstrip("=")
        return f"{random_state}.{encoded}"
    return random_state


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
            self.server.oauth_error = query.get(
                "error_description", query["error"]
            )[0]

        # Send success page
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()

        if self.server.oauth_code:
            html = """
            <html>
            <head><title>Authentication Successful</title></head>
            <body style="font-family: system-ui; text-align: center; padding: 50px;">
                <h1>Authentication Successful</h1>
                <p>You can close this window and return to jaato.</p>
            </body>
            </html>
            """
        else:
            html = f"""
            <html>
            <head><title>Authentication Failed</title></head>
            <body style="font-family: system-ui; text-align: center; padding: 50px;">
                <h1>Authentication Failed</h1>
                <p>{self.server.oauth_error or 'Unknown error'}</p>
            </body>
            </html>
            """

        self.wfile.write(html.encode())

        # Signal completion
        self.server.oauth_complete = True


def build_auth_url(project_id: Optional[str] = None) -> Tuple[str, str, str]:
    """Build OAuth authorization URL.

    Args:
        project_id: Optional project ID to encode in state.

    Returns:
        Tuple of (auth_url, code_verifier, state).
    """
    code_verifier, code_challenge = _generate_pkce_pair()
    state = _generate_state(project_id)

    # Build callback URL
    redirect_uri = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"

    # Build authorization URL
    auth_params = [
        ("client_id", OAUTH_CLIENT_ID),
        ("redirect_uri", redirect_uri),
        ("response_type", "code"),
        ("scope", " ".join(OAUTH_SCOPES)),
        ("access_type", "offline"),  # Request refresh token
        ("prompt", "consent"),  # Force consent to get refresh token
        ("code_challenge", code_challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
    ]

    auth_url = f"{OAUTH_AUTH_URL}?{urllib.parse.urlencode(auth_params)}"

    return auth_url, code_verifier, state


def authorize_interactive(
    timeout: int = 120,
    on_message: Optional[Callable[[str], None]] = None,
    project_id: Optional[str] = None,
) -> Tuple[str, str, str, str]:
    """Run interactive OAuth flow in browser.

    Opens browser to Google's OAuth page and waits for callback.

    Args:
        timeout: Seconds to wait for user to complete auth.
        on_message: Optional callback for status messages.
        project_id: Optional project ID to encode in state.

    Returns:
        Tuple of (authorization_code, code_verifier, state, auth_url)

    Raises:
        TimeoutError: If user doesn't complete auth in time.
        RuntimeError: If OAuth fails.
    """
    # Generate PKCE parameters
    code_verifier, code_challenge = _generate_pkce_pair()
    state = _generate_state(project_id)

    # Build callback URL
    redirect_uri = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"

    # Build authorization URL
    auth_params = [
        ("client_id", OAUTH_CLIENT_ID),
        ("redirect_uri", redirect_uri),
        ("response_type", "code"),
        ("scope", " ".join(OAUTH_SCOPES)),
        ("access_type", "offline"),
        ("prompt", "consent"),
        ("code_challenge", code_challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
    ]

    auth_url = f"{OAUTH_AUTH_URL}?{urllib.parse.urlencode(auth_params)}"

    # Start callback server
    server = http.server.HTTPServer(
        (CALLBACK_HOST, CALLBACK_PORT), _OAuthCallbackHandler
    )
    server.oauth_code = None
    server.oauth_state = None
    server.oauth_error = None
    server.oauth_complete = False
    server.timeout = 1  # Check every second

    # Emit status messages via callback
    msg1 = "Opening browser for Google authentication..."
    msg2 = f"If browser doesn't open, visit:\n{auth_url}"

    if on_message:
        on_message(msg1)
        on_message(msg2)

    # Open browser in background thread
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
    import requests

    # Build callback URL (must match authorization request)
    redirect_uri = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"

    # Build token request
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": OAUTH_CLIENT_ID,
        "client_secret": OAUTH_CLIENT_SECRET,
        "code_verifier": code_verifier,
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    try:
        resp = requests.post(
            OAUTH_TOKEN_URL,
            data=token_data,
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            try:
                error_data = resp.json()
                error_msg = (
                    error_data.get("error_description")
                    or error_data.get("error")
                    or resp.text
                )
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

    if not access_token:
        raise RuntimeError(f"Invalid token response: missing access_token")

    if not refresh_token:
        raise RuntimeError(f"Invalid token response: missing refresh_token")

    # Get user email
    email = _get_user_email(access_token)

    return OAuthTokens(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=time.time() + expires_in,
        email=email,
    )


def _get_user_email(access_token: str) -> Optional[str]:
    """Get user email from Google userinfo endpoint."""
    import requests

    try:
        resp = requests.get(
            OAUTH_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("email")
    except Exception:
        pass
    return None


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

    token_data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": OAUTH_CLIENT_ID,
        "client_secret": OAUTH_CLIENT_SECRET,
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    try:
        resp = requests.post(
            OAUTH_TOKEN_URL,
            data=token_data,
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
    # Google may rotate refresh tokens
    new_refresh_token = data.get("refresh_token", refresh_token)
    expires_in = data.get("expires_in", 3600)

    if not access_token:
        raise RuntimeError(f"Invalid refresh response: {data}")

    # Get user email
    email = _get_user_email(access_token)

    return OAuthTokens(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_at=time.time() + expires_in,
        email=email,
    )


def load_code_assist_project(access_token: str) -> Optional[str]:
    """Load project ID from Code Assist API.

    Tries multiple endpoints with fallback.

    Args:
        access_token: Valid OAuth access token.

    Returns:
        Project ID if found, None otherwise.
    """
    import requests

    for endpoint in CODE_ASSIST_ENDPOINTS:
        url = f"{endpoint}/v1internal/loadCodeAssist"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                url,
                json={},
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("projectId")
        except Exception:
            continue

    return None


# ==================== Token Storage ====================


def _get_token_storage_path(for_write: bool = False) -> Path:
    """Get path to token storage file.

    Follows jaato convention:
    1. Project .jaato/ first (project-specific auth)
    2. Home ~/.jaato/ second (user-level default)

    Args:
        for_write: If True, returns the path to write to (prefers project dir
                   if it exists, otherwise home). If False, returns the first
                   existing file or the default write location.

    Returns:
        Path to token storage file.
    """
    project_path = Path.cwd() / ".jaato" / "antigravity_accounts.json"
    home_path = Path.home() / ".jaato" / "antigravity_accounts.json"

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


def save_accounts(manager: AccountManager) -> None:
    """Save account manager to persistent storage."""
    path = _get_token_storage_path(for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(manager.to_dict(), f, indent=2)

    # Secure permissions on Unix
    if os.name == "posix":
        os.chmod(path, 0o600)


def load_accounts() -> AccountManager:
    """Load account manager from persistent storage."""
    path = _get_token_storage_path()

    if not path.exists():
        return AccountManager()

    try:
        with open(path) as f:
            data = json.load(f)
        return AccountManager.from_dict(data)
    except Exception:
        return AccountManager()


def clear_accounts() -> None:
    """Clear all stored accounts."""
    path = _get_token_storage_path()
    if path.exists():
        path.unlink()


def get_valid_access_token() -> Optional[Tuple[str, str, Optional[str]]]:
    """Get a valid access token, refreshing if needed.

    Returns:
        Tuple of (access_token, email, project_id), or None if no accounts stored.

    Raises:
        RuntimeError: If refresh fails.
    """
    manager = load_accounts()
    account = manager.get_active_account()

    if not account:
        return None

    # Refresh if expired
    if account.tokens.is_expired:
        new_tokens = refresh_tokens(account.tokens.refresh_token)
        # Preserve email and project_id
        new_tokens.email = account.email
        new_tokens.project_id = account.project_id
        account.tokens = new_tokens
        save_accounts(manager)

    return (
        account.tokens.access_token,
        account.email,
        account.project_id or account.tokens.project_id,
    )


def login(
    on_message: Optional[Callable[[str], None]] = None,
) -> Tuple[Account, str]:
    """Run full OAuth login flow.

    Opens browser, waits for auth, exchanges code for tokens.

    Args:
        on_message: Optional callback for status messages.

    Returns:
        Tuple of (Account, auth_url) after successful authentication.
    """
    code, verifier, state, auth_url = authorize_interactive(on_message=on_message)
    tokens = exchange_code_for_tokens(code, verifier)

    # Try to get project ID
    project_id = load_code_assist_project(tokens.access_token)
    if not project_id:
        project_id = DEFAULT_PROJECT_ID

    tokens.project_id = project_id

    # Create account
    account = Account(
        email=tokens.email or "unknown",
        tokens=tokens,
        project_id=project_id,
    )

    # Save to account manager
    manager = load_accounts()
    manager.add_account(account)
    save_accounts(manager)

    if on_message:
        on_message(f"Authenticated as: {account.email}")
        on_message(f"Project ID: {project_id}")

    return account, auth_url


# ==================== Pending Auth State ====================


def _get_pending_auth_path() -> Path:
    """Get path to pending auth state file."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif os.name == "posix" and os.uname().sysname == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

    return base / "jaato" / "antigravity_pending_auth.json"


def save_pending_auth(code_verifier: str, state: str) -> None:
    """Save pending auth state for two-step login flow."""
    path = _get_pending_auth_path()
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
    on_message: Optional[Callable[[str], None]] = None,
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
        on_message("Opening browser for Google authentication...")
        on_message(f"If browser doesn't open, visit:\n{auth_url}")
        on_message("")
        on_message("After authenticating, copy the authorization code and run:")
        on_message("  antigravity-auth code <paste_code_here>")

    # Open browser in background
    threading.Thread(target=webbrowser.open, args=(auth_url,), daemon=True).start()

    return auth_url


def complete_interactive_login(
    auth_code: str,
    on_message: Optional[Callable[[str], None]] = None,
) -> Optional[Account]:
    """Complete two-step interactive login with authorization code.

    Args:
        auth_code: Authorization code from browser callback.
        on_message: Optional callback for status messages.

    Returns:
        Account if successful, None if no pending auth or exchange failed.
    """
    code_verifier, state = load_pending_auth()

    if not code_verifier:
        if on_message:
            on_message("No pending login found. Run 'antigravity-auth login' first.")
        return None

    try:
        tokens = exchange_code_for_tokens(auth_code, code_verifier)

        # Try to get project ID
        project_id = load_code_assist_project(tokens.access_token)
        if not project_id:
            project_id = DEFAULT_PROJECT_ID

        tokens.project_id = project_id

        # Create account
        account = Account(
            email=tokens.email or "unknown",
            tokens=tokens,
            project_id=project_id,
        )

        # Save to account manager
        manager = load_accounts()
        manager.add_account(account)
        save_accounts(manager)
        clear_pending_auth()

        if on_message:
            on_message(f"Authenticated as: {account.email}")
            on_message(f"Project ID: {project_id}")

        return account

    except Exception as e:
        if on_message:
            on_message(f"Token exchange failed: {e}")
        return None
