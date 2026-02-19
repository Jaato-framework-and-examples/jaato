"""Authentication handling for service connector.

Manages authentication credentials from environment variables and
handles OAuth2 token lifecycle.
"""

import base64
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .types import AuthConfig, AuthType, ParameterLocation


class AuthError(Exception):
    """Authentication error."""
    pass


@dataclass
class OAuth2Token:
    """Cached OAuth2 access token.

    Attributes:
        access_token: The access token string.
        token_type: Token type (usually "Bearer").
        expires_at: Unix timestamp when token expires.
        scope: Granted scope (may differ from requested).
    """
    access_token: str
    token_type: str = "Bearer"
    expires_at: Optional[float] = None
    scope: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Check if token has expired (with 60s buffer)."""
        if self.expires_at is None:
            return False
        return time.time() >= (self.expires_at - 60)


class AuthManager:
    """Manages authentication for service requests.

    Handles:
    - Reading credentials from environment variables
    - Building authentication headers
    - OAuth2 token lifecycle (fetch, cache, refresh)

    Attributes:
        token_cache: In-memory cache of OAuth2 tokens by service name.
    """

    def __init__(self):
        """Initialize the auth manager."""
        # OAuth2 token cache: service_name -> OAuth2Token
        self._token_cache: Dict[str, OAuth2Token] = {}

    def get_auth_headers(
        self,
        auth_config: AuthConfig,
        service_name: Optional[str] = None
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Get authentication headers and query params.

        Args:
            auth_config: Authentication configuration.
            service_name: Service name (for OAuth2 token caching).

        Returns:
            Tuple of (headers dict, query_params dict).

        Raises:
            AuthError: If required credentials are missing.
        """
        headers: Dict[str, str] = {}
        query_params: Dict[str, str] = {}

        if auth_config.type == AuthType.NONE:
            return headers, query_params

        elif auth_config.type == AuthType.API_KEY:
            if not auth_config.value_env:
                raise AuthError("API key env var not configured")

            api_key = os.environ.get(auth_config.value_env)
            if not api_key:
                raise AuthError(
                    f"API key not found in environment variable: {auth_config.value_env}"
                )

            if auth_config.key_location == ParameterLocation.HEADER:
                key_name = auth_config.key_name or "X-API-Key"
                headers[key_name] = api_key
            elif auth_config.key_location == ParameterLocation.QUERY:
                key_name = auth_config.key_name or "api_key"
                query_params[key_name] = api_key
            else:
                # Default to header
                key_name = auth_config.key_name or "X-API-Key"
                headers[key_name] = api_key

        elif auth_config.type == AuthType.BEARER:
            if not auth_config.value_env:
                raise AuthError("Bearer token env var not configured")

            token = os.environ.get(auth_config.value_env)
            if not token:
                raise AuthError(
                    f"Bearer token not found in environment variable: {auth_config.value_env}"
                )

            headers["Authorization"] = f"Bearer {token}"

        elif auth_config.type == AuthType.BASIC:
            if not auth_config.username_env or not auth_config.password_env:
                raise AuthError("Basic auth env vars not configured")

            username = os.environ.get(auth_config.username_env)
            password = os.environ.get(auth_config.password_env)

            if not username:
                raise AuthError(
                    f"Username not found in environment variable: {auth_config.username_env}"
                )
            if not password:
                raise AuthError(
                    f"Password not found in environment variable: {auth_config.password_env}"
                )

            credentials = base64.b64encode(
                f"{username}:{password}".encode()
            ).decode("ascii")
            headers["Authorization"] = f"Basic {credentials}"

        elif auth_config.type == AuthType.OAUTH2_CLIENT:
            token = self._get_oauth2_token(auth_config, service_name)
            headers["Authorization"] = f"{token.token_type} {token.access_token}"

        return headers, query_params

    def _get_oauth2_token(
        self,
        auth_config: AuthConfig,
        service_name: Optional[str] = None
    ) -> OAuth2Token:
        """Get OAuth2 token, fetching or refreshing if needed.

        Args:
            auth_config: OAuth2 auth configuration.
            service_name: Service name for caching.

        Returns:
            Valid OAuth2Token.

        Raises:
            AuthError: If token cannot be obtained.
        """
        cache_key = service_name or auth_config.token_url or "default"

        # Check cache
        cached = self._token_cache.get(cache_key)
        if cached and not cached.is_expired:
            return cached

        # Fetch new token
        token = self._fetch_oauth2_token(auth_config)
        self._token_cache[cache_key] = token
        return token

    def _fetch_oauth2_token(self, auth_config: AuthConfig) -> OAuth2Token:
        """Fetch a new OAuth2 token using client credentials flow.

        Args:
            auth_config: OAuth2 auth configuration.

        Returns:
            New OAuth2Token.

        Raises:
            AuthError: If token fetch fails.
        """
        if not auth_config.token_url:
            raise AuthError("OAuth2 token URL not configured")

        if not auth_config.client_id_env or not auth_config.client_secret_env:
            raise AuthError("OAuth2 client credentials env vars not configured")

        client_id = os.environ.get(auth_config.client_id_env)
        client_secret = os.environ.get(auth_config.client_secret_env)

        if not client_id:
            raise AuthError(
                f"Client ID not found in environment variable: {auth_config.client_id_env}"
            )
        if not client_secret:
            raise AuthError(
                f"Client secret not found in environment variable: {auth_config.client_secret_env}"
            )

        # Build token request
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }

        if auth_config.scope:
            data["scope"] = auth_config.scope

        # Make token request
        try:
            import httpx
        except ImportError:
            try:
                import requests
                from shared.http import get_requests_kwargs

                proxy_kwargs = get_requests_kwargs(auth_config.token_url)
                response = requests.post(
                    auth_config.token_url,
                    data=data,
                    timeout=30,
                    **proxy_kwargs
                )
                response.raise_for_status()
                token_data = response.json()
            except ImportError:
                raise AuthError(
                    "httpx or requests is required for OAuth2. "
                    "Install with: pip install httpx"
                )
            except Exception as e:
                raise AuthError(f"OAuth2 token request failed: {e}")
        else:
            from shared.http import get_httpx_kwargs

            proxy_kwargs = get_httpx_kwargs(auth_config.token_url)

            try:
                with httpx.Client(timeout=30, **proxy_kwargs) as client:
                    response = client.post(auth_config.token_url, data=data)
                    response.raise_for_status()
                    token_data = response.json()
            except httpx.HTTPError as e:
                raise AuthError(f"OAuth2 token request failed: {e}")

        # Parse response
        access_token = token_data.get("access_token")
        if not access_token:
            raise AuthError("OAuth2 response missing access_token")

        expires_in = token_data.get("expires_in")
        expires_at = None
        if expires_in:
            expires_at = time.time() + int(expires_in)

        return OAuth2Token(
            access_token=access_token,
            token_type=token_data.get("token_type", "Bearer"),
            expires_at=expires_at,
            scope=token_data.get("scope"),
        )

    def clear_token_cache(self, service_name: Optional[str] = None) -> None:
        """Clear cached OAuth2 tokens.

        Args:
            service_name: Specific service to clear, or None for all.
        """
        if service_name:
            self._token_cache.pop(service_name, None)
        else:
            self._token_cache.clear()

    def check_credentials(
        self,
        auth_config: AuthConfig
    ) -> Dict[str, Any]:
        """Check which credentials are present in environment.

        Args:
            auth_config: Authentication configuration.

        Returns:
            Dict with env_vars_required, env_vars_present, env_vars_missing.
        """
        required: list[str] = []
        present: list[str] = []
        missing: list[str] = []

        if auth_config.type == AuthType.NONE:
            pass

        elif auth_config.type == AuthType.API_KEY:
            if auth_config.value_env:
                required.append(auth_config.value_env)

        elif auth_config.type == AuthType.BEARER:
            if auth_config.value_env:
                required.append(auth_config.value_env)

        elif auth_config.type == AuthType.BASIC:
            if auth_config.username_env:
                required.append(auth_config.username_env)
            if auth_config.password_env:
                required.append(auth_config.password_env)

        elif auth_config.type == AuthType.OAUTH2_CLIENT:
            if auth_config.client_id_env:
                required.append(auth_config.client_id_env)
            if auth_config.client_secret_env:
                required.append(auth_config.client_secret_env)

        # Check which are present
        for env_var in required:
            if os.environ.get(env_var):
                present.append(env_var)
            else:
                missing.append(env_var)

        return {
            "env_vars_required": required,
            "env_vars_present": present,
            "env_vars_missing": missing,
        }

    def redact_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Redact sensitive values from headers for logging.

        Args:
            headers: Headers dict.

        Returns:
            Headers dict with sensitive values redacted.
        """
        sensitive_keys = {"authorization", "x-api-key", "api-key", "apikey"}
        redacted = {}

        for key, value in headers.items():
            if key.lower() in sensitive_keys:
                # Show first/last few characters
                if len(value) > 12:
                    redacted[key] = f"{value[:4]}...{value[-4:]}"
                else:
                    redacted[key] = "***"
            else:
                redacted[key] = value

        return redacted
