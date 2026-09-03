"""Authentication handling for service connector.

Manages authentication credentials from environment variables and
handles OAuth2 token lifecycle.

Credential reads use ``get_session_env()`` which checks the
session-scoped ``ContextVar`` first, avoiding the race where
concurrent sessions clobber each other's ``os.environ`` values.

Each resolution produces an :class:`AuthAttempt` carrying structured
provenance (where the credential came from, what shape it has) that
flows out through ``get_auth_headers`` and back to ``call_service``'s
error path.  This is the data the agent needs to give a useful 401
diagnosis ("the token from ``pass://X`` is stale, rotate with
``pass edit X`` and retry") instead of "Bad credentials."  The
credential value itself is never carried beyond the bearer/basic
header it produces — only safe metadata (length, 4-char prefix, source
URI) lands in :class:`AuthAttempt`.
"""

import base64
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from shared.session_context import get_session_env
from shared.secret_repr import secret_safe_repr
from ..subagent.config import (
    _SECRET_URI_RE,
    _resolve_secret_uri,
    expand_variables,
)
from .types import AuthConfig, AuthType, ParameterLocation

logger = logging.getLogger(__name__)


# Maps secret-URI scheme → command template.  ``{path}`` and ``{key}``
# are substituted from the parsed URI.  When a scheme is absent here,
# rotation_hint stays None and the auth-error hint omits the suggestion
# rather than guessing.  Add entries as new resolvers ship.
_ROTATION_HINTS: Dict[str, str] = {
    "pass": "pass edit {path}",
}


# AuthSource methods.  These are the per-credential roles carried in
# AuthSource.method so consumers can distinguish "the bearer token" from
# "the basic-auth password" without re-deriving the auth type.
METHOD_API_KEY = "api_key"
METHOD_BEARER = "bearer"
METHOD_BASIC_USERNAME = "basic.username"
METHOD_BASIC_PASSWORD = "basic.password"
METHOD_OAUTH2_CLIENT_ID = "oauth2.client_id"
METHOD_OAUTH2_CLIENT_SECRET = "oauth2.client_secret"

# Methods whose values are secret-like (warrant len/prefix capture for
# diagnosis).  Username and OAuth2 client_id are identifiers, not
# secrets, so we never capture their length or prefix.
_SECRET_LIKE_METHODS = frozenset({
    METHOD_API_KEY,
    METHOD_BEARER,
    METHOD_BASIC_PASSWORD,
    METHOD_OAUTH2_CLIENT_SECRET,
})


def _rotation_hint_for_uri(uri: str) -> Optional[str]:
    """Derive a "how to rotate this credential" command from a secret URI.

    Returns a shell command string the user can run to update the
    backing secret (e.g. ``pass edit jaato/github-token`` for
    ``pass://jaato/github-token``).  Returns ``None`` for URIs whose
    scheme isn't in :data:`_ROTATION_HINTS` — we'd rather omit the
    suggestion than guess at an incorrect command.
    """
    m = _SECRET_URI_RE.match(uri)
    if not m:
        return None
    template = _ROTATION_HINTS.get(m.group("scheme"))
    if template is None:
        return None
    return template.format(path=m.group("path"), key=m.group("key") or "")


@dataclass(frozen=True)
class AuthSource:
    """Where one credential was resolved from.

    Captured per credential during ``AuthManager.get_auth_headers`` so
    error consumers can name the exact provenance instead of guessing.
    Never carries the credential value itself.

    Attributes:
        method: The credential role (one of the ``METHOD_*`` constants
            in this module).  Lets consumers tell apart e.g. the bearer
            token from the basic-auth password without re-deriving auth
            type.
        raw_field: The original ``AuthConfig.*_env`` value as the user
            wrote it — env var name OR ``scheme://path[#key]`` URI.
        kind: One of ``"env"`` (raw_field is an env var name),
            ``"uri"`` (raw_field is a secret URI), or ``"unset"``
            (raw_field was empty/None).
        env_var: Populated when ``kind == "env"``.
        uri: Populated when ``kind == "uri"``.
        rotation_hint: Shell command that rotates the underlying
            secret, when the scheme is recognised by
            :data:`_ROTATION_HINTS`.  ``None`` for env vars or
            unknown schemes.
    """
    method: str
    raw_field: str
    kind: str
    env_var: Optional[str] = None
    uri: Optional[str] = None
    rotation_hint: Optional[str] = None

    @property
    def provenance(self) -> str:
        """Human-readable source description, safe to log and surface."""
        if self.kind == "env" and self.env_var:
            return f"env var {self.env_var}"
        if self.kind == "uri" and self.uri:
            return f"secret URI {self.uri}"
        return "(unset)"


@dataclass(frozen=True)
class AuthAttempt:
    """Outcome of one credential resolution attempt.

    Carries the :class:`AuthSource` (where the credential was supposed
    to come from) plus token-shape info (length, 4-char prefix) for
    secret-like fields.  These shape hints let the agent identify the
    credential type (``ghp_``, ``sk-ant-``, ``oauth-``) without ever
    seeing the secret itself.

    Attributes:
        source: Where the credential came from (or was meant to).
        resolved: ``True`` if a non-empty value was obtained.  ``False``
            when the env var is absent, the URI scheme is unregistered,
            or the resolver returned an empty value.
        token_len: Length of the resolved value, or ``None``.  Only
            populated when ``source.method`` is in
            :data:`_SECRET_LIKE_METHODS` AND a value was resolved.
        token_prefix: First 4 characters of the resolved value, or
            ``None``.  Same gating as ``token_len``.
    """
    source: AuthSource
    resolved: bool
    token_len: Optional[int] = None
    token_prefix: Optional[str] = None


def _make_source(name_or_uri: Optional[str], method: str) -> AuthSource:
    """Build an :class:`AuthSource` from a raw ``*_env`` field value."""
    if not name_or_uri:
        return AuthSource(
            method=method,
            raw_field="",
            kind="unset",
        )
    if _SECRET_URI_RE.match(name_or_uri):
        return AuthSource(
            method=method,
            raw_field=name_or_uri,
            kind="uri",
            uri=name_or_uri,
            rotation_hint=_rotation_hint_for_uri(name_or_uri),
        )
    return AuthSource(
        method=method,
        raw_field=name_or_uri,
        kind="env",
        env_var=name_or_uri,
    )


def _make_attempt(value: Optional[str], source: AuthSource) -> AuthAttempt:
    """Wrap a resolution outcome in an :class:`AuthAttempt`.

    Captures token-shape metadata (len, 4-char prefix) only for
    secret-like methods so usernames/client-ids don't get prefix-leaked.
    """
    if not value:
        return AuthAttempt(source=source, resolved=False)
    if source.method in _SECRET_LIKE_METHODS:
        return AuthAttempt(
            source=source,
            resolved=True,
            token_len=len(value),
            token_prefix=value[:4],
        )
    return AuthAttempt(source=source, resolved=True)


def _resolve_credential(
    name_or_uri: Optional[str],
    method: str,
) -> Tuple[Optional[str], AuthSource]:
    """Resolve a credential value from either an env var name or a secret URI.

    The ``AuthConfig`` fields named ``*_env`` historically held literal
    environment variable names.  They now also accept
    ``scheme://path[#key]`` secret URIs resolved via the
    :class:`SecretResolver` registry discovered from the
    ``jaato.premium`` entry point (``pass://`` from jaato-premium's
    ``pass(1)`` resolver, ``vault://`` from a HashiCorp Vault resolver,
    etc.).  This helper centralises the two-mode lookup so every auth
    type benefits uniformly.

    Args:
        name_or_uri: The value of an ``AuthConfig.*_env`` field.
        method: The credential role this resolution is for; flows into
            :attr:`AuthSource.method`.

    Returns:
        Tuple of ``(resolved_value, source)`` where ``source`` is an
        :class:`AuthSource` carrying structured provenance.  Use
        ``source.provenance`` for the human-readable string previously
        returned by this function.

    Raises:
        SecretResolutionError: propagated when a URI matches a
            registered resolver but resolution itself fails.
    """
    source = _make_source(name_or_uri, method)

    if not name_or_uri:
        return None, source

    if source.kind == "uri":
        # Secret URI path — resolve via registered SecretResolver.
        # _resolve_secret_uri returns the URI unchanged when no resolver
        # is registered (with a warning log); treat that as "not resolved".
        resolved = _resolve_secret_uri(name_or_uri)
        if resolved == name_or_uri:
            return None, source
        return resolved, source

    # Env var path (legacy behaviour).
    return get_session_env(name_or_uri), source


class AuthError(Exception):
    """Authentication error.

    Carries the partial list of :class:`AuthAttempt` collected up to
    the point of failure so the ``call_service`` error path can attach
    the same structured ``auth_context`` it would on a 401/403 response.
    Callers building one without any attempts (e.g. unit tests) can
    omit the second argument.
    """

    def __init__(
        self, message: str, attempts: Optional[List[AuthAttempt]] = None
    ) -> None:
        super().__init__(message)
        self.attempts: List[AuthAttempt] = list(attempts) if attempts else []


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

    # Never print the token (#721).
    __repr__ = secret_safe_repr("access_token")

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
    ) -> Tuple[Dict[str, str], Dict[str, str], List[AuthAttempt]]:
        """Get authentication headers, query params, and resolution attempts.

        Args:
            auth_config: Authentication configuration.
            service_name: Service name (for OAuth2 token caching).

        Returns:
            Tuple of (headers, query_params, attempts) where ``attempts``
            is one :class:`AuthAttempt` per credential the request needed
            (one for bearer/api_key, two for basic, two for oauth2 on
            cache miss / zero on cache hit).  The list flows out to
            ``call_service``'s 401/403 path so the model can attach
            rotation hints to the diagnostic.

        Raises:
            AuthError: If required credentials are missing.  The
                exception carries the partial ``attempts`` list so the
                caller can build the same auth_context it would on a
                401 response.
        """
        headers: Dict[str, str] = {}
        query_params: Dict[str, str] = {}
        attempts: List[AuthAttempt] = []

        if auth_config.type == AuthType.NONE:
            return headers, query_params, attempts

        elif auth_config.type == AuthType.API_KEY:
            if not auth_config.value_env:
                raise AuthError("API key env var not configured")

            api_key, source = _resolve_credential(
                auth_config.value_env, METHOD_API_KEY
            )
            attempts.append(_make_attempt(api_key, source))
            if not api_key:
                raise AuthError(
                    f"API key could not be resolved from {source.provenance}",
                    attempts=attempts,
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

            token, source = _resolve_credential(
                auth_config.value_env, METHOD_BEARER
            )
            attempts.append(_make_attempt(token, source))
            if not token:
                raise AuthError(
                    f"Bearer token could not be resolved from {source.provenance}",
                    attempts=attempts,
                )

            logger.info(
                "BEARER auth: source=%s len=%d prefix=%s",
                source.provenance, len(token), token[:4] + "...",
            )
            headers["Authorization"] = f"Bearer {token}"

        elif auth_config.type == AuthType.BASIC:
            if not auth_config.username_env or not auth_config.password_env:
                raise AuthError("Basic auth env vars not configured")

            username, user_source = _resolve_credential(
                auth_config.username_env, METHOD_BASIC_USERNAME
            )
            attempts.append(_make_attempt(username, user_source))
            password, pass_source = _resolve_credential(
                auth_config.password_env, METHOD_BASIC_PASSWORD
            )
            attempts.append(_make_attempt(password, pass_source))

            if not username:
                raise AuthError(
                    f"Username could not be resolved from {user_source.provenance}",
                    attempts=attempts,
                )
            if not password:
                raise AuthError(
                    f"Password could not be resolved from {pass_source.provenance}",
                    attempts=attempts,
                )

            credentials = base64.b64encode(
                f"{username}:{password}".encode()
            ).decode("ascii")
            headers["Authorization"] = f"Basic {credentials}"

        elif auth_config.type == AuthType.OAUTH2_CLIENT:
            token, oauth_attempts = self._get_oauth2_token(
                auth_config, service_name
            )
            attempts.extend(oauth_attempts)
            headers["Authorization"] = f"{token.token_type} {token.access_token}"

        return headers, query_params, attempts

    def _get_oauth2_token(
        self,
        auth_config: AuthConfig,
        service_name: Optional[str] = None
    ) -> Tuple[OAuth2Token, List[AuthAttempt]]:
        """Get OAuth2 token, fetching or refreshing if needed.

        Returns the token plus the resolution attempts that produced
        it.  On cache hit, ``attempts`` is empty — the credentials were
        resolved on a prior turn and not re-checked here.  The agent's
        401 diagnosis still has ``auth_type`` from the calling
        ``get_auth_headers`` to recognise OAuth2.

        Raises:
            AuthError: If token cannot be obtained.  Carries any
                client-credential attempts collected before the failure.
        """
        cache_key = service_name or auth_config.token_url or "default"

        # Check cache
        cached = self._token_cache.get(cache_key)
        if cached and not cached.is_expired:
            return cached, []

        # Fetch new token (returns attempts so cache-miss path surfaces
        # client_id/client_secret provenance to the caller).
        token, attempts = self._fetch_oauth2_token(auth_config)
        self._token_cache[cache_key] = token
        return token, attempts

    def _fetch_oauth2_token(
        self, auth_config: AuthConfig
    ) -> Tuple[OAuth2Token, List[AuthAttempt]]:
        """Fetch a new OAuth2 token using client credentials flow.

        Returns the token plus the :class:`AuthAttempt` list for the
        client_id and client_secret resolutions.  These are the
        credentials whose provenance the agent needs to diagnose a
        client-credentials failure (wrong env var, stale ``pass://``
        entry, etc.) — the issued access token's own metadata is not
        captured here in V1; ``auth_type=oauth2_client_credentials`` in
        the auth_context is enough for the agent to recommend a flow.

        Raises:
            AuthError: If token fetch fails.  Carries the partial
                attempts list (one or two entries depending on which
                credential resolved before the failure).
        """
        attempts: List[AuthAttempt] = []

        if not auth_config.token_url:
            raise AuthError("OAuth2 token URL not configured", attempts=attempts)

        # Expand ${VAR} in token_url at request time so .env values are available
        token_url = expand_variables(auth_config.token_url)

        if not auth_config.client_id_env or not auth_config.client_secret_env:
            raise AuthError(
                "OAuth2 client credentials env vars not configured",
                attempts=attempts,
            )

        client_id, cid_source = _resolve_credential(
            auth_config.client_id_env, METHOD_OAUTH2_CLIENT_ID
        )
        attempts.append(_make_attempt(client_id, cid_source))
        client_secret, cs_source = _resolve_credential(
            auth_config.client_secret_env, METHOD_OAUTH2_CLIENT_SECRET
        )
        attempts.append(_make_attempt(client_secret, cs_source))

        if not client_id:
            raise AuthError(
                f"Client ID could not be resolved from {cid_source.provenance}",
                attempts=attempts,
            )
        if not client_secret:
            raise AuthError(
                f"Client secret could not be resolved from {cs_source.provenance}",
                attempts=attempts,
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

                proxy_kwargs = get_requests_kwargs(token_url)
                response = requests.post(
                    token_url,
                    data=data,
                    timeout=30,
                    **proxy_kwargs
                )
                response.raise_for_status()
                token_data = response.json()
            except ImportError:
                raise AuthError(
                    "httpx or requests is required for OAuth2. "
                    "Install with: pip install httpx",
                    attempts=attempts,
                )
            except Exception as e:
                raise AuthError(
                    f"OAuth2 token request failed: {e}", attempts=attempts
                )
        else:
            from shared.http import get_httpx_kwargs

            proxy_kwargs = get_httpx_kwargs(token_url)

            try:
                with httpx.Client(timeout=30, **proxy_kwargs) as client:
                    response = client.post(token_url, data=data)
                    response.raise_for_status()
                    token_data = response.json()
            except httpx.HTTPError as e:
                raise AuthError(
                    f"OAuth2 token request failed: {e}", attempts=attempts
                )

        # Parse response
        access_token = token_data.get("access_token")
        if not access_token:
            raise AuthError(
                "OAuth2 response missing access_token", attempts=attempts
            )

        expires_in = token_data.get("expires_in")
        expires_at = None
        if expires_in:
            expires_at = time.time() + int(expires_in)

        token = OAuth2Token(
            access_token=access_token,
            token_type=token_data.get("token_type", "Bearer"),
            expires_at=expires_at,
            scope=token_data.get("scope"),
        )
        return token, attempts

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
        """Check which credentials are present.

        For each ``*_env`` field, classifies it as present or missing
        using the same :func:`_resolve_credential` helper that
        :meth:`get_auth_headers` uses at request time — so a value
        registered as a secret URI (``pass://``, ``vault://``, …)
        is correctly reported as present when its resolver succeeds,
        not falsely as missing because ``get_session_env(URI_string)``
        returned ``None``.

        Args:
            auth_config: Authentication configuration.

        Returns:
            Dict with env_vars_required, env_vars_present,
            env_vars_missing.  The list values preserve the original
            ``*_env`` strings (env var names OR URIs) so the user can
            tell at a glance what kind of source each credential uses.
        """
        # Each entry pairs the raw *_env string with the credential
        # role it serves — needed to call _resolve_credential, which now
        # requires the method for AuthSource construction.
        required_with_method: List[Tuple[str, str]] = []

        if auth_config.type == AuthType.NONE:
            pass

        elif auth_config.type == AuthType.API_KEY:
            if auth_config.value_env:
                required_with_method.append(
                    (auth_config.value_env, METHOD_API_KEY)
                )

        elif auth_config.type == AuthType.BEARER:
            if auth_config.value_env:
                required_with_method.append(
                    (auth_config.value_env, METHOD_BEARER)
                )

        elif auth_config.type == AuthType.BASIC:
            if auth_config.username_env:
                required_with_method.append(
                    (auth_config.username_env, METHOD_BASIC_USERNAME)
                )
            if auth_config.password_env:
                required_with_method.append(
                    (auth_config.password_env, METHOD_BASIC_PASSWORD)
                )

        elif auth_config.type == AuthType.OAUTH2_CLIENT:
            if auth_config.client_id_env:
                required_with_method.append(
                    (auth_config.client_id_env, METHOD_OAUTH2_CLIENT_ID)
                )
            if auth_config.client_secret_env:
                required_with_method.append(
                    (auth_config.client_secret_env, METHOD_OAUTH2_CLIENT_SECRET)
                )

        required: List[str] = [s for s, _ in required_with_method]
        present: List[str] = []
        missing: List[str] = []

        # Check which are present — same resolution path as request time.
        for raw, method in required_with_method:
            value, _source = _resolve_credential(raw, method)
            if value:
                present.append(raw)
            else:
                missing.append(raw)

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
