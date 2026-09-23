"""Configuration loading and merging for the webhook plugin.

Config precedence (highest to lowest):
1. Profile plugin_configs.webhook   (per-session override)
2. <workspace>/.jaato/webhook.json  (project-level)
3. ~/.jaato/webhook.json            (user-level)
4. Built-in defaults
"""

import json
import logging
import os
from dataclasses import dataclass, field
from jaato_server.shared.secret_repr import secret_safe_repr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..subagent.config import expand_variables
from .replay import DEFAULT_REPLAY_CACHE_SIZE
from .signature_schemes import (
    DEFAULT_MAX_AGE_SECONDS,
    SCHEME_BODY,
    SCHEME_STRIPE_V1,
    SIGNATURE_SCHEMES,
    TIMESTAMP_BOUND_SCHEMES,
    TIMESTAMP_FROM_HEADER,
    SCHEME_TIMESTAMP_SOURCE,
)

logger = logging.getLogger(__name__)

# The vocabulary of ``RouteConfig.secret_algo`` — how the route's shared secret
# is checked against the request.  Anything outside this set is a hard error at
# validation time and a refusal at request time (see ``verify_signature``); the
# fail-closed posture is what makes the set safe to extend.
#
#   'hmac-sha256' — the secret keys an HMAC over the request BODY, and the
#                   header carries the resulting digest.  Not replayable, not
#                   readable by anything that terminates TLS.  (GitHub, Stripe.)
#   'token'       — the header carries the shared secret VERBATIM and is
#                   compared for equality in constant time.  Strictly weaker:
#                   replayable, and exposed to every hop that terminates TLS.
#                   The only mode a producer that does not sign bodies can use
#                   (GitLab's ``X-Gitlab-Token``, Jira, many internal senders).
SECRET_ALGO_HMAC_SHA256 = 'hmac-sha256'
SECRET_ALGO_TOKEN = 'token'
SECRET_ALGOS = (SECRET_ALGO_HMAC_SHA256, SECRET_ALGO_TOKEN)


def _as_bool_strict(value: Any) -> bool:
    """Fail-closed boolean coercion for security flags.

    Native booleans pass through.  Strings (which is what a value becomes after
    ``${ENV_VAR}`` expansion, or when an operator quotes it in YAML/JSON) count
    as True ONLY for an explicit affirmative token — so a stray ``"false"`` or
    ``""`` never silently enables the flag via Python truthiness (``bool("false")``
    is ``True``).  Anything else is False.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(value, (int, float)):
        return value == 1
    return False


def _as_int_or(value: Any, default: int) -> int:
    """Coerce a config value to int, falling back to ``default``.

    ``${VAR}`` expansion turns every value into a string, so an operator's
    ``"max_age_seconds": "${WEBHOOK_MAX_AGE}"`` arrives as ``'300'`` and must
    still be a number.  A value that is not a whole number of seconds falls
    back to the default rather than to ``0``: ``0`` is this key's explicit
    "no freshness window", and a typo must never land on the disabled posture.
    ``validate_config`` reports the same value as an error, so the fallback is
    a safe landing rather than a silent fix.

    Args:
        value: The raw config value.
        default: What to use when ``value`` is absent or unusable.

    Returns:
        The parsed integer, or ``default``.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        logger.warning(
            "webhook config: %r is not a whole number of seconds; "
            "using the default (%d)", value, default,
        )
        return default


@dataclass
class RouteConfig:
    """Configuration for a single webhook route.

    Each route maps a URL path to an event source with optional shared-secret
    verification and event type extraction from headers.

    Attributes:
        path: URL path for this route (e.g., '/webhook/github').
        secret_header: Header carrying the route's credential — an HMAC digest
            under ``secret_algo='hmac-sha256'`` (e.g., 'X-Hub-Signature-256'),
            or the shared secret itself under ``secret_algo='token'``
            (e.g., 'X-Gitlab-Token').
        secret_algo: How that header is verified — one of ``SECRET_ALGOS``.
            ``'hmac-sha256'`` verifies a digest over the request body;
            ``'token'`` compares the header value to the secret in constant
            time.  ``'token'`` is the WEAKER mode (replayable, and readable by
            anything that terminates TLS) and exists for producers that do not
            sign bodies; pair it with TLS.  Both halves of the pair are
            required — declaring one without the other is refused, never
            downgraded to unsigned.
        signature_scheme: How the signed payload is CONSTRUCTED — one of
            ``SIGNATURE_SCHEMES``.  ``'body'`` (default, and every pre-#713
            route) signs the request body alone, so the signature never
            expires.  ``'slack-v0'`` and ``'stripe-v1'`` bind a timestamp into
            the signature, which is what makes a freshness window enforceable.
            See :mod:`.signature_schemes`.
        timestamp_header: Header carrying the signed timestamp, for a scheme
            that reads it from a separate header (``'slack-v0'``:
            ``X-Slack-Request-Timestamp``).  REFUSED on ``'stripe-v1'``, which
            carries its own ``t=`` field, and on ``'body'``, where the
            timestamp would not be covered by the signature and so could be
            rewritten by whoever is replaying the request — a window that
            checks an attacker-controlled value is theatre, not protection.
        max_age_seconds: Freshness window in seconds (default
            ``DEFAULT_MAX_AGE_SECONDS``, Slack's documented 5 minutes).  A
            signed timestamp more than this far from now — in EITHER direction
            — is refused.  ``0`` disables the window and is announced at
            WARNING.  Also sizes the replay cache's TTL.
        replay_key_header: Header carrying a unique delivery id (GitHub's
            ``X-GitHub-Delivery``, GitLab's ``X-Gitlab-Event-UUID``).  On a
            ``'body'`` or ``'token'`` route this is the ONLY thing a replay
            cache can key on, since the credential there is a pure function of
            the body (or constant).  Bounded protection: it refuses a verbatim
            replay and a sender retry within the TTL, and an attacker who
            rewrites the id defeats it, because no sender here signs headers.
        metadata: Static metadata merged into every event from this route.
        allow_unauthenticated: Explicit opt-in to accept UNSIGNED requests on
            this route. Default False (fail-closed). A route without a shared
            secret is refused unless the deployment provides mutual TLS or an
            IP allowlist, or this flag is set — so an operator cannot expose an
            open ingestion endpoint by omission.
    """
    path: str
    secret_header: Optional[str] = None
    secret_algo: Optional[str] = None
    event_type_header: Optional[str] = None
    signature_scheme: str = SCHEME_BODY
    timestamp_header: Optional[str] = None
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS
    replay_key_header: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    allow_unauthenticated: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RouteConfig':
        """Create RouteConfig from a dictionary."""
        return cls(
            path=data.get('path', '/webhook'),
            secret_header=data.get('secret_header'),
            secret_algo=data.get('secret_algo'),
            event_type_header=data.get('event_type_header'),
            # An absent scheme is 'body' — what every route meant before the
            # key existed.  A present but unrecognised one is carried through
            # VERBATIM rather than coerced, so validate_config() and the
            # request path both see the typo and refuse it (fail-closed);
            # defaulting it here would silently verify a mistyped route the
            # old way.
            signature_scheme=data.get('signature_scheme') or SCHEME_BODY,
            timestamp_header=data.get('timestamp_header'),
            max_age_seconds=_as_int_or(
                data.get('max_age_seconds'), DEFAULT_MAX_AGE_SECONDS
            ),
            replay_key_header=data.get('replay_key_header'),
            metadata=data.get('metadata', {}),
            allow_unauthenticated=_as_bool_strict(data.get('allow_unauthenticated', False)),
        )

    def binds_timestamp(self) -> bool:
        """Whether this route's scheme covers a timestamp with the signature.

        Only such a route can carry a freshness window that means anything:
        elsewhere the timestamp is not signed, so whoever replays the request
        can set it to now.

        Returns:
            True for ``'slack-v0'`` / ``'stripe-v1'``.
        """
        return self.signature_scheme in TIMESTAMP_BOUND_SCHEMES

    def replay_protected(self) -> bool:
        """Whether a delivery on this route can be recorded against replay.

        True when the route has something unique-per-delivery to key on: a
        timestamp-bound signature, or a ``replay_key_header`` the sender fills
        in.  False means a replay inside the freshness window (or, for a
        ``'body'`` route, at any time) is indistinguishable from the original.

        Returns:
            True when the replay cache is active for this route.
        """
        return bool(self.binds_timestamp() or self.replay_key_header)


@dataclass
class TLSConfig:
    """TLS/SSL configuration for the webhook HTTP server.

    When ``enabled`` is True, the server wraps its socket with an
    ``ssl.SSLContext``. Both ``certfile`` and ``keyfile`` are required.

    Attributes:
        enabled: Whether to enable TLS (default: False).
        certfile: Path to PEM certificate file (or chain).
        keyfile: Path to PEM private key file.
        ca_certfile: Optional CA certificate for client certificate verification
            (mutual TLS). When set, clients must present a valid certificate
            signed by this CA.
    """
    enabled: bool = False
    certfile: Optional[str] = None
    keyfile: Optional[str] = None
    ca_certfile: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TLSConfig':
        """Create TLSConfig from a dictionary."""
        return cls(
            enabled=data.get('enabled', False),
            certfile=data.get('certfile'),
            keyfile=data.get('keyfile'),
            ca_certfile=data.get('ca_certfile'),
        )


@dataclass
class WebhookConfig:
    """Top-level configuration for the webhook plugin.

    Each plugin instance owns its own HTTP listener port — never shared
    across the server or other plugins.

    Attributes:
        port: HTTP listener port (default: 9100).
        host: Bind address (default: '127.0.0.1' — localhost only).
        secret: Global shared secret used by whichever verification mode a
            route declares — the HMAC key under ``'hmac-sha256'``, the expected
            header value under ``'token'``. Per-route secrets
            (``routes.<name>.metadata.secret``) override this. Use
            ``${ENV_VAR}`` syntax to avoid storing secrets in plain text.
        routes: Named routes mapping source names to RouteConfig.
        max_body_size: Maximum request body size in bytes (default: 1 MB).
        response_timeout: Seconds before responding to webhook sender (default: 5.0).
        tls: TLS/SSL configuration. When enabled, the server uses HTTPS.
        allowed_ips: List of allowed source IPs or CIDR ranges (e.g.,
            ``['192.168.1.0/24', '10.0.0.5']``). Empty list means all IPs
            are allowed. Supports both IPv4 and IPv6.
        rate_limit_per_second: Maximum requests per second per source IP
            (default: 0 = unlimited). Excess requests receive 429.
        replay_cache_size: Maximum delivery keys the listener remembers for
            replay refusal (default ``DEFAULT_REPLAY_CACHE_SIZE``). One cache
            serves every route; entries expire after the route's
            ``max_age_seconds``, so this only binds the pathological case where
            valid deliveries arrive faster than the window retires them. An
            eviction under this ceiling re-opens a replay window for that key
            and is counted and announced — see :class:`.replay.ReplayCache`.
    """
    port: int = 9100
    host: str = '127.0.0.1'
    secret: Optional[str] = None
    routes: Dict[str, RouteConfig] = field(default_factory=dict)
    max_body_size: int = 1048576  # 1 MB
    response_timeout: float = 5.0
    tls: TLSConfig = field(default_factory=TLSConfig)
    allowed_ips: List[str] = field(default_factory=list)
    rate_limit_per_second: float = 0
    replay_cache_size: int = DEFAULT_REPLAY_CACHE_SIZE

    # Never print the shared secret (#721).  The config is loaded
    # from ``.jaato/webhook.json`` with ``${ENV_VAR}`` expansion, so
    # by the time it is an object the placeholder has been resolved
    # to the real secret.
    __repr__ = secret_safe_repr("secret")

    def __post_init__(self):
        # Intentionally do NOT auto-create a route.  A zero-config default route
        # would be an unauthenticated, open ingestion endpoint (the fail-open
        # this class previously had).  Empty routes means the listener matches
        # nothing (every request 404s); the operator must declare routes
        # explicitly, and each unsigned route must opt in via
        # allow_unauthenticated (or be covered by mutual TLS / an IP allowlist).
        # http_server logs a startup warning when no routes are configured.
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebhookConfig':
        """Create WebhookConfig from a dictionary.

        Args:
            data: Raw config dict (from JSON file or plugin_configs).

        Returns:
            WebhookConfig instance.
        """
        routes = {}
        for name, route_data in data.get('routes', {}).items():
            if isinstance(route_data, dict):
                routes[name] = RouteConfig.from_dict(route_data)

        tls_data = data.get('tls', {})
        tls = TLSConfig.from_dict(tls_data) if isinstance(tls_data, dict) else TLSConfig()

        return cls(
            port=data.get('port', 9100),
            host=data.get('host', '127.0.0.1'),
            secret=data.get('secret'),
            routes=routes,
            max_body_size=data.get('max_body_size', 1048576),
            response_timeout=data.get('response_timeout', 5.0),
            tls=tls,
            allowed_ips=data.get('allowed_ips', []),
            rate_limit_per_second=data.get('rate_limit_per_second', 0),
            replay_cache_size=_as_int_or(
                data.get('replay_cache_size'), DEFAULT_REPLAY_CACHE_SIZE
            ),
        )


def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a JSON file, returning None on any error.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dict or None if file doesn't exist or is invalid.
    """
    if not path.is_file():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        logger.warning("webhook.json at %s is not a JSON object", path)
        return None
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load webhook config from %s: %s", path, e)
        return None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge override into base, returning a new dict.

    For nested dicts, merges recursively. For all other types, override wins.

    Args:
        base: Base configuration dict.
        override: Override dict (values take precedence).

    Returns:
        Merged dict.
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    explicit_config: Optional[Dict[str, Any]] = None,
    workspace_path: Optional[str] = None,
) -> WebhookConfig:
    """Load webhook config with standard precedence.

    Precedence (highest to lowest):
    1. explicit_config (from profile plugin_configs.webhook)
    2. <workspace>/.jaato/webhook.json
    3. ~/.jaato/webhook.json
    4. Built-in defaults

    Each layer is deep-merged, not replaced — a profile can override just
    ``port`` while inheriting routes from the workspace config.

    Args:
        explicit_config: Config dict from profile plugin_configs (highest priority).
        workspace_path: Workspace root for finding .jaato/webhook.json.

    Returns:
        Merged WebhookConfig.
    """
    merged: Dict[str, Any] = {}

    # Layer 4 → 3: user home config
    home_config = _load_json_file(Path.home() / '.jaato' / 'webhook.json')
    if home_config:
        merged = _deep_merge(merged, home_config)
        logger.debug("Loaded user-level webhook config from ~/.jaato/webhook.json")

    # Layer 3 → 2: workspace config
    if workspace_path:
        ws_config = _load_json_file(
            Path(workspace_path) / '.jaato' / 'webhook.json'
        )
        if ws_config:
            merged = _deep_merge(merged, ws_config)
            logger.debug(
                "Loaded workspace webhook config from %s/.jaato/webhook.json",
                workspace_path,
            )

    # Layer 2 → 1: explicit profile config
    if explicit_config:
        merged = _deep_merge(merged, explicit_config)
        logger.debug("Applied explicit webhook config from profile")

    # Expand ${VAR} references in string values (env vars + context like ${workspaceRoot}).
    # Uses the shared expand_variables() from subagent config — same function
    # used by the MCP plugin.
    # NOTE: expansion reads os.environ at call time. When called during
    # set_workspace_path() (init), .env values may not yet be in os.environ.
    # Plugins that need .env values should re-resolve config at tool execution
    # time, when _with_session_env() has populated os.environ.
    merged = expand_variables(merged)

    if merged:
        return WebhookConfig.from_dict(merged)
    return WebhookConfig()




def validate_config(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a webhook configuration dict.

    Args:
        data: Raw config dict.

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors: List[str] = []

    port = data.get('port')
    if port is not None:
        if not isinstance(port, int) or isinstance(port, bool):
            errors.append("'port' must be an integer")
        elif port < 1 or port > 65535:
            errors.append("'port' must be between 1 and 65535")

    host = data.get('host')
    if host is not None and not isinstance(host, str):
        errors.append("'host' must be a string")

    max_body = data.get('max_body_size')
    if max_body is not None:
        if not isinstance(max_body, int) or isinstance(max_body, bool):
            errors.append("'max_body_size' must be an integer")
        elif max_body <= 0:
            errors.append("'max_body_size' must be positive")

    timeout = data.get('response_timeout')
    if timeout is not None:
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            errors.append("'response_timeout' must be a number")
        elif timeout <= 0:
            errors.append("'response_timeout' must be positive")

    rate_limit = data.get('rate_limit_per_second')
    if rate_limit is not None:
        if not isinstance(rate_limit, (int, float)) or isinstance(rate_limit, bool):
            errors.append("'rate_limit_per_second' must be a number")
        elif rate_limit < 0:
            errors.append("'rate_limit_per_second' must be non-negative")

    # Validate TLS config
    tls = data.get('tls')
    if tls is not None:
        if not isinstance(tls, dict):
            errors.append("'tls' must be an object")
        else:
            if tls.get('enabled'):
                if not tls.get('certfile'):
                    errors.append("tls.certfile is required when TLS is enabled")
                if not tls.get('keyfile'):
                    errors.append("tls.keyfile is required when TLS is enabled")

    # Validate allowed_ips
    allowed_ips = data.get('allowed_ips')
    if allowed_ips is not None:
        if not isinstance(allowed_ips, list):
            errors.append("'allowed_ips' must be an array")
        else:
            import ipaddress
            for i, entry in enumerate(allowed_ips):
                if not isinstance(entry, str):
                    errors.append(f"allowed_ips[{i}] must be a string")
                    continue
                try:
                    if '/' in entry:
                        ipaddress.ip_network(entry, strict=False)
                    else:
                        ipaddress.ip_address(entry)
                except ValueError as e:
                    errors.append(f"allowed_ips[{i}] is not a valid IP/CIDR: {e}")

    errors.extend(_validate_replay_cache_size(data.get('replay_cache_size')))
    errors.extend(_validate_routes(data.get('routes')))

    return len(errors) == 0, errors


def _validate_replay_cache_size(value: Any) -> List[str]:
    """Validate the top-level ``replay_cache_size`` knob.

    A free function called with ``extend`` rather than an ``if`` inside
    ``validate_config``: that function's cyclomatic score is a frozen ratchet
    entry, and a plain call adds no decision point to it.

    Args:
        value: The raw ``replay_cache_size`` value, or None when absent.

    Returns:
        Human-readable error strings (empty when valid).
    """
    if value is None:
        return []
    if not isinstance(value, int) or isinstance(value, bool):
        return ["'replay_cache_size' must be an integer"]
    if value < 1:
        return [
            "'replay_cache_size' must be at least 1 — a zero-size cache would "
            "accept every replay. To run a route without replay protection, "
            "leave its replay_key_header unset instead."
        ]
    return []


def _validate_routes(routes: Any) -> List[str]:
    """Validate the ``routes`` block of a webhook config.

    Split out of ``validate_config`` so route rules can grow without growing
    that function (its cyclomatic-complexity baseline is a ratchet).

    Args:
        routes: The raw ``routes`` value, or None when the key is absent.

    Returns:
        A list of human-readable error strings (empty when valid).
    """
    if routes is None:
        return []
    if not isinstance(routes, dict):
        return ["'routes' must be an object"]

    errors: List[str] = []
    for name, route in routes.items():
        if not isinstance(route, dict):
            errors.append(f"routes['{name}'] must be an object")
            continue

        path = route.get('path')
        if not path or not isinstance(path, str):
            errors.append(f"routes['{name}'].path is required and must be a string")
        elif not path.startswith('/'):
            errors.append(f"routes['{name}'].path must start with '/'")

        # An unrecognised algo is a HARD ERROR, not a fallback: a typo must
        # never leave a route unverified.  Widening the vocabulary widens this
        # set and nothing else — the incomplete-pair check in
        # ``parse_webhook_request`` is untouched by it.
        algo = route.get('secret_algo')
        if algo is not None and algo not in SECRET_ALGOS:
            allowed = ' or '.join(repr(a) for a in SECRET_ALGOS)
            errors.append(
                f"routes['{name}'].secret_algo must be {allowed} (got '{algo}')"
            )

        errors.extend(_validate_route_freshness(name, route))

    return errors


def _validate_route_freshness(name: str, route: Dict[str, Any]) -> List[str]:
    """Validate one route's replay-protection keys (#713).

    Every finding here is an **error**, not a warning, and every one of them is
    also refused at request time with a 500.  That is deliberate: each case is
    a configuration that is expressible and *does not do what it says*, which
    is the silent-ignore family this tree keeps closing (#910, #925, #947,
    #950).  A window that never fires reads, to whoever wrote it, exactly like
    a window that does.

    The rules:

    * ``signature_scheme`` outside ``SIGNATURE_SCHEMES`` — a typo must not fall
      back to ``'body'``, or a route meant to verify Slack's construction would
      verify something else entirely and say nothing.
    * A timestamp-bound scheme with ``secret_algo`` other than
      ``'hmac-sha256'`` — there is no constant-time-token form of Slack's or
      Stripe's construction, so this is the cross-mode leniency the plugin
      already refuses between ``token`` and ``hmac-sha256``.
    * ``'slack-v0'`` with no ``timestamp_header`` — the scheme has no other
      source for the timestamp it signs.
    * ``'stripe-v1'`` WITH a ``timestamp_header`` — the scheme takes its
      timestamp from ``t=`` inside the signature header, so a second source
      could disagree with the signed one.
    * ``timestamp_header`` on ``'body'`` — the value would not be covered by
      the signature, so whoever replays the request rewrites it. A freshness
      check against an attacker-controlled value is theatre; refusing is how
      the operator finds out before shipping it.
    * ``max_age_seconds`` that is not a non-negative whole number.

    Args:
        name: The route's config key, for the message.
        route: The raw route dict.

    Returns:
        Human-readable error strings (empty when valid).
    """
    errors: List[str] = []

    scheme = route.get('signature_scheme')
    if scheme is not None and scheme not in SIGNATURE_SCHEMES:
        allowed = ' or '.join(repr(s) for s in SIGNATURE_SCHEMES)
        errors.append(
            f"routes['{name}'].signature_scheme must be {allowed} (got '{scheme}')"
        )
        scheme = None

    max_age = route.get('max_age_seconds')
    if max_age is not None:
        if not isinstance(max_age, int) or isinstance(max_age, bool):
            errors.append(f"routes['{name}'].max_age_seconds must be an integer")
        elif max_age < 0:
            errors.append(
                f"routes['{name}'].max_age_seconds must be non-negative "
                f"(0 disables the freshness window)"
            )

    has_ts_header = bool(route.get('timestamp_header'))

    if scheme in TIMESTAMP_BOUND_SCHEMES:
        errors.extend(
            _validate_timestamp_bound_route(name, route, scheme, has_ts_header)
        )
    elif has_ts_header:
        errors.append(
            f"routes['{name}'] sets timestamp_header on signature_scheme "
            f"'{scheme or SCHEME_BODY}', which does not cover the timestamp "
            f"with the signature — anyone replaying the request can rewrite "
            f"it, so the freshness window would check an attacker-controlled "
            f"value. Use signature_scheme 'slack-v0' or '{SCHEME_STRIPE_V1}' "
            f"if the sender signs a timestamp, or replay_key_header for "
            f"delivery-id deduplication."
        )

    return errors


def _validate_timestamp_bound_route(
    name: str,
    route: Dict[str, Any],
    scheme: str,
    has_ts_header: bool,
) -> List[str]:
    """Validate the keys a timestamp-bound scheme requires.

    Split from :func:`_validate_route_freshness` to keep both under the
    cyclomatic ceiling, and because these three rules share one premise the
    caller has already established: this route's scheme signs a timestamp.

    Args:
        name: The route's config key, for the message.
        route: The raw route dict.
        scheme: The route's (already vocabulary-checked) signature scheme.
        has_ts_header: Whether the route names a ``timestamp_header``.

    Returns:
        Human-readable error strings (empty when valid).
    """
    errors: List[str] = []

    algo = route.get('secret_algo')
    if algo is not None and algo != SECRET_ALGO_HMAC_SHA256:
        errors.append(
            f"routes['{name}'].signature_scheme '{scheme}' signs an "
            f"HMAC-SHA256 digest, so secret_algo must be "
            f"'{SECRET_ALGO_HMAC_SHA256}' (got '{algo}')"
        )

    wants_header = SCHEME_TIMESTAMP_SOURCE[scheme] == TIMESTAMP_FROM_HEADER
    if wants_header and not has_ts_header:
        errors.append(
            f"routes['{name}'].signature_scheme '{scheme}' signs a "
            f"timestamp carried in its own header, so timestamp_header is "
            f"required (Slack sends 'X-Slack-Request-Timestamp')"
        )
    elif not wants_header and has_ts_header:
        errors.append(
            f"routes['{name}'].signature_scheme '{scheme}' reads its "
            f"timestamp from the signature header itself, so "
            f"timestamp_header must not be set"
        )

    return errors
