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
from shared.secret_repr import secret_safe_repr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..subagent.config import expand_variables

logger = logging.getLogger(__name__)


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


@dataclass
class RouteConfig:
    """Configuration for a single webhook route.

    Each route maps a URL path to an event source with optional HMAC
    verification and event type extraction from headers.

    Attributes:
        path: URL path for this route (e.g., '/webhook/github').
        secret_header: Header containing the HMAC signature (e.g., 'X-Hub-Signature-256').
        secret_algo: HMAC algorithm (e.g., 'hmac-sha256'). Only 'hmac-sha256' supported.
        event_type_header: Header to extract event type from (e.g., 'X-GitHub-Event').
        metadata: Static metadata merged into every event from this route.
        allow_unauthenticated: Explicit opt-in to accept UNSIGNED requests on
            this route. Default False (fail-closed). A route without an HMAC
            secret is refused unless the deployment provides mutual TLS or an
            IP allowlist, or this flag is set — so an operator cannot expose an
            open ingestion endpoint by omission.
    """
    path: str
    secret_header: Optional[str] = None
    secret_algo: Optional[str] = None
    event_type_header: Optional[str] = None
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
            metadata=data.get('metadata', {}),
            allow_unauthenticated=_as_bool_strict(data.get('allow_unauthenticated', False)),
        )


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
        secret: Global shared secret for HMAC verification. Per-route secrets
            override this when the route's secret_header is set. Use
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

    # Never print the shared secret (#721).  The config is loaded
    # from ``.jaato/webhook.json`` with ``${ENV_VAR}`` expansion, so
    # by the time it is an object the placeholder has been resolved
    # to the real HMAC key.
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

    routes = data.get('routes')
    if routes is not None:
        if not isinstance(routes, dict):
            errors.append("'routes' must be an object")
        else:
            for name, route in routes.items():
                if not isinstance(route, dict):
                    errors.append(f"routes['{name}'] must be an object")
                    continue
                path = route.get('path')
                if not path or not isinstance(path, str):
                    errors.append(f"routes['{name}'].path is required and must be a string")
                elif not path.startswith('/'):
                    errors.append(f"routes['{name}'].path must start with '/'")

                algo = route.get('secret_algo')
                if algo is not None and algo != 'hmac-sha256':
                    errors.append(
                        f"routes['{name}'].secret_algo must be 'hmac-sha256' "
                        f"(got '{algo}')"
                    )

    return len(errors) == 0, errors
