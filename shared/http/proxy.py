"""Proxy configuration and Kerberos authentication for HTTP clients.

Provides unified proxy support for urllib, requests, and httpx with:
- Standard proxy environment variables (HTTP_PROXY, HTTPS_PROXY, NO_PROXY)
- JAATO_NO_PROXY for exact host matching
- JAATO_KERBEROS_PROXY for Kerberos/SPNEGO proxy authentication

Platform Support for Kerberos:
- Windows: Native SSPI via pyspnego
- macOS: GSS.framework via pyspnego
- Linux: MIT Kerberos via pyspnego
"""

import base64
import logging
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================
# Environment Variables
# ============================================================

ENV_HTTPS_PROXY = "HTTPS_PROXY"
ENV_HTTP_PROXY = "HTTP_PROXY"
ENV_NO_PROXY = "NO_PROXY"
ENV_JAATO_NO_PROXY = "JAATO_NO_PROXY"
ENV_JAATO_KERBEROS_PROXY = "JAATO_KERBEROS_PROXY"


# ============================================================
# Configuration Functions
# ============================================================

def get_proxy_url() -> Optional[str]:
    """Get proxy URL from environment variables.

    Checks HTTPS_PROXY and HTTP_PROXY (both cases).

    Returns:
        Proxy URL or None if not configured.
    """
    for var in [ENV_HTTPS_PROXY, ENV_HTTPS_PROXY.lower(),
                ENV_HTTP_PROXY, ENV_HTTP_PROXY.lower()]:
        url = os.environ.get(var)
        if url:
            return url
    return None


def parse_proxy_url(proxy_url: str) -> Tuple[str, int, str]:
    """Parse proxy URL into components.

    Args:
        proxy_url: Proxy URL like http://proxy.example.com:8080

    Returns:
        Tuple of (hostname, port, scheme)
    """
    parsed = urllib.parse.urlparse(proxy_url)
    hostname = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 8080)
    scheme = parsed.scheme or "http"
    return hostname, port, scheme


def is_kerberos_proxy_enabled() -> bool:
    """Check if Kerberos proxy authentication is enabled.

    Enabled when JAATO_KERBEROS_PROXY is set to 'true', '1', or 'yes'.

    Returns:
        True if Kerberos proxy auth should be used.
    """
    value = os.environ.get(ENV_JAATO_KERBEROS_PROXY, "").lower()
    return value in ("true", "1", "yes")


def _get_jaato_no_proxy_hosts() -> list:
    """Get list of hosts from JAATO_NO_PROXY.

    Returns:
        List of hostnames (lowercase) that should bypass proxy.
    """
    value = os.environ.get(ENV_JAATO_NO_PROXY, "")
    if not value:
        return []
    return [h.strip().lower() for h in value.split(",") if h.strip()]


def should_bypass_proxy(url: str) -> bool:
    """Check if a URL should bypass proxy based on JAATO_NO_PROXY.

    Uses exact host matching (case-insensitive), unlike standard NO_PROXY
    which does suffix matching.

    Args:
        url: The URL to check.

    Returns:
        True if the host matches exactly an entry in JAATO_NO_PROXY.
    """
    no_proxy_hosts = _get_jaato_no_proxy_hosts()
    if not no_proxy_hosts:
        return False

    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if not host:
        return False

    host = host.lower()
    return host in no_proxy_hosts


# ============================================================
# Kerberos/SPNEGO Support
# ============================================================

def generate_spnego_token(proxy_host: str) -> Optional[str]:
    """Generate SPNEGO token for proxy authentication.

    Uses the system's Kerberos credentials to generate a Negotiate token
    for the HTTP/<proxy_host> service principal.

    Args:
        proxy_host: Proxy server hostname.

    Returns:
        Base64-encoded SPNEGO token, or None if generation fails.
    """
    try:
        import spnego
    except ImportError:
        logger.debug("pyspnego not installed, Kerberos proxy auth unavailable")
        return None

    try:
        # Create SPNEGO client context for HTTP service on proxy
        ctx = spnego.client(
            hostname=proxy_host,
            service="HTTP",
            protocol="negotiate",
        )

        # Generate the initial token
        token = ctx.step()
        if token:
            return base64.b64encode(token).decode("ascii")
        return None

    except Exception as e:
        logger.debug(f"Failed to generate SPNEGO token for {proxy_host}: {e}")
        return None


# ============================================================
# urllib Support
# ============================================================

class KerberosProxyHandler(urllib.request.BaseHandler):
    """urllib handler for Kerberos proxy authentication.

    Handles 407 Proxy Authentication Required responses by generating
    SPNEGO tokens and retrying the request.
    """

    handler_order = 400

    def __init__(self, proxy_url: str):
        self.proxy_url = proxy_url
        self.proxy_host, self.proxy_port, self.proxy_scheme = parse_proxy_url(proxy_url)

    def http_error_407(self, req, fp, code, msg, hdrs):
        """Handle 407 Proxy Authentication Required."""
        from urllib.error import HTTPError

        auth_header = hdrs.get("Proxy-Authenticate", "")
        if "Negotiate" not in auth_header:
            logger.debug(f"Proxy does not support Negotiate auth: {auth_header}")
            raise HTTPError(req.full_url, code, msg, hdrs, fp)

        token = generate_spnego_token(self.proxy_host)
        if not token:
            logger.debug("Failed to generate SPNEGO token")
            raise HTTPError(req.full_url, code, msg, hdrs, fp)

        logger.debug(f"Retrying with SPNEGO token for {self.proxy_host}")
        req.add_header("Proxy-Authorization", f"Negotiate {token}")

        return self.parent.open(req, timeout=req.timeout if hasattr(req, 'timeout') else None)

    https_error_407 = http_error_407


def get_url_opener(url: Optional[str] = None) -> urllib.request.OpenerDirector:
    """Get an appropriate urllib opener.

    Priority:
    1. If url provided and JAATO_NO_PROXY matches (exact), bypass proxy
    2. If JAATO_KERBEROS_PROXY=true, use Kerberos proxy authentication
    3. Otherwise use default opener (standard proxy env vars)

    Args:
        url: Optional URL to check for bypass. If None, returns general opener.

    Returns:
        An OpenerDirector configured appropriately.
    """
    # Check bypass first
    if url and should_bypass_proxy(url):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # Check Kerberos proxy
    if is_kerberos_proxy_enabled():
        proxy_url = get_proxy_url()
        if proxy_url:
            proxy_host, _, _ = parse_proxy_url(proxy_url)
            spnego_token = generate_spnego_token(proxy_host)

            if spnego_token:
                class PreAuthProxyHandler(urllib.request.ProxyHandler):
                    def __init__(self, proxies, token):
                        super().__init__(proxies)
                        self.spnego_token = token

                    def proxy_open(self, req, proxy, type):
                        if not req.has_header("Proxy-Authorization"):
                            req.add_header("Proxy-Authorization", f"Negotiate {self.spnego_token}")
                        return super().proxy_open(req, proxy, type)

                proxy_handler = PreAuthProxyHandler(
                    {"http": proxy_url, "https": proxy_url},
                    spnego_token
                )
                kerberos_handler = KerberosProxyHandler(proxy_url)
                return urllib.request.build_opener(proxy_handler, kerberos_handler)
            else:
                proxy_handler = urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
                kerberos_handler = KerberosProxyHandler(proxy_url)
                return urllib.request.build_opener(proxy_handler, kerberos_handler)

    # Default opener with standard proxy env vars
    return urllib.request.build_opener()


# ============================================================
# requests Support
# ============================================================

def get_requests_session() -> "requests.Session":
    """Create a requests Session with appropriate proxy configuration.

    Configures the session based on:
    1. JAATO_KERBEROS_PROXY: Adds SPNEGO token to headers
    2. Standard proxy env vars (requests handles these automatically)

    Returns:
        Configured requests.Session.
    """
    import requests

    session = requests.Session()

    if is_kerberos_proxy_enabled():
        proxy_url = get_proxy_url()
        if proxy_url:
            proxy_host, _, _ = parse_proxy_url(proxy_url)
            session.proxies = {
                "http": proxy_url,
                "https": proxy_url,
            }
            token = generate_spnego_token(proxy_host)
            if token:
                session.headers["Proxy-Authorization"] = f"Negotiate {token}"

    return session


def get_requests_kwargs(url: str) -> Dict[str, Any]:
    """Get kwargs for a requests call with appropriate proxy configuration.

    Use this when you need per-request proxy configuration instead of
    a session-level configuration.

    Args:
        url: The URL being requested.

    Returns:
        Dict with 'proxies' and optionally 'headers' keys.
    """
    kwargs: Dict[str, Any] = {}

    if should_bypass_proxy(url):
        kwargs["proxies"] = {}
        return kwargs

    if is_kerberos_proxy_enabled():
        proxy_url = get_proxy_url()
        if proxy_url:
            proxy_host, _, _ = parse_proxy_url(proxy_url)
            kwargs["proxies"] = {
                "http": proxy_url,
                "https": proxy_url,
            }
            token = generate_spnego_token(proxy_host)
            if token:
                kwargs["headers"] = {"Proxy-Authorization": f"Negotiate {token}"}

    return kwargs


# ============================================================
# httpx Support
# ============================================================

def get_httpx_client(**client_kwargs) -> "httpx.Client":
    """Create an httpx Client with appropriate proxy configuration.

    Configures the client based on:
    1. JAATO_KERBEROS_PROXY: Adds SPNEGO token to headers
    2. Standard proxy env vars

    Args:
        **client_kwargs: Additional kwargs to pass to httpx.Client

    Returns:
        Configured httpx.Client.
    """
    import httpx

    if is_kerberos_proxy_enabled():
        proxy_url = get_proxy_url()
        if proxy_url:
            proxy_host, _, _ = parse_proxy_url(proxy_url)

            # Set proxy
            client_kwargs.setdefault("proxy", proxy_url)

            # Add SPNEGO token to headers
            token = generate_spnego_token(proxy_host)
            if token:
                headers = client_kwargs.get("headers", {})
                if isinstance(headers, dict):
                    headers["Proxy-Authorization"] = f"Negotiate {token}"
                    client_kwargs["headers"] = headers

    return httpx.Client(**client_kwargs)


def get_httpx_kwargs(url: str) -> Dict[str, Any]:
    """Get kwargs for httpx client/request with appropriate proxy configuration.

    Args:
        url: The URL being requested.

    Returns:
        Dict with proxy and header configuration.
    """
    kwargs: Dict[str, Any] = {}

    if should_bypass_proxy(url):
        kwargs["proxy"] = None
        return kwargs

    if is_kerberos_proxy_enabled():
        proxy_url = get_proxy_url()
        if proxy_url:
            proxy_host, _, _ = parse_proxy_url(proxy_url)
            kwargs["proxy"] = proxy_url
            token = generate_spnego_token(proxy_host)
            if token:
                kwargs["headers"] = {"Proxy-Authorization": f"Negotiate {token}"}

    return kwargs


# ============================================================
# AsyncIO Support (httpx async)
# ============================================================

def get_httpx_async_client(**client_kwargs) -> "httpx.AsyncClient":
    """Create an httpx AsyncClient with appropriate proxy configuration.

    Args:
        **client_kwargs: Additional kwargs to pass to httpx.AsyncClient

    Returns:
        Configured httpx.AsyncClient.
    """
    import httpx

    if is_kerberos_proxy_enabled():
        proxy_url = get_proxy_url()
        if proxy_url:
            proxy_host, _, _ = parse_proxy_url(proxy_url)
            client_kwargs.setdefault("proxy", proxy_url)
            token = generate_spnego_token(proxy_host)
            if token:
                headers = client_kwargs.get("headers", {})
                if isinstance(headers, dict):
                    headers["Proxy-Authorization"] = f"Negotiate {token}"
                    client_kwargs["headers"] = headers

    return httpx.AsyncClient(**client_kwargs)
