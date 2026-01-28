"""Kerberos/SPNEGO proxy authentication support.

Provides transparent Kerberos authentication for HTTP proxies that require
SPNEGO negotiation (407 Proxy Authentication Required with Negotiate).

This module supports:
- Windows: Uses native SSPI via pyspnego
- macOS: Uses native GSS.framework via pyspnego
- Linux: Uses MIT Kerberos via pyspnego

Usage:
    # For urllib
    opener = create_kerberos_proxy_opener(proxy_url)
    response = opener.open(request)

    # For requests
    session = create_kerberos_proxy_session(proxy_url)
    response = session.get(url)

Requirements:
    pip install pyspnego
"""

import base64
import logging
import os
import urllib.parse
import urllib.request
from typing import Optional, Tuple
from urllib.error import HTTPError

logger = logging.getLogger(__name__)

# Environment variable for proxy URL
ENV_HTTPS_PROXY = "HTTPS_PROXY"
ENV_HTTP_PROXY = "HTTP_PROXY"


def get_proxy_url() -> Optional[str]:
    """Get proxy URL from environment variables.

    Checks HTTPS_PROXY and HTTP_PROXY (case-insensitive).

    Returns:
        Proxy URL or None if not configured.
    """
    # Check both uppercase and lowercase
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
        # This uses the system's Kerberos ticket cache (kinit/Windows login)
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


class KerberosProxyHandler(urllib.request.BaseHandler):
    """urllib handler for Kerberos proxy authentication.

    Handles 407 Proxy Authentication Required responses by generating
    SPNEGO tokens and retrying the request.
    """

    handler_order = 400  # Before default error handler

    def __init__(self, proxy_url: str):
        """Initialize handler with proxy URL.

        Args:
            proxy_url: Proxy URL like http://proxy.example.com:8080
        """
        self.proxy_url = proxy_url
        self.proxy_host, self.proxy_port, self.proxy_scheme = parse_proxy_url(proxy_url)
        self._token_cache: Optional[str] = None

    def http_error_407(self, req, fp, code, msg, hdrs):
        """Handle 407 Proxy Authentication Required.

        Generates SPNEGO token and retries the request with
        Proxy-Authorization header.
        """
        # Check if Negotiate is offered
        auth_header = hdrs.get("Proxy-Authenticate", "")
        if "Negotiate" not in auth_header:
            logger.debug(f"Proxy does not support Negotiate auth: {auth_header}")
            raise HTTPError(req.full_url, code, msg, hdrs, fp)

        # Generate SPNEGO token
        token = generate_spnego_token(self.proxy_host)
        if not token:
            logger.debug("Failed to generate SPNEGO token")
            raise HTTPError(req.full_url, code, msg, hdrs, fp)

        # Add Proxy-Authorization header and retry
        logger.debug(f"Retrying with SPNEGO token for {self.proxy_host}")
        req.add_header("Proxy-Authorization", f"Negotiate {token}")

        # Retry the request through the parent opener
        return self.parent.open(req, timeout=req.timeout if hasattr(req, 'timeout') else None)

    # Also handle HTTPS errors
    https_error_407 = http_error_407


class KerberosProxyHTTPSHandler(urllib.request.HTTPSHandler):
    """HTTPS handler that adds SPNEGO token proactively.

    For HTTPS through a proxy, we need to add the Proxy-Authorization
    header to the initial CONNECT request, not wait for 407.
    """

    def __init__(self, proxy_url: str, **kwargs):
        super().__init__(**kwargs)
        self.proxy_url = proxy_url
        self.proxy_host, self.proxy_port, _ = parse_proxy_url(proxy_url)

    def https_open(self, req):
        """Open HTTPS connection, adding SPNEGO token if available."""
        # Try to add SPNEGO token proactively for the CONNECT tunnel
        if not req.has_header("Proxy-Authorization"):
            token = generate_spnego_token(self.proxy_host)
            if token:
                req.add_header("Proxy-Authorization", f"Negotiate {token}")

        return super().https_open(req)


def create_kerberos_proxy_opener(proxy_url: Optional[str] = None) -> urllib.request.OpenerDirector:
    """Create urllib opener with Kerberos proxy authentication.

    Args:
        proxy_url: Proxy URL, or None to read from environment.

    Returns:
        OpenerDirector configured for Kerberos proxy auth.
    """
    if proxy_url is None:
        proxy_url = get_proxy_url()

    if not proxy_url:
        # No proxy configured, return default opener
        return urllib.request.build_opener()

    proxy_host, proxy_port, proxy_scheme = parse_proxy_url(proxy_url)

    # Try to generate SPNEGO token proactively
    spnego_token = generate_spnego_token(proxy_host)

    if spnego_token:
        # Create proxy handler with pre-authenticated header
        class PreAuthProxyHandler(urllib.request.ProxyHandler):
            def __init__(self, proxies, token):
                super().__init__(proxies)
                self.spnego_token = token

            def proxy_open(self, req, proxy, type):
                # Add SPNEGO token to the request before proxy handling
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
        # No token available, use standard proxy with 407 handler
        proxy_handler = urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        kerberos_handler = KerberosProxyHandler(proxy_url)

        return urllib.request.build_opener(proxy_handler, kerberos_handler)


def create_kerberos_proxy_session(proxy_url: Optional[str] = None):
    """Create requests Session with Kerberos proxy authentication.

    Args:
        proxy_url: Proxy URL, or None to read from environment.

    Returns:
        requests.Session configured for Kerberos proxy auth.
    """
    import requests
    from requests.adapters import HTTPAdapter

    if proxy_url is None:
        proxy_url = get_proxy_url()

    session = requests.Session()

    if not proxy_url:
        return session

    proxy_host, _, _ = parse_proxy_url(proxy_url)

    # Set proxy
    session.proxies = {
        "http": proxy_url,
        "https": proxy_url,
    }

    # Generate SPNEGO token and add to headers
    token = generate_spnego_token(proxy_host)
    if token:
        session.headers["Proxy-Authorization"] = f"Negotiate {token}"

    return session


def make_request_with_kerberos_proxy(
    url: str,
    method: str = "GET",
    data: Optional[bytes] = None,
    headers: Optional[dict] = None,
    timeout: int = 30,
    proxy_url: Optional[str] = None,
) -> urllib.request.Request:
    """Make HTTP request with Kerberos proxy authentication.

    This is a convenience function that handles the full flow:
    1. Detect proxy from environment or explicit URL
    2. Generate SPNEGO token if Kerberos is available
    3. Make request with proper authentication

    Args:
        url: Request URL.
        method: HTTP method.
        data: Request body.
        headers: Request headers.
        timeout: Request timeout in seconds.
        proxy_url: Explicit proxy URL, or None for environment.

    Returns:
        Response from urllib.request.

    Raises:
        URLError: On network errors.
        HTTPError: On HTTP errors.
    """
    req = urllib.request.Request(url, data=data, method=method)

    if headers:
        for key, value in headers.items():
            req.add_header(key, value)

    opener = create_kerberos_proxy_opener(proxy_url)
    return opener.open(req, timeout=timeout)
