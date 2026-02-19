"""Proxy configuration and Kerberos authentication for HTTP clients.

Provides unified proxy support for requests and httpx with:
- Standard proxy environment variables (HTTP_PROXY, HTTPS_PROXY, NO_PROXY)
- JAATO_NO_PROXY for exact host matching
- JAATO_KERBEROS_PROXY for Kerberos/SPNEGO proxy authentication

Platform Support for Kerberos:
- Windows: Native SSPI via pyspnego, or ctypes secur32.dll fallback
- Windows MSYS2: ctypes SSPI fallback (pyspnego typically unavailable)
- macOS: GSS.framework via pyspnego
- Linux: MIT Kerberos via pyspnego
"""

import base64
import logging
import os
import urllib.parse
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


def _get_no_proxy_entries() -> list:
    """Get list of entries from standard NO_PROXY/no_proxy env var.

    Checks NO_PROXY first, then no_proxy (case-sensitive env var names).

    Returns:
        List of no-proxy entries (lowercase, stripped). Entries may be
        hostnames, domain suffixes (with or without leading dot), '*',
        or host:port pairs.
    """
    value = os.environ.get(ENV_NO_PROXY) or os.environ.get(ENV_NO_PROXY.lower(), "")
    if not value:
        return []
    return [e.strip().lower() for e in value.split(",") if e.strip()]


def _matches_no_proxy(host: str, port: Optional[int], no_proxy_entry: str) -> bool:
    """Check if a host[:port] matches a single NO_PROXY entry.

    Standard NO_PROXY matching rules:
    - '*' matches everything
    - 'hostname' matches that exact hostname
    - '.domain.com' matches any subdomain of domain.com (suffix match)
    - 'domain.com' also matches subdomains (suffix match without leading dot)
    - 'host:port' matches only when both host and port match

    Args:
        host: Lowercase hostname from the URL.
        port: Port from the URL, or None.
        no_proxy_entry: Single lowercase entry from NO_PROXY.

    Returns:
        True if the host (and optionally port) matches the entry.
    """
    if no_proxy_entry == "*":
        return True

    # Split entry into host part and optional port
    entry_host = no_proxy_entry
    entry_port = None
    if ":" in no_proxy_entry:
        parts = no_proxy_entry.rsplit(":", 1)
        try:
            entry_port = int(parts[1])
            entry_host = parts[0]
        except ValueError:
            pass  # Not a valid port, treat whole string as host

    # Port mismatch means no match
    if entry_port is not None and port != entry_port:
        return False

    # Exact match
    if host == entry_host:
        return True

    # Suffix match: ".example.com" matches "foo.example.com"
    if entry_host.startswith("."):
        if host.endswith(entry_host):
            return True
        # Also match the bare domain: ".example.com" matches "example.com"
        if host == entry_host[1:]:
            return True
        return False

    # Without leading dot, standard behavior is also suffix matching:
    # "example.com" matches "foo.example.com"
    if host.endswith("." + entry_host):
        return True

    return False


def should_bypass_proxy(url: str) -> bool:
    """Check if a URL should bypass proxy based on NO_PROXY and JAATO_NO_PROXY.

    Checks both:
    - Standard NO_PROXY/no_proxy (suffix matching per convention)
    - JAATO_NO_PROXY (exact host matching)

    Args:
        url: The URL to check.

    Returns:
        True if the host matches an entry in NO_PROXY or JAATO_NO_PROXY.
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if not host:
        return False

    host = host.lower()
    port = parsed.port

    # Check JAATO_NO_PROXY (exact match)
    jaato_no_proxy_hosts = _get_jaato_no_proxy_hosts()
    if jaato_no_proxy_hosts and host in jaato_no_proxy_hosts:
        return True

    # Check standard NO_PROXY/no_proxy (suffix match)
    no_proxy_entries = _get_no_proxy_entries()
    for entry in no_proxy_entries:
        if _matches_no_proxy(host, port, entry):
            return True

    return False


# ============================================================
# Kerberos/SPNEGO Support
# ============================================================

def _generate_spnego_token_sspi(proxy_host: str) -> Optional[str]:
    """Generate SPNEGO token using Windows SSPI via ctypes.

    Fallback for when pyspnego is not available but we're on Windows
    (including MSYS2).  Uses secur32.dll's AcquireCredentialsHandleW and
    InitializeSecurityContextW to produce a Negotiate token from the
    current user's domain credentials.

    The SSPI flow:
    1. AcquireCredentialsHandleW — get handle to current user's credentials
    2. InitializeSecurityContextW — generate SPNEGO token for HTTP/<proxy_host>
    3. Extract token bytes from output SecBuffer
    4. Clean up: FreeCredentialsHandle, DeleteSecurityContext

    Args:
        proxy_host: Proxy server hostname for the service principal.

    Returns:
        Base64-encoded SPNEGO token, or None if SSPI is unavailable or fails.
    """
    try:
        import ctypes
        # Check that secur32.dll is accessible (Windows or MSYS2 with
        # native Windows Python).  On non-Windows ctypes has no windll
        # attribute; on Cygwin Python secur32.dll may not load.
        secur32 = ctypes.windll.secur32  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None

    from ctypes import (
        POINTER,
        Structure,
        byref,
        c_ulong,
        c_ulonglong,
        c_void_p,
        c_wchar_p,
        pointer,
        string_at,
    )

    # -- Constants from Windows SDK Sspi.h ---------------------------------
    SECPKG_CRED_OUTBOUND = 0x00000002
    ISC_REQ_DELEGATE = 0x00000001
    ISC_REQ_MUTUAL_AUTH = 0x00000002
    ISC_REQ_CONNECTION = 0x00000800
    SECURITY_NETWORK_DREP = 0x00000000
    SECBUFFER_TOKEN = 2
    SECBUFFER_VERSION = 0
    SEC_E_OK = 0x00000000
    SEC_I_CONTINUE_NEEDED = 0x00090312
    MAX_TOKEN_SIZE = 12288  # 12 KiB — generous for Negotiate tokens

    # -- ctypes structure definitions --------------------------------------

    class SecHandle(Structure):
        """SSPI credential / context handle (CredHandle, CtxtHandle)."""

        _fields_ = [("dwLower", c_void_p), ("dwUpper", c_void_p)]

    class SecBuffer(Structure):
        """Single SSPI security buffer."""

        _fields_ = [
            ("cbBuffer", c_ulong),
            ("BufferType", c_ulong),
            ("pvBuffer", c_void_p),
        ]

    class SecBufferDesc(Structure):
        """Descriptor for an array of SecBuffer structures."""

        _fields_ = [
            ("ulVersion", c_ulong),
            ("cBuffers", c_ulong),
            ("pBuffers", POINTER(SecBuffer)),
        ]

    # -- Bind function signatures ------------------------------------------
    secur32.AcquireCredentialsHandleW.argtypes = [
        c_wchar_p, c_wchar_p, c_ulong, c_void_p, c_void_p,
        c_void_p, c_void_p, POINTER(SecHandle), POINTER(c_ulonglong),
    ]
    secur32.AcquireCredentialsHandleW.restype = c_ulong

    secur32.InitializeSecurityContextW.argtypes = [
        POINTER(SecHandle), POINTER(SecHandle), c_wchar_p,
        c_ulong, c_ulong, c_ulong,
        POINTER(SecBufferDesc), c_ulong,
        POINTER(SecHandle), POINTER(SecBufferDesc),
        POINTER(c_ulong), POINTER(c_ulonglong),
    ]
    secur32.InitializeSecurityContextW.restype = c_ulong

    secur32.FreeCredentialsHandle.argtypes = [POINTER(SecHandle)]
    secur32.FreeCredentialsHandle.restype = c_ulong
    secur32.DeleteSecurityContext.argtypes = [POINTER(SecHandle)]
    secur32.DeleteSecurityContext.restype = c_ulong

    # -- Acquire credentials -----------------------------------------------
    cred_handle = SecHandle()
    expiry = c_ulonglong()
    cred_acquired = False
    ctx_initialized = False

    try:
        status = secur32.AcquireCredentialsHandleW(
            None,                   # pszPrincipal (NULL → current user)
            "Negotiate",            # pszPackage
            SECPKG_CRED_OUTBOUND,   # fCredentialUse
            None, None, None, None, # pvLogonId, pAuthData, pGetKeyFn, pvArg
            byref(cred_handle),
            byref(expiry),
        )
        if status != SEC_E_OK:
            logger.debug(
                "SSPI AcquireCredentialsHandleW failed: 0x%08X", status,
            )
            return None
        cred_acquired = True

        # -- Prepare output buffer -----------------------------------------
        token_buf = ctypes.create_string_buffer(MAX_TOKEN_SIZE)
        out_buf = SecBuffer(
            cbBuffer=MAX_TOKEN_SIZE,
            BufferType=SECBUFFER_TOKEN,
            pvBuffer=ctypes.cast(token_buf, c_void_p),
        )
        out_buf_desc = SecBufferDesc(
            ulVersion=SECBUFFER_VERSION,
            cBuffers=1,
            pBuffers=pointer(out_buf),
        )

        # -- Initialize security context -----------------------------------
        ctx_handle = SecHandle()
        ctx_attrs = c_ulong()
        ctx_expiry = c_ulonglong()
        target_name = f"HTTP/{proxy_host}"

        status = secur32.InitializeSecurityContextW(
            byref(cred_handle),
            None,                   # phContext (NULL → first call)
            target_name,
            ISC_REQ_DELEGATE | ISC_REQ_MUTUAL_AUTH | ISC_REQ_CONNECTION,
            0,                      # Reserved1
            SECURITY_NETWORK_DREP,
            None,                   # pInput (NULL → first call)
            0,                      # Reserved2
            byref(ctx_handle),
            byref(out_buf_desc),
            byref(ctx_attrs),
            byref(ctx_expiry),
        )

        if status not in (SEC_E_OK, SEC_I_CONTINUE_NEEDED):
            logger.debug(
                "SSPI InitializeSecurityContextW failed: 0x%08X", status,
            )
            return None
        ctx_initialized = True

        # -- Extract token bytes -------------------------------------------
        token_size = out_buf_desc.pBuffers[0].cbBuffer
        if token_size == 0:
            logger.debug("SSPI produced empty token")
            return None

        token_bytes = string_at(out_buf_desc.pBuffers[0].pvBuffer, token_size)
        return base64.b64encode(token_bytes).decode("ascii")

    except Exception as e:
        logger.debug("SSPI token generation failed: %s", e)
        return None

    finally:
        if ctx_initialized:
            secur32.DeleteSecurityContext(byref(ctx_handle))
        if cred_acquired:
            secur32.FreeCredentialsHandle(byref(cred_handle))


def generate_spnego_token(proxy_host: str) -> Optional[str]:
    """Generate SPNEGO token for proxy authentication.

    Uses the system's Kerberos credentials to generate a Negotiate token
    for the HTTP/<proxy_host> service principal.

    Token generation strategy (in priority order):
    1. pyspnego library (cross-platform, most robust)
    2. Windows SSPI via ctypes (fallback when pyspnego unavailable)

    Args:
        proxy_host: Proxy server hostname.

    Returns:
        Base64-encoded SPNEGO token, or None if generation fails.
    """
    try:
        import spnego
    except ImportError:
        logger.debug(
            "pyspnego not installed, trying native SSPI fallback",
        )
        return _generate_spnego_token_sspi(proxy_host)

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
# requests Support
# ============================================================

def get_requests_session() -> "requests.Session":
    """Create a requests Session with appropriate proxy and SSL configuration.

    Configures the session based on:
    1. Corporate CA certificates (via REQUESTS_CA_BUNDLE / SSL_CERT_FILE)
    2. JAATO_KERBEROS_PROXY: Adds SPNEGO token to headers
    3. Standard proxy env vars (requests handles these automatically)

    Note: The requests library checks NO_PROXY per-request even when
    session-level proxies are set, so NO_PROXY is respected.
    For per-request bypass with JAATO_NO_PROXY, use get_requests_kwargs(url)
    or call should_bypass_proxy(url) and pass proxies={} to the request.

    Returns:
        Configured requests.Session.
    """
    import requests
    from shared.ssl_helper import active_cert_bundle

    session = requests.Session()

    ca_bundle = active_cert_bundle()
    if ca_bundle:
        if os.path.isfile(ca_bundle):
            session.verify = ca_bundle
        else:
            logger.warning(
                "SSL CA bundle not found: %s (from REQUESTS_CA_BUNDLE or "
                "SSL_CERT_FILE). Falling back to default certificate verification.",
                ca_bundle,
            )

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
    """Create an httpx Client with appropriate proxy and SSL configuration.

    Configures the client based on:
    1. Corporate CA certificates (via REQUESTS_CA_BUNDLE / SSL_CERT_FILE)
    2. JAATO_KERBEROS_PROXY: Creates an ``httpx.Proxy`` with the SPNEGO
       Negotiate token in its ``headers`` so the token is sent during the
       CONNECT tunnel handshake (required for HTTPS through a proxy).
    3. Standard proxy env vars (HTTPS_PROXY, HTTP_PROXY): Sets explicit
       proxy URL on the client rather than relying on httpx env detection,
       which may not propagate correctly through all SDK code paths

    Note: When explicit proxy config is set, it overrides httpx's built-in
    NO_PROXY handling. For per-request bypass, use get_httpx_kwargs(url)
    instead, which calls should_bypass_proxy().

    Args:
        **client_kwargs: Additional kwargs to pass to httpx.Client.
            Callers can override SSL via ``verify=False`` or
            ``verify="/custom/path.pem"``; ``setdefault`` preserves
            explicit overrides.

    Returns:
        Configured httpx.Client.
    """
    import httpx
    from shared.ssl_helper import active_cert_bundle

    ca_bundle = active_cert_bundle()
    if ca_bundle:
        if os.path.isfile(ca_bundle):
            client_kwargs.setdefault("verify", ca_bundle)
        else:
            logger.warning(
                "SSL CA bundle not found: %s (from REQUESTS_CA_BUNDLE or "
                "SSL_CERT_FILE). Falling back to default certificate verification.",
                ca_bundle,
            )

    proxy_url = get_proxy_url()
    if proxy_url:
        # Determine the proxy value: either a Proxy object with SPNEGO
        # headers (for Kerberos) or a plain URL string.
        proxy_value: Any = proxy_url
        if is_kerberos_proxy_enabled():
            proxy_host, _, _ = parse_proxy_url(proxy_url)
            token = generate_spnego_token(proxy_host)
            if token:
                # Use httpx.Proxy so the Negotiate header is sent on the
                # CONNECT request that establishes the HTTPS tunnel,
                # not just on the tunnelled request itself.
                proxy_value = httpx.Proxy(
                    url=proxy_url,
                    headers={"Proxy-Authorization": f"Negotiate {token}"},
                )
        client_kwargs.setdefault("proxy", proxy_value)

    return httpx.Client(**client_kwargs)


def get_httpx_kwargs(url: str) -> Dict[str, Any]:
    """Get kwargs for httpx client/request with appropriate proxy configuration.

    When Kerberos is enabled and a SPNEGO token is generated, the returned
    ``proxy`` value is an ``httpx.Proxy`` object whose ``headers`` carry the
    ``Proxy-Authorization: Negotiate …`` header.  This ensures the token is
    sent during the CONNECT tunnel handshake for HTTPS URLs.

    Args:
        url: The URL being requested.

    Returns:
        Dict with ``proxy`` key (and no ``headers`` key for Kerberos —
        the auth header lives on the Proxy object).
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
                import httpx
                kwargs["proxy"] = httpx.Proxy(
                    url=proxy_url,
                    headers={"Proxy-Authorization": f"Negotiate {token}"},
                )

    return kwargs


# ============================================================
# AsyncIO Support (httpx async)
# ============================================================

def get_httpx_async_client(**client_kwargs) -> "httpx.AsyncClient":
    """Create an httpx AsyncClient with appropriate proxy and SSL configuration.

    Same behaviour as :func:`get_httpx_client` — CA bundle, Kerberos SPNEGO
    token, and proxy URL are applied automatically.

    Same caveats regarding NO_PROXY: when explicit proxy config is set, it
    overrides httpx's env-based NO_PROXY handling.  Use
    ``get_httpx_kwargs(url)`` for per-request bypass.

    Args:
        **client_kwargs: Additional kwargs to pass to httpx.AsyncClient.
            Callers can override SSL via ``verify=False`` or
            ``verify="/custom/path.pem"``; ``setdefault`` preserves
            explicit overrides.

    Returns:
        Configured httpx.AsyncClient.
    """
    import httpx
    from shared.ssl_helper import active_cert_bundle

    ca_bundle = active_cert_bundle()
    if ca_bundle:
        if os.path.isfile(ca_bundle):
            client_kwargs.setdefault("verify", ca_bundle)
        else:
            logger.warning(
                "SSL CA bundle not found: %s (from REQUESTS_CA_BUNDLE or "
                "SSL_CERT_FILE). Falling back to default certificate verification.",
                ca_bundle,
            )

    proxy_url = get_proxy_url()
    if proxy_url:
        proxy_value: Any = proxy_url
        if is_kerberos_proxy_enabled():
            proxy_host, _, _ = parse_proxy_url(proxy_url)
            token = generate_spnego_token(proxy_host)
            if token:
                proxy_value = httpx.Proxy(
                    url=proxy_url,
                    headers={"Proxy-Authorization": f"Negotiate {token}"},
                )
        client_kwargs.setdefault("proxy", proxy_value)

    return httpx.AsyncClient(**client_kwargs)
