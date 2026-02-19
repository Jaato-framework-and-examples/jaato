"""Shared HTTP utilities with proxy and Kerberos authentication support.

This module provides unified HTTP client configuration for all providers and plugins,
supporting:
- Standard proxy environment variables (HTTP_PROXY, HTTPS_PROXY, NO_PROXY)
- JAATO_NO_PROXY for exact host matching (unlike standard NO_PROXY suffix matching)
- JAATO_KERBEROS_PROXY for Kerberos/SPNEGO proxy authentication
- JAATO_SSL_VERIFY to disable SSL certificate verification (escape hatch)

Usage:
    # For requests
    from shared.http import get_requests_session
    session = get_requests_session()
    response = session.get(url)

    # For httpx (preferred)
    from shared.http import get_httpx_client
    with get_httpx_client() as client:
        response = client.get(url)

Environment Variables:
    HTTPS_PROXY / HTTP_PROXY: Standard proxy URL
    NO_PROXY: Standard no-proxy hosts (suffix matching)
    JAATO_NO_PROXY: Exact host matching for no-proxy
    JAATO_KERBEROS_PROXY: Enable Kerberos/SPNEGO proxy auth (true/false)
    JAATO_SSL_VERIFY: SSL certificate verification (true/false, default: true)
"""

# Eagerly suppress urllib3 InsecureRequestWarning when SSL verification is
# disabled.  This must happen before any ``requests``/``urllib3`` call so that
# early HTTP traffic (e.g. cached Copilot token refresh during import) doesn't
# emit per-request warnings.  The one-time log message in
# ``_get_ssl_verify_value()`` is the intended notification channel.
from .proxy import is_ssl_verify_disabled as _is_ssl_verify_disabled

if _is_ssl_verify_disabled():
    import warnings as _warnings
    try:
        from urllib3.exceptions import InsecureRequestWarning as _InsecureRequestWarning
        _warnings.filterwarnings("ignore", category=_InsecureRequestWarning)
    except ImportError:
        pass

from .proxy import (
    # Configuration
    get_proxy_url,
    is_kerberos_proxy_enabled,
    is_ssl_verify_disabled,
    should_bypass_proxy,
    # requests support
    get_requests_session,
    get_requests_kwargs,
    # httpx support
    get_httpx_client,
    get_httpx_async_client,
    get_httpx_kwargs,
    # Low-level
    generate_spnego_token,
)

__all__ = [
    # Configuration
    "get_proxy_url",
    "is_kerberos_proxy_enabled",
    "is_ssl_verify_disabled",
    "should_bypass_proxy",
    # requests support
    "get_requests_session",
    "get_requests_kwargs",
    # httpx support
    "get_httpx_client",
    "get_httpx_async_client",
    "get_httpx_kwargs",
    # Low-level
    "generate_spnego_token",
]
