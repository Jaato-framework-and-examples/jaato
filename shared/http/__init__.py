"""Shared HTTP utilities with proxy and Kerberos authentication support.

This module provides unified HTTP client configuration for all providers and plugins,
supporting:
- Standard proxy environment variables (HTTP_PROXY, HTTPS_PROXY, NO_PROXY)
- JAATO_NO_PROXY for exact host matching (unlike standard NO_PROXY suffix matching)
- JAATO_KERBEROS_PROXY for Kerberos/SPNEGO proxy authentication

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
"""

from .proxy import (
    # Configuration
    get_proxy_url,
    is_kerberos_proxy_enabled,
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
