#!/usr/bin/env python3
"""
Simple MITM proxy to capture OAuth token exchange requests.

Usage:
1. Install mitmproxy: pip install mitmproxy
2. Run: mitmdump -s oauth_mitm.py -p 8888
3. In another terminal, run Claude Code with proxy:
   HTTPS_PROXY=http://127.0.0.1:8888 claude login
4. Complete the OAuth flow - this script will log the token exchange request

Note: You may need to trust mitmproxy's CA certificate. See:
https://docs.mitmproxy.org/stable/concepts-certificates/
"""

import json
from mitmproxy import http

def request(flow: http.HTTPFlow) -> None:
    """Log requests to Anthropic OAuth endpoints."""
    if "console.anthropic.com" in flow.request.host:
        print("\n" + "=" * 70)
        print(f"REQUEST: {flow.request.method} {flow.request.url}")
        print("=" * 70)

        print("\nHEADERS:")
        for name, value in flow.request.headers.items():
            print(f"  {name}: {value}")

        if flow.request.content:
            print("\nBODY:")
            try:
                body = json.loads(flow.request.content)
                print(json.dumps(body, indent=2))
            except:
                print(flow.request.content.decode('utf-8', errors='replace'))

        print("=" * 70 + "\n")

def response(flow: http.HTTPFlow) -> None:
    """Log responses from Anthropic OAuth endpoints."""
    if "console.anthropic.com" in flow.request.host:
        print("\n" + "-" * 70)
        print(f"RESPONSE: {flow.response.status_code} {flow.response.reason}")
        print("-" * 70)

        print("\nHEADERS:")
        for name, value in flow.response.headers.items():
            print(f"  {name}: {value}")

        if flow.response.content:
            print("\nBODY:")
            try:
                body = json.loads(flow.response.content)
                print(json.dumps(body, indent=2))
            except:
                print(flow.response.content.decode('utf-8', errors='replace')[:500])

        print("-" * 70 + "\n")
