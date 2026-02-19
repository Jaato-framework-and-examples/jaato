#!/usr/bin/env python3
"""End-to-end health check for GitHub Models authentication and inference.

Traces every HTTP endpoint in the OAuth → Copilot token → model listing →
chat completion pipeline, using jaato's shared HTTP module so proxy, SSL,
and Kerberos configuration is exercised exactly as the real client would.

Usage:
    # From repo root (loads .env automatically):
    .venv/bin/python scripts/github_health_check.py

    # Skip the chat completion test (auth-only check):
    .venv/bin/python scripts/github_health_check.py --auth-only

    # Use a specific .env file:
    .venv/bin/python scripts/github_health_check.py --env-file /path/to/.env

    # Verbose mode (print response bodies):
    .venv/bin/python scripts/github_health_check.py -v

Exit codes:
    0  All checks passed
    1  One or more checks failed
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
_VERBOSE = False


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def info(msg: str) -> None:
    print(f"  {msg}")


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def header(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def verbose(msg: str) -> None:
    if _VERBOSE:
        print(f"  [DEBUG] {msg}")


def mask(token: str, head: int = 6, tail: int = 4) -> str:
    if len(token) <= head + tail + 3:
        return "***"
    return f"{token[:head]}...{token[-tail:]}"


# ---------------------------------------------------------------------------
# Check result tracking
# ---------------------------------------------------------------------------
class CheckResult:
    """Accumulates pass/fail results for a final exit code."""

    def __init__(self) -> None:
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.skipped: List[str] = []

    def record_pass(self, name: str) -> None:
        self.passed.append(name)
        ok(name)

    def record_fail(self, name: str, reason: str = "") -> None:
        self.failed.append(name)
        fail(f"{name}: {reason}" if reason else name)

    def record_skip(self, name: str, reason: str = "") -> None:
        self.skipped.append(name)
        warn(f"{name} (skipped: {reason})" if reason else f"{name} (skipped)")

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


# ---------------------------------------------------------------------------
# Step 0: Environment & shared HTTP module
# ---------------------------------------------------------------------------
def check_environment(results: CheckResult) -> Dict[str, Any]:
    """Check environment variables and shared HTTP configuration."""
    header("Step 0: Environment & HTTP Configuration")

    ctx: Dict[str, Any] = {}

    # .env should already be loaded by caller
    from shared.http import (
        get_proxy_url,
        is_kerberos_proxy_enabled,
        is_ssl_verify_disabled,
        get_httpx_client,
    )
    from shared.ssl_helper import active_cert_bundle

    # SSL
    ssl_verify_off = is_ssl_verify_disabled()
    if ssl_verify_off:
        info("JAATO_SSL_VERIFY: false (certificate verification DISABLED)")
        warn("SSL verification is disabled — do not use this in production")
        results.record_pass("SSL verify configured (disabled via JAATO_SSL_VERIFY)")
    else:
        info("JAATO_SSL_VERIFY: true (default)")

    ca = active_cert_bundle()
    if ca:
        exists = os.path.isfile(ca)
        info(f"CA bundle: {ca} (exists={exists})")
        if ssl_verify_off:
            info("(CA bundle is ignored because SSL verification is disabled)")
            results.record_pass("SSL CA bundle (ignored, verify=false)")
        elif not exists:
            results.record_fail("SSL CA bundle", f"File not found: {ca}")
        else:
            results.record_pass("SSL CA bundle exists")
    else:
        info("CA bundle: (system default)")
        results.record_pass("SSL CA bundle (system default)")

    # Proxy
    proxy = get_proxy_url()
    kerberos = is_kerberos_proxy_enabled()
    info(f"Proxy: {proxy or '(none)'}")
    info(f"Kerberos proxy: {kerberos}")

    # Token sources
    from shared.plugins.model_provider.github_models.env import (
        resolve_token,
        resolve_token_source,
        resolve_endpoint,
        resolve_organization,
        resolve_enterprise,
        get_checked_credential_locations,
    )

    token = resolve_token()
    source = resolve_token_source()
    endpoint = resolve_endpoint()
    org = resolve_organization()
    enterprise = resolve_enterprise()

    info(f"Token source: {source or '(none)'}")
    if token:
        info(f"Token: {mask(token)}")
        results.record_pass(f"Token found ({source})")
    else:
        results.record_fail("Token", "No token found")
        info("Checked locations:")
        for loc in get_checked_credential_locations():
            info(f"  - {loc}")

    info(f"Endpoint: {endpoint}")
    if org:
        info(f"Organization: {org}")
    if enterprise:
        info(f"Enterprise: {enterprise}")

    # Verify httpx client can be constructed
    try:
        client = get_httpx_client(timeout=5.0)
        client.close()
        results.record_pass("httpx client construction")
    except Exception as e:
        results.record_fail("httpx client construction", str(e))

    ctx["token"] = token
    ctx["source"] = source
    ctx["endpoint"] = endpoint
    ctx["org"] = org
    return ctx


# ---------------------------------------------------------------------------
# Step 1: OAuth device code endpoint (smoke test — does not start a login)
# ---------------------------------------------------------------------------
def check_device_code_endpoint(results: CheckResult) -> None:
    """Verify the device code endpoint is reachable (POST with empty scope)."""
    header("Step 1: OAuth Device Code Endpoint")

    from shared.plugins.model_provider.github_models.oauth import (
        DEVICE_CODE_URL,
        OAUTH_CLIENT_ID,
    )

    info(f"POST {DEVICE_CODE_URL}")
    info(f"Client ID: {OAUTH_CLIENT_ID}")

    import httpx
    from shared.http import get_httpx_client

    t0 = time.monotonic()
    try:
        with get_httpx_client(timeout=15.0) as client:
            resp = client.post(
                DEVICE_CODE_URL,
                content=urllib.parse.urlencode({
                    "client_id": OAUTH_CLIENT_ID,
                    "scope": "read:user",
                }),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
        elapsed = time.monotonic() - t0
        info(f"Status: {resp.status_code} ({elapsed:.1f}s)")

        if resp.status_code == 200:
            body = resp.json()
            verbose(f"Response: {json.dumps(body, indent=2)}")
            info(f"user_code: {body.get('user_code', '?')}")
            info(f"verification_uri: {body.get('verification_uri', '?')}")
            info(f"expires_in: {body.get('expires_in', '?')}s")
            results.record_pass(f"POST {DEVICE_CODE_URL} -> 200")
        else:
            results.record_fail(
                f"POST {DEVICE_CODE_URL}",
                f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"POST {DEVICE_CODE_URL}",
            f"{type(e).__name__}: {e} ({elapsed:.1f}s)",
        )


# ---------------------------------------------------------------------------
# Step 2: OAuth token endpoint (smoke test — expects error)
# ---------------------------------------------------------------------------
def check_token_endpoint(results: CheckResult) -> None:
    """Verify the token endpoint is reachable (sends invalid device_code)."""
    header("Step 2: OAuth Token Endpoint")

    from shared.plugins.model_provider.github_models.oauth import (
        TOKEN_URL,
        OAUTH_CLIENT_ID,
    )

    info(f"POST {TOKEN_URL}")

    import httpx
    from shared.http import get_httpx_client

    t0 = time.monotonic()
    try:
        with get_httpx_client(timeout=15.0) as client:
            resp = client.post(
                TOKEN_URL,
                content=urllib.parse.urlencode({
                    "client_id": OAUTH_CLIENT_ID,
                    "device_code": "health_check_invalid_code",
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                }),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
        elapsed = time.monotonic() - t0
        info(f"Status: {resp.status_code} ({elapsed:.1f}s)")

        body = resp.json()
        verbose(f"Response: {json.dumps(body, indent=2)}")
        # We expect an error like "bad_verification_code" — that's a healthy response
        error = body.get("error", "")
        info(f"Error (expected): {error}")
        if resp.status_code == 200 and error:
            results.record_pass(f"POST {TOKEN_URL} -> reachable (error={error})")
        elif resp.status_code == 200:
            results.record_pass(f"POST {TOKEN_URL} -> 200")
        else:
            results.record_fail(
                f"POST {TOKEN_URL}",
                f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"POST {TOKEN_URL}",
            f"{type(e).__name__}: {e} ({elapsed:.1f}s)",
        )


# ---------------------------------------------------------------------------
# Step 3: Copilot token exchange
# ---------------------------------------------------------------------------
def check_copilot_token_exchange(
    results: CheckResult, ctx: Dict[str, Any]
) -> Optional[str]:
    """Exchange OAuth token for Copilot API token.

    Returns the Copilot token on success, None on failure or skip.
    """
    header("Step 3: Copilot Token Exchange")

    from shared.plugins.model_provider.github_models.oauth import (
        COPILOT_TOKEN_URL,
    )

    info(f"GET {COPILOT_TOKEN_URL}")

    if ctx["source"] != "oauth":
        results.record_skip(
            f"GET {COPILOT_TOKEN_URL}",
            f"token source is '{ctx['source']}', not 'oauth'",
        )
        return None

    from shared.plugins.model_provider.github_models.oauth import (
        get_stored_access_token,
        load_tokens,
        load_copilot_token,
        exchange_oauth_for_copilot_token,
    )

    # Show stored token state
    oauth_tokens = load_tokens()
    copilot_token = load_copilot_token()
    if copilot_token:
        info(f"Cached Copilot token: {mask(copilot_token.token)}")
        info(f"  expires_at: {time.ctime(copilot_token.expires_at)}")
        info(f"  expired: {copilot_token.is_expired()}")
        info(f"  needs_refresh: {copilot_token.needs_refresh()}")

    # Force a fresh exchange to test the endpoint
    if not oauth_tokens:
        results.record_fail(f"GET {COPILOT_TOKEN_URL}", "No stored OAuth tokens")
        return None

    t0 = time.monotonic()
    try:
        fresh = exchange_oauth_for_copilot_token(oauth_tokens.access_token)
        elapsed = time.monotonic() - t0
        info(f"Exchange successful ({elapsed:.1f}s)")
        info(f"  token: {mask(fresh.token)}")
        info(f"  expires_at: {time.ctime(fresh.expires_at)}")
        remaining = int(fresh.expires_at - time.time())
        info(f"  remaining: {remaining // 60}m {remaining % 60}s")
        results.record_pass(f"GET {COPILOT_TOKEN_URL} -> token obtained")
        return fresh.token
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"GET {COPILOT_TOKEN_URL}",
            f"{type(e).__name__}: {e} ({elapsed:.1f}s)",
        )
        return None


# ---------------------------------------------------------------------------
# Step 4: Copilot models endpoint
# ---------------------------------------------------------------------------
def check_copilot_models(
    results: CheckResult, copilot_token: Optional[str]
) -> Optional[List[str]]:
    """List models from the Copilot API.

    Returns the model list on success, None on failure or skip.
    """
    header("Step 4a: Copilot Models Endpoint")

    from shared.plugins.model_provider.github_models.copilot_client import (
        COPILOT_MODELS_ENDPOINT,
        COPILOT_HEADERS,
    )

    info(f"GET {COPILOT_MODELS_ENDPOINT}")

    if not copilot_token:
        results.record_skip(
            f"GET {COPILOT_MODELS_ENDPOINT}",
            "no Copilot token available",
        )
        return None

    from shared.http import get_requests_session, should_bypass_proxy

    session = get_requests_session()
    headers = {
        **COPILOT_HEADERS,
        "Authorization": f"Bearer {copilot_token}",
    }
    proxies = {} if should_bypass_proxy(COPILOT_MODELS_ENDPOINT) else None

    t0 = time.monotonic()
    try:
        resp = session.get(
            COPILOT_MODELS_ENDPOINT,
            headers=headers,
            timeout=15,
            proxies=proxies,
        )
        elapsed = time.monotonic() - t0
        info(f"Status: {resp.status_code} ({elapsed:.1f}s)")

        if resp.status_code == 200:
            body = resp.json()
            verbose(f"Response: {json.dumps(body, indent=2)[:2000]}")
            models = [m.get("id") for m in body.get("data", []) if m.get("id")]
            info(f"Models available: {len(models)}")
            for m in models[:10]:
                info(f"  - {m}")
            if len(models) > 10:
                info(f"  ... and {len(models) - 10} more")
            results.record_pass(
                f"GET {COPILOT_MODELS_ENDPOINT} -> {len(models)} models"
            )
            return models
        else:
            results.record_fail(
                f"GET {COPILOT_MODELS_ENDPOINT}",
                f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
            return None
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"GET {COPILOT_MODELS_ENDPOINT}",
            f"{type(e).__name__}: {e} ({elapsed:.1f}s)",
        )
        return None
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Step 4b: GitHub Models catalog endpoint (PAT path)
# ---------------------------------------------------------------------------
def check_catalog_endpoint(
    results: CheckResult, ctx: Dict[str, Any]
) -> Optional[List[str]]:
    """List models from the GitHub Models catalog API (PAT path)."""
    header("Step 4b: GitHub Models Catalog Endpoint")

    from shared.plugins.model_provider.github_models.provider import (
        CATALOG_API_ENDPOINT,
    )

    info(f"GET {CATALOG_API_ENDPOINT}")

    token = ctx.get("token")
    if not token:
        results.record_skip(
            f"GET {CATALOG_API_ENDPOINT}", "no token available"
        )
        return None

    import httpx
    from shared.http import get_httpx_client

    t0 = time.monotonic()
    try:
        with get_httpx_client(timeout=15.0) as client:
            resp = client.get(
                CATALOG_API_ENDPOINT,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
        elapsed = time.monotonic() - t0
        info(f"Status: {resp.status_code} ({elapsed:.1f}s)")

        if resp.status_code == 200:
            body = resp.json()
            verbose(f"Response keys: {list(body.keys()) if isinstance(body, dict) else type(body).__name__}")
            if isinstance(body, list):
                models = [m.get("id") or m.get("name") for m in body if isinstance(m, dict)]
            elif isinstance(body, dict):
                models = [
                    m.get("id") or m.get("name")
                    for m in body.get("data", body.get("models", []))
                    if isinstance(m, dict)
                ]
            else:
                models = []
            info(f"Models in catalog: {len(models)}")
            for m in models[:10]:
                info(f"  - {m}")
            if len(models) > 10:
                info(f"  ... and {len(models) - 10} more")
            results.record_pass(
                f"GET {CATALOG_API_ENDPOINT} -> {len(models)} models"
            )
            return models
        elif resp.status_code == 401:
            results.record_fail(
                f"GET {CATALOG_API_ENDPOINT}",
                f"HTTP 401 Unauthorized (token may lack 'models:read' scope)",
            )
            return None
        else:
            results.record_fail(
                f"GET {CATALOG_API_ENDPOINT}",
                f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
            return None
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"GET {CATALOG_API_ENDPOINT}",
            f"{type(e).__name__}: {e} ({elapsed:.1f}s)",
        )
        return None


# ---------------------------------------------------------------------------
# Step 5a: Copilot chat completion (OAuth path)
# ---------------------------------------------------------------------------
def check_copilot_chat(
    results: CheckResult, copilot_token: Optional[str], model: str = "gpt-4o"
) -> None:
    """Send a minimal chat completion through the Copilot API."""
    header("Step 5a: Copilot Chat Completions Endpoint")

    from shared.plugins.model_provider.github_models.copilot_client import (
        COPILOT_CHAT_ENDPOINT,
        COPILOT_HEADERS,
    )

    info(f"POST {COPILOT_CHAT_ENDPOINT}")
    info(f"Model: {model}")

    if not copilot_token:
        results.record_skip(
            f"POST {COPILOT_CHAT_ENDPOINT}",
            "no Copilot token available",
        )
        return

    from shared.http import get_requests_session, should_bypass_proxy

    session = get_requests_session()
    headers = {
        **COPILOT_HEADERS,
        "Authorization": f"Bearer {copilot_token}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'ok' and nothing else."}],
        "max_tokens": 10,
    }
    proxies = {} if should_bypass_proxy(COPILOT_CHAT_ENDPOINT) else None

    t0 = time.monotonic()
    try:
        resp = session.post(
            COPILOT_CHAT_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=30,
            proxies=proxies,
        )
        elapsed = time.monotonic() - t0
        info(f"Status: {resp.status_code} ({elapsed:.1f}s)")

        if resp.status_code == 200:
            body = resp.json()
            verbose(f"Response: {json.dumps(body, indent=2)[:1000]}")
            text = ""
            for choice in body.get("choices", []):
                text = choice.get("message", {}).get("content", "")
            usage = body.get("usage", {})
            info(f"Response text: {text!r}")
            info(f"Usage: prompt={usage.get('prompt_tokens', '?')}, "
                 f"completion={usage.get('completion_tokens', '?')}, "
                 f"total={usage.get('total_tokens', '?')}")
            results.record_pass(
                f"POST {COPILOT_CHAT_ENDPOINT} -> {resp.status_code}"
            )
        else:
            body_text = resp.text[:300]
            results.record_fail(
                f"POST {COPILOT_CHAT_ENDPOINT}",
                f"HTTP {resp.status_code}: {body_text}",
            )
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"POST {COPILOT_CHAT_ENDPOINT}",
            f"{type(e).__name__}: {e} ({elapsed:.1f}s)",
        )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Step 5b: GitHub Models inference (PAT path)
# ---------------------------------------------------------------------------
def check_models_inference(
    results: CheckResult, ctx: Dict[str, Any], model: str = "openai/gpt-4o"
) -> None:
    """Send a minimal chat completion through the GitHub Models API."""
    header("Step 5b: GitHub Models Inference Endpoint")

    endpoint = ctx.get("endpoint", "https://models.github.ai/inference")
    chat_url = f"{endpoint.rstrip('/')}/chat/completions"
    info(f"POST {chat_url}")
    info(f"Model: {model}")

    token = ctx.get("token")
    source = ctx.get("source")
    if not token:
        results.record_skip(f"POST {chat_url}", "no token available")
        return
    if source == "oauth":
        results.record_skip(
            f"POST {chat_url}",
            "OAuth tokens use Copilot API path (step 5a), not Models API",
        )
        return

    import httpx
    from shared.http import get_httpx_client

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'ok' and nothing else."}],
        "max_tokens": 10,
    }

    t0 = time.monotonic()
    try:
        with get_httpx_client(timeout=30.0) as client:
            resp = client.post(
                chat_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
        elapsed = time.monotonic() - t0
        info(f"Status: {resp.status_code} ({elapsed:.1f}s)")

        if resp.status_code == 200:
            body = resp.json()
            verbose(f"Response: {json.dumps(body, indent=2)[:1000]}")
            text = ""
            for choice in body.get("choices", []):
                text = choice.get("message", {}).get("content", "")
            usage = body.get("usage", {})
            info(f"Response text: {text!r}")
            info(f"Usage: prompt={usage.get('prompt_tokens', '?')}, "
                 f"completion={usage.get('completion_tokens', '?')}, "
                 f"total={usage.get('total_tokens', '?')}")
            results.record_pass(f"POST {chat_url} -> {resp.status_code}")
        elif resp.status_code == 401:
            results.record_fail(
                f"POST {chat_url}",
                "HTTP 401 Unauthorized (token may lack 'models:read' scope)",
            )
        else:
            results.record_fail(
                f"POST {chat_url}",
                f"HTTP {resp.status_code}: {resp.text[:300]}",
            )
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"POST {chat_url}",
            f"{type(e).__name__}: {e} ({elapsed:.1f}s)",
        )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(results: CheckResult) -> None:
    header("Summary")
    total = len(results.passed) + len(results.failed) + len(results.skipped)
    info(f"Total checks: {total}")
    info(f"  Passed:  {len(results.passed)}")
    info(f"  Failed:  {len(results.failed)}")
    info(f"  Skipped: {len(results.skipped)}")

    if results.failed:
        print()
        fail("Failed checks:")
        for name in results.failed:
            print(f"    - {name}")

    print()
    if results.exit_code == 0:
        ok("All checks passed!")
    else:
        fail("Some checks failed. See details above.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    global _VERBOSE

    parser = argparse.ArgumentParser(
        description="GitHub Models end-to-end health check",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env in CWD)",
    )
    parser.add_argument(
        "--auth-only",
        action="store_true",
        help="Only check authentication, skip chat completion",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print response bodies",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use for chat test (default: auto-select based on auth path)",
    )
    args = parser.parse_args()
    _VERBOSE = args.verbose

    # Load .env
    try:
        from dotenv import load_dotenv
        env_path = Path(args.env_file)
        if env_path.exists():
            load_dotenv(env_path)
            info(f"Loaded {env_path.resolve()}")
        else:
            info(f".env file not found at {env_path.resolve()}, using environment as-is")
    except ImportError:
        info("python-dotenv not installed, using environment as-is")

    results = CheckResult()

    # Step 0: Environment
    ctx = check_environment(results)

    # Step 1: Device code endpoint
    check_device_code_endpoint(results)

    # Step 2: Token endpoint
    check_token_endpoint(results)

    # Step 3: Copilot token exchange (OAuth path only)
    copilot_token = check_copilot_token_exchange(results, ctx)

    # Step 4a: Copilot models (OAuth path)
    copilot_models = check_copilot_models(results, copilot_token)

    # Step 4b: Catalog endpoint (PAT path)
    catalog_models = check_catalog_endpoint(results, ctx)

    # Step 5: Chat completion
    if not args.auth_only:
        if copilot_token:
            model = args.model or "gpt-4o"
            check_copilot_chat(results, copilot_token, model)
        if ctx.get("source") != "oauth" and ctx.get("token"):
            model = args.model or "openai/gpt-4o"
            check_models_inference(results, ctx, model)
        if not copilot_token and not ctx.get("token"):
            results.record_skip(
                "Chat completion", "no valid token for either path"
            )

    print_summary(results)
    return results.exit_code


if __name__ == "__main__":
    sys.exit(main())
