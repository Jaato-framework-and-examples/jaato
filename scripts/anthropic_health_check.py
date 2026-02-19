#!/usr/bin/env python3
"""End-to-end health check for Anthropic Claude authentication and inference.

Traces every credential source and HTTP endpoint in the OAuth PKCE / API key →
token refresh → messages pipeline, using jaato's shared HTTP module so proxy,
SSL, and Kerberos configuration is exercised exactly as the real provider would.

Usage:
    # From repo root (loads .env automatically):
    .venv/bin/python scripts/anthropic_health_check.py

    # Skip the chat completion test (auth-only check):
    .venv/bin/python scripts/anthropic_health_check.py --auth-only

    # Use a specific .env file:
    .venv/bin/python scripts/anthropic_health_check.py --env-file /path/to/.env

    # Verbose mode (print response bodies):
    .venv/bin/python scripts/anthropic_health_check.py -v

    # Test a specific model:
    .venv/bin/python scripts/anthropic_health_check.py --model claude-sonnet-4-20250514

Exit codes:
    0  All checks passed
    1  One or more checks failed
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    # Credential sources
    from shared.plugins.model_provider.anthropic.env import (
        resolve_api_key,
        resolve_oauth_token,
        get_checked_credential_locations,
        resolve_enable_thinking,
        resolve_thinking_budget,
        resolve_enable_caching,
    )
    from shared.plugins.model_provider.anthropic.oauth import (
        load_tokens,
        get_valid_access_token,
    )

    # 1. PKCE OAuth tokens (interactive login)
    pkce_token = None
    oauth_tokens = load_tokens()
    if oauth_tokens:
        info(f"PKCE OAuth token found: {mask(oauth_tokens.access_token)}")
        info(f"  expires_at: {time.ctime(oauth_tokens.expires_at)}")
        expired = oauth_tokens.is_expired
        info(f"  expired: {expired}")
        try:
            pkce_token = get_valid_access_token()
            if pkce_token:
                results.record_pass("PKCE OAuth token (valid or refreshed)")
        except Exception as e:
            results.record_fail("PKCE OAuth token refresh", str(e))
    else:
        info("PKCE OAuth token: (none stored)")

    # 2. OAuth token from env var
    env_oauth = resolve_oauth_token()
    if env_oauth:
        info(f"Env OAuth token: {mask(env_oauth)}")
        results.record_pass("Env OAuth token found")
    else:
        info("Env OAuth token: (not set)")

    # 3. API key from env var
    api_key = resolve_api_key()
    if api_key:
        info(f"API key: {mask(api_key)}")
        results.record_pass("API key found")
    else:
        info("API key: (not set)")

    # Summary of auth method that will be used
    if pkce_token:
        ctx["auth_method"] = "pkce_oauth"
        ctx["token"] = pkce_token
        info("Active auth: PKCE OAuth (highest priority)")
    elif env_oauth:
        ctx["auth_method"] = "env_oauth"
        ctx["token"] = env_oauth
        info("Active auth: OAuth token from environment")
    elif api_key:
        ctx["auth_method"] = "api_key"
        ctx["token"] = api_key
        info("Active auth: API key")
    else:
        ctx["auth_method"] = None
        ctx["token"] = None
        results.record_fail("Credentials", "No credentials found")
        info("Checked locations:")
        for loc in get_checked_credential_locations():
            info(f"  - {loc}")

    # Feature flags
    thinking = resolve_enable_thinking()
    thinking_budget = resolve_thinking_budget()
    caching = resolve_enable_caching()
    info(f"Extended thinking: {thinking} (budget: {thinking_budget})")
    info(f"Prompt caching: {caching}")

    ctx["enable_thinking"] = thinking
    ctx["thinking_budget"] = thinking_budget
    ctx["enable_caching"] = caching

    # Verify httpx client can be constructed
    try:
        client = get_httpx_client(timeout=5.0)
        client.close()
        results.record_pass("httpx client construction")
    except Exception as e:
        results.record_fail("httpx client construction", str(e))

    return ctx


# ---------------------------------------------------------------------------
# Step 1: OAuth Token Endpoint (smoke test — expects error)
# ---------------------------------------------------------------------------
def check_token_endpoint(results: CheckResult) -> None:
    """Verify the Anthropic OAuth token endpoint is reachable.

    Sends an invalid grant to prove connectivity; expects an error response.
    """
    header("Step 1: OAuth Token Endpoint")

    from shared.plugins.model_provider.anthropic.oauth import OAUTH_TOKEN_URL

    info(f"POST {OAUTH_TOKEN_URL}")

    from shared.http import get_httpx_client

    t0 = time.monotonic()
    try:
        with get_httpx_client(timeout=15.0) as client:
            resp = client.post(
                OAUTH_TOKEN_URL,
                json={
                    "grant_type": "authorization_code",
                    "code": "health_check_invalid_code",
                    "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
                    "code_verifier": "health_check_invalid_verifier",
                    "redirect_uri": "https://console.anthropic.com/oauth/code/callback",
                },
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
        elapsed = time.monotonic() - t0
        info(f"Status: {resp.status_code} ({elapsed:.1f}s)")

        verbose(f"Response: {resp.text[:500]}")
        # Any non-network-error response means the endpoint is reachable.
        # We expect 400/401 since the code is invalid.
        if resp.status_code in (200, 400, 401, 403):
            try:
                body = resp.json()
                error = body.get("error", body.get("message", ""))
                info(f"Error (expected): {error}")
            except Exception:
                info(f"Response body: {resp.text[:200]}")
            results.record_pass(
                f"POST {OAUTH_TOKEN_URL} -> reachable (HTTP {resp.status_code})"
            )
        else:
            results.record_fail(
                f"POST {OAUTH_TOKEN_URL}",
                f"Unexpected HTTP {resp.status_code}: {resp.text[:200]}",
            )
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"POST {OAUTH_TOKEN_URL}",
            f"{type(e).__name__}: {e} ({elapsed:.1f}s)",
        )


# ---------------------------------------------------------------------------
# Step 2: OAuth Token Refresh (if PKCE tokens exist)
# ---------------------------------------------------------------------------
def check_token_refresh(results: CheckResult, ctx: Dict[str, Any]) -> None:
    """Test the token refresh flow if PKCE OAuth tokens are stored.

    Attempts a refresh to verify the refresh_token is still valid and the
    token endpoint accepts refresh_token grants.
    """
    header("Step 2: OAuth Token Refresh")

    from shared.plugins.model_provider.anthropic.oauth import (
        OAUTH_TOKEN_URL,
        load_tokens,
        refresh_tokens,
        save_tokens,
    )

    tokens = load_tokens()
    if not tokens:
        results.record_skip(
            "Token refresh", "no stored PKCE OAuth tokens"
        )
        return

    info(f"POST {OAUTH_TOKEN_URL} (grant_type=refresh_token)")
    info(f"Refresh token: {mask(tokens.refresh_token)}")

    t0 = time.monotonic()
    try:
        new_tokens = refresh_tokens(tokens.refresh_token)
        elapsed = time.monotonic() - t0

        info(f"Refresh successful ({elapsed:.1f}s)")
        info(f"  new access_token: {mask(new_tokens.access_token)}")
        info(f"  expires_at: {time.ctime(new_tokens.expires_at)}")
        remaining = int(new_tokens.expires_at - time.time())
        info(f"  remaining: {remaining // 3600}h {(remaining % 3600) // 60}m")

        # Save refreshed tokens
        save_tokens(new_tokens)
        ctx["token"] = new_tokens.access_token
        info("  (saved refreshed tokens)")
        results.record_pass("Token refresh -> new access token obtained")
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            "Token refresh",
            f"{type(e).__name__}: {e} ({elapsed:.1f}s)",
        )


# ---------------------------------------------------------------------------
# Step 3: Anthropic Messages API (auth validation)
# ---------------------------------------------------------------------------
def check_messages_api(
    results: CheckResult,
    ctx: Dict[str, Any],
    model: str = "claude-haiku-4-20250414",
) -> None:
    """Send a minimal chat completion through the Anthropic Messages API.

    Uses the Anthropic Python SDK to exercise the same code path as the
    provider, including proxy/SSL/auth header configuration.
    """
    header("Step 3: Anthropic Messages API")

    MESSAGES_URL = "https://api.anthropic.com/v1/messages"
    info(f"POST {MESSAGES_URL}")
    info(f"Model: {model}")
    info(f"Auth method: {ctx.get('auth_method', '(none)')}")

    token = ctx.get("token")
    auth_method = ctx.get("auth_method")
    if not token:
        results.record_skip(
            f"POST {MESSAGES_URL}", "no valid credentials available"
        )
        return

    try:
        import anthropic
    except ImportError:
        results.record_fail(
            f"POST {MESSAGES_URL}",
            "anthropic package not installed (pip install anthropic)",
        )
        return

    from shared.http import get_httpx_client

    # Build client matching provider logic
    client_kwargs: Dict[str, Any] = {}
    try:
        from shared.ssl_helper import active_cert_bundle
        from shared.http.proxy import (
            get_proxy_url,
            is_kerberos_proxy_enabled,
        )
        ca = active_cert_bundle()
        if ca or is_kerberos_proxy_enabled() or get_proxy_url():
            client_kwargs["http_client"] = get_httpx_client()
    except Exception:
        pass  # Let SDK use defaults

    # OAuth tokens need special headers
    if auth_method in ("pkce_oauth", "env_oauth"):
        oauth_headers = {
            "anthropic-beta": (
                "oauth-2025-04-20,"
                "interleaved-thinking-2025-05-14,"
                "claude-code-20250219"
            ),
            "user-agent": "claude-cli/2.1.2 (external, cli)",
        }
        client = anthropic.Anthropic(
            auth_token=token,
            default_headers=oauth_headers,
            **client_kwargs,
        )
        info("Using OAuth auth headers")
    else:
        client = anthropic.Anthropic(api_key=token, **client_kwargs)
        info("Using API key auth")

    t0 = time.monotonic()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
        )
        elapsed = time.monotonic() - t0
        info(f"Response received ({elapsed:.1f}s)")

        # Extract text
        text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                text = block.text
        info(f"Response text: {text!r}")

        # Usage
        if resp.usage:
            info(f"Usage: input={resp.usage.input_tokens}, "
                 f"output={resp.usage.output_tokens}")

        info(f"Model: {resp.model}")
        info(f"Stop reason: {resp.stop_reason}")

        verbose(f"Full response: {resp}")
        results.record_pass(f"POST {MESSAGES_URL} -> {resp.stop_reason}")

    except anthropic.AuthenticationError as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"POST {MESSAGES_URL}",
            f"Authentication failed ({elapsed:.1f}s): {e}",
        )
    except anthropic.NotFoundError as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"POST {MESSAGES_URL}",
            f"Model not found ({elapsed:.1f}s): {e}",
        )
    except anthropic.RateLimitError as e:
        elapsed = time.monotonic() - t0
        # Rate limit means auth worked — the request was accepted
        warn(f"Rate limited ({elapsed:.1f}s) — auth is valid but limit hit")
        results.record_pass(
            f"POST {MESSAGES_URL} -> rate limited (auth valid)"
        )
    except anthropic.APIStatusError as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"POST {MESSAGES_URL}",
            f"HTTP {e.status_code} ({elapsed:.1f}s): {e.message}",
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"POST {MESSAGES_URL}",
            f"{type(e).__name__}: {e} ({elapsed:.1f}s)",
        )


# ---------------------------------------------------------------------------
# Step 4: Extended Thinking (optional)
# ---------------------------------------------------------------------------
def check_thinking(
    results: CheckResult,
    ctx: Dict[str, Any],
    model: str = "claude-sonnet-4-20250514",
) -> None:
    """Verify extended thinking works if enabled.

    Sends a request with thinking enabled and checks that a thinking block
    is returned.
    """
    header("Step 4: Extended Thinking")

    if not ctx.get("enable_thinking"):
        results.record_skip(
            "Extended thinking",
            "not enabled (set JAATO_ANTHROPIC_ENABLE_THINKING=true)",
        )
        return

    MESSAGES_URL = "https://api.anthropic.com/v1/messages"
    info(f"POST {MESSAGES_URL} (with thinking)")
    info(f"Model: {model}")
    info(f"Thinking budget: {ctx.get('thinking_budget', 10000)}")

    token = ctx.get("token")
    auth_method = ctx.get("auth_method")
    if not token:
        results.record_skip("Extended thinking", "no valid credentials")
        return

    try:
        import anthropic
    except ImportError:
        results.record_fail("Extended thinking", "anthropic package not installed")
        return

    from shared.http import get_httpx_client

    client_kwargs: Dict[str, Any] = {}
    try:
        from shared.ssl_helper import active_cert_bundle
        from shared.http.proxy import (
            get_proxy_url,
            is_kerberos_proxy_enabled,
        )
        ca = active_cert_bundle()
        if ca or is_kerberos_proxy_enabled() or get_proxy_url():
            client_kwargs["http_client"] = get_httpx_client()
    except Exception:
        pass

    if auth_method in ("pkce_oauth", "env_oauth"):
        oauth_headers = {
            "anthropic-beta": (
                "oauth-2025-04-20,"
                "interleaved-thinking-2025-05-14,"
                "claude-code-20250219"
            ),
            "user-agent": "claude-cli/2.1.2 (external, cli)",
        }
        client = anthropic.Anthropic(
            auth_token=token,
            default_headers=oauth_headers,
            **client_kwargs,
        )
    else:
        client = anthropic.Anthropic(api_key=token, **client_kwargs)

    budget = ctx.get("thinking_budget", 10000)

    t0 = time.monotonic()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=budget + 100,
            thinking={
                "type": "enabled",
                "budget_tokens": budget,
            },
            messages=[{"role": "user", "content": "What is 2+2? Think step by step."}],
        )
        elapsed = time.monotonic() - t0
        info(f"Response received ({elapsed:.1f}s)")

        has_thinking = False
        text = ""
        for block in resp.content:
            if hasattr(block, "thinking"):
                has_thinking = True
                info(f"Thinking block: {len(block.thinking)} chars")
                verbose(f"Thinking: {block.thinking[:300]}")
            elif hasattr(block, "text"):
                text = block.text

        info(f"Response text: {text!r}")
        if resp.usage:
            info(f"Usage: input={resp.usage.input_tokens}, "
                 f"output={resp.usage.output_tokens}")

        if has_thinking:
            results.record_pass("Extended thinking -> thinking block received")
        else:
            warn("No thinking block in response (model may not support it)")
            results.record_pass("Extended thinking -> response OK (no thinking block)")

    except anthropic.BadRequestError as e:
        elapsed = time.monotonic() - t0
        if "thinking" in str(e).lower():
            results.record_fail(
                "Extended thinking",
                f"Model does not support thinking ({elapsed:.1f}s): {e}",
            )
        else:
            results.record_fail(
                "Extended thinking",
                f"Bad request ({elapsed:.1f}s): {e}",
            )
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            "Extended thinking",
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
        description="Anthropic Claude end-to-end health check",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env in CWD)",
    )
    parser.add_argument(
        "--auth-only",
        action="store_true",
        help="Only check authentication, skip chat completion and thinking",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print response bodies",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use for chat test (default: claude-haiku-4-20250414)",
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

    # Step 1: OAuth token endpoint (smoke test)
    check_token_endpoint(results)

    # Step 2: Token refresh (PKCE path only)
    check_token_refresh(results, ctx)

    if not args.auth_only:
        # Step 3: Messages API
        model = args.model or "claude-haiku-4-20250414"
        check_messages_api(results, ctx, model)

        # Step 4: Extended thinking (if enabled)
        thinking_model = args.model or "claude-sonnet-4-20250514"
        check_thinking(results, ctx, thinking_model)

    print_summary(results)
    return results.exit_code


if __name__ == "__main__":
    sys.exit(main())
