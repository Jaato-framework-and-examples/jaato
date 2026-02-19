#!/usr/bin/env python3
"""End-to-end health check for Zhipu AI (Z.AI) authentication and inference.

Traces every credential source and HTTP endpoint in the API key → model
listing → chat completion pipeline, using jaato's shared HTTP module so
proxy, SSL, and Kerberos configuration is exercised exactly as the real
provider would.

Usage:
    # From repo root (loads .env automatically):
    .venv/bin/python scripts/zhipuai_health_check.py

    # Skip the chat completion test (auth-only check):
    .venv/bin/python scripts/zhipuai_health_check.py --auth-only

    # Use a specific .env file:
    .venv/bin/python scripts/zhipuai_health_check.py --env-file /path/to/.env

    # Verbose mode (print response bodies):
    .venv/bin/python scripts/zhipuai_health_check.py -v

    # Test a specific model:
    .venv/bin/python scripts/zhipuai_health_check.py --model glm-5

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
    from shared.plugins.model_provider.zhipuai.env import (
        resolve_api_key,
        resolve_base_url,
        resolve_model,
        resolve_context_length,
        resolve_enable_thinking,
        resolve_thinking_budget,
    )
    from shared.plugins.model_provider.zhipuai.auth import (
        load_credentials,
        get_stored_api_key,
        get_stored_base_url,
    )

    # 1. API key from environment
    env_key = resolve_api_key()
    if env_key:
        info(f"Env API key (ZHIPUAI_API_KEY): {mask(env_key)}")
        results.record_pass("Env API key found")
    else:
        info("Env API key (ZHIPUAI_API_KEY): (not set)")

    # 2. Stored credentials
    creds = load_credentials()
    if creds:
        info(f"Stored API key: {mask(creds.api_key)}")
        info(f"  saved: {time.ctime(creds.created_at)}")
        if creds.base_url:
            info(f"  stored base_url: {creds.base_url}")
        results.record_pass("Stored credentials found")
    else:
        info("Stored credentials: (none)")

    # Effective key (env takes precedence over stored)
    api_key = env_key or (creds.api_key if creds else None)
    if api_key:
        source = "environment" if env_key else "stored credentials"
        info(f"Effective API key: {mask(api_key)} (from {source})")
        ctx["api_key"] = api_key
        ctx["key_source"] = source
    else:
        ctx["api_key"] = None
        ctx["key_source"] = None
        results.record_fail(
            "Credentials",
            "No API key found (set ZHIPUAI_API_KEY or run zhipuai-auth login)",
        )

    # Base URL
    base_url = resolve_base_url()
    stored_base = get_stored_base_url()
    if stored_base and base_url == "https://api.z.ai/api/anthropic":
        base_url = stored_base
    ctx["base_url"] = base_url
    info(f"Base URL: {base_url}")

    # Model override
    env_model = resolve_model()
    if env_model:
        info(f"Default model (ZHIPUAI_MODEL): {env_model}")

    # Context length override
    ctx_len = resolve_context_length()
    if ctx_len:
        info(f"Context length override: {ctx_len}")

    # Thinking config
    thinking = resolve_enable_thinking()
    thinking_budget = resolve_thinking_budget()
    info(f"Extended thinking: {thinking} (budget: {thinking_budget})")
    ctx["enable_thinking"] = thinking
    ctx["thinking_budget"] = thinking_budget

    # Verify httpx client can be constructed
    try:
        client = get_httpx_client(timeout=5.0)
        client.close()
        results.record_pass("httpx client construction")
    except Exception as e:
        results.record_fail("httpx client construction", str(e))

    return ctx


# ---------------------------------------------------------------------------
# Step 1: API key validation (lightweight auth check)
# ---------------------------------------------------------------------------
def check_api_key_validation(results: CheckResult, ctx: Dict[str, Any]) -> None:
    """Validate the API key by sending a minimal request to the messages endpoint.

    Uses the same validation logic as the auth module to exercise the real
    proxy/SSL path.
    """
    header("Step 1: API Key Validation")

    api_key = ctx.get("api_key")
    base_url = ctx.get("base_url", "https://api.z.ai/api/anthropic")
    test_url = f"{base_url.rstrip('/')}/v1/messages"

    info(f"POST {test_url}")

    if not api_key:
        results.record_skip(f"POST {test_url}", "no API key available")
        return

    from shared.http import get_httpx_client

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": "glm-4.7",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }

    t0 = time.monotonic()
    try:
        with get_httpx_client(timeout=30.0) as client:
            resp = client.post(test_url, headers=headers, json=body, timeout=30)
        elapsed = time.monotonic() - t0
        info(f"Status: {resp.status_code} ({elapsed:.1f}s)")
        verbose(f"Response: {resp.text[:500]}")

        if resp.status_code in (401, 403):
            results.record_fail(
                f"POST {test_url}",
                f"Authentication failed (HTTP {resp.status_code}) — check your API key",
            )
        elif resp.status_code == 200:
            results.record_pass(f"POST {test_url} -> 200 (key valid)")
        else:
            # Any non-auth error means the key was accepted
            try:
                body_json = resp.json()
                verbose(f"Response body: {json.dumps(body_json, indent=2)[:500]}")
            except Exception:
                pass
            results.record_pass(
                f"POST {test_url} -> HTTP {resp.status_code} (key accepted)"
            )
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"POST {test_url}",
            f"{type(e).__name__}: {e} ({elapsed:.1f}s)",
        )


# ---------------------------------------------------------------------------
# Step 2: Models listing endpoint (OpenAI-compatible)
# ---------------------------------------------------------------------------
def check_models_endpoint(
    results: CheckResult, ctx: Dict[str, Any]
) -> Optional[List[str]]:
    """List models from Z.AI's OpenAI-compatible /models endpoint.

    Returns the model list on success, None on failure or skip.
    """
    header("Step 2: Models Listing Endpoint")

    api_key = ctx.get("api_key")
    base_url = ctx.get("base_url", "https://api.z.ai/api/anthropic")

    # Derive the OpenAI-compatible models URL from the Anthropic base URL
    from shared.plugins.model_provider.zhipuai.provider import _openai_models_url
    models_url = _openai_models_url(base_url)

    info(f"GET {models_url}")

    if not api_key:
        results.record_skip(f"GET {models_url}", "no API key available")
        return None

    from shared.http import get_httpx_client

    t0 = time.monotonic()
    try:
        with get_httpx_client(timeout=15.0) as client:
            resp = client.get(
                models_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
            )
        elapsed = time.monotonic() - t0
        info(f"Status: {resp.status_code} ({elapsed:.1f}s)")

        if resp.status_code == 200:
            body = resp.json()
            verbose(f"Response: {json.dumps(body, indent=2)[:2000]}")
            models = [m.get("id") for m in body.get("data", []) if m.get("id")]
            info(f"Models available: {len(models)}")
            for m in sorted(models)[:15]:
                info(f"  - {m}")
            if len(models) > 15:
                info(f"  ... and {len(models) - 15} more")
            results.record_pass(f"GET {models_url} -> {len(models)} models")
            return models
        elif resp.status_code in (401, 403):
            results.record_fail(
                f"GET {models_url}",
                f"Authentication failed (HTTP {resp.status_code})",
            )
            return None
        else:
            results.record_fail(
                f"GET {models_url}",
                f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
            return None
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"GET {models_url}",
            f"{type(e).__name__}: {e} ({elapsed:.1f}s)",
        )
        return None


# ---------------------------------------------------------------------------
# Step 3: Chat completion (Anthropic-compatible messages API)
# ---------------------------------------------------------------------------
def check_chat_completion(
    results: CheckResult,
    ctx: Dict[str, Any],
    model: str = "glm-4.7",
) -> None:
    """Send a minimal chat completion through the Anthropic-compatible API.

    Uses the Anthropic Python SDK pointed at Z.AI's base URL, exercising the
    same code path as the ZhipuAIProvider.
    """
    header("Step 3: Chat Completion (Anthropic-compatible)")

    api_key = ctx.get("api_key")
    base_url = ctx.get("base_url", "https://api.z.ai/api/anthropic")
    messages_url = f"{base_url.rstrip('/')}/v1/messages"

    info(f"POST {messages_url}")
    info(f"Model: {model}")

    if not api_key:
        results.record_skip(f"POST {messages_url}", "no API key available")
        return

    try:
        import anthropic
    except ImportError:
        results.record_fail(
            f"POST {messages_url}",
            "anthropic package not installed (pip install anthropic)",
        )
        return

    from shared.http import get_httpx_client

    # Build client matching ZhipuAIProvider._create_client() logic
    client_kwargs: Dict[str, Any] = {
        "base_url": base_url,
        "api_key": api_key,
    }
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

    client = anthropic.Anthropic(**client_kwargs)

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
        results.record_pass(f"POST {messages_url} -> {resp.stop_reason}")

    except anthropic.AuthenticationError as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"POST {messages_url}",
            f"Authentication failed ({elapsed:.1f}s): {e}",
        )
    except anthropic.NotFoundError as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"POST {messages_url}",
            f"Model not found ({elapsed:.1f}s): {e}",
        )
    except anthropic.RateLimitError as e:
        elapsed = time.monotonic() - t0
        # Rate limit means auth worked
        warn(f"Rate limited ({elapsed:.1f}s) — auth is valid but limit hit")
        results.record_pass(
            f"POST {messages_url} -> rate limited (auth valid)"
        )
    except anthropic.APIStatusError as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"POST {messages_url}",
            f"HTTP {e.status_code} ({elapsed:.1f}s): {e.message}",
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        results.record_fail(
            f"POST {messages_url}",
            f"{type(e).__name__}: {e} ({elapsed:.1f}s)",
        )


# ---------------------------------------------------------------------------
# Step 4: Extended Thinking (optional)
# ---------------------------------------------------------------------------
def check_thinking(
    results: CheckResult,
    ctx: Dict[str, Any],
    model: str = "glm-4.7",
) -> None:
    """Verify extended thinking works if enabled.

    GLM-5 and GLM-4.7 (non-flash) support native chain-of-thought reasoning.
    """
    header("Step 4: Extended Thinking")

    if not ctx.get("enable_thinking"):
        results.record_skip(
            "Extended thinking",
            "not enabled (set ZHIPUAI_ENABLE_THINKING=true)",
        )
        return

    api_key = ctx.get("api_key")
    base_url = ctx.get("base_url", "https://api.z.ai/api/anthropic")
    messages_url = f"{base_url.rstrip('/')}/v1/messages"

    info(f"POST {messages_url} (with thinking)")
    info(f"Model: {model}")
    info(f"Thinking budget: {ctx.get('thinking_budget', 10000)}")

    if not api_key:
        results.record_skip("Extended thinking", "no API key available")
        return

    try:
        import anthropic
    except ImportError:
        results.record_fail("Extended thinking", "anthropic package not installed")
        return

    from shared.http import get_httpx_client

    client_kwargs: Dict[str, Any] = {
        "base_url": base_url,
        "api_key": api_key,
    }
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

    client = anthropic.Anthropic(**client_kwargs)
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
        description="Zhipu AI (Z.AI) end-to-end health check",
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
        help="Model to use for chat test (default: glm-4.7)",
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

    # Step 1: API key validation
    check_api_key_validation(results, ctx)

    # Step 2: Models listing
    check_models_endpoint(results, ctx)

    if not args.auth_only:
        # Step 3: Chat completion
        model = args.model or "glm-4.7"
        check_chat_completion(results, ctx, model)

        # Step 4: Extended thinking (if enabled)
        thinking_model = args.model or "glm-4.7"
        check_thinking(results, ctx, thinking_model)

    print_summary(results)
    return results.exit_code


if __name__ == "__main__":
    sys.exit(main())
