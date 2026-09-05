"""Chrome Built-in AI (Gemini Nano) Model Provider Plugin.

Drives the on-device LLM embedded in Google Chrome — Gemini Nano, exposed
through the built-in AI **Prompt API** (the ``LanguageModel`` global,
stable for web pages since Chrome 148) — by launching or attaching to a
real Chrome (or Microsoft Edge, which ships the same-shaped API) over the
Chrome DevTools Protocol.  No new Python dependencies: the CDP transport
reuses the core ``websockets`` package.

Benefits:
- Zero-cost local inference through a browser most users already have —
  no API keys, no separate model runtime, data never leaves the machine.

Requirements & limits (know before you fly):
- **Branded Google Chrome 138+/148+ or recent Edge only** — plain
  Chromium has no on-device model (Google-proprietary component).
- The model (~2-4 GB component) must be downloaded into the browser
  profile; the provider uses a persistent profile
  (``~/.jaato/chrome_ai/profile`` by default) and can trigger the
  download with the ``auto_download`` knob.
- The context window is TINY (~6-9k tokens, detected from the session
  quota) and shared between input and output — pair with an aggressive
  GC strategy.
- Tool calling is prompt-injected (no native tool API on stable Chrome);
  small-model quality applies.  Structured output uses the API's native
  ``responseConstraint`` JSON-Schema support.

Configuration:
    Environment variables:
        JAATO_CHROME_AI_BINARY: Browser binary path (default: search
            PATH / well-known install locations for Chrome, then Edge)
        JAATO_CHROME_AI_CDP_URL: Attach to a running browser
            (http://host:port DevTools endpoint or ws:// URL) instead of
            launching one
        JAATO_CHROME_AI_USER_DATA_DIR: Persistent profile directory
        JAATO_CHROME_AI_HEADLESS: Run headless (default: true; the model
            must already be downloaded in the profile — first-time setup
            may need a headed run)
        JAATO_CHROME_AI_CONTEXT_LENGTH: Manual context-window override
        JAATO_CHROME_AI_MODEL: Nominal model name (default: gemini-nano)
        JAATO_CHROME_AI_PAGE_URL: Page hosting the API calls (default:
            about:blank)

Example:
    from shared.plugins.model_provider.chrome_ai import ChromeAIProvider

    provider = ChromeAIProvider()
    provider.initialize()          # no credentials needed
    provider.connect('gemini-nano')
    result = provider.complete(messages=[...])
"""

from .env import (
    DEFAULT_MODEL,
    DEFAULT_PAGE_URL,
    find_browser_binary,
    resolve_binary,
    resolve_cdp_url,
    resolve_context_length,
    resolve_model,
    resolve_user_data_dir,
)
from .errors import (
    ChromeAIBinaryNotFoundError,
    ChromeAIConnectionError,
    ChromeAIError,
    ChromeAIResponseError,
    ChromeAIUnavailableError,
)
from .provider import ChromeAIProvider, create_provider

__all__ = [
    # Main provider
    "ChromeAIProvider",
    "create_provider",
    # Errors
    "ChromeAIError",
    "ChromeAIBinaryNotFoundError",
    "ChromeAIConnectionError",
    "ChromeAIResponseError",
    "ChromeAIUnavailableError",
    # Environment
    "DEFAULT_MODEL",
    "DEFAULT_PAGE_URL",
    "find_browser_binary",
    "resolve_binary",
    "resolve_cdp_url",
    "resolve_context_length",
    "resolve_model",
    "resolve_user_data_dir",
]


# --- Provider capability contract (see docs/model-provider-capabilities.md) ---
from ..base import (  # noqa: E402
    ProviderCapabilities, ProviderKnobs, KnobLayer, KnobSpec, AuthSource,
)

PROVIDER_CAPABILITIES = ProviderCapabilities(
    user_message_images=False,   # Prompt API has image input; adapter is text-only (v1)
    tool_result_images=False,
    pdf_input=False,
    tool_choice_forwarding=False,  # no such control in the Prompt API
    thinking=False,
    prompt_caching=False,
    streaming=True,                # promptStreaming -> on_chunk
    cancellation=True,             # cancel token -> page-side AbortController
    output_media=False,
)

# --- Provider config-knob contract (authored from provider.py read sites) ---
PROVIDER_KNOBS = ProviderKnobs(layers=(
    KnobLayer("top_level", (
        KnobSpec("binary", "str", None,
                 "Browser binary path (JAATO_CHROME_AI_BINARY override)"),
        KnobSpec("cdp_url", "str", None,
                 "Attach to a running browser (http://host:port or ws:// "
                 "DevTools URL) instead of launching one"),
        KnobSpec("user_data_dir", "str", None,
                 "Persistent Chrome profile dir (default "
                 "~/.jaato/chrome_ai/profile; the model download is "
                 "profile-bound)"),
        KnobSpec("headless", "bool", True,
                 "Launch with --headless=new (model must already be in "
                 "the profile)"),
        KnobSpec("page_url", "str", "about:blank",
                 "Page hosting the Prompt API calls (point at an https "
                 "origin if the build gates the API)"),
        KnobSpec("extra_args", "list", None,
                 "Additional Chrome CLI switches (e.g. --enable-features=...)"),
        KnobSpec("auto_download", "bool", False,
                 "Trigger the on-device model component download at "
                 "connect() when the model is 'downloadable'"),
        KnobSpec("download_timeout", "int", 1800,
                 "Seconds to wait for the model download"),
        KnobSpec("connect_timeout", "int", 30,
                 "Seconds for browser launch / CDP attach"),
        KnobSpec("turn_timeout", "int", 300,
                 "Seconds of mid-turn event silence before aborting"),
        KnobSpec("context_length", "int", None,
                 "Manual context-window override (normally detected from "
                 "the session quota)"),
        KnobSpec("warmup", "bool", True,
                 "Run one throwaway generation at connect() to absorb the "
                 "on-device model's cold-start (~11s first-token) off the "
                 "first real turn; set false for fastest connect"),
    ), description="browser connection"),
    KnobLayer("api_params", (
        KnobSpec("temperature", "float", None,
                 "Sampling temperature (LanguageModel.create option; "
                 "unset = browser default)"),
        KnobSpec("top_k", "int", None,
                 "Top-K sampling (LanguageModel.create option; unset = "
                 "browser default)"),
    ), description="LanguageModel.create() sampling options"),
))
PROVIDER_QUIRKS = frozenset()

# --- Provider credential-resolution contract (from verify_auth) ---
PROVIDER_AUTH_RESOLUTION = (
    AuthSource("none", "", "local browser over CDP — no credential"),
)
