"""Environment-variable resolution for the Chrome built-in AI provider.

Follows the newer ``JAATO_``-prefixed convention (nim, openrouter, nebius,
ovhcloud).  Each helper returns ``None`` when the variable is unset so the
provider's own precedence chain (profile knob → env var → default /
fail-loud) stays explicit at the read site.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

from shared.session_context import get_session_env

#: The single on-device model the Prompt API exposes.  There is no model
#: selection in the API; this name is nominal (used for display, profiles,
#: and the ledger).
DEFAULT_MODEL = "gemini-nano"

#: Default page the provider evaluates the Prompt API on.  A top-level
#: ``about:blank`` tab is a secure context in Chrome, which the built-in AI
#: APIs require.  Override via the ``page_url`` knob if a Chrome build gates
#: the web-page API on a real origin.
DEFAULT_PAGE_URL = "about:blank"

#: Well-known browser binary names/paths, tried in order by
#: :func:`find_browser_binary`.  Microsoft Edge is included because it ships
#: the same-shaped Prompt API (backed by Phi-4-mini / Aion-1.0-Instruct)
#: since Edge 138 — a CDP bridge written against ``LanguageModel`` works on
#: both.  Plain Chromium is deliberately ABSENT: the on-device model is
#: Google-proprietary and never available in unbranded builds.
BROWSER_CANDIDATES: List[str] = [
    "google-chrome",
    "google-chrome-stable",
    "google-chrome-beta",
    "google-chrome-unstable",
    "chrome",
    "msedge",
    "microsoft-edge",
    "microsoft-edge-stable",
]

#: Absolute install locations checked after PATH lookup (macOS / Windows).
BROWSER_ABSOLUTE_PATHS: List[str] = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def resolve_binary() -> Optional[str]:
    """``JAATO_CHROME_AI_BINARY`` — explicit browser binary path."""
    return get_session_env("JAATO_CHROME_AI_BINARY") or None


def resolve_cdp_url() -> Optional[str]:
    """``JAATO_CHROME_AI_CDP_URL`` — attach to a running browser.

    Accepts ``http(s)://host:port`` (the DevTools HTTP endpoint; the
    browser WebSocket URL is discovered via ``/json/version``) or a full
    ``ws://`` DevTools URL.  When set, the provider never launches its
    own browser process.
    """
    return get_session_env("JAATO_CHROME_AI_CDP_URL") or None


def resolve_user_data_dir() -> Optional[str]:
    """``JAATO_CHROME_AI_USER_DATA_DIR`` — persistent profile directory.

    The Gemini Nano component download is profile-bound, so the profile
    must persist across sessions (a fresh ephemeral profile reports the
    model as ``downloadable`` every time).
    """
    return get_session_env("JAATO_CHROME_AI_USER_DATA_DIR") or None


def default_user_data_dir() -> str:
    """The provider-owned default profile: ``~/.jaato/chrome_ai/profile``."""
    return str(Path.home() / ".jaato" / "chrome_ai" / "profile")


def resolve_headless() -> Optional[bool]:
    """``JAATO_CHROME_AI_HEADLESS`` — parse a boolean-ish env value."""
    raw = get_session_env("JAATO_CHROME_AI_HEADLESS")
    if raw is None or raw == "":
        return None
    return raw.strip().lower() not in ("0", "false", "no", "off")


def resolve_context_length() -> Optional[int]:
    """``JAATO_CHROME_AI_CONTEXT_LENGTH`` — manual context-window override."""
    raw = get_session_env("JAATO_CHROME_AI_CONTEXT_LENGTH")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def resolve_model() -> Optional[str]:
    """``JAATO_CHROME_AI_MODEL`` — nominal default model name."""
    return get_session_env("JAATO_CHROME_AI_MODEL") or None


def resolve_page_url() -> Optional[str]:
    """``JAATO_CHROME_AI_PAGE_URL`` — page to host the Prompt API calls."""
    return get_session_env("JAATO_CHROME_AI_PAGE_URL") or None


def find_browser_binary(explicit: Optional[str] = None) -> Optional[str]:
    """Locate a launchable Chrome/Edge binary.

    Precedence: ``explicit`` (profile knob) → ``JAATO_CHROME_AI_BINARY``
    → PATH lookup over :data:`BROWSER_CANDIDATES` → the well-known
    absolute install paths.  Returns ``None`` when nothing is found so
    the caller can raise the provider-specific fail-loud error.
    """
    for candidate in (explicit, resolve_binary()):
        if candidate:
            resolved = shutil.which(candidate) or (
                candidate if os.path.isfile(candidate) else None
            )
            if resolved:
                return resolved
            return None  # explicitly named but missing: don't silently fall through
    for name in BROWSER_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    for path in BROWSER_ABSOLUTE_PATHS:
        if os.path.isfile(path):
            return path
    return None
