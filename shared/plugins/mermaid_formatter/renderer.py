# shared/plugins/mermaid_formatter/renderer.py
"""Mermaid diagram rendering: source text → PNG bytes.

Tries multiple rendering strategies in order of preference:
1. mmdc (mermaid-cli) - gold standard, requires Node.js
2. kroki.io POST API - remote rendering, no local dependencies
3. Returns None if no renderer is available

The caller is responsible for graceful degradation when rendering
is unavailable (e.g., showing the raw mermaid source as a code block).
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# Cache renderer availability checks
_mmdc_path: Optional[str] = None
_mmdc_checked = False
_kroki_available: Optional[bool] = None


def render(source: str, theme: str = "default", scale: int = 2,
           background: str = "white") -> Optional[bytes]:
    """Render Mermaid diagram source to PNG bytes.

    Args:
        source: Mermaid diagram source text.
        theme: Mermaid theme (default, dark, forest, neutral).
        scale: Scale factor for rasterization (default 2 for retina).
        background: Background color (default white).

    Returns:
        PNG image bytes, or None if no renderer is available.
    """
    # Try mmdc first (local, gold standard)
    png = _render_mmdc(source, theme, scale, background)
    if png is not None:
        return png

    # Try kroki.io POST API (remote, no local deps)
    png = _render_kroki(source, theme)
    if png is not None:
        return png

    return None


def is_renderer_available() -> bool:
    """Check if any Mermaid renderer is available.

    Checks local mmdc first, then probes kroki.io reachability
    (cached after first check).
    """
    if _find_mmdc() is not None:
        return True
    return _check_kroki()


def _find_mmdc() -> Optional[str]:
    """Find mmdc binary, caching the result."""
    global _mmdc_path, _mmdc_checked
    if not _mmdc_checked:
        _mmdc_path = shutil.which("mmdc")
        _mmdc_checked = True
    return _mmdc_path


def _get_kroki_url() -> str:
    """Get the kroki base URL (configurable for self-hosted instances).

    Set JAATO_KROKI_URL to point to a self-hosted kroki instance,
    e.g. http://localhost:8000 — useful when the public kroki.io
    is blocked by an enterprise firewall.
    """
    return os.environ.get("JAATO_KROKI_URL", "https://kroki.io").rstrip("/")


def _check_kroki() -> bool:
    """Check if kroki is reachable by rendering a tiny test diagram.

    Caches the result process-wide to avoid repeated network checks.
    Called once during session setup when building system instructions.
    """
    global _kroki_available
    if _kroki_available is not None:
        return _kroki_available

    url = f"{_get_kroki_url()}/mermaid/png"
    try:
        req = urllib.request.Request(
            url,
            data=b"graph TD\n    A-->B",
            headers={"Content-Type": "text/plain"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            _kroki_available = resp.status == 200
    except Exception:
        _kroki_available = False

    if _kroki_available:
        logger.debug("kroki available at %s", _get_kroki_url())
    else:
        logger.debug("kroki not available at %s", _get_kroki_url())

    return _kroki_available


def _render_mmdc(source: str, theme: str, scale: int,
                 background: str) -> Optional[bytes]:
    """Render using mermaid-cli (mmdc).

    mmdc is the official Mermaid CLI tool from @mermaid-js/mermaid-cli.
    It uses Puppeteer/Playwright to render diagrams in a headless browser.
    """
    mmdc = _find_mmdc()
    if mmdc is None:
        return None

    with tempfile.TemporaryDirectory(prefix="jaato_mermaid_") as tmpdir:
        input_path = os.path.join(tmpdir, "input.mmd")
        output_path = os.path.join(tmpdir, "output.png")
        config_path = os.path.join(tmpdir, "config.json")

        # Write input file
        with open(input_path, "w") as f:
            f.write(source)

        # Write config for puppeteer args (no-sandbox for containers)
        config = {
            "puppeteerConfig": {
                "args": ["--no-sandbox", "--disable-setuid-sandbox"]
            }
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        cmd = [
            mmdc,
            "-i", input_path,
            "-o", output_path,
            "-t", theme,
            "-s", str(scale),
            "-b", background,
            "-p", config_path,
            "--quiet",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.debug("mmdc failed: %s", result.stderr)
                return None

            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    return f.read()
            return None

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug("mmdc execution error: %s", e)
            return None


def _render_kroki(source: str, theme: str) -> Optional[bytes]:
    """Render using kroki POST API.

    Sends the mermaid source as plain text to kroki's POST endpoint
    and receives PNG bytes back.  Theme is injected via mermaid's
    init directive when not 'default'.

    Args:
        source: Mermaid diagram source text.
        theme: Mermaid theme name.

    Returns:
        PNG image bytes, or None if rendering failed.
    """
    # Inject theme via mermaid init directive (unless already present)
    if theme != "default" and not source.lstrip().startswith("%%{"):
        source = f"%%{{init: {{'theme': '{theme}'}}}}%%\n{source}"

    url = f"{_get_kroki_url()}/mermaid/png"
    try:
        req = urllib.request.Request(
            url,
            data=source.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                return resp.read()
        return None
    except Exception as e:
        logger.debug("kroki rendering failed: %s", e)
        return None
