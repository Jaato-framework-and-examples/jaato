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
import re
import shutil
import subprocess
import tempfile
from typing import NamedTuple, Optional

from shared.http import get_httpx_client

logger = logging.getLogger(__name__)


class RenderResult(NamedTuple):
    """Result of a render attempt."""
    png: Optional[bytes] = None
    error: Optional[str] = None

# Cache renderer availability checks
_mmdc_path: Optional[str] = None
_mmdc_checked = False
_kroki_available: Optional[bool] = None


def render(source: str, theme: str = "default", scale: int = 2,
           background: str = "white") -> RenderResult:
    """Render Mermaid diagram source to PNG bytes.

    Args:
        source: Mermaid diagram source text.
        theme: Mermaid theme (default, dark, forest, neutral).
        scale: Scale factor for rasterization (default 2 for retina).
        background: Background color (default white).

    Returns:
        RenderResult with .png (bytes) on success, or .error (str)
        with a syntax error message when the diagram source is invalid.
    """
    # Try mmdc first (local, gold standard)
    result = _render_mmdc(source, theme, scale, background)
    if result.png is not None:
        return result

    # Try kroki.io POST API (remote, no local deps)
    result = _render_kroki(source, theme)
    if result.png is not None or result.error is not None:
        return result

    return RenderResult()


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
        with get_httpx_client(timeout=5.0) as client:
            resp = client.post(
                url,
                content=b"graph TD\n    A-->B",
                headers={"Content-Type": "text/plain",
                         "User-Agent": "jaato/1.0"},
            )
            _kroki_available = resp.status_code == 200
    except Exception:
        _kroki_available = False

    if _kroki_available:
        logger.debug("kroki available at %s", _get_kroki_url())
    else:
        logger.debug("kroki not available at %s", _get_kroki_url())

    return _kroki_available


def _render_mmdc(source: str, theme: str, scale: int,
                 background: str) -> RenderResult:
    """Render using mermaid-cli (mmdc).

    mmdc is the official Mermaid CLI tool from @mermaid-js/mermaid-cli.
    It uses Puppeteer/Playwright to render diagrams in a headless browser.
    """
    mmdc = _find_mmdc()
    if mmdc is None:
        return RenderResult()

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
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode != 0:
                logger.debug("mmdc failed: %s", proc.stderr)
                error = _extract_mmdc_error(proc.stderr)
                return RenderResult(error=error)

            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    return RenderResult(png=f.read())
            return RenderResult()

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug("mmdc execution error: %s", e)
            return RenderResult()


def _render_kroki(source: str, theme: str) -> RenderResult:
    """Render using kroki POST API.

    Sends the mermaid source as plain text to kroki's POST endpoint
    and receives PNG bytes back.  Theme is injected via mermaid's
    init directive when not 'default'.

    Args:
        source: Mermaid diagram source text.
        theme: Mermaid theme name.

    Returns:
        RenderResult with .png on success, .error on syntax error (400),
        or empty on other failures.
    """
    # Inject theme via mermaid init directive (unless already present)
    if theme != "default" and not source.lstrip().startswith("%%{"):
        source = f"%%{{init: {{'theme': '{theme}'}}}}%%\n{source}"

    import httpx

    url = f"{_get_kroki_url()}/mermaid/png"
    try:
        with get_httpx_client(timeout=30.0) as client:
            resp = client.post(
                url,
                content=source.encode("utf-8"),
                headers={"Content-Type": "text/plain",
                         "User-Agent": "jaato/1.0"},
            )
            if resp.status_code == 200:
                return RenderResult(png=resp.content)
            if resp.status_code == 400:
                body = resp.text
                logger.debug("kroki rendering failed: %s %s", resp.status_code, body)
                return RenderResult(error=_extract_kroki_error(body))
        return RenderResult()
    except httpx.HTTPError as e:
        logger.debug("kroki rendering failed: %s", e)
        return RenderResult()
    except Exception as e:
        logger.debug("kroki rendering failed: %s", e)
        return RenderResult()


def _extract_kroki_error(body: str) -> Optional[str]:
    """Extract the mermaid syntax error from a kroki 400 response body.

    Kroki returns bodies like:
        Error 400: SyntaxError: Parse error on line 8:
        ...alls renderer.render(...)| renderer_py
        -----------------------^
        Expecting 'SQE', ... got 'PS'
        Error: Syntax error in graph
        ...stack trace...

    We strip the "Error 400: " prefix and the stack trace, keeping
    just the SyntaxError through the "Expecting..." line.
    """
    if not body:
        return None
    # Strip "Error 400: " prefix
    msg = re.sub(r"^Error \d+:\s*", "", body)
    # Cut at stack trace ("    at ...")
    cut = re.search(r"\n\s+at ", msg)
    if cut:
        msg = msg[:cut.start()]
    return msg.strip() or None


def _extract_mmdc_error(stderr: str) -> Optional[str]:
    """Extract a useful error message from mmdc stderr."""
    if not stderr:
        return None
    # mmdc outputs "Error: ..." lines with parse errors
    for line in stderr.splitlines():
        if "error" in line.lower():
            return line.strip()
    return stderr.strip()[:200] or None
