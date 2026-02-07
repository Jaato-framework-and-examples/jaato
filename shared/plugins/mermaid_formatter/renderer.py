# shared/plugins/mermaid_formatter/renderer.py
"""Mermaid diagram rendering: source text → PNG bytes.

Tries multiple rendering strategies in order of preference:
1. mmdc (mermaid-cli) - gold standard, requires Node.js
2. mermaid Python package - if available
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
from typing import Optional

logger = logging.getLogger(__name__)

# Cache renderer availability check
_mmdc_path: Optional[str] = None
_mmdc_checked = False


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
    # Try mmdc first
    png = _render_mmdc(source, theme, scale, background)
    if png is not None:
        return png

    # Try mermaid Python package
    png = _render_mermaid_py(source, theme, scale, background)
    if png is not None:
        return png

    return None


def is_renderer_available() -> bool:
    """Check if any Mermaid renderer is available."""
    if _find_mmdc() is not None:
        return True
    try:
        import mermaid  # noqa: F401
        return True
    except ImportError:
        pass
    return False


def _find_mmdc() -> Optional[str]:
    """Find mmdc binary, caching the result."""
    global _mmdc_path, _mmdc_checked
    if not _mmdc_checked:
        _mmdc_path = shutil.which("mmdc")
        _mmdc_checked = True
    return _mmdc_path


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


def _render_mermaid_py(source: str, theme: str, scale: int,
                       background: str) -> Optional[bytes]:
    """Render using the mermaid Python package (if installed).

    This is an optional fallback when mmdc is not available.
    The mermaid Python package may use its own rendering engine
    or call out to a server.
    """
    try:
        import mermaid as md
        from mermaid.graph import Graph

        graph = Graph("diagram", source)
        result = md.Mermaid(graph)

        # The mermaid package may return SVG; convert to PNG if needed
        svg_data = str(result)
        if svg_data and svg_data.startswith("<svg") or svg_data.startswith("<?xml"):
            return _svg_to_png(svg_data.encode("utf-8"), scale)

        return None
    except (ImportError, Exception) as e:
        logger.debug("mermaid-py rendering failed: %s", e)
        return None


def _svg_to_png(svg_data: bytes, scale: int = 2) -> Optional[bytes]:
    """Convert SVG bytes to PNG using cairosvg if available."""
    try:
        import cairosvg
        return cairosvg.svg2png(
            bytestring=svg_data,
            scale=scale,
        )
    except ImportError:
        logger.debug("cairosvg not available for SVG→PNG conversion")
        return None
    except Exception as e:
        logger.debug("SVG→PNG conversion failed: %s", e)
        return None
