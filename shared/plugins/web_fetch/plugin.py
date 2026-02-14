"""Web fetch plugin for retrieving and parsing web page content."""

import os
import re
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional
from urllib.parse import urljoin, urlparse

from ..base import UserCommand
from ..background import BackgroundCapableMixin
from ..model_provider.types import ToolSchema
from shared.trace import trace as _trace_write


DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_MAX_LENGTH = 100000  # max characters to return
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; JaatoBot/1.0; +https://github.com/apanoia/jaato)"

# Auto-background threshold for web_fetch (covers PDF download + conversion).
# Regular HTML fetches complete well within this; only PDFs are likely to exceed it.
DEFAULT_WEB_FETCH_AUTO_BACKGROUND_THRESHOLD = 10.0

# Content types that can be converted to markdown (not treated as opaque binary).
PDF_CONTENT_TYPES = {'application/pdf'}
PDF_EXTENSIONS = {'.pdf'}

# Maximum PDF file size to attempt conversion (50 MB)
MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024


class WebFetchPlugin(BackgroundCapableMixin):
    """Plugin that fetches and parses web page content.

    Supports multiple output modes:
        - markdown: Clean markdown conversion (default, best for reading)
        - structured: JSON with extracted components (links, images, tables, etc.)
        - raw: Raw HTML content

    PDF support:
        When the URL points to a PDF, the plugin downloads the file and converts
        it to markdown using pymupdf4llm (if installed). Falls back to returning
        binary metadata when the library is unavailable.

    Inherits from BackgroundCapableMixin so that slow fetches (e.g. large PDFs)
    are automatically backgrounded by the ToolExecutor after a threshold.

    Configuration:
        timeout: Request timeout in seconds (default: 30).
        max_length: Maximum content length to return (default: 100000).
        user_agent: Custom User-Agent string.
        follow_redirects: Whether to follow redirects (default: True).
    """

    def __init__(self):
        # Initialize BackgroundCapableMixin for auto-background support
        super().__init__(max_workers=2)

        self._timeout: int = DEFAULT_TIMEOUT
        self._max_length: int = DEFAULT_MAX_LENGTH
        self._user_agent: str = DEFAULT_USER_AGENT
        self._follow_redirects: bool = True
        self._initialized = False
        self._auto_background_threshold: float = DEFAULT_WEB_FETCH_AUTO_BACKGROUND_THRESHOLD
        # Cache for recently fetched pages (simple in-memory cache)
        self._cache: Dict[str, tuple] = {}  # url -> (content, timestamp)
        self._cache_ttl: int = 300  # 5 minutes
        # Agent context for trace logging
        self._agent_name: Optional[str] = None

    @property
    def name(self) -> str:
        return "web_fetch"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        _trace_write("WEB_FETCH", msg)

    # --- BackgroundCapableMixin overrides ---

    def supports_background(self, tool_name: str) -> bool:
        """web_fetch supports background execution for slow fetches (e.g. PDFs)."""
        return tool_name == 'web_fetch'

    def get_auto_background_threshold(self, tool_name: str) -> Optional[float]:
        """Return threshold in seconds before auto-backgrounding web_fetch.

        Regular HTML fetches complete well within this window; only large
        PDF downloads + conversion are likely to exceed it.
        """
        if tool_name == 'web_fetch':
            return self._auto_background_threshold
        return None

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the web fetch plugin.

        Args:
            config: Optional dict with:
                - timeout: Request timeout in seconds (default: 30)
                - max_length: Max content length to return (default: 100000)
                - user_agent: Custom User-Agent string
                - follow_redirects: Whether to follow redirects (default: True)
                - cache_ttl: Cache TTL in seconds (default: 300)
        """
        if config:
            self._agent_name = config.get("agent_name")
            if 'timeout' in config:
                self._timeout = config['timeout']
            if 'max_length' in config:
                self._max_length = config['max_length']
            if 'user_agent' in config:
                self._user_agent = config['user_agent']
            if 'follow_redirects' in config:
                self._follow_redirects = config['follow_redirects']
            if 'cache_ttl' in config:
                self._cache_ttl = config['cache_ttl']
        self._initialized = True
        self._trace(f"initialize: timeout={self._timeout}, max_length={self._max_length}")

    def shutdown(self) -> None:
        """Shutdown the web fetch plugin."""
        self._trace("shutdown: clearing cache")
        self._cache.clear()
        self._initialized = False

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return the ToolSchema for the web fetch tool."""
        return [ToolSchema(
            name='web_fetch',
            description='Fetch a web page or PDF and return its content in a readable format. '
                       'Supports markdown conversion (including PDF-to-markdown), '
                       'structured data extraction, and CSS selectors.',
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["markdown", "structured", "raw"],
                        "description": "Output format: 'markdown' (clean text, default), "
                                      "'structured' (JSON with components), 'raw' (HTML)"
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector to extract specific element(s). "
                                      "Example: 'main', '.content', '#article'"
                    },
                    "extract": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["links", "images", "tables", "forms", "headings", "metadata"]
                        },
                        "description": "Specific components to extract (used with structured mode)"
                    },
                    "include_links": {
                        "type": "boolean",
                        "description": "Include a list of links found on the page (default: false)"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Custom HTTP headers to send with the request (e.g., for authentication). "
                                      "Example: {\"Authorization\": \"Bearer token\"}"
                    },
                    "no_cache": {
                        "type": "boolean",
                        "description": "Bypass the cache and fetch fresh content (default: false)"
                    },
                    "include_headers": {
                        "type": "boolean",
                        "description": "Include response headers in the result (default: false)"
                    },
                    "insecure": {
                        "type": "boolean",
                        "description": (
                            "Skip SSL certificate verification for this request. "
                            "Only use after the user explicitly confirms they trust "
                            "the target host. Default: false."
                        )
                    },
                    "no_proxy": {
                        "type": "boolean",
                        "description": (
                            "Bypass the configured HTTP proxy for this request. "
                            "Only use after the user confirms the host should be "
                            "reached directly. Default: false."
                        )
                    }
                },
                "required": ["url"]
            },
            category="web",
        )]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executor mapping."""
        return {'web_fetch': self._execute}

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the web fetch tool."""
        return """You have access to `web_fetch` which retrieves and parses web page content.

**Modes:**
- `markdown` (default): Returns clean, readable text converted from HTML
- `structured`: Returns JSON with extracted components (links, images, tables, etc.) or parsed JSON data
- `raw`: Returns the raw content (use sparingly, token-heavy)

**Content-Type Detection:** The tool automatically detects JSON, XML, PDF, and plain text content:
- JSON APIs return pretty-printed JSON (markdown mode) or parsed data (structured mode)
- XML feeds are returned as-is in readable format
- PDF documents are downloaded and converted to markdown (requires pymupdf4llm)
- HTML is converted to clean markdown

**PDF Support:**
- PDF URLs are automatically detected and converted to markdown
- Tables, headers, bold/italic formatting are preserved
- Large PDFs may be auto-backgrounded; use `getBackgroundTask()` to retrieve the result
- If pymupdf4llm is not installed, returns binary metadata with an install hint

**Examples:**
```
# Read an article as markdown
web_fetch(url="https://example.com/article")

# Read a PDF document
web_fetch(url="https://example.com/document.pdf")

# Extract just the main content area
web_fetch(url="https://example.com", selector=".main-content")

# Get structured data about the page
web_fetch(url="https://example.com", mode="structured", extract=["links", "images"])

# Get page content with links list
web_fetch(url="https://example.com", include_links=true)

# Fetch with authentication
web_fetch(url="https://api.example.com/data", headers={"Authorization": "Bearer token"})

# Bypass cache for fresh content
web_fetch(url="https://example.com/live-data", no_cache=true)

# Include response headers (for debugging or cache inspection)
web_fetch(url="https://example.com", include_headers=true)
```

**Tips:**
- Use `selector` to focus on specific page sections (server-rendered HTML only, no JS execution)
- Use `structured` mode when you need to analyze page components or JSON API data
- The tool follows redirects and handles common encodings
- Results are cached briefly; use `no_cache=true` to bypass
- PDF URLs are converted to markdown when pymupdf4llm is installed
- Binary content (images, archives, etc.) returns metadata instead of garbled text
- Use `headers` parameter for authentication (Bearer tokens, API keys, etc.)
- Use `include_headers=true` to see response headers like Last-Modified, ETag, Cache-Control

**SSL certificate issues:**
- If a request fails with `ssl_error: true`, ask the user if they trust the host
- If the user confirms, retry with `insecure=true` to skip SSL verification
- Never set `insecure=true` without explicit user confirmation

**Proxy issues:**
- If a request fails with `proxy_error: true`, ask the user if the host should be reached directly
- If the user confirms, retry with `no_proxy=true` to bypass the proxy
- Never set `no_proxy=true` without explicit user confirmation"""

    def get_auto_approved_tools(self) -> List[str]:
        """Web fetch is read-only and safe - auto-approve it."""
        return ['web_fetch']

    def get_user_commands(self) -> List[UserCommand]:
        """Web fetch plugin provides model tools only, no user commands."""
        return []

    def _get_cached(self, url: str) -> Optional[str]:
        """Get cached content if still valid."""
        if url in self._cache:
            content, timestamp = self._cache[url]
            age = (datetime.now() - timestamp).total_seconds()
            if age < self._cache_ttl:
                self._trace(f"cache hit: {url} (age={age:.1f}s)")
                return content
            else:
                del self._cache[url]
        return None

    def _set_cached(self, url: str, content: str) -> None:
        """Cache content for URL."""
        # Simple LRU: remove oldest if cache too large
        if len(self._cache) > 50:
            oldest_url = min(self._cache.keys(), key=lambda u: self._cache[u][1])
            del self._cache[oldest_url]
        self._cache[url] = (content, datetime.now())

    # Content types that should be treated as binary (return metadata only).
    # Note: application/pdf is NOT here — PDFs are handled separately for conversion.
    BINARY_CONTENT_TYPES = {
        'image/', 'audio/', 'video/', 'application/octet-stream',
        'application/zip', 'application/gzip',
        'application/x-tar', 'application/x-rar', 'application/x-7z',
        'application/vnd.', 'application/x-executable',
        'font/', 'model/',
    }

    # File extensions that indicate binary content.
    # Note: .pdf is NOT here — PDFs are handled separately for conversion.
    BINARY_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico', '.svg',
        '.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a',
        '.mp4', '.webm', '.avi', '.mov', '.mkv',
        '.zip', '.gz', '.tar', '.rar', '.7z',
        '.exe', '.dll', '.so', '.dylib',
        '.woff', '.woff2', '.ttf', '.otf', '.eot',
        '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    }

    def _is_binary_content_type(self, content_type: str) -> bool:
        """Check if content type indicates binary content (excluding PDFs)."""
        if not content_type:
            return False
        content_type = content_type.lower().split(';')[0].strip()
        for binary_prefix in self.BINARY_CONTENT_TYPES:
            if content_type.startswith(binary_prefix):
                return True
        return False

    def _is_binary_url(self, url: str) -> bool:
        """Check if URL extension indicates binary content (excluding PDFs)."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        for ext in self.BINARY_EXTENSIONS:
            if path.endswith(ext):
                return True
        return False

    @staticmethod
    def _is_pdf_content_type(content_type: str) -> bool:
        """Check if content type indicates a PDF document."""
        if not content_type:
            return False
        ct = content_type.lower().split(';')[0].strip()
        return ct in PDF_CONTENT_TYPES

    @staticmethod
    def _is_pdf_url(url: str) -> bool:
        """Check if URL extension indicates a PDF document."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        return any(path.endswith(ext) for ext in PDF_EXTENSIONS)

    def _save_pdf_to_temp(self, content: bytes, size_limit: int = MAX_PDF_SIZE_BYTES) -> tuple[Optional[str], Optional[str]]:
        """Save PDF bytes to a temporary file.

        Args:
            content: Raw PDF bytes.
            size_limit: Maximum size in bytes to accept.

        Returns:
            Tuple of (temp_file_path, error_message).
        """
        if len(content) > size_limit:
            return None, f"PDF too large ({len(content)} bytes, limit {size_limit} bytes)"
        try:
            fd, path = tempfile.mkstemp(suffix='.pdf', prefix='jaato_web_fetch_')
            with os.fdopen(fd, 'wb') as f:
                f.write(content)
            return path, None
        except Exception as e:
            return None, f"Failed to save PDF: {e}"

    def _fetch_url(
        self,
        url: str,
        custom_headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
        use_proxy: bool = True,
    ) -> tuple[str, str, Optional[str], Optional[Dict[str, Any]]]:
        """Fetch URL and return (content, final_url, error, metadata).

        Args:
            url: The URL to fetch.
            custom_headers: Optional dict of custom HTTP headers (e.g., for auth).
            verify_ssl: Whether to verify SSL certificates. Defaults to True.
                Set to False only for explicitly user-trusted hosts with
                certificate issues (e.g., weak key, self-signed).
            use_proxy: Whether to use the configured proxy. Defaults to True.
                Set to False for hosts that should connect directly,
                bypassing any configured HTTP/HTTPS proxy.

        Returns:
            Tuple of (content, final_url, error_message, metadata)
            - For text content: (html_content, final_url, None, {"response_content_type": "...", "response_headers": {...}})
            - For PDF content: ("", final_url, None, {"is_pdf": True, "pdf_path": "/tmp/...", ...})
            - For binary content: ("", final_url, None, {"is_binary": True, ...})
            - On error: ("", url, error_message, None)
        """
        # Build headers with User-Agent and any custom headers
        headers = {'User-Agent': self._user_agent}
        if custom_headers:
            headers.update(custom_headers)

        try:
            import httpx
        except ImportError:
            try:
                import requests
                from shared.http import get_requests_kwargs

                # Get proxy configuration
                proxy_kwargs = get_requests_kwargs(url) if use_proxy else {"proxies": {}}

                # Merge custom headers with proxy headers
                request_headers = headers.copy()
                if 'headers' in proxy_kwargs:
                    request_headers.update(proxy_kwargs.pop('headers'))

                # Fallback to requests if httpx not available
                response = requests.get(
                    url,
                    timeout=self._timeout,
                    headers=request_headers,
                    allow_redirects=self._follow_redirects,
                    stream=True,  # Don't download body yet
                    verify=verify_ssl,
                    **proxy_kwargs,
                )
                response.raise_for_status()

                content_type = response.headers.get('Content-Type', '')
                content_length = response.headers.get('Content-Length')
                final_url = response.url

                # Check for PDF content — download and save to temp file
                if self._is_pdf_content_type(content_type) or self._is_pdf_url(str(final_url)):
                    pdf_bytes = response.content  # Download full body
                    pdf_path, pdf_err = self._save_pdf_to_temp(pdf_bytes)
                    if pdf_err:
                        response.close()
                        return "", str(final_url), pdf_err, None
                    self._trace(f"PDF saved to {pdf_path} ({len(pdf_bytes)} bytes)")
                    return "", str(final_url), None, {
                        'is_pdf': True,
                        'pdf_path': pdf_path,
                        'content_type': content_type,
                        'size_bytes': len(pdf_bytes),
                    }

                # Check for binary content
                if self._is_binary_content_type(content_type) or self._is_binary_url(str(final_url)):
                    metadata = {
                        'is_binary': True,
                        'content_type': content_type,
                        'size_bytes': int(content_length) if content_length else None,
                    }
                    response.close()
                    return "", str(final_url), None, metadata

                # Read text content - include content_type and headers in metadata
                response_headers = dict(response.headers)
                return response.text, str(final_url), None, {
                    'response_content_type': content_type,
                    'response_headers': response_headers
                }
            except ImportError:
                return "", url, "Neither httpx nor requests package installed. Install with: pip install httpx", None
            except Exception as e:
                return "", url, f"Request failed: {str(e)}", None

        from shared.http import get_httpx_kwargs

        # Get proxy configuration for httpx
        proxy_kwargs = get_httpx_kwargs(url) if use_proxy else {"proxy": None}

        # Merge custom headers with proxy headers
        request_headers = headers.copy()
        if 'headers' in proxy_kwargs:
            request_headers.update(proxy_kwargs.pop('headers'))

        try:
            with httpx.Client(
                timeout=self._timeout,
                follow_redirects=self._follow_redirects,
                headers=request_headers,
                verify=verify_ssl,
                **proxy_kwargs,
            ) as client:
                # First, do a HEAD request to check content type (if supported)
                # Fall back to GET with stream if HEAD fails
                head_content_type = None
                head_content_length = None
                head_final_url = None
                try:
                    head_response = client.head(url)
                    head_content_type = head_response.headers.get('Content-Type', '')
                    head_content_length = head_response.headers.get('Content-Length')
                    head_final_url = str(head_response.url)

                    # For non-PDF binary: early return without downloading body
                    if self._is_binary_content_type(head_content_type) or (
                        self._is_binary_url(head_final_url)
                        and not self._is_pdf_content_type(head_content_type)
                        and not self._is_pdf_url(head_final_url)
                    ):
                        metadata = {
                            'is_binary': True,
                            'content_type': head_content_type,
                            'size_bytes': int(head_content_length) if head_content_length else None,
                        }
                        return "", head_final_url, None, metadata
                except httpx.HTTPStatusError:
                    # HEAD not supported, continue with GET
                    pass

                # Fetch the content
                response = client.get(url)
                response.raise_for_status()

                content_type = response.headers.get('Content-Type', '')
                content_length = response.headers.get('Content-Length')
                final_url = str(response.url)

                # Check for PDF content — save to temp file for conversion
                if self._is_pdf_content_type(content_type) or self._is_pdf_url(final_url):
                    pdf_bytes = response.content
                    pdf_path, pdf_err = self._save_pdf_to_temp(pdf_bytes)
                    if pdf_err:
                        return "", final_url, pdf_err, None
                    self._trace(f"PDF saved to {pdf_path} ({len(pdf_bytes)} bytes)")
                    return "", final_url, None, {
                        'is_pdf': True,
                        'pdf_path': pdf_path,
                        'content_type': content_type,
                        'size_bytes': len(pdf_bytes),
                    }

                # Check for binary content
                if self._is_binary_content_type(content_type) or self._is_binary_url(final_url):
                    metadata = {
                        'is_binary': True,
                        'content_type': content_type,
                        'size_bytes': int(content_length) if content_length else len(response.content),
                    }
                    return "", final_url, None, metadata

                # Return text content with content_type and headers in metadata
                response_headers = dict(response.headers)
                return response.text, final_url, None, {
                    'response_content_type': content_type,
                    'response_headers': response_headers
                }
        except httpx.TimeoutException:
            return "", url, f"Request timed out after {self._timeout}s", None
        except httpx.HTTPStatusError as e:
            return "", url, f"HTTP {e.response.status_code}: {e.response.reason_phrase}", None
        except Exception as e:
            return "", url, f"Request failed: {str(e)}", None

    def _pdf_to_markdown(self, pdf_path: str) -> tuple[Optional[str], Optional[str]]:
        """Convert a PDF file to markdown using pymupdf4llm.

        Args:
            pdf_path: Path to the PDF file on disk.

        Returns:
            Tuple of (markdown_text, error_message). On success error is None;
            on failure markdown_text is None.
        """
        try:
            import pymupdf4llm
        except ImportError:
            return None, (
                "pymupdf4llm is not installed. "
                "Install with: pip install pymupdf4llm"
            )

        try:
            self._trace(f"pdf_to_markdown: converting {pdf_path}")
            md_text = pymupdf4llm.to_markdown(pdf_path)
            self._trace(f"pdf_to_markdown: converted {len(md_text)} chars")
            return md_text, None
        except Exception as e:
            return None, f"PDF conversion failed: {e}"

    def _html_to_markdown(self, html: str, base_url: str) -> str:
        """Convert HTML to clean markdown.

        Uses a tiered approach:
        1. trafilatura for article extraction (best for news/blog content)
        2. html2text for general HTML conversion
        3. BeautifulSoup text extraction as fallback
        4. Basic regex stripping as last resort
        """
        trafilatura_result = None
        html2text_result = None

        # Try trafilatura first (best for article extraction)
        try:
            import trafilatura
            # Extract with trafilatura - it handles boilerplate removal
            trafilatura_result = trafilatura.extract(
                html,
                include_links=True,
                include_images=True,
                include_tables=True,
                output_format='markdown',
                favor_recall=True,  # Get more content rather than less
            )
            if trafilatura_result and len(trafilatura_result.strip()) > 100:
                return trafilatura_result
        except ImportError:
            pass
        except Exception as e:
            self._trace(f"trafilatura failed: {e}")

        # Try html2text for general conversion
        try:
            import html2text
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.ignore_tables = False
            h.body_width = 0  # Don't wrap lines
            h.skip_internal_links = True
            h.inline_links = True
            h.protect_links = True
            h.baseurl = base_url
            html2text_result = h.handle(html)
            if html2text_result and len(html2text_result.strip()) > 50:
                return html2text_result
        except ImportError:
            pass
        except Exception as e:
            self._trace(f"html2text failed: {e}")

        # BeautifulSoup text extraction fallback (handles JS-heavy sites better)
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # Remove script, style, nav, header, footer elements
            for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript']):
                tag.decompose()

            # Try to find main content areas
            main_content = (
                soup.find('main') or
                soup.find('article') or
                soup.find(id='content') or
                soup.find(id='main-content') or
                soup.find(class_='content') or
                soup.find(id='mw-content-text') or  # Wikipedia
                soup.find(id='bodyContent') or      # Wikipedia alternate
                soup.body or
                soup
            )

            # Extract text with some structure preservation
            text_parts = []
            for element in main_content.descendants:
                if element.name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                    level = int(element.name[1])
                    text = element.get_text(strip=True)
                    if text:
                        text_parts.append(f"\n{'#' * level} {text}\n")
                elif element.name == 'p':
                    text = element.get_text(strip=True)
                    if text:
                        text_parts.append(f"{text}\n")
                elif element.name == 'li':
                    text = element.get_text(strip=True)
                    if text:
                        text_parts.append(f"- {text}\n")
                elif element.name == 'br':
                    text_parts.append("\n")

            bs_result = ''.join(text_parts).strip()
            if bs_result and len(bs_result) > 50:
                return bs_result

            # Fallback: just get all text
            all_text = main_content.get_text(separator='\n', strip=True)
            if all_text and len(all_text) > 50:
                return all_text
        except ImportError:
            pass
        except Exception as e:
            self._trace(f"beautifulsoup fallback failed: {e}")

        # Return best available result even if short
        if html2text_result and html2text_result.strip():
            return html2text_result
        if trafilatura_result and trafilatura_result.strip():
            return trafilatura_result

        # Last resort: basic regex stripping
        return self._basic_html_strip(html)

    def _basic_html_strip(self, html: str) -> str:
        """Basic HTML to text conversion (fallback)."""
        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML comments
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        # Remove tags but keep content
        html = re.sub(r'<[^>]+>', ' ', html)
        # Decode common entities
        html = html.replace('&nbsp;', ' ')
        html = html.replace('&amp;', '&')
        html = html.replace('&lt;', '<')
        html = html.replace('&gt;', '>')
        html = html.replace('&quot;', '"')
        # Collapse whitespace
        html = re.sub(r'\s+', ' ', html)
        return html.strip()

    def _extract_with_selector(self, html: str, selector: str, base_url: str) -> tuple[str, Optional[str]]:
        """Extract content matching CSS selector.

        Returns:
            Tuple of (extracted_html, error_message)
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return html, "beautifulsoup4 not installed for selector support"

        try:
            soup = BeautifulSoup(html, 'html.parser')
            elements = soup.select(selector)
            if not elements:
                return "", f"No elements found matching selector: {selector}"

            # Combine all matching elements
            combined = '\n'.join(str(el) for el in elements)
            return combined, None
        except Exception as e:
            return html, f"Selector error: {str(e)}"

    def _extract_structured(
        self,
        html: str,
        base_url: str,
        components: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Extract structured data from HTML.

        Args:
            html: The HTML content
            base_url: Base URL for resolving relative links
            components: List of components to extract, or None for all

        Returns:
            Dict with extracted components
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {'error': 'beautifulsoup4 not installed for structured extraction'}

        soup = BeautifulSoup(html, 'html.parser')
        result: Dict[str, Any] = {}

        # Default: extract common components
        if components is None:
            components = ['metadata', 'headings', 'links', 'images']

        if 'metadata' in components:
            result['metadata'] = self._extract_metadata(soup)

        if 'headings' in components:
            result['headings'] = self._extract_headings(soup)

        if 'links' in components:
            result['links'] = self._extract_links(soup, base_url)

        if 'images' in components:
            result['images'] = self._extract_images(soup, base_url)

        if 'tables' in components:
            result['tables'] = self._extract_tables(soup)

        if 'forms' in components:
            result['forms'] = self._extract_forms(soup, base_url)

        return result

    def _extract_metadata(self, soup) -> Dict[str, Any]:
        """Extract page metadata."""
        metadata = {}

        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)

        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')
            if name and content:
                # Common useful meta tags
                if name in ['description', 'og:description', 'twitter:description']:
                    metadata['description'] = content
                elif name in ['og:title', 'twitter:title']:
                    metadata.setdefault('title', content)
                elif name in ['author', 'og:author']:
                    metadata['author'] = content
                elif name in ['keywords']:
                    metadata['keywords'] = content
                elif name in ['og:image', 'twitter:image']:
                    metadata['image'] = content
                elif name in ['og:type']:
                    metadata['type'] = content

        # Canonical URL
        canonical = soup.find('link', rel='canonical')
        if canonical and canonical.get('href'):
            metadata['canonical_url'] = canonical['href']

        return metadata

    def _extract_headings(self, soup) -> List[Dict[str, str]]:
        """Extract headings hierarchy."""
        headings = []
        for level in range(1, 7):
            for h in soup.find_all(f'h{level}'):
                text = h.get_text(strip=True)
                if text:
                    headings.append({
                        'level': level,
                        'text': text[:200]  # Truncate long headings
                    })
        return headings

    def _extract_links(self, soup, base_url: str) -> List[Dict[str, str]]:
        """Extract links from page."""
        links = []
        seen_urls = set()

        for a in soup.find_all('a', href=True):
            href = a['href']
            # Skip anchors and javascript
            if href.startswith('#') or href.startswith('javascript:'):
                continue

            # Resolve relative URLs
            full_url = urljoin(base_url, href)

            # Deduplicate
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            text = a.get_text(strip=True)
            link_data = {'url': full_url}
            if text:
                link_data['text'] = text[:100]  # Truncate long link text

            links.append(link_data)

        return links[:100]  # Limit to 100 links

    def _extract_images(self, soup, base_url: str) -> List[Dict[str, str]]:
        """Extract images from page."""
        images = []
        seen_urls = set()

        for img in soup.find_all('img', src=True):
            src = img['src']
            # Skip data URIs and tracking pixels
            if src.startswith('data:') or '1x1' in src:
                continue

            full_url = urljoin(base_url, src)

            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            img_data = {'src': full_url}
            if img.get('alt'):
                img_data['alt'] = img['alt'][:200]
            if img.get('title'):
                img_data['title'] = img['title'][:200]

            images.append(img_data)

        return images[:50]  # Limit to 50 images

    def _extract_tables(self, soup) -> List[Dict[str, Any]]:
        """Extract tables from page."""
        tables = []

        for table in soup.find_all('table')[:10]:  # Limit to 10 tables
            table_data = {'rows': []}

            # Extract headers
            headers = []
            for th in table.find_all('th'):
                headers.append(th.get_text(strip=True))
            if headers:
                table_data['headers'] = headers

            # Extract rows
            for tr in table.find_all('tr')[:50]:  # Limit rows
                cells = []
                for td in tr.find_all(['td', 'th']):
                    cells.append(td.get_text(strip=True)[:200])
                if cells and cells != headers:
                    table_data['rows'].append(cells)

            if table_data['rows'] or headers:
                tables.append(table_data)

        return tables

    def _extract_forms(self, soup, base_url: str) -> List[Dict[str, Any]]:
        """Extract forms from page."""
        forms = []

        for form in soup.find_all('form')[:5]:  # Limit to 5 forms
            form_data = {}

            if form.get('action'):
                form_data['action'] = urljoin(base_url, form['action'])
            if form.get('method'):
                form_data['method'] = form['method'].upper()

            # Extract fields
            fields = []
            for inp in form.find_all(['input', 'textarea', 'select']):
                field = {}
                if inp.name == 'input':
                    field['type'] = inp.get('type', 'text')
                elif inp.name == 'textarea':
                    field['type'] = 'textarea'
                elif inp.name == 'select':
                    field['type'] = 'select'
                    # Get options
                    options = [opt.get_text(strip=True) for opt in inp.find_all('option')[:10]]
                    if options:
                        field['options'] = options

                if inp.get('name'):
                    field['name'] = inp['name']
                if inp.get('placeholder'):
                    field['placeholder'] = inp['placeholder']
                if inp.get('required'):
                    field['required'] = True

                if field.get('name') or field.get('type') not in ['hidden', 'submit']:
                    fields.append(field)

            if fields:
                form_data['fields'] = fields

            if form_data:
                forms.append(form_data)

        return forms

    def _detect_content_type(self, response_content_type: str, content: str) -> str:
        """Detect the actual content type from response header and content.

        Returns one of: 'html', 'json', 'xml', 'text'
        """
        # Normalize the content type header
        ct = response_content_type.lower().split(';')[0].strip() if response_content_type else ''

        # Check by content-type header first
        if 'json' in ct or ct == 'application/json':
            return 'json'
        if 'xml' in ct or ct in ('application/xml', 'text/xml'):
            return 'xml'
        if ct == 'text/plain':
            return 'text'
        if 'html' in ct:
            return 'html'

        # Fall back to content inspection
        content_stripped = content.strip() if content else ''
        if content_stripped:
            # Check for JSON (starts with { or [)
            if (content_stripped.startswith('{') and content_stripped.endswith('}')) or \
               (content_stripped.startswith('[') and content_stripped.endswith(']')):
                try:
                    import json
                    json.loads(content_stripped)
                    return 'json'
                except (json.JSONDecodeError, ValueError):
                    pass

            # Check for XML (starts with < and has xml declaration or root element)
            if content_stripped.startswith('<?xml') or \
               (content_stripped.startswith('<') and not content_stripped.startswith('<!DOCTYPE html') and
                not content_stripped.startswith('<html')):
                return 'xml'

            # Check for HTML
            if '<!DOCTYPE html' in content_stripped[:100].lower() or \
               '<html' in content_stripped[:100].lower():
                return 'html'

        return 'html'  # Default to html

    @staticmethod
    def _is_ssl_error(error_msg: str) -> bool:
        """Check whether an error message indicates an SSL certificate failure."""
        ssl_indicators = (
            "CERTIFICATE_VERIFY_FAILED",
            "certificate verify failed",
            "SSL: CERTIFICATE_VERIFY_FAILED",
            "[SSL]",
            "certificate key too weak",
            "self-signed certificate",
            "unable to get local issuer certificate",
        )
        return any(indicator in error_msg for indicator in ssl_indicators)

    @staticmethod
    def _is_proxy_error(error_msg: str) -> bool:
        """Check whether an error message indicates an HTTP proxy failure."""
        proxy_indicators = (
            "ProxyError",
            "407 Proxy Authentication Required",
            "Proxy Authentication Required",
            "Tunnel connection failed",
            "Cannot connect to proxy",
            "proxy",
        )
        error_lower = error_msg.lower()
        return any(indicator.lower() in error_lower for indicator in proxy_indicators)

    def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web fetch.

        Args:
            args: Dict containing:
                - url: The URL to fetch (required)
                - mode: Output format ('markdown', 'structured', 'raw')
                - selector: CSS selector for targeted extraction
                - extract: List of components to extract (for structured mode)
                - include_links: Whether to include links list
                - headers: Custom HTTP headers (e.g., for authentication)
                - no_cache: Bypass cache and fetch fresh content
                - include_headers: Include response headers in result
                - insecure: Skip SSL certificate verification (default: false)
                - no_proxy: Bypass the configured HTTP proxy (default: false)

        Returns:
            Dict containing fetched content or error.
        """
        url = args.get('url', '').strip()
        mode = args.get('mode', 'markdown')
        selector = args.get('selector')
        extract_components = args.get('extract')
        include_links = args.get('include_links', False)
        custom_headers = args.get('headers')
        no_cache = args.get('no_cache', False)
        include_headers = args.get('include_headers', False)
        insecure = bool(args.get('insecure', False))
        no_proxy = bool(args.get('no_proxy', False))

        self._trace(
            f"web_fetch: url={url!r}, mode={mode}, selector={selector}, "
            f"no_cache={no_cache}, insecure={insecure}, no_proxy={no_proxy}"
        )

        # Validate URL
        if not url:
            return {'error': 'web_fetch: url must be provided'}

        # Basic URL validation
        parsed = urlparse(url)
        if not parsed.scheme:
            url = 'https://' + url
            parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return {'error': f'web_fetch: unsupported URL scheme: {parsed.scheme}'}
        if not parsed.netloc:
            return {'error': 'web_fetch: invalid URL'}

        # Check cache (skip if custom headers are provided, or no_cache is True)
        cache_key = f"{url}|{selector or ''}"
        cached_html = None
        if not custom_headers and not no_cache:
            cached_html = self._get_cached(cache_key)

        response_content_type = ''
        response_headers = {}
        if cached_html:
            html = cached_html
            final_url = url
            fetch_metadata = None
        else:
            # Fetch the URL
            verify_ssl = not insecure
            html, final_url, error, fetch_metadata = self._fetch_url(
                url, custom_headers, verify_ssl=verify_ssl, use_proxy=not no_proxy,
            )
            if error:
                result: Dict[str, Any] = {'error': error, 'url': url}
                if self._is_ssl_error(error):
                    result["ssl_error"] = True
                    result["hint"] = (
                        "The SSL certificate for this URL could not be verified. "
                        "Ask the user whether they trust this host. If they confirm, "
                        "retry with insecure=true to skip SSL verification."
                    )
                elif self._is_proxy_error(error):
                    result["proxy_error"] = True
                    result["hint"] = (
                        "The request failed due to a proxy issue. Ask the user "
                        "whether this host should be reached directly (bypassing "
                        "the proxy). If they confirm, retry with no_proxy=true."
                    )
                return result

            # Handle PDF content — convert to markdown
            if fetch_metadata and fetch_metadata.get('is_pdf'):
                pdf_path = fetch_metadata.get('pdf_path')
                try:
                    md_text, conv_error = self._pdf_to_markdown(pdf_path)
                    if conv_error:
                        # Conversion failed — fall back to binary metadata
                        self._trace(f"PDF conversion failed: {conv_error}")
                        return {
                            'url': final_url,
                            'is_binary': True,
                            'content_type': fetch_metadata.get('content_type', 'application/pdf'),
                            'size_bytes': fetch_metadata.get('size_bytes'),
                            'message': f'PDF detected but conversion failed: {conv_error}',
                            'hint': 'Install pymupdf4llm (pip install pymupdf4llm) to '
                                    'enable PDF-to-markdown conversion.',
                        }

                    # Truncate if needed
                    result: Dict[str, Any] = {'url': final_url}
                    if final_url != url:
                        result['original_url'] = url
                        result['redirected'] = True
                    if len(md_text) > self._max_length:
                        md_text = md_text[:self._max_length]
                        result['truncated'] = True
                    result['content'] = md_text
                    result['content_type'] = 'markdown'
                    result['source_type'] = 'pdf'
                    result['size_bytes'] = fetch_metadata.get('size_bytes')
                    return result
                finally:
                    # Always clean up the temp file
                    if pdf_path:
                        try:
                            os.unlink(pdf_path)
                        except OSError:
                            pass

            # Handle binary content - return metadata instead of garbled text
            if fetch_metadata and fetch_metadata.get('is_binary'):
                return {
                    'url': final_url,
                    'is_binary': True,
                    'content_type': fetch_metadata.get('content_type', 'unknown'),
                    'size_bytes': fetch_metadata.get('size_bytes'),
                    'message': 'Binary content detected. Cannot parse as text.',
                    'hint': 'This URL points to a binary file (image, etc.). '
                           'Use a download tool or check the content_type for more info.'
                }

            # Extract response content type and headers for later use
            if fetch_metadata:
                response_content_type = fetch_metadata.get('response_content_type', '')
                response_headers = fetch_metadata.get('response_headers', {})

            # Cache the result (only for non-authenticated requests and when caching is enabled)
            if not custom_headers and not no_cache:
                self._set_cached(cache_key, html)

        # Determine the actual content type from response
        actual_type = self._detect_content_type(response_content_type, html)

        # Apply CSS selector if provided
        if selector:
            html, selector_error = self._extract_with_selector(html, selector, final_url)
            if selector_error:
                return {'error': selector_error, 'url': final_url}
            if not html:
                return {
                    'error': f'No content found for selector: {selector}',
                    'url': final_url
                }

        # Build response based on mode
        result: Dict[str, Any] = {'url': final_url}

        if final_url != url:
            result['original_url'] = url
            result['redirected'] = True

        if mode == 'raw':
            # Return raw content (truncated if too long)
            content = html[:self._max_length]
            if len(html) > self._max_length:
                result['truncated'] = True
                result['original_length'] = len(html)
            result['content'] = content
            # Use detected content type instead of always 'html'
            result['content_type'] = actual_type

        elif mode == 'structured':
            # Handle JSON content specially - parse and return as structured data
            if actual_type == 'json':
                try:
                    import json
                    parsed_json = json.loads(html)
                    result['data'] = parsed_json
                    result['content_type'] = 'json'
                except (json.JSONDecodeError, ValueError) as e:
                    result['error'] = f'Failed to parse JSON: {e}'
                    result['content'] = html[:self._max_length]
                    result['content_type'] = 'json'
            else:
                # Extract structured components from HTML
                structured = self._extract_structured(html, final_url, extract_components)
                result.update(structured)
                result['content_type'] = 'structured'

        else:  # markdown (default)
            # Handle JSON content specially - format it nicely
            if actual_type == 'json':
                try:
                    import json
                    parsed_json = json.loads(html)
                    # Pretty-print JSON as content
                    result['content'] = json.dumps(parsed_json, indent=2)
                    result['content_type'] = 'json'
                except (json.JSONDecodeError, ValueError):
                    result['content'] = html[:self._max_length]
                    result['content_type'] = 'json'
            elif actual_type == 'xml':
                # Return XML as-is (it's already readable)
                content = html[:self._max_length]
                if len(html) > self._max_length:
                    result['truncated'] = True
                result['content'] = content
                result['content_type'] = 'xml'
            elif actual_type == 'text':
                # Plain text - return as-is
                content = html[:self._max_length]
                if len(html) > self._max_length:
                    result['truncated'] = True
                result['content'] = content
                result['content_type'] = 'text'
            else:
                # Convert HTML to markdown
                markdown = self._html_to_markdown(html, final_url)

                # Truncate if needed
                if len(markdown) > self._max_length:
                    markdown = markdown[:self._max_length]
                    result['truncated'] = True

                result['content'] = markdown
                result['content_type'] = 'markdown'

            # Optionally include links (only for HTML content)
            if include_links and actual_type == 'html':
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    result['links'] = self._extract_links(soup, final_url)
                except ImportError:
                    result['links_error'] = 'beautifulsoup4 not installed'

        # Optionally include response headers
        if include_headers and response_headers:
            # Filter to useful headers (exclude internal/sensitive ones)
            useful_headers = {
                k: v for k, v in response_headers.items()
                if k.lower() in (
                    'content-type', 'content-length', 'last-modified', 'etag',
                    'cache-control', 'expires', 'date', 'server', 'x-powered-by',
                    'content-encoding', 'content-language', 'vary', 'age',
                    'x-cache', 'x-cache-hits', 'cf-cache-status', 'x-request-id'
                )
            }
            result['response_headers'] = useful_headers

        return result


def create_plugin() -> WebFetchPlugin:
    """Factory function to create the web fetch plugin instance."""
    return WebFetchPlugin()
