# Web Fetch Plugin

Fetches and parses web page content, returning it in model-friendly formats.

## Features

- **Multiple output modes**: markdown (default), structured JSON, or raw HTML
- **CSS selector support**: Extract specific page sections
- **Component extraction**: Links, images, tables, forms, headings, metadata
- **Content extraction**: Uses trafilatura for article extraction with html2text fallback
- **Caching**: Brief in-memory cache to avoid redundant requests
- **Redirect handling**: Follows redirects and reports final URL

## Installation

The plugin has optional dependencies for full functionality:

```bash
# Core (at least one HTTP client required)
pip install httpx  # Recommended
# OR
pip install requests  # Fallback

# Content extraction (recommended)
pip install trafilatura  # Best for articles
pip install html2text    # General HTML→markdown

# Structured extraction (required for structured mode and selectors)
pip install beautifulsoup4 lxml
```

## Usage

### Basic Fetch (Markdown Mode)

```python
# Returns clean markdown text
web_fetch(url="https://example.com/article")
```

### CSS Selector

```python
# Extract only the main content area
web_fetch(url="https://example.com", selector=".main-content")

# Extract article body
web_fetch(url="https://example.com", selector="article")
```

> **Note:** CSS selectors work on server-rendered HTML only. The plugin does not
> execute JavaScript, so content loaded dynamically via JS won't be available.
> For JS-heavy single-page applications, consider the initial HTML structure
> or use selectors that match server-rendered content.

### Authenticated Requests

```python
# Bearer token (OAuth, JWT)
web_fetch(
    url="https://api.example.com/data",
    headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiIs..."}
)

# Basic authentication
import base64
credentials = base64.b64encode(b"username:password").decode()
web_fetch(
    url="https://example.com/protected",
    headers={"Authorization": f"Basic {credentials}"}
)

# API key in header
web_fetch(
    url="https://api.example.com/data",
    headers={"X-API-Key": "your-api-key-here"}
)

# API key in custom header
web_fetch(
    url="https://api.example.com/data",
    headers={"Api-Token": "your-token"}
)
```

### Structured Mode

```python
# Get all components
web_fetch(url="https://example.com", mode="structured")

# Get specific components only
web_fetch(
    url="https://example.com",
    mode="structured",
    extract=["links", "images", "metadata"]
)
```

### Include Links List

```python
# Markdown content plus a list of all links
web_fetch(url="https://example.com", include_links=True)
```

### Cache Control

```python
# Bypass cache for fresh content
web_fetch(url="https://example.com/live-data", no_cache=True)

# Cached results are used by default (5-minute TTL)
# Cache is also bypassed when custom headers are provided
```

### Response Headers

```python
# Include response headers in the result
result = web_fetch(url="https://example.com", include_headers=True)
# result['response_headers'] contains: Content-Type, Last-Modified, ETag, Cache-Control, etc.
```

## Output Modes

### `markdown` (default)

Returns clean, readable text converted from HTML. Best for:
- Reading articles and documentation
- Summarizing page content
- General content analysis

Uses trafilatura (if available) for intelligent content extraction that removes boilerplate, navigation, ads, etc. Falls back to html2text for general conversion.

### `structured`

Returns JSON with extracted page components:

```json
{
  "url": "https://example.com",
  "content_type": "structured",
  "metadata": {
    "title": "Page Title",
    "description": "Meta description",
    "author": "Author Name"
  },
  "headings": [
    {"level": 1, "text": "Main Heading"},
    {"level": 2, "text": "Subheading"}
  ],
  "links": [
    {"url": "https://...", "text": "Link text"}
  ],
  "images": [
    {"src": "https://...", "alt": "Alt text"}
  ],
  "tables": [...],
  "forms": [...]
}
```

### `raw`

Returns the raw HTML content. Use sparingly as it's token-heavy. Best for:
- Debugging extraction issues
- When you need exact HTML structure
- Custom parsing requirements

## Automatic Content-Type Detection

The plugin automatically detects the content type from the HTTP response headers
and content inspection, then handles each type appropriately:

| Content Type | Detection | Behavior |
|--------------|-----------|----------|
| **HTML** | `text/html` header or `<html>` tag | Converted to markdown (default mode) |
| **JSON** | `application/json` header or `{}`/`[]` structure | Pretty-printed in markdown mode; parsed in structured mode |
| **XML** | `application/xml`/`text/xml` header or `<?xml` declaration | Returned as-is (readable format) |
| **Plain Text** | `text/plain` header | Returned as-is |

### JSON API Example

```python
# Fetching a JSON API endpoint
result = web_fetch(url="https://api.example.com/posts/1")
# Returns: {"content": "{\n  \"id\": 1,\n  \"title\": \"...\"\n}", "content_type": "json"}

# Structured mode parses JSON into data field
result = web_fetch(url="https://api.example.com/posts/1", mode="structured")
# Returns: {"data": {"id": 1, "title": "..."}, "content_type": "json"}
```

### XML Feed Example

```python
# Fetching an RSS/Atom feed
result = web_fetch(url="https://example.com/feed.xml")
# Returns: {"content": "<?xml version=\"1.0\"?>...", "content_type": "xml"}
```

## Configuration

```python
plugin.initialize({
    'timeout': 30,           # Request timeout in seconds
    'max_length': 100000,    # Max content length to return
    'user_agent': '...',     # Custom User-Agent string
    'follow_redirects': True, # Whether to follow redirects
    'cache_ttl': 300,        # Cache TTL in seconds
})
```

## Extractable Components

| Component | Description |
|-----------|-------------|
| `metadata` | Title, description, author, keywords, canonical URL, Open Graph data |
| `headings` | H1-H6 hierarchy with level and text |
| `links` | All links with URL and anchor text (max 100) |
| `images` | Image sources with alt text (max 50) |
| `tables` | Table data with headers and rows (max 10 tables) |
| `forms` | Form actions, methods, and fields (max 5 forms) |

## Error Handling

The plugin returns structured errors:

```json
{
  "error": "Request timed out after 30s",
  "url": "https://example.com"
}
```

Common errors:
- HTTP errors (404, 500, etc.)
- Timeout errors
- Invalid URL
- Missing dependencies
- Selector not found

## Tips

1. **Use selectors** to reduce noise and focus on relevant content
2. **Prefer markdown mode** for reading - it's token-efficient
3. **Use structured mode** when analyzing page components
4. **Check for redirects** - the response includes `redirected: true` and `original_url`
5. **Handle missing dependencies gracefully** - the plugin reports what's missing
