"""Tests for the web_fetch plugin."""

import pytest
from unittest.mock import patch, MagicMock

from ..plugin import WebFetchPlugin, create_plugin


class TestWebFetchPlugin:
    """Test suite for WebFetchPlugin."""

    def test_create_plugin(self):
        """Test plugin creation."""
        plugin = create_plugin()
        assert plugin is not None
        assert plugin.name == "web_fetch"

    def test_plugin_name(self):
        """Test plugin name property."""
        plugin = WebFetchPlugin()
        assert plugin.name == "web_fetch"

    def test_get_tool_schemas(self):
        """Test tool schema definition."""
        plugin = WebFetchPlugin()
        schemas = plugin.get_tool_schemas()

        assert len(schemas) == 1
        schema = schemas[0]
        assert schema.name == "web_fetch"
        assert schema.category == "web"
        assert "url" in schema.parameters["properties"]
        assert "mode" in schema.parameters["properties"]
        assert "selector" in schema.parameters["properties"]
        assert "extract" in schema.parameters["properties"]

    def test_get_executors(self):
        """Test executor mapping."""
        plugin = WebFetchPlugin()
        executors = plugin.get_executors()

        assert "web_fetch" in executors
        assert callable(executors["web_fetch"])

    def test_get_auto_approved_tools(self):
        """Test auto-approved tools list."""
        plugin = WebFetchPlugin()
        auto_approved = plugin.get_auto_approved_tools()

        assert "web_fetch" in auto_approved

    def test_get_system_instructions(self):
        """Test system instructions."""
        plugin = WebFetchPlugin()
        instructions = plugin.get_system_instructions()

        assert instructions is not None
        assert "web_fetch" in instructions
        assert "markdown" in instructions
        assert "structured" in instructions

    def test_initialize_with_config(self):
        """Test plugin initialization with config."""
        plugin = WebFetchPlugin()
        plugin.initialize({
            "timeout": 60,
            "max_length": 50000,
            "cache_ttl": 600,
        })

        assert plugin._timeout == 60
        assert plugin._max_length == 50000
        assert plugin._cache_ttl == 600

    def test_shutdown_clears_cache(self):
        """Test that shutdown clears the cache."""
        plugin = WebFetchPlugin()
        plugin._cache["test"] = ("content", None)

        plugin.shutdown()

        assert len(plugin._cache) == 0
        assert not plugin._initialized


class TestUrlValidation:
    """Test URL validation."""

    def test_missing_url(self):
        """Test error when URL is missing."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        result = plugin._execute({})
        assert "error" in result
        assert "url must be provided" in result["error"]

    def test_empty_url(self):
        """Test error when URL is empty."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        result = plugin._execute({"url": ""})
        assert "error" in result

    def test_invalid_scheme(self):
        """Test error for unsupported URL scheme."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        result = plugin._execute({"url": "ftp://example.com"})
        assert "error" in result
        assert "unsupported URL scheme" in result["error"]

    def test_auto_adds_https(self):
        """Test that https is added for URLs without scheme."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        # Mock the fetch to avoid actual network call
        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = ("<html></html>", "https://example.com", None, {'response_content_type': 'text/html'})
            result = plugin._execute({"url": "example.com"})

            # Should have called with https
            mock_fetch.assert_called_once()
            assert result.get("url") == "https://example.com"


class TestHtmlToMarkdown:
    """Test HTML to markdown conversion."""

    def test_basic_html_strip(self):
        """Test basic HTML stripping fallback."""
        plugin = WebFetchPlugin()

        html = "<p>Hello <strong>world</strong>!</p>"
        result = plugin._basic_html_strip(html)

        assert "Hello" in result
        assert "world" in result
        assert "<p>" not in result
        assert "<strong>" not in result

    def test_strips_script_tags(self):
        """Test that script tags are removed."""
        plugin = WebFetchPlugin()

        html = "<p>Text</p><script>alert('bad')</script><p>More</p>"
        result = plugin._basic_html_strip(html)

        assert "alert" not in result
        assert "Text" in result
        assert "More" in result

    def test_strips_style_tags(self):
        """Test that style tags are removed."""
        plugin = WebFetchPlugin()

        html = "<style>.foo { color: red; }</style><p>Content</p>"
        result = plugin._basic_html_strip(html)

        assert "color" not in result
        assert "Content" in result


class TestStructuredExtraction:
    """Test structured data extraction."""

    @pytest.fixture
    def sample_html(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="A test page">
            <meta name="author" content="Test Author">
        </head>
        <body>
            <h1>Main Heading</h1>
            <h2>Subheading</h2>
            <p>Some content with <a href="/link1">Link 1</a></p>
            <img src="/image.jpg" alt="Test image">
            <table>
                <tr><th>Header 1</th><th>Header 2</th></tr>
                <tr><td>Data 1</td><td>Data 2</td></tr>
            </table>
            <form action="/submit" method="POST">
                <input type="text" name="username" placeholder="Username">
                <input type="submit" value="Submit">
            </form>
        </body>
        </html>
        """

    def test_extract_metadata(self, sample_html):
        """Test metadata extraction."""
        pytest.importorskip("bs4")
        from bs4 import BeautifulSoup

        plugin = WebFetchPlugin()
        soup = BeautifulSoup(sample_html, 'html.parser')
        metadata = plugin._extract_metadata(soup)

        assert metadata["title"] == "Test Page"
        assert metadata["description"] == "A test page"
        assert metadata["author"] == "Test Author"

    def test_extract_headings(self, sample_html):
        """Test headings extraction."""
        pytest.importorskip("bs4")
        from bs4 import BeautifulSoup

        plugin = WebFetchPlugin()
        soup = BeautifulSoup(sample_html, 'html.parser')
        headings = plugin._extract_headings(soup)

        assert len(headings) == 2
        assert headings[0]["level"] == 1
        assert headings[0]["text"] == "Main Heading"
        assert headings[1]["level"] == 2
        assert headings[1]["text"] == "Subheading"

    def test_extract_links(self, sample_html):
        """Test links extraction."""
        pytest.importorskip("bs4")
        from bs4 import BeautifulSoup

        plugin = WebFetchPlugin()
        soup = BeautifulSoup(sample_html, 'html.parser')
        links = plugin._extract_links(soup, "https://example.com")

        assert len(links) >= 1
        link = links[0]
        assert "url" in link
        assert link["url"] == "https://example.com/link1"
        assert link["text"] == "Link 1"

    def test_extract_images(self, sample_html):
        """Test images extraction."""
        pytest.importorskip("bs4")
        from bs4 import BeautifulSoup

        plugin = WebFetchPlugin()
        soup = BeautifulSoup(sample_html, 'html.parser')
        images = plugin._extract_images(soup, "https://example.com")

        assert len(images) == 1
        assert images[0]["src"] == "https://example.com/image.jpg"
        assert images[0]["alt"] == "Test image"

    def test_extract_tables(self, sample_html):
        """Test tables extraction."""
        pytest.importorskip("bs4")
        from bs4 import BeautifulSoup

        plugin = WebFetchPlugin()
        soup = BeautifulSoup(sample_html, 'html.parser')
        tables = plugin._extract_tables(soup)

        assert len(tables) == 1
        assert "headers" in tables[0]
        assert tables[0]["headers"] == ["Header 1", "Header 2"]
        assert len(tables[0]["rows"]) >= 1

    def test_extract_forms(self, sample_html):
        """Test forms extraction."""
        pytest.importorskip("bs4")
        from bs4 import BeautifulSoup

        plugin = WebFetchPlugin()
        soup = BeautifulSoup(sample_html, 'html.parser')
        forms = plugin._extract_forms(soup, "https://example.com")

        assert len(forms) == 1
        assert forms[0]["action"] == "https://example.com/submit"
        assert forms[0]["method"] == "POST"
        assert "fields" in forms[0]


class TestCaching:
    """Test caching behavior."""

    def test_cache_hit(self):
        """Test that cached content is returned."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        # Pre-populate cache
        from datetime import datetime
        plugin._cache["https://example.com|"] = ("<html>cached</html>", datetime.now())

        with patch.object(plugin, '_fetch_url') as mock_fetch:
            with patch.object(plugin, '_html_to_markdown') as mock_convert:
                mock_convert.return_value = "cached content"
                result = plugin._execute({"url": "https://example.com"})

                # Should not have called fetch
                mock_fetch.assert_not_called()

    def test_cache_miss_expired(self):
        """Test that expired cache triggers new fetch."""
        plugin = WebFetchPlugin()
        plugin.initialize({"cache_ttl": 0})  # Immediate expiry

        from datetime import datetime, timedelta
        plugin._cache["https://example.com|"] = (
            "<html>old</html>",
            datetime.now() - timedelta(seconds=10)
        )

        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = ("<html>new</html>", "https://example.com", None, {'response_content_type': 'text/html'})
            with patch.object(plugin, '_html_to_markdown') as mock_convert:
                mock_convert.return_value = "new content"
                result = plugin._execute({"url": "https://example.com"})

                # Should have called fetch due to expired cache
                mock_fetch.assert_called_once()


class TestModes:
    """Test different output modes."""

    def test_markdown_mode_default(self):
        """Test that markdown is the default mode."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = ("<html><body><p>Hello</p></body></html>", "https://example.com", None, {'response_content_type': 'text/html'})
            with patch.object(plugin, '_html_to_markdown') as mock_convert:
                mock_convert.return_value = "Hello"
                result = plugin._execute({"url": "https://example.com"})

                assert result["content_type"] == "markdown"
                mock_convert.assert_called_once()

    def test_raw_mode(self):
        """Test raw HTML mode."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        html = "<html><body><p>Hello</p></body></html>"
        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = (html, "https://example.com", None, {'response_content_type': 'text/html'})
            result = plugin._execute({"url": "https://example.com", "mode": "raw"})

            assert result["content_type"] == "html"
            assert result["content"] == html

    def test_raw_mode_xml_content_type(self):
        """Test raw mode returns xml content type for XML content."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        xml = '<?xml version="1.0"?><root><item>Hello</item></root>'
        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = (xml, "https://example.com/feed.xml", None, {'response_content_type': 'application/xml'})
            result = plugin._execute({"url": "https://example.com/feed.xml", "mode": "raw"})

            assert result["content_type"] == "xml"
            assert result["content"] == xml

    def test_structured_mode(self):
        """Test structured extraction mode."""
        pytest.importorskip("bs4")

        plugin = WebFetchPlugin()
        plugin.initialize()

        html = """
        <html>
        <head><title>Test</title></head>
        <body><h1>Hello</h1><a href="/link">Link</a></body>
        </html>
        """
        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = (html, "https://example.com", None, {'response_content_type': 'text/html'})
            result = plugin._execute({
                "url": "https://example.com",
                "mode": "structured",
                "extract": ["metadata", "headings", "links"]
            })

            assert result["content_type"] == "structured"
            assert "metadata" in result
            assert "headings" in result
            assert "links" in result

    def test_structured_mode_json(self):
        """Test structured mode parses JSON content."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        json_content = '{"id": 1, "title": "Test Post", "body": "Content"}'
        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = (json_content, "https://api.example.com/posts/1", None, {'response_content_type': 'application/json'})
            result = plugin._execute({
                "url": "https://api.example.com/posts/1",
                "mode": "structured"
            })

            assert result["content_type"] == "json"
            assert "data" in result
            assert result["data"]["id"] == 1
            assert result["data"]["title"] == "Test Post"

    def test_markdown_mode_json(self):
        """Test markdown mode pretty-prints JSON content."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        json_content = '{"id":1,"title":"Test"}'
        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = (json_content, "https://api.example.com/data", None, {'response_content_type': 'application/json'})
            result = plugin._execute({"url": "https://api.example.com/data"})

            assert result["content_type"] == "json"
            # Should be pretty-printed with indentation
            assert '"id": 1' in result["content"]
            assert '"title": "Test"' in result["content"]


class TestSelector:
    """Test CSS selector functionality."""

    def test_selector_extraction(self):
        """Test extracting content with CSS selector."""
        pytest.importorskip("bs4")

        plugin = WebFetchPlugin()
        plugin.initialize()

        html = """
        <html>
        <body>
            <nav>Navigation</nav>
            <main class="content"><p>Main content here</p></main>
            <footer>Footer</footer>
        </body>
        </html>
        """
        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = (html, "https://example.com", None, {'response_content_type': 'text/html'})
            with patch.object(plugin, '_html_to_markdown') as mock_convert:
                mock_convert.return_value = "Main content here"
                result = plugin._execute({
                    "url": "https://example.com",
                    "selector": ".content"
                })

                # The selector should have filtered the HTML before conversion
                call_args = mock_convert.call_args[0]
                assert "Main content" in call_args[0]

    def test_selector_not_found(self):
        """Test error when selector doesn't match."""
        pytest.importorskip("bs4")

        plugin = WebFetchPlugin()
        plugin.initialize()

        html = "<html><body><p>Content</p></body></html>"
        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = (html, "https://example.com", None, {'response_content_type': 'text/html'})
            result = plugin._execute({
                "url": "https://example.com",
                "selector": ".nonexistent"
            })

            assert "error" in result
            assert "No elements found" in result["error"]


class TestCacheControl:
    """Test cache control features."""

    def test_no_cache_bypasses_cache(self):
        """Test that no_cache=True bypasses the cache."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        # Pre-populate cache
        from datetime import datetime
        plugin._cache["https://example.com|"] = ("<html>cached</html>", datetime.now())

        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = ("<html>fresh</html>", "https://example.com", None, {'response_content_type': 'text/html', 'response_headers': {}})
            with patch.object(plugin, '_html_to_markdown') as mock_convert:
                mock_convert.return_value = "fresh content"
                result = plugin._execute({"url": "https://example.com", "no_cache": True})

                # Should have called fetch despite cache existing
                mock_fetch.assert_called_once()

    def test_no_cache_prevents_caching(self):
        """Test that no_cache=True prevents storing in cache."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = ("<html>content</html>", "https://example.com", None, {'response_content_type': 'text/html', 'response_headers': {}})
            with patch.object(plugin, '_html_to_markdown') as mock_convert:
                mock_convert.return_value = "content"
                result = plugin._execute({"url": "https://example.com", "no_cache": True})

                # Cache should remain empty
                assert len(plugin._cache) == 0


class TestResponseHeaders:
    """Test response headers feature."""

    def test_include_headers_false_by_default(self):
        """Test that headers are not included by default."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = ("<html>test</html>", "https://example.com", None, {
                'response_content_type': 'text/html',
                'response_headers': {'Content-Type': 'text/html', 'Server': 'nginx'}
            })
            with patch.object(plugin, '_html_to_markdown') as mock_convert:
                mock_convert.return_value = "test"
                result = plugin._execute({"url": "https://example.com"})

                assert 'response_headers' not in result

    def test_include_headers_returns_filtered_headers(self):
        """Test that include_headers=True returns filtered headers."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = ("<html>test</html>", "https://example.com", None, {
                'response_content_type': 'text/html',
                'response_headers': {
                    'Content-Type': 'text/html; charset=utf-8',
                    'Last-Modified': 'Wed, 15 Jan 2025 10:00:00 GMT',
                    'ETag': '"abc123"',
                    'Cache-Control': 'max-age=3600',
                    'Server': 'nginx',
                    'Set-Cookie': 'session=xyz',  # Should be filtered out
                    'X-Internal-Id': '12345',     # Should be filtered out
                }
            })
            with patch.object(plugin, '_html_to_markdown') as mock_convert:
                mock_convert.return_value = "test"
                result = plugin._execute({"url": "https://example.com", "include_headers": True})

                assert 'response_headers' in result
                headers = result['response_headers']
                # These should be included
                assert 'Content-Type' in headers
                assert 'Last-Modified' in headers
                assert 'ETag' in headers
                assert 'Cache-Control' in headers
                assert 'Server' in headers
                # These should be filtered out
                assert 'Set-Cookie' not in headers
                assert 'X-Internal-Id' not in headers


class TestContentTypeDetection:
    """Test content type detection."""

    def test_detect_json_from_header(self):
        """Test JSON detection from content-type header."""
        plugin = WebFetchPlugin()

        assert plugin._detect_content_type('application/json', '{}') == 'json'
        assert plugin._detect_content_type('application/json; charset=utf-8', '{}') == 'json'

    def test_detect_xml_from_header(self):
        """Test XML detection from content-type header."""
        plugin = WebFetchPlugin()

        assert plugin._detect_content_type('application/xml', '<root/>') == 'xml'
        assert plugin._detect_content_type('text/xml', '<root/>') == 'xml'

    def test_detect_html_from_header(self):
        """Test HTML detection from content-type header."""
        plugin = WebFetchPlugin()

        assert plugin._detect_content_type('text/html', '<html></html>') == 'html'
        assert plugin._detect_content_type('text/html; charset=utf-8', '<html></html>') == 'html'

    def test_detect_text_from_header(self):
        """Test plain text detection from content-type header."""
        plugin = WebFetchPlugin()

        assert plugin._detect_content_type('text/plain', 'Hello world') == 'text'

    def test_detect_json_from_content(self):
        """Test JSON detection from content inspection."""
        plugin = WebFetchPlugin()

        # Empty header, valid JSON content
        assert plugin._detect_content_type('', '{"key": "value"}') == 'json'
        assert plugin._detect_content_type('', '[1, 2, 3]') == 'json'

    def test_detect_xml_from_content(self):
        """Test XML detection from content inspection."""
        plugin = WebFetchPlugin()

        assert plugin._detect_content_type('', '<?xml version="1.0"?><root/>') == 'xml'

    def test_detect_html_from_content(self):
        """Test HTML detection from content inspection."""
        plugin = WebFetchPlugin()

        assert plugin._detect_content_type('', '<!DOCTYPE html><html></html>') == 'html'
        assert plugin._detect_content_type('', '<html><body></body></html>') == 'html'

    def test_default_to_html(self):
        """Test that unknown content defaults to HTML."""
        plugin = WebFetchPlugin()

        assert plugin._detect_content_type('', 'Some random text') == 'html'
