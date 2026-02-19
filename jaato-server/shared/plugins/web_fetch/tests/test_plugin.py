"""Tests for the web_fetch plugin."""

import sys
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


class TestPdfDetection:
    """Test PDF-specific content detection."""

    def test_pdf_content_type_detection(self):
        """Test that application/pdf is detected as PDF, not generic binary."""
        plugin = WebFetchPlugin()

        assert plugin._is_pdf_content_type('application/pdf')
        assert plugin._is_pdf_content_type('application/pdf; charset=binary')
        assert not plugin._is_pdf_content_type('text/html')
        assert not plugin._is_pdf_content_type('application/zip')
        assert not plugin._is_pdf_content_type('')

    def test_pdf_url_detection(self):
        """Test that .pdf URLs are detected."""
        plugin = WebFetchPlugin()

        assert plugin._is_pdf_url('https://example.com/doc.pdf')
        assert plugin._is_pdf_url('https://example.com/path/doc.PDF')
        assert not plugin._is_pdf_url('https://example.com/page.html')
        assert not plugin._is_pdf_url('https://example.com/image.png')

    def test_pdf_not_in_binary_sets(self):
        """Test that PDF is excluded from generic binary detection."""
        plugin = WebFetchPlugin()

        # PDF should NOT be detected as generic binary
        assert not plugin._is_binary_content_type('application/pdf')
        assert not plugin._is_binary_url('https://example.com/doc.pdf')

    def test_other_binary_still_detected(self):
        """Test that non-PDF binaries are still detected."""
        plugin = WebFetchPlugin()

        assert plugin._is_binary_content_type('image/png')
        assert plugin._is_binary_content_type('application/zip')
        assert plugin._is_binary_url('https://example.com/image.png')
        assert plugin._is_binary_url('https://example.com/archive.zip')


class TestPdfConversion:
    """Test PDF-to-markdown conversion flow."""

    def test_pdf_url_returns_markdown_content(self):
        """Test that PDF URLs return markdown content when conversion succeeds."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = ("", "https://example.com/doc.pdf", None, {
                'is_pdf': True,
                'pdf_path': '/tmp/test.pdf',
                'content_type': 'application/pdf',
                'size_bytes': 1024,
            })
            with patch.object(plugin, '_pdf_to_markdown') as mock_convert:
                mock_convert.return_value = ("# Document Title\n\nSome content here.", None)
                with patch('os.unlink'):  # Don't actually delete
                    result = plugin._execute({"url": "https://example.com/doc.pdf"})

                assert result['content_type'] == 'markdown'
                assert result['source_type'] == 'pdf'
                assert '# Document Title' in result['content']
                assert result['size_bytes'] == 1024
                mock_convert.assert_called_once_with('/tmp/test.pdf')

    def test_pdf_conversion_failure_returns_binary_metadata(self):
        """Test that failed PDF conversion falls back to binary metadata."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = ("", "https://example.com/doc.pdf", None, {
                'is_pdf': True,
                'pdf_path': '/tmp/test.pdf',
                'content_type': 'application/pdf',
                'size_bytes': 2048,
            })
            with patch.object(plugin, '_pdf_to_markdown') as mock_convert:
                mock_convert.return_value = (None, "pymupdf4llm is not installed. Install with: pip install pymupdf4llm")
                with patch('os.unlink'):
                    result = plugin._execute({"url": "https://example.com/doc.pdf"})

                assert result.get('is_binary') is True
                assert 'pymupdf4llm' in result.get('hint', '')

    def test_pdf_temp_file_cleanup(self):
        """Test that temp PDF files are cleaned up after conversion."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = ("", "https://example.com/doc.pdf", None, {
                'is_pdf': True,
                'pdf_path': '/tmp/test_cleanup.pdf',
                'content_type': 'application/pdf',
                'size_bytes': 512,
            })
            with patch.object(plugin, '_pdf_to_markdown') as mock_convert:
                mock_convert.return_value = ("Some text", None)
                with patch('os.unlink') as mock_unlink:
                    plugin._execute({"url": "https://example.com/doc.pdf"})
                    mock_unlink.assert_called_once_with('/tmp/test_cleanup.pdf')

    def test_pdf_temp_file_cleanup_on_error(self):
        """Test that temp PDF files are cleaned up even when conversion fails."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = ("", "https://example.com/doc.pdf", None, {
                'is_pdf': True,
                'pdf_path': '/tmp/test_cleanup_err.pdf',
                'content_type': 'application/pdf',
                'size_bytes': 512,
            })
            with patch.object(plugin, '_pdf_to_markdown') as mock_convert:
                mock_convert.return_value = (None, "Conversion failed")
                with patch('os.unlink') as mock_unlink:
                    plugin._execute({"url": "https://example.com/doc.pdf"})
                    mock_unlink.assert_called_once_with('/tmp/test_cleanup_err.pdf')

    def test_pdf_truncation(self):
        """Test that large PDF markdown output is truncated."""
        plugin = WebFetchPlugin()
        plugin.initialize({"max_length": 100})

        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = ("", "https://example.com/big.pdf", None, {
                'is_pdf': True,
                'pdf_path': '/tmp/big.pdf',
                'content_type': 'application/pdf',
                'size_bytes': 10000,
            })
            with patch.object(plugin, '_pdf_to_markdown') as mock_convert:
                mock_convert.return_value = ("x" * 200, None)
                with patch('os.unlink'):
                    result = plugin._execute({"url": "https://example.com/big.pdf"})

                assert result.get('truncated') is True
                assert len(result['content']) == 100

    def test_pdf_redirect_tracking(self):
        """Test that redirected PDF URLs track original URL."""
        plugin = WebFetchPlugin()
        plugin.initialize()

        with patch.object(plugin, '_fetch_url') as mock_fetch:
            mock_fetch.return_value = ("", "https://cdn.example.com/doc.pdf", None, {
                'is_pdf': True,
                'pdf_path': '/tmp/redir.pdf',
                'content_type': 'application/pdf',
                'size_bytes': 512,
            })
            with patch.object(plugin, '_pdf_to_markdown') as mock_convert:
                mock_convert.return_value = ("Content", None)
                with patch('os.unlink'):
                    result = plugin._execute({"url": "https://example.com/doc.pdf"})

                assert result['url'] == "https://cdn.example.com/doc.pdf"
                assert result['original_url'] == "https://example.com/doc.pdf"
                assert result['redirected'] is True


class TestPdfToMarkdownMethod:
    """Test the _pdf_to_markdown method directly."""

    def test_returns_error_when_pymupdf4llm_not_installed(self):
        """Test graceful fallback when pymupdf4llm is not installed."""
        plugin = WebFetchPlugin()

        with patch.dict('sys.modules', {'pymupdf4llm': None}):
            md, err = plugin._pdf_to_markdown('/tmp/test.pdf')
            assert md is None
            assert 'pymupdf4llm is not installed' in err

    def test_returns_markdown_on_success(self):
        """Test successful conversion."""
        plugin = WebFetchPlugin()

        mock_pymupdf4llm = MagicMock()
        mock_pymupdf4llm.to_markdown.return_value = "# Title\n\nParagraph text."

        with patch.dict('sys.modules', {'pymupdf4llm': mock_pymupdf4llm}):
            md, err = plugin._pdf_to_markdown('/tmp/test.pdf')
            assert err is None
            assert md == "# Title\n\nParagraph text."
            mock_pymupdf4llm.to_markdown.assert_called_once_with('/tmp/test.pdf')

    def test_returns_error_on_exception(self):
        """Test error handling when conversion raises."""
        plugin = WebFetchPlugin()

        mock_pymupdf4llm = MagicMock()
        mock_pymupdf4llm.to_markdown.side_effect = RuntimeError("corrupt PDF")

        with patch.dict('sys.modules', {'pymupdf4llm': mock_pymupdf4llm}):
            md, err = plugin._pdf_to_markdown('/tmp/corrupt.pdf')
            assert md is None
            assert 'corrupt PDF' in err


class TestBackgroundSupport:
    """Test BackgroundCapableMixin integration."""

    def test_supports_background(self):
        """Test that web_fetch declares background support."""
        plugin = WebFetchPlugin()

        assert plugin.supports_background('web_fetch') is True
        assert plugin.supports_background('other_tool') is False

    def test_auto_background_threshold(self):
        """Test auto-background threshold for web_fetch."""
        plugin = WebFetchPlugin()

        assert plugin.get_auto_background_threshold('web_fetch') == 10.0
        assert plugin.get_auto_background_threshold('other_tool') is None

    def test_custom_threshold_via_config(self):
        """Test that threshold can be configured."""
        plugin = WebFetchPlugin()
        plugin._auto_background_threshold = 5.0

        assert plugin.get_auto_background_threshold('web_fetch') == 5.0


class TestSavePdfToTemp:
    """Test the _save_pdf_to_temp helper."""

    def test_saves_to_temp_file(self):
        """Test that PDF bytes are saved to a temp file."""
        import os
        plugin = WebFetchPlugin()

        pdf_bytes = b'%PDF-1.4 fake content'
        path, err = plugin._save_pdf_to_temp(pdf_bytes)

        assert err is None
        assert path is not None
        assert path.endswith('.pdf')
        assert os.path.exists(path)

        # Read back and verify
        with open(path, 'rb') as f:
            assert f.read() == pdf_bytes

        os.unlink(path)

    def test_rejects_oversized_pdf(self):
        """Test that oversized PDFs are rejected."""
        plugin = WebFetchPlugin()

        large_bytes = b'x' * 100
        path, err = plugin._save_pdf_to_temp(large_bytes, size_limit=50)

        assert path is None
        assert 'too large' in err
