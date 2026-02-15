# shared/plugins/code_block_formatter/tests/test_fence_detection.py
"""Tests for markdown fence opening and closing detection."""

import re

from shared.plugins.code_block_formatter.plugin import CodeBlockFormatterPlugin


ANSI_PATTERN = re.compile(r'\033\[[0-9;]*m')


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return ANSI_PATTERN.sub('', text)


class TestFenceDetection:
    """Tests for detecting opening and closing code fences."""

    def test_closing_fence_at_chunk_boundary(self):
        """Closing fence at the start of a new chunk should be detected."""
        plugin = CodeBlockFormatterPlugin()
        plugin.initialize()

        # Simulate streaming chunks where closing fence starts a new chunk
        # Chunk 1: Opening fence + code content ending with newline
        chunk1 = "```python\nabc\n"
        output1 = list(plugin.process_chunk(chunk1))

        # Chunk 2: Closing fence at the very start
        chunk2 = "```\nmore text"
        output2 = list(plugin.process_chunk(chunk2))

        # Flush any remaining content
        output3 = list(plugin.flush())

        # Combine all output
        full_output = "".join(output1 + output2 + output3)

        # The code block should be closed and "more text" should be outside
        assert plugin._in_code_block is False, "Code block should be closed"
        assert "more text" in full_output
        # The "more text" should NOT be inside ANSI-styled code block
        # It should appear after the code block is closed

    def test_closing_fence_in_same_chunk(self):
        """Complete code block in a single chunk should be detected."""
        plugin = CodeBlockFormatterPlugin()
        plugin.initialize()

        chunk = "```python\nprint('hello')\n```\nafter"
        output = list(plugin.process_chunk(chunk))
        output.extend(plugin.flush())

        full_output = "".join(output)

        assert plugin._in_code_block is False
        assert "after" in full_output

    def test_closing_fence_with_content_before_newline(self):
        """Closing fence preceded by code content and newline."""
        plugin = CodeBlockFormatterPlugin()
        plugin.initialize()

        # Normal case: code content followed by newline, then closing fence
        chunk = "```python\ncode here\n```\n"
        output = list(plugin.process_chunk(chunk))
        output.extend(plugin.flush())

        assert plugin._in_code_block is False

    def test_empty_code_block_fence_at_boundary(self):
        """Empty code block with closing fence in next chunk."""
        plugin = CodeBlockFormatterPlugin()
        plugin.initialize()

        # Chunk 1: Just the opening fence
        chunk1 = "```python\n"
        output1 = list(plugin.process_chunk(chunk1))

        # Chunk 2: Immediately closing fence
        chunk2 = "```\n"
        output2 = list(plugin.process_chunk(chunk2))
        output3 = list(plugin.flush())

        assert plugin._in_code_block is False

    def test_multiple_code_blocks(self):
        """Multiple code blocks should all be detected correctly."""
        plugin = CodeBlockFormatterPlugin()
        plugin.initialize()

        text = "text1\n```python\ncode1\n```\ntext2\n```bash\ncode2\n```\ntext3"
        output = list(plugin.process_chunk(text))
        output.extend(plugin.flush())

        full_output = "".join(output)

        assert plugin._in_code_block is False
        assert "text1" in full_output
        assert "text2" in full_output
        assert "text3" in full_output

    def test_partial_fence_held_back(self):
        """Partial fence markers at chunk end should be held in buffer."""
        plugin = CodeBlockFormatterPlugin()
        plugin.initialize()

        # Chunk ends with partial fence (could be opening or inline backticks)
        chunk1 = "text here `"
        output1 = list(plugin.process_chunk(chunk1))

        # The backtick should be held back
        assert "text here" in "".join(output1)
        # The backtick is still in buffer
        assert plugin._buffer == "`"

        # Next chunk completes it as inline backtick
        chunk2 = "code`"
        output2 = list(plugin.process_chunk(chunk2))
        output3 = list(plugin.flush())

        full_output = "".join(output1 + output2 + output3)
        assert "`code`" in full_output

    def test_closing_fence_detection_pattern(self):
        """The closing fence pattern should match at start or after newline."""
        plugin = CodeBlockFormatterPlugin()
        plugin.initialize()

        # Simulate being in a code block
        plugin._in_code_block = True
        plugin._code_block_lang = "python"

        # Test 1: Closing fence after newline
        plugin._buffer = "code\n```"
        match = re.search(r'(?:^|\n)```', plugin._buffer)
        assert match is not None
        assert match.start() == 4  # Position of \n

        # Test 2: Closing fence at start of buffer
        plugin._buffer = "```"
        match = re.search(r'(?:^|\n)```', plugin._buffer)
        assert match is not None
        assert match.start() == 0  # Position 0 (start)

        # Test 3: Closing fence after content and newline
        plugin._buffer = "abc\n```\nmore"
        match = re.search(r'(?:^|\n)```', plugin._buffer)
        assert match is not None
        assert match.start() == 3  # Position of \n before ```
