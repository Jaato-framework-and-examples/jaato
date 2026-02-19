"""Tests for the prompt validation module."""

import pytest

from ..validation import (
    PromptValidator,
    ValidationResult,
    format_validation_error,
)


class TestPromptValidator:
    """Tests for PromptValidator class."""

    def test_accepts_valid_skill_with_frontmatter(self):
        """Valid skill with proper frontmatter should pass validation."""
        validator = PromptValidator()
        content = """---
name: code-review
description: Review code for issues
tags: [review, quality]
---

Review the following code for bugs and security issues.
"""
        result = validator.validate(content)

        assert result.valid is True
        assert result.errors == []
        assert result.frontmatter["name"] == "code-review"
        assert result.frontmatter["description"] == "Review code for issues"
        assert "Review the following" in result.body

    def test_accepts_plain_markdown_with_warning(self):
        """Plain markdown without frontmatter should pass with warning."""
        validator = PromptValidator()
        content = "# Simple Prompt\n\nJust do the thing."

        result = validator.validate(content)

        assert result.valid is True
        assert result.errors == []
        assert len(result.warnings) > 0
        assert any("frontmatter" in w.lower() for w in result.warnings)

    def test_rejects_html_content_doctype(self):
        """HTML content with DOCTYPE should be rejected."""
        validator = PromptValidator()
        content = """<!DOCTYPE html>
<html>
<head><title>Not a prompt</title></head>
<body>This is a web page</body>
</html>"""

        result = validator.validate(content)

        assert result.valid is False
        assert len(result.errors) > 0
        assert any("HTML" in e for e in result.errors)
        assert any("<!DOCTYPE" in e for e in result.errors)

    def test_rejects_html_content_html_tag(self):
        """HTML content with <html> tag should be rejected."""
        validator = PromptValidator()
        content = "<html><body>Hello</body></html>"

        result = validator.validate(content)

        assert result.valid is False
        assert any("HTML" in e for e in result.errors)

    def test_rejects_html_content_head_tag(self):
        """HTML content with <head> tag should be rejected."""
        validator = PromptValidator()
        content = "<head><meta charset='utf-8'></head>"

        result = validator.validate(content)

        assert result.valid is False
        assert any("HTML" in e for e in result.errors)

    def test_rejects_html_content_case_insensitive(self):
        """HTML detection should be case-insensitive."""
        validator = PromptValidator()
        content = "<!doctype HTML>\n<HTML><BODY>Test</BODY></HTML>"

        result = validator.validate(content)

        assert result.valid is False
        assert any("HTML" in e for e in result.errors)

    def test_accepts_markdown_with_html_snippets(self):
        """Markdown with inline HTML snippets (not at start) should pass."""
        validator = PromptValidator()
        content = """---
name: test
description: Test prompt
---

Here's some code:
<div class="example">Example</div>
"""
        result = validator.validate(content)

        # Should pass because HTML markers not at start
        assert result.valid is True

    def test_size_limit_enforced(self):
        """Content exceeding size limit should be rejected."""
        validator = PromptValidator()
        # Create content larger than 100KB
        content = "x" * 150_000

        result = validator.validate(content)

        assert result.valid is False
        assert any("size" in e.lower() for e in result.errors)
        assert any("100,000" in e for e in result.errors)

    def test_line_count_warning(self):
        """Content with many lines should generate warning."""
        validator = PromptValidator()
        # Create content with > 500 lines
        content = "\n".join([f"Line {i}" for i in range(600)])

        result = validator.validate(content)

        assert result.valid is True  # Warning, not error
        assert len(result.warnings) > 0
        assert any("lines" in w.lower() for w in result.warnings)

    def test_warns_missing_frontmatter(self):
        """Missing frontmatter should generate warning."""
        validator = PromptValidator()
        content = "Just plain text without any frontmatter."

        result = validator.validate(content)

        assert result.valid is True
        assert any("frontmatter" in w.lower() for w in result.warnings)

    def test_warns_missing_description(self):
        """Frontmatter without description should generate warning."""
        validator = PromptValidator()
        content = """---
name: test-prompt
---

Do something.
"""
        result = validator.validate(content)

        assert result.valid is True
        assert any("description" in w.lower() for w in result.warnings)

    def test_warns_missing_name(self):
        """Frontmatter without name should generate warning."""
        validator = PromptValidator()
        content = """---
description: A test prompt
---

Do something.
"""
        result = validator.validate(content)

        assert result.valid is True
        assert any("name" in w.lower() for w in result.warnings)

    def test_validates_frontmatter_structure(self):
        """Valid frontmatter with params should parse correctly."""
        validator = PromptValidator()
        content = """---
name: parameterized-prompt
description: A prompt with parameters
params:
  file:
    required: true
    description: File to process
  format:
    default: json
---

Process {{file}} as {{format}}.
"""
        result = validator.validate(content)

        assert result.valid is True
        assert result.frontmatter["name"] == "parameterized-prompt"
        assert "params" in result.frontmatter
        assert "file" in result.frontmatter["params"]

    def test_handles_malformed_yaml(self):
        """Malformed YAML should result in empty frontmatter."""
        validator = PromptValidator()
        content = """---
name: broken
  indentation: wrong: here
---

Content.
"""
        result = validator.validate(content)

        # Should pass with warning about no frontmatter
        assert result.valid is True
        assert result.frontmatter == {}

    def test_empty_content(self):
        """Empty content should pass validation."""
        validator = PromptValidator()
        content = ""

        result = validator.validate(content)

        assert result.valid is True
        assert any("frontmatter" in w.lower() for w in result.warnings)


class TestFormatValidationError:
    """Tests for format_validation_error function."""

    def test_formats_errors(self):
        """Should format errors in readable way."""
        result = ValidationResult(
            valid=False,
            errors=["Content appears to be HTML, not markdown (found '<!DOCTYPE')"],
            warnings=[],
            frontmatter={},
            body=""
        )

        formatted = format_validation_error(result)

        assert "Errors:" in formatted
        assert "HTML" in formatted
        assert "<!DOCTYPE" in formatted

    def test_formats_warnings(self):
        """Should format warnings."""
        result = ValidationResult(
            valid=True,
            errors=[],
            warnings=["Missing recommended frontmatter fields: name, description"],
            frontmatter={},
            body=""
        )

        formatted = format_validation_error(result)

        assert "Warnings:" in formatted
        assert "frontmatter" in formatted

    def test_github_url_hint(self):
        """Should provide hint for GitHub blob URLs."""
        result = ValidationResult(
            valid=False,
            errors=["Content appears to be HTML"],
            warnings=[],
            frontmatter={},
            body=""
        )

        formatted = format_validation_error(
            result,
            source_url="https://github.com/user/repo/blob/main/SKILL.md"
        )

        assert "Hint:" in formatted
        assert "raw.githubusercontent.com" in formatted
        # The raw URL should not have /blob/
        raw_url_line = [l for l in formatted.split('\n') if 'raw.githubusercontent.com' in l][0]
        assert "/blob/" not in raw_url_line

    def test_no_github_hint_for_raw_url(self):
        """Should not add hint for already-raw URLs."""
        result = ValidationResult(
            valid=False,
            errors=["Some error"],
            warnings=[],
            frontmatter={},
            body=""
        )

        formatted = format_validation_error(
            result,
            source_url="https://raw.githubusercontent.com/user/repo/main/SKILL.md"
        )

        assert "Hint:" not in formatted

    def test_no_github_hint_for_other_urls(self):
        """Should not add GitHub hint for non-GitHub URLs."""
        result = ValidationResult(
            valid=False,
            errors=["Some error"],
            warnings=[],
            frontmatter={},
            body=""
        )

        formatted = format_validation_error(
            result,
            source_url="https://example.com/prompt.md"
        )

        assert "Hint:" not in formatted
