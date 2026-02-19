"""Validation module for prompt library content.

Validates fetched content to ensure it's a valid prompt/skill format,
not HTML or other invalid content.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import yaml


@dataclass
class ValidationResult:
    """Result of content validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    body: str = ""


class PromptValidator:
    """Validates prompt/skill content format.

    Checks that content is:
    - Not HTML (common mistake with GitHub blob URLs)
    - Within size limits
    - Has valid YAML frontmatter (if present)
    - Contains recommended metadata fields
    """

    MAX_CONTENT_SIZE = 100_000  # 100KB
    MAX_RECOMMENDED_LINES = 500
    REQUIRED_SKILL_FIELDS = ["name", "description"]
    HTML_MARKERS = ["<!DOCTYPE", "<html", "<head>", "<body>", "<!doctype"]

    def validate(self, content: str, source_hint: str = "") -> ValidationResult:
        """Validate prompt/skill content.

        Args:
            content: The raw content to validate
            source_hint: Optional hint about source (e.g., URL) for error messages

        Returns:
            ValidationResult with errors, warnings, and parsed frontmatter
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check for HTML content
        html_errors = self._check_not_html(content)
        errors.extend(html_errors)

        # If HTML detected, no point checking further
        if html_errors:
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                frontmatter={},
                body=content
            )

        # Check size limits
        size_errors, size_warnings = self._check_size(content)
        errors.extend(size_errors)
        warnings.extend(size_warnings)

        # Parse and check frontmatter
        frontmatter, body = self._parse_frontmatter(content)
        fm_errors, fm_warnings = self._check_frontmatter(frontmatter)
        errors.extend(fm_errors)
        warnings.extend(fm_warnings)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            frontmatter=frontmatter,
            body=body
        )

    def _check_not_html(self, content: str) -> List[str]:
        """Check that content is not HTML.

        Returns list of errors if HTML detected.
        """
        content_start = content.lstrip()[:500].lower()

        for marker in self.HTML_MARKERS:
            if marker.lower() in content_start:
                return [
                    f"Content appears to be HTML, not markdown (found '{marker}')"
                ]

        return []

    def _check_size(self, content: str) -> Tuple[List[str], List[str]]:
        """Check content size limits.

        Returns (errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Hard limit
        if len(content) > self.MAX_CONTENT_SIZE:
            errors.append(
                f"Content exceeds maximum size of {self.MAX_CONTENT_SIZE:,} bytes "
                f"(got {len(content):,} bytes)"
            )

        # Soft limit on lines
        line_count = content.count('\n') + 1
        if line_count > self.MAX_RECOMMENDED_LINES:
            warnings.append(
                f"Content has {line_count} lines (recommended max: {self.MAX_RECOMMENDED_LINES})"
            )

        return errors, warnings

    def _parse_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Parse YAML frontmatter from markdown content.

        Returns (frontmatter_dict, body_content).
        """
        if not content.startswith('---'):
            return {}, content

        # Find the closing ---
        lines = content.split('\n')
        end_idx = None
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == '---':
                end_idx = i
                break

        if end_idx is None:
            return {}, content

        frontmatter_text = '\n'.join(lines[1:end_idx])
        body = '\n'.join(lines[end_idx + 1:]).lstrip()

        try:
            frontmatter = yaml.safe_load(frontmatter_text) or {}
            return frontmatter, body
        except yaml.YAMLError:
            # Malformed YAML - return empty, let _check_frontmatter handle it
            return {}, content

    def _check_frontmatter(
        self, frontmatter: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Check frontmatter for recommended fields.

        Returns (errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []

        # No frontmatter is just a warning, not an error
        if not frontmatter:
            warnings.append(
                "No YAML frontmatter found. Consider adding name and description."
            )
            return errors, warnings

        # Check for recommended fields
        missing_fields = []
        for field_name in self.REQUIRED_SKILL_FIELDS:
            if field_name not in frontmatter:
                missing_fields.append(field_name)

        if missing_fields:
            warnings.append(
                f"Missing recommended frontmatter fields: {', '.join(missing_fields)}"
            )

        return errors, warnings


def format_validation_error(result: ValidationResult, source_url: str = "") -> str:
    """Format a validation error for user display.

    Args:
        result: The validation result with errors
        source_url: Optional URL for providing hints

    Returns:
        Formatted error message string
    """
    lines = ["Content is not a valid prompt/skill", ""]

    if result.errors:
        lines.append("Errors:")
        for error in result.errors:
            lines.append(f"  - {error}")
        lines.append("")

    if result.warnings:
        lines.append("Warnings:")
        for warning in result.warnings:
            lines.append(f"  - {warning}")
        lines.append("")

    # Add hint for GitHub URLs
    if source_url and "github.com" in source_url and "/blob/" in source_url:
        lines.append("Hint: For GitHub files, use the raw URL:")
        raw_url = source_url.replace("github.com", "raw.githubusercontent.com")
        raw_url = raw_url.replace("/blob/", "/")
        lines.append(f"  {raw_url}")
        lines.append("Instead of:")
        lines.append(f"  {source_url}")

    return '\n'.join(lines)
