"""Utilities for editing tool content in an external editor.

This module provides functionality to open tool arguments in an external editor
(via $EDITOR or $VISUAL) and parse the edited content back.

Used by the permission system to allow users to edit tool content before
approving execution.
"""

import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class EditResult:
    """Result of an edit operation."""

    def __init__(
        self,
        success: bool,
        arguments: Dict[str, Any],
        was_modified: bool,
        error: Optional[str] = None,
    ):
        self.success = success
        self.arguments = arguments
        self.was_modified = was_modified
        self.error = error


def get_editor() -> str:
    """Get the user's preferred editor.

    Checks $EDITOR, then $VISUAL, then falls back to 'vi'.
    """
    return os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"


def get_file_suffix(format: str) -> str:
    """Get file suffix for a format type."""
    return {
        "yaml": ".yaml",
        "json": ".json",
        "markdown": ".md",
        "text": ".txt",
    }.get(format, ".txt")


def format_for_editing(
    arguments: Dict[str, Any],
    parameters: List[str],
    format: str,
    template: Optional[str] = None,
) -> str:
    """Format tool arguments for editing in external editor.

    Args:
        arguments: Full tool arguments dict.
        parameters: List of parameter names that are editable.
        format: Format type ('yaml', 'json', 'text', 'markdown').
        template: Optional header/instructions to prepend.

    Returns:
        Formatted string ready for editing.
    """
    # Extract only the editable parameters
    editable_args = {k: arguments[k] for k in parameters if k in arguments}

    # Format based on type
    if format == "yaml":
        if not HAS_YAML:
            # Fall back to JSON if YAML not available
            content = json.dumps(editable_args, indent=2, ensure_ascii=False)
        else:
            content = yaml.safe_dump(
                editable_args,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
    elif format == "json":
        content = json.dumps(editable_args, indent=2, ensure_ascii=False)
    elif format == "markdown":
        # For markdown, format each parameter as a section
        lines = []
        for key in parameters:
            if key in editable_args:
                value = editable_args[key]
                lines.append(f"## {key}")
                lines.append("")
                if isinstance(value, list):
                    for item in value:
                        lines.append(f"- {item}")
                else:
                    lines.append(str(value))
                lines.append("")
        content = "\n".join(lines)
    else:
        # Plain text - just stringify
        content = str(editable_args)

    # Prepend template if provided
    if template:
        content = template + "\n" + content

    return content


def parse_edited_content(
    content: str,
    parameters: List[str],
    format: str,
    template: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Parse edited content back to arguments dict.

    Args:
        content: The edited content string.
        parameters: List of expected parameter names.
        format: Format type ('yaml', 'json', 'text', 'markdown').
        template: Template header that should be stripped.

    Returns:
        Tuple of (parsed_arguments, error_message).
        If successful, error_message is None.
    """
    # Strip template header if present
    if template and content.startswith(template):
        content = content[len(template):].lstrip("\n")

    try:
        if format == "yaml":
            if not HAS_YAML:
                # Try parsing as JSON
                parsed = json.loads(content)
            else:
                parsed = yaml.safe_load(content)
        elif format == "json":
            parsed = json.loads(content)
        elif format == "markdown":
            # Parse markdown sections back to dict
            parsed = _parse_markdown_sections(content, parameters)
        else:
            # Plain text - return as-is under first parameter
            if parameters:
                parsed = {parameters[0]: content.strip()}
            else:
                parsed = {"content": content.strip()}

        if not isinstance(parsed, dict):
            return {}, f"Expected dict, got {type(parsed).__name__}"

        return parsed, None

    except yaml.YAMLError as e:
        return {}, f"YAML parse error: {e}"
    except json.JSONDecodeError as e:
        return {}, f"JSON parse error: {e}"
    except Exception as e:
        return {}, f"Parse error: {e}"


def _parse_markdown_sections(content: str, parameters: List[str]) -> Dict[str, Any]:
    """Parse markdown with ## headers back to dict."""
    result = {}
    current_key = None
    current_lines = []

    for line in content.split("\n"):
        if line.startswith("## "):
            # Save previous section
            if current_key:
                result[current_key] = _parse_section_value(current_lines)
            # Start new section
            current_key = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_key:
        result[current_key] = _parse_section_value(current_lines)

    return result


def _parse_section_value(lines: List[str]) -> Any:
    """Parse section lines into appropriate value type."""
    # Remove leading/trailing empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return ""

    # Check if it's a list (all lines start with "- ")
    if all(line.startswith("- ") or not line.strip() for line in lines):
        return [line[2:].strip() for line in lines if line.startswith("- ")]

    # Otherwise return as joined text
    return "\n".join(lines).strip()


def edit_tool_content(
    arguments: Dict[str, Any],
    editable: Any,  # EditableContent from types.py
    session_dir: Optional[Path] = None,
) -> EditResult:
    """Open tool arguments in external editor and return edited version.

    Args:
        arguments: Current tool arguments.
        editable: EditableContent metadata from the tool schema.
        session_dir: Optional session directory for edit history.

    Returns:
        EditResult with success status, arguments, and whether modified.
    """
    if editable is None:
        return EditResult(
            success=False,
            arguments=arguments,
            was_modified=False,
            error="Tool does not have editable content",
        )

    parameters = getattr(editable, 'parameters', [])
    format = getattr(editable, 'format', 'yaml')
    template = getattr(editable, 'template', None)

    if not parameters:
        return EditResult(
            success=False,
            arguments=arguments,
            was_modified=False,
            error="No editable parameters defined",
        )

    # Format content for editing
    content = format_for_editing(arguments, parameters, format, template)

    # Create temp file with appropriate suffix
    suffix = get_file_suffix(format)
    editor = get_editor()

    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=suffix,
            delete=False,
            encoding='utf-8',
        ) as f:
            f.write(content)
            temp_path = f.name

        # Record original content for comparison
        original_content = content

        # Open in editor (blocking)
        result = subprocess.run([editor, temp_path], check=False)

        if result.returncode != 0:
            os.unlink(temp_path)
            return EditResult(
                success=False,
                arguments=arguments,
                was_modified=False,
                error=f"Editor exited with code {result.returncode}",
            )

        # Read back edited content
        with open(temp_path, 'r', encoding='utf-8') as f:
            edited_content = f.read()

        # Clean up temp file
        os.unlink(temp_path)

        # Check if content was modified
        was_modified = edited_content.strip() != original_content.strip()

        if not was_modified:
            return EditResult(
                success=True,
                arguments=arguments,
                was_modified=False,
            )

        # Parse edited content
        parsed_args, error = parse_edited_content(
            edited_content, parameters, format, template
        )

        if error:
            return EditResult(
                success=False,
                arguments=arguments,
                was_modified=False,
                error=error,
            )

        # Merge edited parameters back into original arguments
        new_arguments = arguments.copy()
        new_arguments.update(parsed_args)

        # Save to edit history if session_dir provided
        if session_dir and was_modified:
            _save_edit_history(
                session_dir,
                original_args=arguments,
                edited_args=new_arguments,
                parameters=parameters,
            )

        return EditResult(
            success=True,
            arguments=new_arguments,
            was_modified=True,
        )

    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        return EditResult(
            success=False,
            arguments=arguments,
            was_modified=False,
            error=str(e),
        )


def _save_edit_history(
    session_dir: Path,
    original_args: Dict[str, Any],
    edited_args: Dict[str, Any],
    parameters: List[str],
) -> None:
    """Save edit to session history for recovery.

    Args:
        session_dir: Session directory path.
        original_args: Original arguments before edit.
        edited_args: Edited arguments.
        parameters: List of edited parameter names.
    """
    history_dir = session_dir / "edit_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    history_file = history_dir / f"edit_{timestamp}.json"

    # Extract only the edited parameters for diff
    original_subset = {k: original_args.get(k) for k in parameters}
    edited_subset = {k: edited_args.get(k) for k in parameters}

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "parameters": parameters,
        "original": original_subset,
        "edited": edited_subset,
    }

    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
    except (IOError, OSError):
        pass  # Best-effort history saving
