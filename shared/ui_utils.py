"""Shared UI formatting utilities.

This module provides common UI formatting functions that can be used
by both IPC mode (client-side) and direct mode (embedded) code.
"""

from typing import Any, Dict, List, Optional, Union


def format_permission_options(
    response_options: List[Union[Dict[str, Any], Any]],
    use_brackets: bool = True
) -> str:
    """Format permission response options for display.

    Works with both dict format (from IPC events) and object format
    (from PermissionResponseOption).

    Args:
        response_options: List of options - either dicts with key/label
            or objects with short/full attributes.
        use_brackets: Whether to wrap keys in brackets like [y]es.

    Returns:
        Formatted options string like "[y]es [n]o [a]lways [once] [never]"
    """
    parts = []
    for opt in response_options:
        # Handle both dict and object formats
        if isinstance(opt, dict):
            key = opt.get('key', opt.get('short', '?'))
            label = opt.get('label', opt.get('full', '?'))
        else:
            key = getattr(opt, 'short', getattr(opt, 'key', '?'))
            label = getattr(opt, 'full', getattr(opt, 'label', '?'))

        # Format: [y]es or [once] depending on whether key differs from label
        if use_brackets:
            if key != label and label.lower().startswith(key.lower()):
                # Key is prefix of label: [y]es, [n]o, [a]lways
                parts.append(f"[{key}]{label[len(key):]}")
            else:
                # Key equals label or is not prefix: [once], [all]
                parts.append(f"[{label}]")
        else:
            parts.append(label)

    return " ".join(parts)


def format_tool_args_summary(tool_args: Dict[str, Any], max_length: int = 60) -> str:
    """Format tool arguments as a truncated summary string.

    Args:
        tool_args: Dictionary of tool arguments.
        max_length: Maximum length before truncation.

    Returns:
        Truncated string representation of args.
    """
    args_str = str(tool_args)
    if len(args_str) > max_length:
        return args_str[:max_length - 3] + "..."
    return args_str


def format_duration(seconds: Optional[float]) -> str:
    """Format a duration in seconds for display.

    Args:
        seconds: Duration in seconds, or None.

    Returns:
        Formatted string like "1.2s" or "".
    """
    if seconds is None:
        return ""
    return f"{seconds:.1f}s"


def format_token_count(count: int) -> str:
    """Format a token count with K/M suffixes for large numbers.

    Args:
        count: Token count.

    Returns:
        Formatted string like "1.2K" or "1.5M".
    """
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)


def format_percent(value: float, decimals: int = 1) -> str:
    """Format a percentage value.

    Args:
        value: Percentage value (0-100).
        decimals: Number of decimal places.

    Returns:
        Formatted string like "75.5%".
    """
    return f"{value:.{decimals}f}%"


def build_permission_prompt_lines(
    tool_args: Optional[Dict[str, Any]],
    response_options: List[Union[Dict[str, Any], Any]],
    include_tool_name: bool = False,
    tool_name: Optional[str] = None,
) -> List[str]:
    """Build permission prompt lines for display in tool tree.

    This function creates a consistent format for permission prompts
    that works for both IPC mode and direct mode.

    Args:
        tool_args: Arguments passed to the tool.
        response_options: List of valid response options.
        include_tool_name: Whether to include "Tool: <name>" line.
        tool_name: Tool name (required if include_tool_name is True).

    Returns:
        List of prompt lines for display.
    """
    lines = []

    # Optionally include tool name (usually redundant since tool tree shows it)
    if include_tool_name and tool_name:
        lines.append(f"Tool: {tool_name}")

    # Add args summary
    if tool_args:
        lines.append(f"Args: {format_tool_args_summary(tool_args, max_length=100)}")

    # Blank line before options
    lines.append("")

    # Add formatted options
    lines.append(format_permission_options(response_options))

    return lines


def build_clarification_prompt_lines(
    question_text: str,
    question_index: Optional[int] = None,
    total_questions: Optional[int] = None,
) -> List[str]:
    """Build clarification prompt lines for display in tool tree.

    Args:
        question_text: The question text (may contain newlines).
        question_index: Current question number (1-based).
        total_questions: Total number of questions.

    Returns:
        List of prompt lines for display.
    """
    lines = []

    # Add progress header if we have question count
    if question_index is not None and total_questions is not None:
        lines.append(f"Question {question_index}/{total_questions}")
        lines.append("")

    # Split question text into lines
    if question_text:
        lines.extend(question_text.split("\n"))

    return lines
