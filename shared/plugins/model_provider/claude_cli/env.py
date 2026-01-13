"""Environment variable resolution for Claude CLI provider."""

import os
import shutil
from typing import Optional

from .types import CLIMode


def resolve_cli_path(config_path: Optional[str] = None) -> str:
    """Resolve the path to the claude CLI executable.

    Resolution order:
    1. Explicit config_path parameter
    2. JAATO_CLAUDE_CLI_PATH environment variable
    3. "claude" (relies on PATH lookup)

    Args:
        config_path: Explicit path from ProviderConfig.extra["cli_path"]

    Returns:
        Path to the claude CLI executable.

    Raises:
        FileNotFoundError: If the CLI cannot be found.
    """
    # 1. Explicit config
    if config_path:
        if os.path.isfile(config_path) and os.access(config_path, os.X_OK):
            return config_path
        raise FileNotFoundError(f"Claude CLI not found at: {config_path}")

    # 2. Environment variable
    env_path = os.environ.get("JAATO_CLAUDE_CLI_PATH")
    if env_path:
        if os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            return env_path
        raise FileNotFoundError(
            f"Claude CLI not found at JAATO_CLAUDE_CLI_PATH: {env_path}"
        )

    # 3. PATH lookup
    cli_path = shutil.which("claude")
    if cli_path:
        return cli_path

    raise FileNotFoundError(
        "Claude CLI not found. Install it with: npm install -g @anthropic-ai/claude-code\n"
        "Or set JAATO_CLAUDE_CLI_PATH to the full path."
    )


def resolve_cli_mode(config_mode: Optional[str] = None) -> CLIMode:
    """Resolve the CLI operating mode.

    Resolution order:
    1. Explicit config_mode parameter
    2. JAATO_CLAUDE_CLI_MODE environment variable
    3. Default: CLIMode.DELEGATED

    Args:
        config_mode: Explicit mode from ProviderConfig.extra["cli_mode"]

    Returns:
        The resolved CLI mode.

    Raises:
        ValueError: If an invalid mode is specified.
    """
    # 1. Explicit config
    if config_mode:
        return _parse_mode(config_mode)

    # 2. Environment variable
    env_mode = os.environ.get("JAATO_CLAUDE_CLI_MODE")
    if env_mode:
        return _parse_mode(env_mode)

    # 3. Default
    return CLIMode.DELEGATED


def _parse_mode(mode_str: str) -> CLIMode:
    """Parse a mode string into CLIMode enum."""
    mode_lower = mode_str.lower().strip()

    if mode_lower in ("delegated", "delegate"):
        return CLIMode.DELEGATED
    elif mode_lower in ("passthrough", "pass-through", "pass_through"):
        return CLIMode.PASSTHROUGH
    else:
        valid_modes = ", ".join([m.value for m in CLIMode])
        raise ValueError(
            f"Invalid CLI mode: {mode_str!r}. Valid modes: {valid_modes}"
        )


def resolve_max_turns(config_max_turns: Optional[int] = None) -> Optional[int]:
    """Resolve the maximum number of agentic turns.

    Resolution order:
    1. Explicit config_max_turns parameter
    2. JAATO_CLAUDE_CLI_MAX_TURNS environment variable
    3. Default: None (unlimited)

    Args:
        config_max_turns: Explicit value from ProviderConfig.extra["max_turns"]

    Returns:
        Maximum turns, or None for unlimited.
    """
    # 1. Explicit config
    if config_max_turns is not None:
        return config_max_turns

    # 2. Environment variable
    env_max_turns = os.environ.get("JAATO_CLAUDE_CLI_MAX_TURNS")
    if env_max_turns:
        try:
            return int(env_max_turns)
        except ValueError:
            pass

    # 3. Default: unlimited
    return None


def resolve_permission_mode(config_mode: Optional[str] = None) -> Optional[str]:
    """Resolve the CLI permission mode.

    Resolution order:
    1. Explicit config_mode parameter
    2. JAATO_CLAUDE_CLI_PERMISSION_MODE environment variable
    3. Default: None (use CLI default)

    Valid modes (from Claude Code docs):
    - "default": Standard permission behavior
    - "acceptEdits": Auto-accept file edits
    - "plan": Planning mode - no execution
    - "bypassPermissions": Bypass all permission checks

    Args:
        config_mode: Explicit mode from ProviderConfig.extra["permission_mode"]

    Returns:
        Permission mode string, or None to use CLI default.
    """
    # 1. Explicit config
    if config_mode:
        return config_mode

    # 2. Environment variable
    env_mode = os.environ.get("JAATO_CLAUDE_CLI_PERMISSION_MODE")
    if env_mode:
        return env_mode

    # 3. Default
    return None
