"""Thinking mode plugin for controlling extended reasoning.

This plugin provides a user command to control thinking/reasoning modes
in AI providers. It is explicitly NOT shared with the model - thinking
mode is entirely under user control.

Supported providers:
- Anthropic: Extended thinking with configurable budget
- Google Gemini: Thinking mode (Gemini 2.0+)
"""

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ..base import (
    CommandCompletion,
    CommandParameter,
    OutputCallback,
    ToolSchema,
    UserCommand,
)
from ..model_provider.types import ThinkingConfig
from .config import (
    ThinkingPluginConfig,
    ThinkingPreset,
    load_config,
)

if TYPE_CHECKING:
    from ...jaato_session import JaatoSession


class ThinkingPlugin:
    """Plugin for controlling extended thinking/reasoning modes.

    This plugin provides the /thinking command for users to control
    thinking modes without model interference. The command is:
    - NOT shared with the model (share_with_model=False)
    - Auto-approved (no permission prompts needed)
    - Config-driven (presets loaded from .jaato/thinking.json)

    Usage:
        plugin = ThinkingPlugin()
        plugin.initialize()
        plugin.set_session(session)

        # User can then run:
        # /thinking          - Show current status
        # /thinking off      - Disable thinking
        # /thinking deep     - Enable deep preset
        # /thinking 50000    - Custom budget
    """

    def __init__(self):
        """Initialize the plugin."""
        self._config: ThinkingPluginConfig = ThinkingPluginConfig.default()
        self._current_config: ThinkingConfig = ThinkingConfig()
        self._session: Optional['JaatoSession'] = None
        self._output_callback: Optional[OutputCallback] = None

    @property
    def name(self) -> str:
        """Plugin identifier."""
        return "thinking"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin.

        Args:
            config: Optional configuration dict. Supports:
                - config_path: Path to thinking.json config file
        """
        config = config or {}
        config_path = config.get("config_path")

        # Load configuration from file or defaults
        self._config = load_config(config_path)

        # Apply default preset
        default_preset = self._config.get_default_preset()
        if default_preset:
            self._current_config = default_preset.to_config()

    def shutdown(self) -> None:
        """Clean up resources."""
        self._session = None
        self._output_callback = None

    def set_session(self, session: 'JaatoSession') -> None:
        """Set the session for applying thinking configuration.

        Args:
            session: The JaatoSession to configure.
        """
        self._session = session

        # Apply current config to session if already configured
        if self._current_config and self._session:
            self._apply_config_to_session()

    def set_output_callback(self, callback: Optional[OutputCallback]) -> None:
        """Set callback for command output.

        Args:
            callback: Function to receive output.
        """
        self._output_callback = callback

    def _apply_config_to_session(self) -> None:
        """Apply current thinking config to the session."""
        if not self._session:
            return

        # Check if session's provider supports thinking
        if hasattr(self._session, '_provider') and self._session._provider:
            provider = self._session._provider
            if hasattr(provider, 'set_thinking_config'):
                provider.set_thinking_config(self._current_config)
            elif hasattr(provider, 'supports_thinking'):
                if not provider.supports_thinking() and self._current_config.enabled:
                    # Provider doesn't support thinking - warn but continue
                    self._emit_output(
                        f"Warning: Current provider does not support thinking mode",
                        "system"
                    )

    def _emit_output(self, text: str, source: str = "thinking") -> None:
        """Emit output via callback or print."""
        if self._output_callback:
            self._output_callback(source, text, "write")
        else:
            print(text)

    # ==================== Plugin Protocol ====================

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas - none for this plugin.

        Thinking control is user-only, not model-accessible.
        """
        return []

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executors for user commands."""
        return {
            "thinking": self._execute_thinking_command,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions - none for this plugin."""
        return None

    def get_auto_approved_tools(self) -> List[str]:
        """Return auto-approved tools.

        The thinking command is user-invoked, so it's auto-approved.
        """
        return ["thinking"]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands.

        The thinking command is explicitly NOT shared with the model.
        """
        # Build description from available presets
        preset_names = list(self._config.presets.keys())
        preset_list = ", ".join(preset_names)

        return [
            UserCommand(
                name="thinking",
                description=f"Set thinking mode: {preset_list}, or custom budget",
                share_with_model=False,  # Critical: user-only control
                parameters=[
                    CommandParameter(
                        name="preset_or_budget",
                        description="Preset name or token budget",
                        required=False
                    )
                ]
            )
        ]

    def get_command_completions(
        self,
        command: str,
        args: List[str]
    ) -> List[CommandCompletion]:
        """Return completion options for the thinking command.

        Completions are dynamically generated from config presets.
        """
        if command != "thinking":
            return []

        # If no args yet, show all presets
        prefix = args[0].lower() if args else ""

        completions = []
        for name, preset in self._config.presets.items():
            if not prefix or name.startswith(prefix):
                if preset.enabled:
                    desc = f"Enable thinking ({preset.budget:,} token budget)"
                else:
                    desc = "Disable thinking"
                completions.append(CommandCompletion(name, desc))

        return completions

    # ==================== Command Execution ====================

    def _execute_thinking_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the thinking command.

        Args:
            args: Command arguments with optional 'preset_or_budget'.

        Returns:
            Dict with command result.
        """
        preset_or_budget = args.get("preset_or_budget", "").strip()

        # No argument - show current status
        if not preset_or_budget:
            return self._get_status()

        # Try as preset name first
        preset = self._config.get_preset(preset_or_budget.lower())
        if preset:
            return self._apply_preset(preset_or_budget.lower(), preset)

        # Try as numeric budget
        try:
            budget = int(preset_or_budget.replace(",", "").replace("_", ""))
            if budget < 0:
                return {"error": "Budget must be non-negative"}
            if budget == 0:
                return self._apply_preset("off", ThinkingPreset(enabled=False, budget=0))
            return self._apply_custom_budget(budget)
        except ValueError:
            pass

        # Unknown preset
        preset_names = list(self._config.presets.keys())
        return {
            "error": f"Unknown preset: {preset_or_budget}",
            "available_presets": preset_names,
            "hint": "Use a preset name or a numeric budget (e.g., 50000)"
        }

    def _get_status(self) -> Dict[str, Any]:
        """Get current thinking mode status."""
        # Find which preset matches current config (if any)
        current_preset = None
        for name, preset in self._config.presets.items():
            if (preset.enabled == self._current_config.enabled and
                    preset.budget == self._current_config.budget):
                current_preset = name
                break

        # Check provider support
        provider_supports = True
        provider_name = None
        if self._session and hasattr(self._session, '_provider') and self._session._provider:
            provider = self._session._provider
            provider_name = getattr(provider, 'name', None)
            if hasattr(provider, 'supports_thinking'):
                provider_supports = provider.supports_thinking()

        status = {
            "enabled": self._current_config.enabled,
            "budget": self._current_config.budget,
            "preset": current_preset or "custom",
            "available_presets": list(self._config.presets.keys()),
        }

        if provider_name:
            status["provider"] = provider_name
        if not provider_supports:
            status["warning"] = "Current provider does not support thinking mode"

        return status

    def _apply_preset(self, name: str, preset: ThinkingPreset) -> Dict[str, Any]:
        """Apply a named preset."""
        old_enabled = self._current_config.enabled
        old_budget = self._current_config.budget

        self._current_config = preset.to_config()
        self._apply_config_to_session()

        result = {
            "success": True,
            "preset": name,
            "enabled": self._current_config.enabled,
            "budget": self._current_config.budget,
        }

        if old_enabled != self._current_config.enabled:
            if self._current_config.enabled:
                result["message"] = f"Thinking enabled: {name} ({self._current_config.budget:,} token budget)"
            else:
                result["message"] = "Thinking disabled"
        elif old_budget != self._current_config.budget:
            result["message"] = f"Thinking budget changed to {self._current_config.budget:,} tokens"
        else:
            result["message"] = f"Thinking mode unchanged: {name}"

        return result

    def _apply_custom_budget(self, budget: int) -> Dict[str, Any]:
        """Apply a custom budget."""
        old_enabled = self._current_config.enabled

        self._current_config = ThinkingConfig(enabled=True, budget=budget)
        self._apply_config_to_session()

        result = {
            "success": True,
            "preset": "custom",
            "enabled": True,
            "budget": budget,
        }

        if not old_enabled:
            result["message"] = f"Thinking enabled with custom budget: {budget:,} tokens"
        else:
            result["message"] = f"Thinking budget changed to {budget:,} tokens"

        return result

    # ==================== Programmatic API ====================

    def get_current_config(self) -> ThinkingConfig:
        """Get the current thinking configuration."""
        return ThinkingConfig(
            enabled=self._current_config.enabled,
            budget=self._current_config.budget
        )

    def set_thinking_mode(self, preset_or_budget: str | int) -> Dict[str, Any]:
        """Programmatically set thinking mode.

        Args:
            preset_or_budget: Preset name (str) or custom budget (int).

        Returns:
            Result dict (same as command execution).
        """
        if isinstance(preset_or_budget, int):
            if preset_or_budget == 0:
                return self._apply_preset("off", ThinkingPreset(enabled=False, budget=0))
            return self._apply_custom_budget(preset_or_budget)

        preset = self._config.get_preset(preset_or_budget.lower())
        if preset:
            return self._apply_preset(preset_or_budget.lower(), preset)

        return {"error": f"Unknown preset: {preset_or_budget}"}


def create_plugin() -> ThinkingPlugin:
    """Factory function for plugin discovery."""
    return ThinkingPlugin()
