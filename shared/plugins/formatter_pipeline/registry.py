# shared/plugins/formatter_pipeline/registry.py
"""Formatter registry for dynamic discovery and configuration of formatter plugins.

Provides auto-discovery of formatter plugins and configuration-based registration,
allowing both server and clients to dynamically load formatters without hardcoding.

Usage:
    from shared.plugins.formatter_pipeline import FormatterRegistry

    # Auto-discover and create pipeline with defaults
    registry = FormatterRegistry()
    registry.discover()
    pipeline = registry.create_pipeline()

    # Or load from configuration file
    registry = FormatterRegistry()
    registry.load_config(".jaato/formatters.json")
    pipeline = registry.create_pipeline()

Configuration file format (.jaato/formatters.json):
    {
      "formatters": [
        {"name": "hidden_content_filter", "enabled": true},
        {"name": "diff_formatter", "enabled": true},
        {"name": "table_formatter", "enabled": true, "config": {}},
        {"name": "code_block_formatter", "config": {"line_numbers": true}}
      ]
    }
"""

import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .pipeline import FormatterPipeline
from .protocol import FormatterPlugin, ConfigurableFormatter


# Known formatter plugins and their module paths
# This serves as a fallback when auto-discovery isn't available
KNOWN_FORMATTERS: Dict[str, str] = {
    "hidden_content_filter": "shared.plugins.hidden_content_filter",
    "diff_formatter": "shared.plugins.diff_formatter",
    "table_formatter": "shared.plugins.table_formatter",
    "code_block_formatter": "shared.plugins.code_block_formatter",
    "code_validation_formatter": "shared.plugins.code_validation_formatter",
}

# Default formatters to enable when no config is provided
DEFAULT_FORMATTERS = [
    {"name": "hidden_content_filter", "enabled": True},
    {"name": "diff_formatter", "enabled": True},
    {"name": "table_formatter", "enabled": True},
    {"name": "code_block_formatter", "enabled": True, "config": {"line_numbers": True}},
]


class FormatterRegistry:
    """Registry for discovering and managing formatter plugins.

    Supports:
    - Auto-discovery of formatter plugins from shared/plugins/
    - Configuration file loading for formatter settings
    - Dynamic pipeline creation based on enabled formatters
    - Custom formatter registration for specialized formatters
    """

    def __init__(self):
        """Initialize the formatter registry."""
        self._discovered: Dict[str, str] = {}  # name -> module path
        self._config: List[Dict[str, Any]] = []
        self._custom_formatters: Dict[str, FormatterPlugin] = {}
        self._wiring_callbacks: Dict[str, Any] = {}  # For formatters needing external deps

    def discover(self) -> List[str]:
        """Discover available formatter plugins.

        Scans the known formatters and verifies they can be imported.

        Returns:
            List of discovered formatter names.
        """
        self._discovered = {}

        for name, module_path in KNOWN_FORMATTERS.items():
            try:
                # Try to import the module to verify it exists
                importlib.import_module(module_path)
                self._discovered[name] = module_path
            except ImportError:
                pass  # Formatter not available

        return list(self._discovered.keys())

    def list_available(self) -> List[str]:
        """List all available formatter names.

        Returns:
            List of formatter names that can be used.
        """
        return list(self._discovered.keys()) + list(self._custom_formatters.keys())

    def load_config(self, config_path: str) -> bool:
        """Load formatter configuration from a JSON file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            True if config was loaded successfully, False otherwise.
        """
        path = Path(config_path)
        if not path.exists():
            return False

        try:
            with open(path) as f:
                data = json.load(f)
                self._config = data.get("formatters", [])
                return True
        except (json.JSONDecodeError, IOError):
            return False

    def load_config_from_dict(self, config: Dict[str, Any]) -> None:
        """Load formatter configuration from a dictionary.

        Args:
            config: Configuration dictionary with 'formatters' key.
        """
        self._config = config.get("formatters", [])

    def use_defaults(self) -> None:
        """Use the default formatter configuration."""
        self._config = DEFAULT_FORMATTERS.copy()

    def register_custom(self, name: str, formatter: FormatterPlugin) -> None:
        """Register a custom formatter instance.

        Use this for formatters that need special initialization or wiring
        (e.g., code_validation_formatter needs LSP plugin).

        Args:
            name: Unique name for the formatter.
            formatter: Formatter instance implementing FormatterPlugin.
        """
        self._custom_formatters[name] = formatter

    def set_wiring_callback(self, formatter_name: str, callback: Any) -> None:
        """Set a callback for wiring a formatter with external dependencies.

        Some formatters need external dependencies (like code_validation_formatter
        needing the LSP plugin). This allows the caller to provide those deps.

        Args:
            formatter_name: Name of the formatter needing wiring.
            callback: Callable that receives (formatter_instance) and wires it.
        """
        self._wiring_callbacks[formatter_name] = callback

    def create_pipeline(self, console_width: int = 120) -> FormatterPipeline:
        """Create a formatter pipeline based on current configuration.

        If no configuration is loaded, uses defaults. Custom formatters
        registered via register_custom() are always included unless
        explicitly disabled in config.

        Args:
            console_width: Console width for formatters that need it.

        Returns:
            Configured FormatterPipeline ready for use.
        """
        from .pipeline import create_pipeline

        pipeline = create_pipeline()

        # Use defaults if no config loaded
        config = self._config if self._config else DEFAULT_FORMATTERS

        # Track which formatters we've added from config
        added_names = set()

        for entry in config:
            name = entry.get("name")
            if not name:
                continue

            # Skip if explicitly disabled
            if entry.get("enabled") is False:
                continue

            formatter = self._create_formatter(name, entry.get("config", {}))
            if formatter:
                pipeline.register(formatter)
                added_names.add(name)

        # Build set of explicitly disabled formatters
        disabled_names = {
            entry.get("name")
            for entry in config
            if entry.get("enabled") is False
        }

        # Add any custom formatters not already in config and not disabled
        # (e.g., code_validation_formatter wired by server)
        for name, formatter in self._custom_formatters.items():
            if name not in added_names and name not in disabled_names:
                pipeline.register(formatter)

        pipeline.set_console_width(console_width)
        return pipeline

    def _create_formatter(
        self, name: str, config: Dict[str, Any]
    ) -> Optional[FormatterPlugin]:
        """Create a formatter instance by name.

        Args:
            name: Formatter name.
            config: Configuration dict for the formatter.

        Returns:
            Formatter instance or None if creation failed.
        """
        # Check custom formatters first
        if name in self._custom_formatters:
            formatter = self._custom_formatters[name]
            if hasattr(formatter, "initialize"):
                formatter.initialize(config)
            return formatter

        # Check discovered formatters
        if name not in self._discovered:
            return None

        module_path = self._discovered[name]

        try:
            module = importlib.import_module(module_path)
            create_fn = getattr(module, "create_plugin", None)

            if not create_fn:
                return None

            formatter = create_fn()

            # Apply wiring callback if registered
            if name in self._wiring_callbacks:
                self._wiring_callbacks[name](formatter)

            # Initialize with config if formatter supports it
            if hasattr(formatter, "initialize"):
                formatter.initialize(config)

            return formatter

        except (ImportError, AttributeError):
            return None

    def get_formatter_info(self) -> List[Dict[str, Any]]:
        """Get information about available formatters.

        Returns:
            List of dicts with name, priority, and availability status.
        """
        info = []

        for name in self.list_available():
            formatter = self._create_formatter(name, {})
            if formatter:
                info.append({
                    "name": name,
                    "priority": formatter.priority,
                    "available": True,
                })
            else:
                info.append({
                    "name": name,
                    "priority": None,
                    "available": False,
                })

        return sorted(info, key=lambda x: (x["priority"] or 999, x["name"]))


def create_registry() -> FormatterRegistry:
    """Factory function to create a FormatterRegistry instance."""
    return FormatterRegistry()


def create_default_pipeline(console_width: int = 120) -> FormatterPipeline:
    """Convenience function to create a pipeline with default formatters.

    Args:
        console_width: Console width for formatters.

    Returns:
        FormatterPipeline with default formatters registered.
    """
    registry = FormatterRegistry()
    registry.discover()
    registry.use_defaults()
    return registry.create_pipeline(console_width)
