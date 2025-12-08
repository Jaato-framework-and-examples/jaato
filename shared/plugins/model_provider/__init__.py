"""Model Provider plugin infrastructure.

This module provides the base types and protocol for implementing
model provider plugins that encapsulate AI SDK interactions.

Model providers abstract away provider-specific details:
- Google GenAI SDK (Vertex AI, Gemini)
- Anthropic SDK (Claude)
- OpenAI SDK (GPT models)
- etc.

Usage:
    from shared.plugins.model_provider import (
        ModelProviderPlugin,
        ProviderConfig,
        discover_providers,
        load_provider,
    )

    # Discover available providers
    providers = discover_providers()
    print(providers)  # {'google_genai': <factory>, 'anthropic': <factory>}

    # Load and configure a provider
    provider = load_provider('google_genai')
    provider.initialize(ProviderConfig(project='my-project', location='us-central1'))
    provider.connect('gemini-2.5-flash')

    # Use the provider
    provider.create_session(system_instruction="You are helpful.")
    response = provider.send_message("Hello!")
"""

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "model_provider"

import sys
from typing import Callable, Dict, Optional

from .base import (
    ModelProviderPlugin,
    OutputCallback,
    ProviderConfig,
)
from .types import (
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    TokenUsage,
    ToolResult,
    ToolSchema,
)


# Entry point group for model provider plugins
MODEL_PROVIDER_ENTRY_POINT = "jaato.model_providers"


def discover_providers() -> Dict[str, Callable[[], ModelProviderPlugin]]:
    """Discover all available model provider plugins via entry points.

    Returns:
        Dict mapping provider names to their factory functions.

    Example:
        providers = discover_providers()
        # {'google_genai': <function>, 'anthropic': <function>}
    """
    try:
        if sys.version_info >= (3, 10):
            from importlib.metadata import entry_points
            eps = entry_points(group=MODEL_PROVIDER_ENTRY_POINT)
        else:
            from importlib.metadata import entry_points
            all_eps = entry_points()
            eps = all_eps.get(MODEL_PROVIDER_ENTRY_POINT, [])
    except Exception:
        eps = []

    providers: Dict[str, Callable[[], ModelProviderPlugin]] = {}
    for ep in eps:
        try:
            factory = ep.load()
            providers[ep.name] = factory
        except Exception:
            # Skip providers that fail to load
            pass

    # Also try to discover via directory scanning for development
    providers.update(_discover_via_directory())

    return providers


def _discover_via_directory() -> Dict[str, Callable[[], ModelProviderPlugin]]:
    """Discover providers by scanning the model_provider directory.

    Used during development when packages aren't installed via entry points.

    Returns:
        Dict mapping provider names to their factory functions.
    """
    import importlib
    import pkgutil
    from pathlib import Path

    providers: Dict[str, Callable[[], ModelProviderPlugin]] = {}
    plugins_dir = Path(__file__).parent

    for item in plugins_dir.iterdir():
        if not item.is_dir():
            continue
        if item.name.startswith('_') or item.name == 'tests':
            continue

        # Try to import the module
        module_name = f"shared.plugins.model_provider.{item.name}"
        try:
            module = importlib.import_module(module_name)

            # Look for create_provider or create_plugin function
            factory = getattr(module, 'create_provider', None)
            if factory is None:
                factory = getattr(module, 'create_plugin', None)

            if factory and callable(factory):
                # Try to get the provider name
                try:
                    instance = factory()
                    providers[instance.name] = factory
                except Exception:
                    # Use directory name as fallback
                    providers[item.name] = factory
        except Exception:
            # Skip modules that fail to import
            pass

    return providers


def load_provider(
    name: str,
    config: Optional[ProviderConfig] = None
) -> ModelProviderPlugin:
    """Load a model provider by name and optionally initialize it.

    Args:
        name: The provider name (e.g., 'google_genai', 'anthropic').
        config: Optional configuration to pass to initialize().

    Returns:
        An initialized ModelProviderPlugin instance.

    Raises:
        ValueError: If the provider is not found.
    """
    providers = discover_providers()

    if name not in providers:
        available = list(providers.keys())
        raise ValueError(
            f"Model provider '{name}' not found. Available: {available}"
        )

    provider = providers[name]()
    if config:
        provider.initialize(config)
    return provider


__all__ = [
    # Protocol and config
    "ModelProviderPlugin",
    "ProviderConfig",
    "OutputCallback",
    # Types
    "Message",
    "Part",
    "Role",
    "ToolSchema",
    "ToolResult",
    "FunctionCall",
    "ProviderResponse",
    "TokenUsage",
    "FinishReason",
    # Discovery
    "discover_providers",
    "load_provider",
    "MODEL_PROVIDER_ENTRY_POINT",
]
