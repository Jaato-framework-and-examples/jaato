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

PLUGIN_TIER = "daemon"
import logging
import sys
import traceback
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

from .base import (
    ModelProviderPlugin,
    OutputCallback,
    ProviderConfig,
)
from jaato_sdk.plugins.model_provider.types import (
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
    except Exception as exc:
        logger.debug(f"Failed to load entry points: {exc}")
        eps = []

    providers: Dict[str, Callable[[], ModelProviderPlugin]] = {}
    for ep in eps:
        try:
            factory = ep.load()
            providers[ep.name] = factory
        except Exception as exc:
            logger.warning(f"Failed to load provider entry point '{ep.name}'", exc_info=True)

    # Also try to discover via directory scanning for development
    providers.update(_discover_via_directory())

    return providers


# Track import errors for better error messages
_provider_import_errors: Dict[str, str] = {}


def _discover_via_directory() -> Dict[str, Callable[[], ModelProviderPlugin]]:
    """Discover providers by scanning the model_provider directory.

    Used during development when packages aren't installed via entry points.

    Returns:
        Dict mapping provider names to their factory functions.
    """
    import importlib
    import pkgutil
    from pathlib import Path

    global _provider_import_errors
    _provider_import_errors.clear()

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
                except Exception as exc:
                    logger.debug(f"Could not instantiate provider from {item.name}, using directory name: {exc}")
                    # Use directory name as fallback
                    providers[item.name] = factory
        except ImportError as e:
            # Track import errors for better error messages
            _provider_import_errors[item.name] = str(e)
        except Exception as e:
            # Track other errors
            _provider_import_errors[item.name] = f"Failed to load: {e}"

    return providers


def get_provider_import_errors() -> Dict[str, str]:
    """Get any import errors that occurred during provider discovery.

    Returns:
        Dict mapping provider names to error messages.
    """
    return _provider_import_errors.copy()


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
        # Check if the provider failed to import
        if name in _provider_import_errors:
            error = _provider_import_errors[name]
            raise ValueError(
                f"Model provider '{name}' failed to load: {error}\n"
                f"Hint: Run 'pip install -r requirements.txt' to install dependencies."
            )
        raise ValueError(
            f"Model provider '{name}' not found. Available: {available}"
        )

    provider = providers[name]()
    if config:
        provider.initialize(config)
    return provider


def list_provider_models(
    provider_name: str,
    workspace_path: Optional[str] = None,
    prefix: Optional[str] = None,
) -> list:
    """List available models for a provider, with workspace-aware credentials.

    Cross-provider helper that instantiates a provider, temporarily sets
    the workspace context so credential discovery finds workspace-specific
    auth files, and calls ``list_models()``.

    Works without a session — intended for daemon extensions, profile
    managers, and other contexts outside the normal session lifecycle.

    Args:
        provider_name: Provider identifier (e.g. ``"zhipuai"``, ``"anthropic"``).
        workspace_path: Workspace directory for credential file lookup.
            If provided, temporarily sets ``JAATO_WORKSPACE_ROOT`` so the
            provider's credential discovery finds workspace-specific auth.
        prefix: Optional model name prefix filter.

    Returns:
        Sorted list of model name strings, or empty list on failure.

    Example::

        from shared.plugins.model_provider import list_provider_models

        models = list_provider_models("zhipuai", workspace_path="/home/user/.jaato/workspaces/sessions/ws_abc")
        # → ["glm-4.5", "glm-4.7", "glm-5", "glm-5.1", ...]
    """
    import os
    from shared.session_context import set_workspace_root, reset_workspace_root

    providers = discover_providers()
    if provider_name not in providers:
        return []

    # Server 0.6.68+: prefer the per-task ContextVar over os.environ
    # mutation.  The os.environ writes are kept for third-party libs
    # that read it directly, but jaato-side reads now go through
    # ``get_workspace_root()`` which honors the ContextVar first.
    ws_token = None
    old_ws = os.environ.get("JAATO_WORKSPACE_ROOT")
    if workspace_path:
        ws_token = set_workspace_root(workspace_path)
        os.environ["JAATO_WORKSPACE_ROOT"] = workspace_path
    try:
        provider = providers[provider_name]()
        return provider.list_models(prefix=prefix)
    except Exception:
        return []
    finally:
        if workspace_path:
            if old_ws is not None:
                os.environ["JAATO_WORKSPACE_ROOT"] = old_ws
            else:
                os.environ.pop("JAATO_WORKSPACE_ROOT", None)
            if ws_token is not None:
                reset_workspace_root(ws_token)


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
    "list_provider_models",
    "get_provider_import_errors",
    "MODEL_PROVIDER_ENTRY_POINT",
]
