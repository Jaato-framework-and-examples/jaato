"""
jaato — Just Another Agentic Tool Orchestrator (public convenience facade).

This top-level ``jaato`` package is the public entry point. It ships in
**jaato-sdk**, so ``import jaato`` and ``jaato.session(mode="ipc"|"ws")`` work
with only the SDK installed — the daemon it talks to may be on another host, and
a thin client needs no server runtime.

Two tiers of names:

* **Eager, sdk-only** — provider-agnostic types / events / plugin base classes
  (re-exported from ``jaato_sdk``) and ``session`` (the transport dispatcher).
  These never import the server runtime.
* **Lazy, server-only** — ``JaatoClient`` / ``PluginRegistry`` (the embedded
  runtime) and ``InProcessClient`` (the ``mode="in_process"`` backend). Resolved
  on first attribute access via PEP 562 ``__getattr__`` from **jaato-server**
  (``shared`` / ``jaato_embedded``); accessing one without jaato-server installed
  raises a clear ``ImportError`` with an install hint. They are intentionally
  **not** in ``__all__`` so ``from jaato import *`` stays sdk-only-safe.

See ``docs/design/in-process-facade.md``.
"""

from typing import TYPE_CHECKING, Any

# The transport-agnostic session entry (sdk-only for ipc/ws; lazy server import
# for in_process). Lives in _session so __init__ stays a thin re-export surface.
from jaato._session import session, _bundle_inline_profile

# Provider-agnostic types for plugin development
from jaato_sdk.plugins.model_provider.types import (
    # Enums
    Role,
    FinishReason,

    # Message types
    Message,
    Part,

    # Tool types
    ToolSchema,
    FunctionCall,
    ToolResult,

    # Response types
    ProviderResponse,
    TokenUsage,

    # Attachments
    Attachment,

    # Cancellation
    CancelToken,
    CancelledException,
)

# Presentation context for display-aware agents
from jaato_sdk.events import PresentationContext, ClientType

# Plugin base classes
from jaato_sdk.plugins.base import UserCommand, CommandParameter


# Lazy server-only names: resolved from jaato-server on first access. Kept out of
# the eager import list (and out of __all__) so ``import jaato`` never pulls the
# server runtime — the whole point of the sdk-only facade.
_LAZY = {
    "JaatoClient": ("shared", "JaatoClient"),
    "PluginRegistry": ("shared", "PluginRegistry"),
    "InProcessClient": ("jaato_embedded", "InProcessClient"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for the server-only names.

    ``from jaato import JaatoClient`` (or attribute access) triggers this for a
    name not defined eagerly above. We import it from jaato-server on demand and
    re-raise a missing server as an actionable ImportError.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'jaato' has no attribute {name!r}")
    module_name, attr = target
    try:
        import importlib
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(
            f"jaato.{name} is provided by jaato-server (module {module_name!r}), "
            f"which is not installed. Install it (pip install jaato-server), or "
            f"use jaato.session(mode='ipc'/'ws') for a thin sdk-only client."
        ) from None
    value = getattr(module, attr)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__() -> list:
    """Include the lazy server-only names in dir() for REPL/IDE discoverability
    (without importing them)."""
    return sorted(list(globals().keys()) + list(_LAZY))


if TYPE_CHECKING:  # static analyzers / IDEs — no runtime import
    from shared import JaatoClient, PluginRegistry
    from jaato_embedded import InProcessClient


# Public API. NOTE: the lazy server-only names (JaatoClient, PluginRegistry,
# InProcessClient) are DELIBERATELY excluded — putting them here would make
# ``from jaato import *`` call __getattr__ and fail on an sdk-only install.
__all__ = [
    # Transport-agnostic session entry
    "session",

    # Enums
    "Role",
    "FinishReason",

    # Message types
    "Message",
    "Part",

    # Tool types
    "ToolSchema",
    "FunctionCall",
    "ToolResult",

    # Response types
    "ProviderResponse",
    "TokenUsage",

    # Attachments
    "Attachment",

    # Cancellation
    "CancelToken",
    "CancelledException",

    # Presentation
    "PresentationContext",
    "ClientType",

    # Plugin support
    "UserCommand",
    "CommandParameter",
]
