"""Lazy loading for OpenAI SDK.

This module defers importing the openai SDK until it's actually needed,
improving startup time when this provider isn't used.

IT IS ALSO WHERE THE DEPENDENCY LIVES for every provider that inherits
``OpenAICompatProvider``.  Most of them — `azure_openai` and the native
`openai` provider included — contain no ``import openai`` of their own, so
this is where the failure surfaces and the install advice has to be legible
from here: hence the extras named in the ImportError below, and why
``jaato-scaffold explain provider <name> deps`` follows first-party imports
instead of scanning one directory (it reported `azure` as the missing package
while the runtime died on `openai`).
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI

# Cached module reference
_openai_module = None


def get_openai_module() -> Any:
    """Get the openai module, importing it lazily.

    Returns:
        The openai module.

    Raises:
        ImportError: If the openai package is not installed.
    """
    global _openai_module
    if _openai_module is None:
        try:
            import openai
            _openai_module = openai
        except ImportError as e:
            raise ImportError(
                "openai package not installed.  It is an optional EXTRA rather "
                "than a core dependency — most providers do not speak this wire "
                "— so an install that named no OpenAI-compatible extra has no "
                "`openai`.\n"
                "  jaato-scaffold explain provider <name> deps   # names the extra for the provider you are on\n"
                "  pip install 'jaato-server[openai]'            # e.g. the native OpenAI provider\n"
                "  pip install 'openai>=1.66'                    # the package alone"
            ) from e
    return _openai_module


def get_openai_client_class() -> Any:
    """Get the OpenAI client class, importing it lazily.

    Returns:
        The OpenAI class.

    Raises:
        ImportError: If the openai package is not installed.
    """
    return get_openai_module().OpenAI
