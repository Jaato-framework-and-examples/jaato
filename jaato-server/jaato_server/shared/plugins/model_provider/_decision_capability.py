"""Whether a provider can carry images in a decision, without building it.

The ``decide`` tool shapes its schema per session (decision-models.md
§12.5): the ``images`` parameter and the ``box`` question type are offered
only when an entry's provider declares ``decision_images``.  Building the
provider to ask would need its key and a network handshake on every schema
build, so the answer is read from the provider package's
``PROVIDER_CAPABILITIES``, the declaration the conformance guard holds a
built-in provider to.

The package is found where :func:`load_provider` finds the provider:

1. the built-in package ``jaato_server.shared.plugins.model_provider.<name>``
   (the directory scan, which wins on a name clash, as in
   ``discover_providers``);
2. otherwise the ``jaato.model_providers`` entry point of that name, from
   another installed distribution.  ``PROVIDER_CAPABILITIES`` is read from
   the module the entry point names, then from each package above it, so a
   provider shipped as ``my_pkg/provider.py`` may declare it in
   ``my_pkg/__init__.py`` as built-in providers do.

Entry-point modules are imported (never their factory called); discovery
already imports them to build its provider table.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

_PACKAGE = "jaato_server.shared.plugins.model_provider"


def _capabilities_of(module_name: str) -> Any:
    """``PROVIDER_CAPABILITIES`` of ``module_name`` or its nearest package."""
    parts = module_name.split(".")
    while parts:
        try:
            module = importlib.import_module(".".join(parts))
        except ImportError:
            return None
        caps = getattr(module, "PROVIDER_CAPABILITIES", None)
        if caps is not None:
            return caps
        parts.pop()
    return None


def _entry_point_module(provider_name: str) -> Optional[str]:
    """The module of the ``jaato.model_providers`` entry point ``provider_name``."""
    from . import MODEL_PROVIDER_ENTRY_POINT
    from ..registry import scan_entry_points

    try:
        eps = scan_entry_points(MODEL_PROVIDER_ENTRY_POINT)
    except Exception:
        return None
    for ep in eps:
        if ep.name == provider_name:
            module = getattr(ep, "module", None) or ep.value.split(":", 1)[0]
            return module.strip() or None
    return None


def provider_capabilities(provider_name: Optional[str]) -> Any:
    """The ``ProviderCapabilities`` a provider declares, or ``None``.

    ``None`` for no name, a name that is not an identifier, a provider
    found in neither source, or one whose package declares nothing.
    """
    if not provider_name or not provider_name.isidentifier():
        return None
    try:
        module = importlib.import_module(f"{_PACKAGE}.{provider_name}")
    except ImportError:
        module = None
    if module is not None:
        caps = getattr(module, "PROVIDER_CAPABILITIES", None)
        if caps is not None:
            return caps
    entry_module = _entry_point_module(provider_name)
    if entry_module is None:
        return None
    return _capabilities_of(entry_module)


def provider_declares_decision_images(provider_name: Optional[str]) -> bool:
    """True when the provider's package declares ``decision_images``.

    ``False`` whenever :func:`provider_capabilities` finds nothing: the
    parameter is then absent, and an image sent anyway is refused by the
    provider itself.
    """
    caps = provider_capabilities(provider_name)
    return bool(getattr(caps, "decision_images", False))


__all__ = ["provider_capabilities", "provider_declares_decision_images"]
