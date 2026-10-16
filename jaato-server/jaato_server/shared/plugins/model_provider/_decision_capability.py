"""Whether a provider can carry images in a decision, without building it.

The ``decide`` tool shapes its schema per session (decision-models.md
§12.5): the ``images`` parameter and the ``box`` question type are offered
only when an entry's provider declares ``decision_images``.  Building the
provider to ask would need its key and a network handshake on every schema
build, so the answer is read from the package's ``PROVIDER_CAPABILITIES``,
the declaration the conformance guard holds the provider to.
"""

from __future__ import annotations

import importlib
from typing import Optional

_PACKAGE = "jaato_server.shared.plugins.model_provider"


def provider_declares_decision_images(provider_name: Optional[str]) -> bool:
    """True when the built-in provider package declares ``decision_images``.

    ``False`` for no name, a name that is not an identifier, a package that
    does not import, or one declaring no capabilities: the parameter is then
    absent, and an image sent anyway is refused by the provider itself.
    """
    if not provider_name or not provider_name.isidentifier():
        return False
    try:
        module = importlib.import_module(f"{_PACKAGE}.{provider_name}")
    except ImportError:
        return False
    caps = getattr(module, "PROVIDER_CAPABILITIES", None)
    return bool(getattr(caps, "decision_images", False))


__all__ = ["provider_declares_decision_images"]
