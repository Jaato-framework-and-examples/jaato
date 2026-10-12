"""Environment variable resolution for the Laya decision provider.

Laya is served two ways (``plugin_configs.laya.mode``):

* ``http`` (the default): a ``laya-serve`` process the operator started,
  answering ``POST /v1/systemone`` with TypeSafe Jev's request and
  response shape.
* ``local``: the ``laya`` package loaded into the RUNNER process.  Torch
  and the weights then live in that process, so the pool slot is retired
  rather than reused (docs/design/decision-models.md §9).

Environment variables (each beneath its ``plugin_configs.laya`` knob):

    JAATO_LAYA_HOST: ``laya-serve`` URL (default ``http://localhost:8000``,
        the address ``laya-serve`` binds).
    JAATO_LAYA_API_KEY: bearer token, when the server was started with
        ``LAYA_API_KEY`` (``laya-serve``'s own variable, honoured beneath
        this one so a host configured for the server needs nothing else).
    JAATO_LAYA_MODE: ``http`` or ``local``.
    JAATO_LAYA_CONTEXT_LENGTH: the window, when the built-in table does
        not know the model.
"""

import os
from typing import Optional

ENV_HOST = "JAATO_LAYA_HOST"
ENV_API_KEY = "JAATO_LAYA_API_KEY"
ENV_VENDOR_API_KEY = "LAYA_API_KEY"
ENV_MODE = "JAATO_LAYA_MODE"
ENV_CONTEXT_LENGTH = "JAATO_LAYA_CONTEXT_LENGTH"

#: Where ``laya-serve`` listens when started with no arguments.
DEFAULT_HOST = "http://localhost:8000"


def resolve_host() -> Optional[str]:
    """The ``laya-serve`` URL from the environment, or ``None``."""
    return os.environ.get(ENV_HOST)  # env: laya-serve URL (default http://localhost:8000)


def resolve_api_key() -> Optional[str]:
    """The bearer token: ``JAATO_LAYA_API_KEY``, then ``LAYA_API_KEY``."""
    return (
        os.environ.get(ENV_API_KEY)  # env: laya-serve bearer token
        or os.environ.get(ENV_VENDOR_API_KEY)  # env: laya-serve's own key variable
    )


def resolve_mode() -> Optional[str]:
    """``http`` or ``local`` from the environment, or ``None``."""
    return os.environ.get(ENV_MODE)  # env: http (laya-serve) or local (in-process)


def resolve_context_length() -> Optional[int]:
    """The context-window override from the environment, if a number."""
    val = os.environ.get(ENV_CONTEXT_LENGTH)  # env: window override (built-in table otherwise)
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return None
