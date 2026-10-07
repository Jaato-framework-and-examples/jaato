"""Shared trace logging utility — re-export shim.

The implementation lives in jaato_sdk.trace. This module re-exports
everything so that existing ``from shared.trace import ...`` statements
continue to work without modification.
"""

from jaato_sdk.trace import (  # noqa: F401
    trace,
    provider_trace,
    trace_write,
    resolve_trace_path,
)

__all__ = [
    "trace",
    "provider_trace",
    "trace_write",
    "resolve_trace_path",
]
