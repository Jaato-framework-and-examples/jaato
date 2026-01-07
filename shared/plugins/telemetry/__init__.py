"""Telemetry plugin for OpenTelemetry-based tracing.

This plugin provides opt-in distributed tracing for jaato operations:
- Turn-level spans for send_message() calls
- LLM API call spans with token usage
- Tool execution spans with timing
- Retry attempt spans with backoff details
- GC operation spans

Usage:
    from shared.plugins.telemetry import create_plugin, create_otel_plugin

    # Default (no-op when OTel not installed)
    telemetry = create_plugin()

    # Explicit OTel plugin
    telemetry = create_otel_plugin()
    telemetry.initialize({
        "enabled": True,
        "exporter": "otlp",
        "endpoint": "http://localhost:4317",
    })

Environment Variables:
    JAATO_TELEMETRY_ENABLED: Enable telemetry (default: false)
    JAATO_TELEMETRY_EXPORTER: Exporter type (otlp, console, none)
    JAATO_TELEMETRY_REDACT_CONTENT: Redact prompts/responses (default: true)
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
    OTEL_EXPORTER_OTLP_HEADERS: Auth headers (key=value,key2=value2)
    OTEL_SERVICE_NAME: Service name (default: jaato)
"""

from .plugin import TelemetryPlugin, SpanContext
from .null_plugin import NullTelemetryPlugin

__all__ = [
    "TelemetryPlugin",
    "SpanContext",
    "NullTelemetryPlugin",
    "create_plugin",
    "create_otel_plugin",
]


def create_plugin() -> TelemetryPlugin:
    """Create a telemetry plugin instance.

    Returns OTelPlugin if opentelemetry is installed and JAATO_TELEMETRY_ENABLED
    is set, otherwise returns NullTelemetryPlugin (zero overhead).
    """
    import os

    enabled = os.environ.get("JAATO_TELEMETRY_ENABLED", "").lower() in ("1", "true", "yes")
    if not enabled:
        return NullTelemetryPlugin()

    try:
        from .otel_plugin import OTelPlugin
        return OTelPlugin()
    except ImportError:
        # OTel not installed, return no-op
        return NullTelemetryPlugin()


def create_otel_plugin() -> TelemetryPlugin:
    """Create an OTelPlugin instance.

    Raises ImportError if opentelemetry packages are not installed.
    """
    from .otel_plugin import OTelPlugin
    return OTelPlugin()
