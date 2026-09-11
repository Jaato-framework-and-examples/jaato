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

Configuration precedence (#858):
    ``create_plugin(config)`` takes the session's ``plugin_configs.telemetry``
    block.  A key present there wins; the matching environment variable below
    is the lower-precedence default; the framework default is last.  With
    neither set, telemetry is off and ``redact_content`` is ``True``.

    ```yaml
    plugin_configs:
      telemetry:
        enabled: true
        backend: langfuse        # otel (default) | langfuse
        exporter: file           # otlp (default) | file | console | none
        file_path: /var/log/jaato-traces.jsonl
        redact_content: false    # withhold prompt/response content (default true)
    ```

Environment Variables:
    JAATO_TELEMETRY_ENABLED: Enable telemetry (default: false)
    JAATO_TELEMETRY_BACKEND: Backend — otel (default) or langfuse
    JAATO_TELEMETRY_EXPORTER: Exporter type (otlp, console, file, none)
    JAATO_TELEMETRY_FILE: Output file path for file exporter (default: /tmp/jaato-traces.jsonl)
    JAATO_TELEMETRY_REDACT_CONTENT: Redact prompts/responses (default: true)
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
    OTEL_EXPORTER_OTLP_PROTOCOL: OTLP wire protocol — grpc (default) or http/protobuf
    OTEL_EXPORTER_OTLP_HEADERS: Auth headers (key=value,key2=value2)
    OTEL_SERVICE_NAME: Service name (default: jaato)
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST: Langfuse backend
"""

from typing import Any, Dict, Optional

from .plugin import TelemetryPlugin, SpanContext
from .null_plugin import NullTelemetryPlugin

__all__ = [
    "TelemetryPlugin",
    "SpanContext",
    "NullTelemetryPlugin",
    "create_plugin",
    "create_otel_plugin",
    "create_langfuse_plugin",
]


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce a profile/env value to a bool, falling back to *default*.

    Profile YAML yields a real ``bool``; a JSON round-trip or an env var
    yields a string.  ``None`` means "not configured" and takes *default*,
    which is what keeps an absent ``redact_content`` at the safe ``True``
    rather than at a falsy-value accident.  A string nobody recognises is
    also *default* — a typo must not silently disable redaction.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("1", "true", "yes", "on"):
            return True
        if token in ("0", "false", "no", "off"):
            return False
        return default
    return bool(value)


def _env_enabled() -> bool:
    """The ``JAATO_TELEMETRY_ENABLED`` gate, read exactly as it always was."""
    import os

    return os.environ.get("JAATO_TELEMETRY_ENABLED", "").lower() in ("1", "true", "yes")  # env: enable OpenTelemetry tracing (needs requirements-telemetry.txt installed)


def _env_redact_content() -> bool:
    """The ``JAATO_TELEMETRY_REDACT_CONTENT`` default — ``True`` when unset.

    Redaction is a privacy control, so the framework default is the safe
    one and stays the safe one: with neither the profile key nor the env
    var set, prompt/response content is withheld from the collector.
    """
    import os

    return os.environ.get("JAATO_TELEMETRY_REDACT_CONTENT", "true").lower() not in ("0", "false", "no")  # env: redact prompt/response content from spans (default true)


def _resolve_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Is telemetry on for this session?

    ``plugin_configs.telemetry.enabled`` wins when the key is present; the
    ``JAATO_TELEMETRY_ENABLED`` env var is the lower-precedence default, and
    with neither set telemetry stays **off** — the opt-in posture this
    plugin has always had.
    """
    if config is not None and "enabled" in config:
        return _as_bool(config.get("enabled"), _env_enabled())
    return _env_enabled()


def _select_backend(config: Optional[Dict[str, Any]] = None) -> str:
    """Resolve which telemetry backend create_plugin() should build.

    Precedence:
      1. ``plugin_configs.telemetry.backend`` — the typed knob.
      2. ``JAATO_TELEMETRY_BACKEND`` — explicit (``otel`` / ``langfuse``).
      3. Auto-detect Langfuse: when the backend is unset, Langfuse keys are
         present, and no generic OTLP endpoint is configured, prefer the
         Langfuse backend so its keys "just work" without extra env.
      4. Default ``otel``.

    Step 3 reads the profile block and the environment under the SAME rule —
    a Langfuse key with no OTLP endpoint selects Langfuse, wherever it was
    written — because the profile outranks the environment everywhere else
    and a Langfuse setup expressed purely in a profile would otherwise be
    served by the generic backend.  A session whose profile carries no
    ``telemetry`` block therefore selects exactly the backend it always did
    (#858).
    """
    import os

    cfg = config or {}
    declared = str(cfg.get("backend") or "").strip().lower()
    if declared:
        return declared
    backend = os.environ.get("JAATO_TELEMETRY_BACKEND", "").strip().lower()  # env: telemetry backend — "otel" (default) or "langfuse"
    if backend:
        return backend
    has_langfuse_keys = bool(
        cfg.get("public_key") or os.environ.get("LANGFUSE_PUBLIC_KEY")
    )
    has_otlp_endpoint = bool(
        cfg.get("endpoint") or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    if has_langfuse_keys and not has_otlp_endpoint:
        return "langfuse"
    return "otel"


def _build_init_config(
    config: Optional[Dict[str, Any]], enabled: bool
) -> Dict[str, Any]:
    """The dict handed to :meth:`TelemetryPlugin.initialize`.

    The whole ``plugin_configs.telemetry`` block is carried through — not a
    hand-picked pair of keys — so every knob ``initialize()`` already
    understands (``endpoint``, ``headers``, ``protocol``, ``service_name``,
    ``instance_id``, ``sample_rate``, ``batch_export``, ``file_path``, and
    Langfuse's ``public_key`` / ``secret_key`` / ``host``) becomes reachable
    from a profile by the same act that makes ``redact_content`` reachable.
    Repairing one key and leaving its neighbours inert would be the same
    defect wearing a smaller number.

    ``backend`` is consumed by :func:`_select_backend` and ``enabled`` is
    already resolved, so neither is forwarded verbatim.

    Precedence is expressed by reading the profile's value FIRST and using
    the env var only as its fallback — ``setdefault`` for ``exporter``,
    whose value is passed through verbatim, and ``_as_bool(init.get(...),
    env)`` for ``redact_content``, which must also coerce a string spelling
    and fall back safely on one it does not recognise.

    Exactly those two defaults are spelled out here — the two the factory
    has always resolved eagerly.  Every other key is left ABSENT rather than
    filled in, so ``initialize()`` applies its own
    ``config.get(key, os.environ[...])`` fallback instead of having a value
    forced on it.
    """
    import os

    init: Dict[str, Any] = {
        k: v for k, v in (config or {}).items()
        if k not in ("backend", "enabled")
    }
    init["enabled"] = enabled
    init.setdefault(
        "exporter",
        os.environ.get("JAATO_TELEMETRY_EXPORTER", "otlp"),  # env: span exporter: otlp (default), file, console, or none
    )
    init["redact_content"] = _as_bool(
        init.get("redact_content"), _env_redact_content()
    )
    return init


def create_plugin(
    config: Optional[Dict[str, Any]] = None,
) -> TelemetryPlugin:
    """Create a telemetry plugin instance.

    Returns an ``OTelPlugin`` (or ``LangfusePlugin``) when opentelemetry is
    installed and telemetry is enabled, otherwise a ``NullTelemetryPlugin``
    (zero overhead).

    Args:
        config: The session's ``plugin_configs.telemetry`` block, or ``None``
            when the caller has no profile.  Before #858 this parameter did
            not exist: the factory assembled its own config dict from the
            environment and passed it to ``initialize()``, so every key a
            profile wrote under ``plugin_configs.telemetry`` was overwritten
            before the plugin saw it.  ``redact_content`` is the one that
            mattered — an operator who set it in the typed place believed
            prompt and response content was being withheld from the
            collector, and it was being exported, with nothing to say so.

    Precedence for every key, highest first:

      1. ``plugin_configs.telemetry.<key>`` (this argument),
      2. the corresponding ``JAATO_TELEMETRY_*`` / ``OTEL_*`` env var,
      3. the framework default.

    With no config and no env vars the behaviour is byte-identical to
    before: telemetry off, and when switched on by env, ``exporter=otlp``
    and ``redact_content=True``.

    Environment variables consulted (all below the profile block):
        JAATO_TELEMETRY_ENABLED: Enable telemetry
        JAATO_TELEMETRY_BACKEND: Backend — otel (default) or langfuse
        JAATO_TELEMETRY_EXPORTER: Exporter type (otlp, console, file, none)
        JAATO_TELEMETRY_REDACT_CONTENT: Redact prompts/responses (default: true)
        JAATO_TELEMETRY_FILE: Output path for the file exporter
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
        OTEL_EXPORTER_OTLP_HEADERS: Auth headers (key=value,key2=value2)
        OTEL_SERVICE_NAME: Service name (default: jaato)
    """
    if not _resolve_enabled(config):
        return NullTelemetryPlugin()

    try:
        backend = _select_backend(config)
        if backend == "langfuse":
            from .langfuse_plugin import LangfusePlugin
            plugin: TelemetryPlugin = LangfusePlugin()
        else:
            from .otel_plugin import OTelPlugin
            plugin = OTelPlugin()

        # Keys this dict does NOT carry are read inside initialize() from
        # env vars: OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_EXPORTER_OTLP_HEADERS,
        # OTEL_SERVICE_NAME, JAATO_TELEMETRY_FILE, JAATO_INSTANCE_ID.
        # LangfusePlugin additionally derives endpoint/protocol/auth from
        # LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST.
        plugin.initialize(_build_init_config(config, True))

        return plugin
    except ImportError:
        # OTel not installed, return no-op
        return NullTelemetryPlugin()


def create_otel_plugin() -> TelemetryPlugin:
    """Create an OTelPlugin instance.

    Raises ImportError if opentelemetry packages are not installed.
    """
    from .otel_plugin import OTelPlugin
    return OTelPlugin()


def create_langfuse_plugin() -> TelemetryPlugin:
    """Create a LangfusePlugin instance (OTelPlugin preconfigured for Langfuse).

    Derives the OTLP endpoint, HTTP protocol, and Basic-auth header from
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST at
    ``initialize()``. Raises ImportError if opentelemetry packages are not
    installed.
    """
    from .langfuse_plugin import LangfusePlugin
    return LangfusePlugin()
