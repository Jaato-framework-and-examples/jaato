"""Langfuse-preconfigured telemetry backend.

jaato already emits OpenInference spans over OTLP, and
[Langfuse](https://langfuse.com) ingests OpenInference directly — so
Langfuse support is a *configuration* of the existing :class:`OTelPlugin`,
not a second emission path. This subclass exists only to remove the three
Langfuse-specific footguns that otherwise make the generic OTLP path fail
silently against Langfuse:

1. **Endpoint path** — Langfuse's OTLP traces endpoint is
   ``<host>/api/public/otel/v1/traces``. jaato passes this to the OTLP/HTTP
   exporter as an explicit endpoint (POSTed verbatim, no ``/v1/traces``
   auto-append), so the full signal path — not the bare host, and not just the
   ``/api/public/otel`` base — must be baked in or Langfuse returns 404.
2. **Transport** — Langfuse is OTLP/HTTP only; jaato's exporter is
   gRPC-first, so the protocol must be pinned to ``http/protobuf``.
3. **Auth** — HTTP Basic over ``base64("<public_key>:<secret_key>")`` (not
   the Bearer scheme collectors usually want).

With those derived from the user's Langfuse keys, configuration is just the
keys (env or profile). All span logic — OpenInference conventions, token
counts, provider/pricing-table cost, redaction — is inherited unchanged from
:class:`OTelPlugin`.

Out of scope: Langfuse's non-tracing features (prompt management, datasets,
score/eval ingestion) live behind the Langfuse SDK/API, not OTLP, and are a
separate follow-up — this backend covers tracing only.

Environment variables:
    LANGFUSE_PUBLIC_KEY: Langfuse project public key (``pk-lf-...``).
    LANGFUSE_SECRET_KEY: Langfuse project secret key (``sk-lf-...``).
    LANGFUSE_HOST: Base URL of the Langfuse instance
        (default ``https://cloud.langfuse.com``; use your host when
        self-hosting or on the EU/US regional clouds).
"""

import base64
import logging
import os
from typing import Any, Dict, Optional

from .otel_plugin import OTelPlugin

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "https://cloud.langfuse.com"
# Full OTLP/HTTP *traces* signal path, not just Langfuse's ``/api/public/otel``
# base. ``OTelPlugin._create_exporter`` hands this to the OTLP/HTTP exporter as
# its explicit ``endpoint=`` kwarg, which is POSTed verbatim — the exporter only
# appends ``/v1/traces`` when it reads a *base* from ``OTEL_EXPORTER_OTLP_ENDPOINT``
# itself, never for an explicit endpoint. So the base alone POSTs to
# ``/api/public/otel`` and Langfuse returns 404; the signal path must be baked in.
_OTEL_PATH = "/api/public/otel/v1/traces"


class LangfusePlugin(OTelPlugin):
    """:class:`OTelPlugin` preconfigured for Langfuse's OTLP endpoint.

    Overrides only :meth:`initialize`: it derives the OTLP endpoint,
    HTTP protocol, and Basic-auth header from Langfuse keys, then delegates
    to the parent. Everything else (span creation, attributes, redaction,
    shutdown) is inherited.

    Explicit config/env always wins: if an ``endpoint`` / ``headers`` /
    ``protocol`` is already supplied (or ``OTEL_EXPORTER_OTLP_ENDPOINT`` is
    set), the derived Langfuse default for that field is not applied — so
    this class is safe to select even behind a custom OTLP collector/proxy.
    """

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().initialize(self._resolve_settings(config))

    @staticmethod
    def _resolve_settings(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Derive an :class:`OTelPlugin` config dict for Langfuse.

        Fills in the Langfuse-specific defaults (endpoint path, HTTP
        protocol, Basic-auth header) from ``config`` keys or environment,
        without clobbering any value the caller supplied explicitly. Pure
        and side-effect-free apart from a warning when no credentials are
        resolvable — safe to unit-test without constructing a provider.
        """
        settings = dict(config or {})

        public_key = settings.pop("public_key", None) or os.environ.get("LANGFUSE_PUBLIC_KEY")  # env: Langfuse project public key
        secret_key = settings.pop("secret_key", None) or os.environ.get("LANGFUSE_SECRET_KEY")  # env: Langfuse project secret key
        host = (
            settings.pop("host", None)
            or os.environ.get("LANGFUSE_HOST")  # env: Langfuse base URL (default https://cloud.langfuse.com)
            or _DEFAULT_HOST
        )

        # Langfuse speaks OTLP/HTTP only — pin the protocol unless the caller
        # explicitly asked for something else.
        settings.setdefault("exporter", "otlp")
        if "protocol" not in settings and not os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL"):
            settings["protocol"] = "http/protobuf"

        # Endpoint: derive <host>/api/public/otel/v1/traces unless one is already
        # given (config key or the standard OTLP env var).
        if "endpoint" not in settings and not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            settings["endpoint"] = host.rstrip("/") + _OTEL_PATH

        # Auth header: Basic base64("<public>:<secret>"). Only inject when the
        # caller hasn't supplied headers and the standard env var is unset, so
        # an explicitly configured Authorization is never clobbered.
        has_env_headers = bool(os.environ.get("OTEL_EXPORTER_OTLP_HEADERS"))
        if "headers" not in settings and not has_env_headers:
            if public_key and secret_key:
                token = base64.b64encode(
                    f"{public_key}:{secret_key}".encode("utf-8")
                ).decode("ascii")
                settings["headers"] = {"Authorization": f"Basic {token}"}
            elif settings.get("enabled", True):
                # No keys and no other auth source — spans will be rejected by
                # Langfuse. Warn loudly rather than silently dropping traffic.
                logger.warning(
                    "LangfusePlugin: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
                    "not set and no Authorization header provided; Langfuse "
                    "will reject spans until credentials are configured."
                )

        return settings


def create_plugin() -> "LangfusePlugin":
    """Create a LangfusePlugin instance.

    Raises ImportError (from the parent's lazy OTel imports at
    ``initialize()``) if the opentelemetry packages are not installed.
    """
    return LangfusePlugin()
