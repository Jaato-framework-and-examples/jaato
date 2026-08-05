"""Tests for the Langfuse telemetry backend.

Cover the pure settings derivation (endpoint / protocol / Basic-auth), the
explicit-config-wins precedence, and the create_plugin() backend selection.
These need no live Langfuse instance — they assert the config jaato would
send, plus (where OTel is installed) that the derived config selects the
HTTP exporter Langfuse requires.
"""

import base64

import pytest

from shared.plugins.telemetry.langfuse_plugin import LangfusePlugin
from shared.plugins.telemetry import _select_backend

try:
    import opentelemetry.sdk  # noqa: F401
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

_LANGFUSE_ENV = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "JAATO_TELEMETRY_BACKEND",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start each test from a clean Langfuse/OTLP environment."""
    for var in _LANGFUSE_ENV:
        monkeypatch.delenv(var, raising=False)


def _basic(public, secret):
    token = base64.b64encode(f"{public}:{secret}".encode()).decode()
    return f"Basic {token}"


class TestSettingsDerivation:
    def test_keys_from_config_derive_endpoint_protocol_auth(self):
        s = LangfusePlugin._resolve_settings(
            {"public_key": "pk-lf-1", "secret_key": "sk-lf-2"}
        )
        assert s["exporter"] == "otlp"
        assert s["protocol"] == "http/protobuf"
        assert s["endpoint"] == "https://cloud.langfuse.com/api/public/otel"
        assert s["headers"]["Authorization"] == _basic("pk-lf-1", "sk-lf-2")

    def test_keys_from_env(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-env")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-env")
        s = LangfusePlugin._resolve_settings({})
        assert s["headers"]["Authorization"] == _basic("pk-env", "sk-env")
        assert s["endpoint"] == "https://cloud.langfuse.com/api/public/otel"

    def test_custom_host_self_hosted(self):
        s = LangfusePlugin._resolve_settings(
            {"public_key": "p", "secret_key": "s", "host": "https://lf.internal:3000/"}
        )
        # Trailing slash trimmed; /api/public/otel appended.
        assert s["endpoint"] == "https://lf.internal:3000/api/public/otel"

    def test_explicit_endpoint_preserved(self):
        s = LangfusePlugin._resolve_settings(
            {"public_key": "p", "secret_key": "s",
             "endpoint": "http://collector.local:4318/v1/traces"}
        )
        assert s["endpoint"] == "http://collector.local:4318/v1/traces"

    def test_env_otlp_endpoint_suppresses_derivation(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
        s = LangfusePlugin._resolve_settings({"public_key": "p", "secret_key": "s"})
        assert "endpoint" not in s  # left for OTelPlugin to read from env

    def test_explicit_headers_not_clobbered(self):
        s = LangfusePlugin._resolve_settings(
            {"public_key": "p", "secret_key": "s",
             "headers": {"Authorization": "Bearer custom"}}
        )
        assert s["headers"] == {"Authorization": "Bearer custom"}

    def test_missing_keys_warns_and_omits_headers(self, caplog):
        with caplog.at_level("WARNING"):
            s = LangfusePlugin._resolve_settings({})
        assert "headers" not in s
        # Endpoint/protocol are still derived; only auth is absent.
        assert s["protocol"] == "http/protobuf"
        assert any("LANGFUSE_PUBLIC_KEY" in r.message for r in caplog.records)

    def test_explicit_protocol_preserved(self):
        s = LangfusePlugin._resolve_settings(
            {"public_key": "p", "secret_key": "s", "protocol": "grpc"}
        )
        assert s["protocol"] == "grpc"


class TestBackendSelection:
    def test_default_is_otel(self):
        assert _select_backend() == "otel"

    def test_explicit_langfuse(self, monkeypatch):
        monkeypatch.setenv("JAATO_TELEMETRY_BACKEND", "langfuse")
        assert _select_backend() == "langfuse"

    def test_explicit_otel_overrides_autodetect(self, monkeypatch):
        monkeypatch.setenv("JAATO_TELEMETRY_BACKEND", "otel")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf")
        assert _select_backend() == "otel"

    def test_autodetect_langfuse_from_keys(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf")
        assert _select_backend() == "langfuse"

    def test_no_autodetect_when_otlp_endpoint_set(self, monkeypatch):
        # A generic OTLP endpoint means the user is wiring their own collector;
        # don't hijack it just because Langfuse keys happen to be present.
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
        assert _select_backend() == "otel"


@pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not installed")
class TestDerivedConfigSelectsHttpExporter:
    def test_langfuse_config_yields_http_exporter(self):
        # End-to-end: the derived settings must select the HTTP OTLP exporter,
        # since Langfuse rejects gRPC. Feed them through the real exporter
        # factory and check the exporter class.
        s = LangfusePlugin._resolve_settings(
            {"public_key": "p", "secret_key": "s"}
        )
        exporter = LangfusePlugin()._create_exporter("otlp", s)
        assert type(exporter).__module__ == (
            "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        )
