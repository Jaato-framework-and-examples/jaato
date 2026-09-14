"""The telemetry factory honours ``plugin_configs.telemetry`` (#858).

``plugin_configs.telemetry.redact_content`` was a documented typed key that
``OTelPlugin.initialize()`` reads (``config.get("redact_content", True)``) and
that nothing ever delivered: ``create_plugin()`` assembled its own config dict
from the environment and passed *that* to ``initialize()``, so the profile
block was overwritten before the plugin saw it.

Why that is a privacy defect rather than a papercut: an operator who writes
``redact_content: true`` in the typed place believes prompt and response
content is withheld from the collector.  It was being exported.  The env
default failing in the safe direction does not make the ignore correct —
``redact_content: false`` was ignored exactly as completely, and nothing at any
severity said so.

The RED these assert on ``main``: every test in
:class:`TestProfileBlockReachesThePlugin` fails at the ``create_plugin(...)``
call, which on ``main`` takes no argument at all (``TypeError``).  The
behavioural pair the issue asks for —
:meth:`TestRedactContentPrecedence.test_profile_false_beats_env_default` and
:meth:`~TestRedactContentPrecedence.test_profile_true_beats_env_false` — are
the two that would still fail if the parameter existed and were dropped.

Nothing here touches the network: the ``none`` exporter means
``initialize()`` builds a ``TracerProvider`` with no span processor.
"""

from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

from shared.plugins.telemetry import (
    _build_init_config,
    _resolve_enabled,
    _select_backend,
    create_plugin,
)
from shared.plugins.telemetry.null_plugin import NullTelemetryPlugin


_TELEMETRY_ENV = (
    "JAATO_TELEMETRY_ENABLED",
    "JAATO_TELEMETRY_BACKEND",
    "JAATO_TELEMETRY_EXPORTER",
    "JAATO_TELEMETRY_REDACT_CONTENT",
    "JAATO_TELEMETRY_FILE",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
)


@pytest.fixture(autouse=True)
def _clean_telemetry_env(monkeypatch):
    """Start every test from an environment that configures nothing.

    The factory's whole subject is precedence between a profile block and the
    environment, so an inherited ``JAATO_TELEMETRY_*`` from the developer's
    shell would make these tests assert about the wrong tier.
    """
    for name in _TELEMETRY_ENV:
        monkeypatch.delenv(name, raising=False)


def _block(**kwargs: Any) -> Dict[str, Any]:
    """A ``plugin_configs.telemetry`` block that exports nowhere.

    ``exporter: none`` keeps ``initialize()`` from constructing an OTLP
    exporter, so these tests neither open a socket nor need a collector.
    """
    block: Dict[str, Any] = {"enabled": True, "exporter": "none"}
    block.update(kwargs)
    return block


# --------------------------------------------------------------------------
# The knob is live
# --------------------------------------------------------------------------

class TestProfileBlockReachesThePlugin:
    """The block a profile wrote arrives at ``initialize()``."""

    def test_profile_redact_false_reaches_the_plugin(self):
        """``redact_content: false`` in a profile turns redaction OFF.

        THE ISSUE.  On ``main`` the factory overwrote this with the env
        default (``True``), so the knob read as configured and did nothing.
        """
        plugin = create_plugin(_block(redact_content=False))

        assert plugin.enabled
        assert plugin._redact_content is False

    def test_profile_redact_true_reaches_the_plugin(self):
        """...and ``true`` arrives too, rather than coinciding with a default."""
        plugin = create_plugin(_block(redact_content=True))

        assert plugin._redact_content is True

    def test_profile_exporter_reaches_the_plugin(self):
        """``exporter`` had the identical defect and is fixed by the same act.

        A fix that repaired ``redact_content`` and left its neighbour inert
        would be the same defect wearing a smaller number, so the whole block
        is carried through rather than a hand-picked pair of keys.
        """
        captured: Dict[str, Any] = {}

        class _Recorder:
            enabled = True

            def initialize(self, config):
                captured.update(config)

        with patch("shared.plugins.telemetry.otel_plugin.OTelPlugin", _Recorder):
            create_plugin(_block(exporter="console"))

        assert captured["exporter"] == "console"

    def test_profile_file_path_reaches_the_plugin(self, tmp_path):
        """``file_path`` already won inside ``initialize()`` — unreachably."""
        target = tmp_path / "traces.jsonl"
        plugin = create_plugin(
            _block(exporter="file", file_path=str(target))
        )

        assert plugin.enabled
        plugin.shutdown()

    def test_a_span_redacts_or_not_according_to_the_profile(self):
        """End to end: the knob decides what a span actually carries.

        ``_redact_content`` is an implementation detail; what an operator is
        promised is that message content does not leave the process.
        """
        revealing = create_plugin(_block(redact_content=False))
        with revealing.turn_span("s1", "main") as span:
            span.set_attribute("input.value", "the user's secret prompt")
            assert span._span.attributes["input.value"] == (
                "the user's secret prompt"
            )

        withholding = create_plugin(_block(redact_content=True))
        with withholding.turn_span("s1", "main") as span:
            span.set_attribute("input.value", "the user's secret prompt")
            assert span._span.attributes["input.value"].startswith("[REDACTED")


# --------------------------------------------------------------------------
# Precedence
# --------------------------------------------------------------------------

class TestRedactContentPrecedence:
    """Profile above env above the framework default."""

    def test_profile_false_beats_env_default(self, monkeypatch):
        """A profile saying ``false`` is obeyed although env says nothing.

        The env default is ``true``, which is exactly the coincidence that let
        the defect hide: the WRONG behaviour also looked safe.
        """
        plugin = create_plugin(_block(redact_content=False))

        assert plugin._redact_content is False

    def test_profile_true_beats_env_false(self, monkeypatch):
        """A profile saying ``true`` outranks ``JAATO_...=false``.

        The issue's second direction: the typed key must WIN, not merely be
        consulted when the env var is absent.
        """
        monkeypatch.setenv("JAATO_TELEMETRY_REDACT_CONTENT", "false")

        plugin = create_plugin(_block(redact_content=True))

        assert plugin._redact_content is True

    def test_env_false_applies_when_the_profile_is_silent(self, monkeypatch):
        """The env var stays the lower-precedence default, not dead."""
        monkeypatch.setenv("JAATO_TELEMETRY_REDACT_CONTENT", "false")

        plugin = create_plugin(_block())

        assert plugin._redact_content is False

    def test_neither_set_stays_true(self):
        """THE SAFE DEFAULT.  Absent all configuration, content is withheld."""
        plugin = create_plugin(_block())

        assert plugin._redact_content is True

    def test_no_config_at_all_stays_true(self, monkeypatch):
        """A caller with no profile is unchanged by #858."""
        monkeypatch.setenv("JAATO_TELEMETRY_ENABLED", "true")
        monkeypatch.setenv("JAATO_TELEMETRY_EXPORTER", "none")

        plugin = create_plugin()

        assert plugin.enabled
        assert plugin._redact_content is True

    @pytest.mark.parametrize("written,expected", [
        (False, False), ("false", False), ("no", False), ("0", False),
        (True, True), ("true", True), ("yes", True), ("on", True),
    ])
    def test_bool_spellings(self, written, expected):
        """YAML gives a bool; a JSON round-trip or an env map gives a string."""
        plugin = create_plugin(_block(redact_content=written))

        assert plugin._redact_content is expected

    def test_an_unrecognised_value_falls_back_to_the_safe_default(self):
        """A typo must not silently DISABLE a privacy control.

        ``bool("maybe")`` is ``True`` and ``bool("")`` is ``False``; neither is
        an answer to what the operator meant, so an unrecognised token takes
        the tier beneath it — here the ``True`` framework default.
        """
        plugin = create_plugin(_block(redact_content="maybe"))

        assert plugin._redact_content is True


# --------------------------------------------------------------------------
# enabled
# --------------------------------------------------------------------------

class TestEnabled:
    """The gate that decides whether a real plugin is built at all."""

    def test_profile_enables_without_the_env_var(self):
        assert _resolve_enabled({"enabled": True}) is True

    def test_profile_disables_despite_the_env_var(self, monkeypatch):
        monkeypatch.setenv("JAATO_TELEMETRY_ENABLED", "true")

        assert _resolve_enabled({"enabled": False}) is False
        assert isinstance(
            create_plugin({"enabled": False}), NullTelemetryPlugin
        )

    def test_env_applies_when_the_profile_is_silent(self, monkeypatch):
        monkeypatch.setenv("JAATO_TELEMETRY_ENABLED", "true")

        assert _resolve_enabled({"exporter": "none"}) is True

    def test_off_by_default(self):
        """Telemetry stays opt-in: no block, no env, no plugin."""
        assert _resolve_enabled(None) is False
        assert _resolve_enabled({}) is False
        assert isinstance(create_plugin(None), NullTelemetryPlugin)


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------

class TestBackendSelection:
    """Honouring the profile must not silently repoint an existing deployment."""

    def test_no_block_selects_what_it_always_did(self, monkeypatch):
        """The regression guard the issue asks for.

        Every pre-#858 selection rule, exercised with ``config=None`` so that
        the answer cannot depend on the new argument.
        """
        assert _select_backend(None) == "otel"

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
        assert _select_backend(None) == "langfuse"

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://c:4317")
        assert _select_backend(None) == "otel"

        monkeypatch.setenv("JAATO_TELEMETRY_BACKEND", "langfuse")
        assert _select_backend(None) == "langfuse"

    def test_an_empty_block_selects_what_it_always_did(self, monkeypatch):
        """A profile with a ``telemetry`` block that names no backend."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")

        assert _select_backend({"redact_content": False}) == "langfuse"

    def test_profile_backend_outranks_env(self, monkeypatch):
        monkeypatch.setenv("JAATO_TELEMETRY_BACKEND", "langfuse")

        assert _select_backend({"backend": "otel"}) == "otel"

    def test_profile_langfuse_key_auto_detects(self):
        """The auto-detect rule reads the block under the SAME rule as env.

        The profile outranks the environment everywhere else, so a Langfuse
        setup written entirely in a profile must not be served by the generic
        backend.
        """
        assert _select_backend({"public_key": "pk-lf-x"}) == "langfuse"

    def test_profile_endpoint_suppresses_auto_detect(self, monkeypatch):
        """...and the same rule's other half: an explicit endpoint wins."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")

        assert _select_backend({"endpoint": "http://collector:4317"}) == "otel"

    def test_langfuse_backend_is_built_from_the_profile(self):
        from shared.plugins.telemetry.langfuse_plugin import LangfusePlugin

        plugin = create_plugin(
            _block(backend="langfuse", redact_content=False)
        )

        assert isinstance(plugin, LangfusePlugin)
        assert plugin._redact_content is False


# --------------------------------------------------------------------------
# The config dict handed to initialize()
# --------------------------------------------------------------------------

class TestBuildInitConfig:
    """What the plugin is handed, and — as importantly — what it is not."""

    def test_backend_and_enabled_are_not_forwarded_verbatim(self):
        """``backend`` is the factory's own key; ``enabled`` is resolved."""
        init = _build_init_config({"backend": "langfuse", "enabled": False}, True)

        assert "backend" not in init
        assert init["enabled"] is True

    def test_unnamed_keys_are_left_absent_for_initialize_to_resolve(self):
        """A key the profile did not write must NOT be filled in here.

        ``initialize()`` resolves ``config.get(key, os.environ[...])`` per key,
        so forcing an env-derived value in would make the profile's silence
        outrank the env var it is the default for — and a later env read
        (the daemon overlays a session's ``env:`` map per turn) would be dead.
        """
        init = _build_init_config({"exporter": "file"}, True)

        for absent in ("endpoint", "headers", "service_name", "file_path",
                       "protocol", "sample_rate", "batch_export"):
            assert absent not in init

    def test_every_other_key_passes_through_untouched(self):
        block = {
            "service_name": "my-app",
            "endpoint": "http://collector:4317",
            "headers": {"Authorization": "Bearer x"},
            "sample_rate": 0.25,
            "batch_export": False,
            "public_key": "pk-lf-x",
        }

        init = _build_init_config(dict(block), True)

        for key, value in block.items():
            assert init[key] == value

    def test_the_caller_s_dict_is_not_mutated(self):
        """The block belongs to the profile snapshot, not to this factory."""
        block = {"redact_content": False}

        _build_init_config(block, True)

        assert block == {"redact_content": False}


# --------------------------------------------------------------------------
# The plumbing: the block has to reach the factory
# --------------------------------------------------------------------------

def _envelope(plugin_configs: Dict[str, Any]):
    """A minimal :class:`SessionInitEnvelope` carrying *plugin_configs*."""
    from shared.session_envelope import SessionInitEnvelope

    return SessionInitEnvelope(
        session_id="s1",
        workspace_path=None,
        profile_name=None,
        provider_name="anthropic",
        model_name="claude-sonnet-4-20250514",
        plugin_configs=plugin_configs,
    )


class TestPlumbing:
    """Telemetry is runtime-scoped, so the block arrives at construction.

    ``JaatoSession._apply_plugin_configs`` (#950) applies a profile's configs
    to plugins the REGISTRY knows; telemetry is not a registry plugin (no
    ``PLUGIN_KIND``), it is built in ``JaatoRuntime.__init__`` before any
    session exists, and it is shared by the main session and every in-process
    subagent.  So the plumbing is a constructor argument, and these assert the
    seam rather than the wiring of any one caller.
    """

    def test_runtime_forwards_its_telemetry_config(self):
        import shared.jaato_runtime as runtime_module

        seen: Dict[str, Optional[Dict[str, Any]]] = {}

        def _fake_factory(config=None):
            seen["config"] = config
            return NullTelemetryPlugin()

        with patch.object(
            runtime_module, "create_telemetry_plugin", _fake_factory
        ):
            runtime_module.JaatoRuntime(
                telemetry_config={"redact_content": False}
            )

        assert seen["config"] == {"redact_content": False}

    def test_runtime_without_a_profile_forwards_none(self):
        import shared.jaato_runtime as runtime_module

        seen: Dict[str, Optional[Dict[str, Any]]] = {}

        def _fake_factory(config=None):
            seen["config"] = config
            return NullTelemetryPlugin()

        with patch.object(
            runtime_module, "create_telemetry_plugin", _fake_factory
        ):
            runtime_module.JaatoRuntime()

        assert seen["config"] is None

    def test_runner_envelope_carries_the_block_to_the_runtime(self):
        """The default production path: a runner-served session.

        ``SessionInitEnvelope.plugin_configs`` carries configs for ALL plugins,
        including ones no ``plugins:`` list names — which is exactly telemetry's
        shape.
        """
        from server.runner.session import _default_runtime_factory

        envelope = _envelope({"telemetry": {"redact_content": False}})

        seen: Dict[str, Any] = {}

        class _FakeRuntime:
            def __init__(self, **kwargs):
                seen.update(kwargs)

        with patch("shared.jaato_runtime.JaatoRuntime", _FakeRuntime):
            _default_runtime_factory(envelope)

        assert seen["telemetry_config"] == {"redact_content": False}

    def test_runner_envelope_without_a_telemetry_block(self):
        from server.runner.session import _default_runtime_factory

        envelope = _envelope({"cli": {"timeout": 5}})

        seen: Dict[str, Any] = {}

        class _FakeRuntime:
            def __init__(self, **kwargs):
                seen.update(kwargs)

        with patch("shared.jaato_runtime.JaatoRuntime", _FakeRuntime):
            _default_runtime_factory(envelope)

        assert seen["telemetry_config"] is None

    def test_client_setter_reaches_the_runtime_at_connect(self):
        """``JaatoClient.set_telemetry_config`` is the in-process route.

        The embedded client builds its ``JaatoClient`` through a pluggable
        factory whose signature predates the block, so a fourth constructor
        argument would break every injected test seam.
        """
        from shared.jaato_client import JaatoClient

        client = JaatoClient(provider_name="anthropic")
        client.set_telemetry_config({"redact_content": False})

        assert client._telemetry_config == {"redact_content": False}
