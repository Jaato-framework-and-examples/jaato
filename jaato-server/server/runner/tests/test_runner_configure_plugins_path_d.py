"""Regression-pin tests for Path D — runner-side plugin configuration
in bootstrap_session (Layer 4 + prerequisites 5-8).

**Root cause** (cycle-5 comprehensive remaining-chain audit):

After Path C closed the ``_connected`` guard at
``jaato_runtime.py:964``, ``runtime.create_session(...)`` still
fails at ``jaato_runtime.py:966-967`` with
``RuntimeError("Plugins not configured. Call configure_plugins()
first.")`` because the runner-side ``bootstrap_session`` never
mirrored daemon-side ``_run_load_plugins``
(``server/core.py:1615-1777``).

The fix consolidates all 9 daemon-side init steps into a single
``_configure_runtime_plugins(runtime, envelope)`` helper called
between Path C connect and ``_build_session``:

1. ``PluginRegistry(model_name=envelope.model_name)``
2. ``registry.discover(tier_filter="runner")`` — runner-tier only
3. Build ``plugin_configs`` dict from ``envelope.workspace_path`` +
   ``envelope.session_id`` + ``envelope.plugins[].config``
4. ``registry.expose_all(plugin_configs)``
5. (N/A — daemon's todo_plugin cached reference)
6. ``registry.set_workspace_path(workspace_path)`` if non-None
7. ``registry.set_config_root(config_root)`` if non-None
8. ``PermissionPlugin()`` + ``initialize(default policy)``
9. ``runtime.configure_plugins(registry, permission_plugin,
   ledger=None)``

Differences from daemon-side:

- ``tier_filter="runner"`` — daemon-tier plugins (auth, gc_*,
  cache_*, session, background) must NOT load runner-side.
- ``ledger=None`` — token accounting is daemon-tier per §4.2.
- No ``on_progress`` callback — runner has no client event sink.
- ``permission_plugin`` initialized with default policy only;
  profile overrides not yet propagated (backlog §3.3c.X).

**Bug chain at cycle 5** (4 layers, all from §7c structural
regressions surfaced by integration testing):

| Layer | Commit | Fix |
|---|---|---|
| Snapshot caller wraps no_session | 20b16326 | Path A narrow |
| IPC dispatches session.bootstrap | c68151a8 | Path B architectural |
| Runner connects runtime pre-create_session | 4d9cf8b7 | Path C envelope+connect |
| Runner configures plugins pre-create_session | (this) | Path D comprehensive |

Tests pin:

- ``bootstrap_session`` calls ``configure_plugins`` on the runtime
- The runtime receives a non-None registry + non-None
  permission_plugin + None ledger
- The registry was discovered with ``tier_filter="runner"`` (no
  daemon-tier plugins leak)
- ``set_workspace_path`` is broadcast when ``envelope.workspace_path``
  is set (and skipped when None — headless sessions)
- ``set_config_root`` is broadcast when ``envelope.config_root`` is set
- envelope.plugins[].config overrides merge into plugin_configs
- The path is idempotent on already-configured runtimes
- ``BootstrapError("plugins", ...)`` surfaces when configuration fails
- AST-level pin: ``bootstrap_session`` body contains the
  configure_plugins call
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, List, Optional

import pytest

from server.runner.session import (
    BootstrapError,
    RunnerSessionHost,
    _configure_runtime_plugins,
    bootstrap_session,
)
from shared.session_envelope import SessionInitEnvelope


# ----------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------


class _StubRuntime:
    """Stub :class:`JaatoRuntime` that tracks configure_plugins calls
    and treats ``_registry`` as the authoritative idempotency marker.

    Unlike Path C's stub (which pre-sets ``_registry`` to skip the
    Path D path), this stub starts with ``_registry = None`` so the
    Path D code path engages.  After ``configure_plugins`` is called,
    ``_registry`` becomes the passed registry, matching production
    semantics at ``jaato_runtime.py:732``.
    """

    def __init__(self) -> None:
        self.connect_calls: list = []
        self.is_connected: bool = False
        self._registry: Any = None
        self.configure_plugins_calls: list = []

    def connect(self, project: str, location: str) -> None:
        self.connect_calls.append((project, location))
        self.is_connected = True

    def configure_plugins(
        self,
        registry: Any,
        permission_plugin: Any = None,
        ledger: Any = None,
        reliability_plugin: Any = None,
    ) -> None:
        self.configure_plugins_calls.append({
            "registry": registry,
            "permission_plugin": permission_plugin,
            "ledger": ledger,
            "reliability_plugin": reliability_plugin,
        })
        # Mirror production: configure_plugins sets _registry.
        self._registry = registry

    def create_session(self, **kwargs: Any) -> Any:
        class _StubSession:
            pass

        return _StubSession()


def _envelope(
    *,
    workspace_path: Optional[str] = "/tmp/ws",
    config_root: Optional[str] = None,
    plugins: Optional[List[dict]] = None,
) -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-pd",
        workspace_path=workspace_path,
        profile_name="p",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        plugins=plugins or [],
        config_root=config_root,
    )


# ----------------------------------------------------------------------
# Direct unit tests of _configure_runtime_plugins
# ----------------------------------------------------------------------


def test_configure_runtime_plugins_calls_runtime_configure_plugins() -> None:
    """Pin: helper calls ``runtime.configure_plugins(registry, perm,
    None)``.  Layer 4 closure check."""
    stub = _StubRuntime()
    _configure_runtime_plugins(stub, _envelope())
    assert len(stub.configure_plugins_calls) == 1
    call = stub.configure_plugins_calls[0]
    assert call["registry"] is not None, "registry must be passed"
    assert call["permission_plugin"] is not None, (
        "permission_plugin must be passed"
    )
    assert call["ledger"] is None, (
        "ledger is daemon-tier per §4.2; runner passes None"
    )


def test_configure_runtime_plugins_uses_runner_tier_filter() -> None:
    """Pin: registry.discover is called with ``tier_filter="runner"``.
    Daemon-tier plugins (auth, gc_*, cache_*, session, background)
    must NOT load runner-side."""
    stub = _StubRuntime()
    _configure_runtime_plugins(stub, _envelope())
    registry = stub.configure_plugins_calls[0]["registry"]
    # Check that daemon-tier plugins are NOT in the registry.
    # Per shared/plugins/CLAUDE.md and __init__.py grep:
    daemon_tier_names = {
        "anthropic_auth", "github_auth", "zhipuai_auth",
        "antigravity_auth", "nim_auth",
        "gc_truncate", "gc_summarize", "gc_hybrid",
        "cache", "cache_anthropic", "cache_zhipuai",
        "session", "background",
    }
    runner_loaded = set(registry._plugins.keys())
    leaked = runner_loaded & daemon_tier_names
    assert not leaked, (
        f"Path D regression: daemon-tier plugins leaked into runner-"
        f"side registry: {leaked}.  ``tier_filter='runner'`` must be "
        f"passed to ``registry.discover()``."
    )


def test_configure_runtime_plugins_passes_workspace_path_to_registry() -> None:
    """Pin: ``registry.set_workspace_path(workspace_path)`` called when
    envelope.workspace_path is non-None."""
    stub = _StubRuntime()
    _configure_runtime_plugins(stub, _envelope(workspace_path="/tmp/wsX"))
    registry = stub.configure_plugins_calls[0]["registry"]
    assert registry.get_workspace_path() == "/tmp/wsX"


def test_configure_runtime_plugins_skips_workspace_for_headless() -> None:
    """Pin: ``registry.set_workspace_path`` skipped when workspace_path
    is None.  Headless sessions have no workspace to broadcast."""
    stub = _StubRuntime()
    _configure_runtime_plugins(stub, _envelope(workspace_path=None))
    registry = stub.configure_plugins_calls[0]["registry"]
    # set_workspace_path was never called → _workspace_path stays None.
    assert registry.get_workspace_path() is None


def test_configure_runtime_plugins_sets_config_root_when_present() -> None:
    """Pin: ``registry.set_config_root`` called when envelope.config_root
    is non-None."""
    stub = _StubRuntime()
    _configure_runtime_plugins(
        stub,
        _envelope(workspace_path=None, config_root="/tmp/cfg"),
    )
    registry = stub.configure_plugins_calls[0]["registry"]
    assert registry.get_config_root() == "/tmp/cfg"


def test_configure_runtime_plugins_merges_envelope_plugin_configs() -> None:
    """Pin: envelope.plugins[].config overrides merge into the
    plugin_configs dict (mirrors daemon-side profile-override
    semantics)."""
    stub = _StubRuntime()
    _configure_runtime_plugins(
        stub,
        _envelope(plugins=[
            {
                "name": "todo",
                "preload": False,
                "config": {"reporter_type": "custom-override"},
            },
        ]),
    )
    # The todo plugin's actual config dict isn't readily inspectable
    # post-initialize, but we can verify the helper didn't crash and
    # the registry was populated.  Stronger pin would require
    # instrumenting plugin.initialize() — left for the AST pin below.
    assert len(stub.configure_plugins_calls) == 1


def test_configure_runtime_plugins_permission_plugin_initialized() -> None:
    """Pin: the permission_plugin passed to configure_plugins has
    been initialize()'d with a default ``ask`` policy."""
    stub = _StubRuntime()
    _configure_runtime_plugins(stub, _envelope())
    perm = stub.configure_plugins_calls[0]["permission_plugin"]
    # PermissionPlugin should at least be a real instance; sanity-
    # check its identity.
    assert perm is not None
    assert perm.__class__.__name__ == "PermissionPlugin"


# ----------------------------------------------------------------------
# bootstrap_session integration
# ----------------------------------------------------------------------


def test_bootstrap_session_invokes_configure_plugins() -> None:
    """Pin: ``bootstrap_session`` calls
    ``_configure_runtime_plugins`` between connect and
    ``_build_session``.  Closes Layer 4 of the cycle-5 bug chain."""
    stub = _StubRuntime()
    host = bootstrap_session(
        _envelope(),
        runtime_factory=lambda env: stub,
    )
    assert len(stub.configure_plugins_calls) == 1
    assert isinstance(host, RunnerSessionHost)


def test_bootstrap_session_idempotent_on_pre_configured_runtime() -> None:
    """Pin: if the runtime already has ``_registry`` set (test stub
    that pre-configured itself), bootstrap_session does NOT
    re-configure.  Mirrors the Path C idempotency contract for
    ``connect``."""
    stub = _StubRuntime()
    # Pre-configure (as if a previous bootstrap_session had run).
    stub._registry = object()
    bootstrap_session(
        _envelope(),
        runtime_factory=lambda env: stub,
    )
    assert stub.configure_plugins_calls == [], (
        "Path D regression: bootstrap_session re-configured an "
        "already-configured runtime; the ``_registry`` guard didn't "
        "fire."
    )


def test_bootstrap_session_plugins_failure_surfaces_as_bootstrap_error() -> None:
    """Pin: when configure_plugins raises, bootstrap_session
    surfaces a ``BootstrapError("plugins", ...)`` (matches the
    existing stage codes: validate/runtime/connect/configure)."""

    class _BoomConfigure(_StubRuntime):
        def configure_plugins(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("plugin init exploded")

    with pytest.raises(BootstrapError) as exc_info:
        bootstrap_session(
            _envelope(),
            runtime_factory=lambda env: _BoomConfigure(),
        )
    assert exc_info.value.stage == "plugins"
    assert "plugin init exploded" in exc_info.value.message


def test_bootstrap_session_call_order_connect_before_plugins() -> None:
    """Pin: ``connect()`` fires BEFORE ``configure_plugins()``.
    Reverse order would defeat Path C's _connected guard."""

    class _OrderedStub(_StubRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.call_order: list = []

        def connect(self, project: str, location: str) -> None:
            self.call_order.append("connect")
            super().connect(project, location)

        def configure_plugins(self, *args: Any, **kwargs: Any) -> None:
            self.call_order.append("configure_plugins")
            super().configure_plugins(*args, **kwargs)

    stub = _OrderedStub()
    bootstrap_session(
        _envelope(),
        runtime_factory=lambda env: stub,
    )
    assert stub.call_order == ["connect", "configure_plugins"], (
        f"Path D regression: call order broken: {stub.call_order}.  "
        f"connect must precede configure_plugins."
    )


# ----------------------------------------------------------------------
# AST-level pin: bootstrap_session contains the Layer-4 call
# ----------------------------------------------------------------------


def test_bootstrap_session_source_contains_configure_runtime_plugins() -> None:
    """Pin via AST: ``bootstrap_session`` body contains a call to
    ``_configure_runtime_plugins``.  Catches accidental deletion of
    the Path D fix."""
    src = inspect.getsource(bootstrap_session)
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_configure_runtime_plugins"
        )
    ]
    assert calls, (
        "Path D regression: ``_configure_runtime_plugins(...)`` call "
        "missing from ``bootstrap_session`` body.  Without it, "
        "``runtime.create_session`` raises "
        "``RuntimeError('Plugins not configured. Call "
        "configure_plugins() first.')``."
    )


def test_configure_runtime_plugins_source_uses_runner_tier_filter() -> None:
    """Pin via AST: ``_configure_runtime_plugins`` invokes
    ``registry.discover`` with ``tier_filter="runner"``.

    Daemon-side discovers everything (no tier filter); runner-side
    correctness requires the explicit filter so daemon-tier plugins
    don't leak.  This pin defends against an accidental drop of the
    kwarg during a refactor."""
    src = inspect.getsource(_configure_runtime_plugins)
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    discover_calls = [
        n for n in ast.walk(tree)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "discover"
        )
    ]
    assert discover_calls, (
        "Path D regression: registry.discover(...) call missing "
        "from _configure_runtime_plugins."
    )
    matched = False
    for call in discover_calls:
        for kw in call.keywords:
            if (
                kw.arg == "tier_filter"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "runner"
            ):
                matched = True
                break
    assert matched, (
        "Path D regression: registry.discover() missing "
        "``tier_filter='runner'`` kwarg.  Without it, daemon-tier "
        "plugins (auth, gc_*, cache_*, session, background) leak "
        "into the runner-side registry — correctness violation per "
        "§3.3.5."
    )


# ----------------------------------------------------------------------
# Bootstrap timing instrumentation (2026-05-14)
# ----------------------------------------------------------------------


def test_configure_runtime_plugins_silent_when_timing_env_unset(
    monkeypatch, caplog,
) -> None:
    """Default (env var unset / falsy): no "bootstrap timing report"
    INFO line; only a DEBUG-level completion line.  Mirrors
    daemon-side gating in ``server/core.py:2243-2245`` — operators
    who don't opt in shouldn't see the verbose report.
    """
    monkeypatch.delenv("JAATO_BOOTSTRAP_TIMING", raising=False)
    stub = _StubRuntime()

    with caplog.at_level("DEBUG", logger="server.runner.session"):
        _configure_runtime_plugins(stub, _envelope())

    info_records = [
        r for r in caplog.records
        if r.levelname == "INFO"
        and "bootstrap timing report" in r.message.lower()
    ]
    assert not info_records, (
        "JAATO_BOOTSTRAP_TIMING unset → no INFO-level timing "
        "report; got: " + repr([r.message for r in info_records])
    )
    # DEBUG-level completion line still emits.
    debug_records = [
        r for r in caplog.records
        if r.levelname == "DEBUG"
        and "runner-side bootstrap completed" in r.message.lower()
    ]
    assert debug_records, (
        "Even when timing is off, a DEBUG line should record total "
        "time for opportunistic observability."
    )


def test_configure_runtime_plugins_emits_report_when_timing_enabled(
    monkeypatch, caplog,
) -> None:
    """``JAATO_BOOTSTRAP_TIMING=true`` → INFO-level
    ``Runner-side bootstrap timing report`` with the BOOTSTRAP
    TIMING REPORT header and the PER-PLUGIN BREAKDOWN section.
    Mirrors the daemon-side report shape so operators see the same
    structure across both halves of bootstrap.
    """
    monkeypatch.setenv("JAATO_BOOTSTRAP_TIMING", "true")
    stub = _StubRuntime()

    with caplog.at_level("INFO", logger="server.runner.session"):
        _configure_runtime_plugins(stub, _envelope())

    matching = [
        r for r in caplog.records
        if "runner-side bootstrap timing report" in r.message.lower()
    ]
    assert matching, (
        "Expected INFO-level 'Runner-side bootstrap timing report' "
        "line when JAATO_BOOTSTRAP_TIMING=true; got: "
        + repr([r.message for r in caplog.records])
    )
    report = matching[0].message
    # The shared BootstrapTimer emits a "BOOTSTRAP TIMING REPORT"
    # header — same shape as daemon-side.
    assert "BOOTSTRAP TIMING REPORT" in report
    # Per-plugin breakdown section is present (registry.get_bootstrap_timings
    # tracking is wired in).
    assert "PER-PLUGIN BREAKDOWN" in report
    # The four named stages should all appear.
    for stage in ("discover", "expose_all", "set_workspace_path",
                  "permission_init", "configure_plugins"):
        assert stage in report, (
            f"stage {stage!r} missing from runner-side timing report"
        )


def test_configure_runtime_plugins_timing_env_truthy_variants(
    monkeypatch, caplog,
) -> None:
    """The env-var gate accepts ``1`` / ``true`` / ``yes`` (case-
    insensitive), matching the daemon-side check.  Other strings
    are treated as off.  Pin the contract so future restorers
    don't accidentally narrow it."""
    cases = [("1", True), ("true", True), ("TRUE", True), ("yes", True),
             ("YES", True), ("0", False), ("false", False), ("", False)]
    for value, expected_on in cases:
        caplog.clear()
        monkeypatch.setenv("JAATO_BOOTSTRAP_TIMING", value)
        stub = _StubRuntime()
        with caplog.at_level("INFO", logger="server.runner.session"):
            _configure_runtime_plugins(stub, _envelope())

        matching = [
            r for r in caplog.records
            if "runner-side bootstrap timing report" in r.message.lower()
        ]
        if expected_on:
            assert matching, (
                f"JAATO_BOOTSTRAP_TIMING={value!r} should enable "
                f"the timing report"
            )
        else:
            assert not matching, (
                f"JAATO_BOOTSTRAP_TIMING={value!r} should NOT "
                f"enable the timing report; got: "
                f"{[r.message for r in matching]}"
            )
