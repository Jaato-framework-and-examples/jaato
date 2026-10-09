"""A subagent inherits the secret its parent already resolved (#1605).

A session's ``pass://`` credential is resolved DAEMON-side (the daemon is
unconfined and can run ``pass``) and reaches the runner in plaintext on the
session envelope.  A subagent the runner spawns re-expanded ITS profile's
``plugin_configs`` inside the runner, where a confined process cannot run
``pass``: the resolver's constructor failed, discovery cached an empty
registry, and the same URI that worked for the parent reached the provider
boundary literally and was refused.  Granting the sandbox the password
store would have exposed every secret the account owns to deliver one.

Now the daemon records the URIs it resolved for the session and ships them
as ``SessionInitEnvelope.inherited_secrets``; the runner installs them at
bootstrap and ``_resolve_secret_uri`` answers those URIs before discovery.

And an empty registry says WHY: "not installed" only when no entry point
exists; an installed provider that constructed nothing is named with each
construction failure it reported.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, FrozenSet, Optional
from unittest.mock import MagicMock, patch

import pytest

from jaato_server.shared.plugins.subagent import config as cfg
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_CONFIG = "jaato-server/jaato_server/shared/plugins/subagent/config.py"
_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"
_RUNNER_SESSION = "jaato-server/jaato_server/server/runner/session.py"
_CORE = "jaato-server/jaato_server/server/core.py"

_URI = "pass://jaato/openrouter/api-key"
_VALUE = "sk-or-v1-resolved-on-the-daemon-0123456789"


REVERSIONS = [
    Reversion(
        target=_CONFIG,
        find=(
            "    inherited = inherited_secret(value)\n"
            "    if inherited is not None:\n"
            "        return inherited\n"
        ),
        replace="",
        test="test_a_subagent_naming_the_parents_uri_resolves_in_a_sandbox",
        because="the runner re-resolves inside the sandbox, where it cannot",
    ),
    Reversion(
        target=_SPAWN,
        find="        inherited_secrets=_inherited_secrets_of(server, plugin_config_secrets),\n",
        replace="",
        test="test_the_envelope_ships_the_secrets_the_daemon_resolved",
        because="the daemon resolves the URI and never tells the runner",
    ),
    Reversion(
        target=_CORE,
        find="        with capture_resolved_secrets(captured):\n",
        replace="        if True:\n",
        test="test_the_session_env_records_the_uris_it_resolved",
        because="a pass:// in the .env or the profile env: is never inherited",
    ),
    Reversion(
        target=_RUNNER_SESSION,
        find="    set_inherited_secrets(inherited)\n",
        replace="    if inherited:\n        set_inherited_secrets(inherited)\n",
        test="test_a_reused_slot_does_not_answer_with_the_last_sessions_secrets",
        because="a pool slot keeps answering with the previous session's values",
    ),
    Reversion(
        target=_RUNNER_SESSION,
        find="    _install_inherited_secrets(envelope)\n    # #1215: build",
        replace="    # #1215: build",
        test="test_bootstrap_installs_the_inherited_secrets",
        because="the map arrives on the envelope and nobody installs it",
    ),
    Reversion(
        target=_CONFIG,
        find="    if not matches:\n        logger.warning(",
        replace="    if True:\n        logger.warning(",
        test="test_an_installed_provider_that_constructs_nothing_is_not_called_missing",
        because="the operator is told to install a package that is installed",
    ),
]


class _PassResolver:
    @property
    def schemes(self) -> FrozenSet[str]:
        return frozenset({"pass"})

    def resolve(self, scheme: str, path: str, key: Optional[str] = None) -> str:
        return _VALUE


@pytest.fixture(autouse=True)
def _clean_state():
    cfg.reset_secret_resolvers()
    cfg.set_inherited_secrets(None)
    yield
    cfg.reset_secret_resolvers()
    cfg.set_inherited_secrets(None)


def _daemon_with_pass(monkeypatch) -> None:
    """The daemon's registry: ``pass`` resolves."""
    monkeypatch.setattr(cfg, "_resolvers", {"pass": _PassResolver()})


def _sandbox_with_no_backend(monkeypatch) -> None:
    """A confined runner: discovery ran and constructed nothing."""
    monkeypatch.setattr(cfg, "_resolvers", {})


# ----------------------------------------------------------------------
# Daemon side
# ----------------------------------------------------------------------


def _stub_server(profile: Any) -> MagicMock:
    server = MagicMock()
    server._profile = profile
    server.config_root = None
    server._cascade_budget_pool = None
    server._suppress_base_instructions = frozenset()
    server._session_env = {}
    return server


def test_the_envelope_ships_the_secrets_the_daemon_resolved(monkeypatch) -> None:
    from jaato_server.server.runner_spawn import build_session_envelope

    _daemon_with_pass(monkeypatch)
    profile = SimpleNamespace(
        provider="openrouter", model="m", plugins=[],
        plugin_configs={"openrouter": {"api_key": _URI}},
        preloaded_plugins=set(), system_instructions=None, gc=None, env={},
    )
    env = build_session_envelope(
        server=_stub_server(profile), session_id="s",
        workspace_path="/tmp/ws", profile_name="p",
    )
    assert env.plugin_configs["openrouter"]["api_key"] == _VALUE
    assert env.inherited_secrets == {_URI: _VALUE}


def test_only_uris_the_session_named_are_captured(monkeypatch) -> None:
    """``${VAR}`` and literal values are not recorded; nothing outside the block."""
    _daemon_with_pass(monkeypatch)
    captured: dict = {}
    with cfg.capture_resolved_secrets(captured):
        out = cfg.expand_plugin_configs({
            "openrouter": {"api_key": _URI, "base_url": "https://x"},
            "cli": {"max_workers": 4},
        })
    cfg.expand_variables(_URI)  # outside the block: not recorded
    assert out["openrouter"]["api_key"] == _VALUE
    assert captured == {_URI: _VALUE}


def test_the_session_env_records_the_uris_it_resolved(monkeypatch) -> None:
    """``JaatoServer._resolve_session_env`` (the .env and profile ``env:``)
    records what it resolved, and the envelope ships it."""
    from jaato_server.server.core import JaatoServer
    from jaato_server.server.runner_spawn import _inherited_secrets_of

    _daemon_with_pass(monkeypatch)
    server = JaatoServer.__new__(JaatoServer)
    server._session_env_resolved = False
    server.env_file = None
    server._workspace_path = None
    server._env_overrides = {}
    server._profile = SimpleNamespace(env={"OPENROUTER_KEY": _URI}, trace=None)
    server._resolve_session_env()
    assert server._session_env["OPENROUTER_KEY"] == _VALUE
    assert _inherited_secrets_of(server, {}) == {_URI: _VALUE}


def test_the_envelope_round_trips_the_map_and_its_repr_hides_it() -> None:
    env = SessionInitEnvelope(
        session_id="s", workspace_path=None, profile_name="p",
        provider_name="openrouter", model_name="m",
        inherited_secrets={_URI: _VALUE},
    )
    assert SessionInitEnvelope.from_dict(env.to_dict()).inherited_secrets == {
        _URI: _VALUE}
    assert _VALUE not in repr(env)
    # An older daemon sends none: the runner then resolves as before.
    d = env.to_dict()
    d.pop("inherited_secrets")
    assert SessionInitEnvelope.from_dict(d).inherited_secrets == {}


# ----------------------------------------------------------------------
# Runner side
# ----------------------------------------------------------------------


def _envelope(inherited: Optional[dict]) -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="s", workspace_path=None, profile_name="",
        provider_name="openrouter", model_name="m",
        inherited_secrets=dict(inherited or {}),
    )


def test_a_subagent_naming_the_parents_uri_resolves_in_a_sandbox(monkeypatch) -> None:
    """The subagent's plugin_configs expansion gets the parent's value."""
    from jaato_server.server.runner.session import _install_inherited_secrets

    _sandbox_with_no_backend(monkeypatch)
    _install_inherited_secrets(_envelope({_URI: _VALUE}))
    # Exactly what ``SubagentPlugin`` does at spawn, in the runner.
    expanded = cfg.expand_plugin_configs(
        {"openrouter": {"api_key": _URI}}, {}, "/tmp/ws")
    assert expanded["openrouter"]["api_key"] == _VALUE
    assert not cfg.looks_like_unresolved_secret_uri(
        expanded["openrouter"]["api_key"])


def test_a_uri_the_parent_never_resolved_still_reaches_the_boundary_literally(
    monkeypatch,
) -> None:
    """Inheritance is not a resolver: an unknown URI is untouched, so the
    strict refusal at the provider boundary still applies to it."""
    from jaato_server.server.runner.session import _install_inherited_secrets

    _sandbox_with_no_backend(monkeypatch)
    _install_inherited_secrets(_envelope({_URI: _VALUE}))
    other = "pass://jaato/someone-else/api-key"
    assert cfg.expand_variables(other) == other
    assert cfg.looks_like_unresolved_secret_uri(other)


def test_a_reused_slot_does_not_answer_with_the_last_sessions_secrets(
    monkeypatch,
) -> None:
    from jaato_server.server.runner.session import _install_inherited_secrets

    _sandbox_with_no_backend(monkeypatch)
    _install_inherited_secrets(_envelope({_URI: _VALUE}))
    _install_inherited_secrets(_envelope(None))  # next session, older daemon
    assert cfg.inherited_secret(_URI) is None
    assert cfg.expand_variables(_URI) == _URI


def test_bootstrap_installs_the_inherited_secrets() -> None:
    """``bootstrap_session`` calls the installer (the #1133 call-site check:
    a test that calls the installer directly passes on a tree where nothing
    invokes it)."""
    src = (Path(__file__).resolve().parents[2]
           / "server" / "runner" / "session.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "bootstrap_session")
    called = {
        n.func.id for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_install_inherited_secrets" in called


def test_inherited_values_are_redacted_under_their_uri() -> None:
    from jaato_server.shared.secret_redaction import (
        configure_redaction_sources, current_redactor, reset_redaction_sources,
    )

    try:
        configure_redaction_sources(
            {}, plugin_configs={"mcp": {"token": _VALUE}},
            inherited_secrets={_URI: _VALUE},
        )
        assert _URI in current_redactor().names
    finally:
        reset_redaction_sources()


# ----------------------------------------------------------------------
# An empty registry says why
# ----------------------------------------------------------------------


class _EP:
    def __init__(self, factory):
        self.name = "secret_resolvers"
        self.value = "jaato_premium.secret_resolvers:get_resolvers"
        self.dist = SimpleNamespace(name="jaato-premium")
        self._factory = factory

    def load(self):
        return self._factory


class _EPs:
    def __init__(self, eps):
        self._eps = eps

    def select(self, group=None, name=None):
        return [e for e in self._eps if name is None or e.name == name]


def _warnings(caplog) -> list:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]


def test_an_installed_provider_that_constructs_nothing_is_not_called_missing(
    caplog,
) -> None:
    def get_resolvers(errors=None):
        errors.append(("PassResolver", PermissionError(13, "pass version")))
        return []

    with patch("importlib.metadata.entry_points",
               return_value=_EPs([_EP(get_resolvers)])), \
            caplog.at_level(logging.WARNING):
        assert cfg._discover_secret_resolvers_uncached() == {}
    msgs = _warnings(caplog)
    assert len(msgs) == 1
    assert "PassResolver" in msgs[0] and "pass version" in msgs[0]
    assert "INSTALLED" in msgs[0]
    assert "NOT INSTALLED" not in msgs[0]
    assert "Install the package" not in msgs[0]


def test_a_factory_that_reports_nothing_is_still_not_called_missing(caplog) -> None:
    """The original zero-argument factory is called as before."""
    with patch("importlib.metadata.entry_points",
               return_value=_EPs([_EP(lambda: [])])), \
            caplog.at_level(logging.WARNING):
        assert cfg._discover_secret_resolvers_uncached() == {}
    msgs = _warnings(caplog)
    assert len(msgs) == 1 and "NOT INSTALLED" not in msgs[0]
    assert "jaato-premium" in msgs[0]


def test_no_entry_point_is_called_not_installed(caplog) -> None:
    with patch("importlib.metadata.entry_points", return_value=_EPs([])), \
            caplog.at_level(logging.WARNING):
        assert cfg._discover_secret_resolvers_uncached() == {}
    assert any("NOT INSTALLED" in m for m in _warnings(caplog))
