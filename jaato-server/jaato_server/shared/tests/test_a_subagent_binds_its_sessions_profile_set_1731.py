"""A subagent's profile key: the right file, and a loud empty answer (#1731).

Reported as "a subagent's provider gets an empty api_key": the child profile
carried ``plugin_configs.openrouter.api_key: pass://...``, and the provider
said ``plugin_configs.openrouter.api_key (profile): not set``.  Two defects
on that path, guarded here.  Which one the reporter hit is not settled: on
their host the set's file WAS the one discovered, so the first does not
explain their report, and the second is what makes the next run say.

1. The wrong FILE.  A session created from a qualified ``<set>/<name>``
   ref (#1588) runs with no ``JAATO_PROFILE_SET``, so the runner's subagent
   plugin discovered profiles outside the set and a same-named flat profile
   (no provider, no ``plugin_configs.<provider>``) bound instead.  The
   daemon now derives the set from the ref (``server._profile_set``, on
   creation and on a revive), ships it as
   ``plugin_configs.subagent.profile_set``, both discoveries scan it, and a
   later discovery replaces what the first found instead of only adding.

2. An EMPTY answer passed on as a key.  A secret resolver (or the
   inherited map) answering ``""`` / ``None`` became an empty ``api_key``
   that ``create_provider`` silently did not promote, so the provider's
   "not set" could not be told from a profile that declared no key.  Both
   resolution routes and the provider boundary now refuse it by name, and
   the spawn's INFO line marks each provider key ``<empty>``, and a secret
   URI ``inherited`` (answered by the daemon's map, #1605) or
   ``resolved-here`` (the runner asked a resolver itself).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import patch

import pytest

from jaato_server.server import record_seal
from jaato_server.server.runner_spawn import build_session_envelope
from jaato_server.server.session_manager import SessionManager
from jaato_server.shared.plugins.subagent.plugin import SubagentPlugin
from jaato_server.shared.session_context import (
    reset_config_root, reset_workspace_root, set_config_root,
    set_workspace_root,
)
from jaato_server.shared.session_envelope import BootstrapEnvelope
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/subagent/plugin.py"
_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"
_SM = "jaato-server/jaato_server/server/session_manager.py"
_CFG = "jaato-server/jaato_server/shared/plugins/subagent/config.py"
_RT = "jaato-server/jaato_server/shared/jaato_runtime.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find=(
            "            discovery = discover_profiles(\n"
            "                self._config.profiles_dir,\n"
            "                force_profile_set=self._config.profile_set,\n"
            "            )\n"
        ),
        replace=(
            "            discovery = discover_profiles(\n"
            "                self._config.profiles_dir,\n"
            "            )\n"
        ),
        test="TestThePlugin::test_the_set_file_binds_the_name",
        because="the subagent binds the flat profile of the same name",
    ),
    Reversion(
        target=_PLUGIN,
        find="            if self._config.profiles.get(name) is not profile:\n",
        replace="            if name not in self._config.profiles:\n",
        test="TestThePlugin::test_a_later_discovery_replaces_what_the_first_found",
        because="a profile found before the config root was known shadows "
                "the workspace set's file",
    ),
    Reversion(
        target=_SPAWN,
        find="        _inject_subagent_profile_set(server, plugin_configs_dict)\n",
        replace="",
        test="TestTheEnvelope::test_the_set_reaches_the_subagent_plugin",
        because="the runner never learns which set the session's profile is in",
    ),
    Reversion(
        target=_SM,
        find=(
            "        server._profile_set = _split_profile_ref(\n"
            "            getattr(envelope, \"profile_ref\", None) or \"\")[0]\n"
        ),
        replace="        server._profile_set = None\n",
        test="TestTheDaemon::test_a_qualified_ref_names_its_set",
        because="the daemon never derives the set from the qualified ref",
    ),
    Reversion(
        target=_SM,
        find=(
            "            # #1731: the set of a qualified ref reaches the runner's subagent\n"
            "            # discovery on a revive as it does at creation.\n"
            "            profile_ref=getattr(state, \"profile_ref\", None),\n"
        ),
        replace="",
        test="TestTheDaemon::test_a_revive_carries_the_ref",
        because="a revived session's subagents resolve outside its set again",
    ),
    Reversion(
        target=_CFG,
        find=(
            "    resolved = _non_empty_secret(value, resolved, f\"resolver '{scheme}'\")\n"
        ),
        replace="",
        test="TestAnEmptyAnswer::test_an_empty_resolver_answer_is_refused",
        because="a resolver answering nothing becomes an empty api_key the "
                "provider reports as 'not set'",
    ),
    Reversion(
        target=_CFG,
        find=(
            "        return _non_empty_secret(value, inherited, \"the value the daemon \"\n"
            "                                 \"resolved for this session (#1605)\")\n"
        ),
        replace="        return inherited\n",
        test="TestAnEmptyAnswer::test_an_empty_inherited_value_is_refused",
        because="an empty inherited value becomes an empty api_key",
    ),
    Reversion(
        target=_RT,
        find="        _refuse_empty_profile_api_key(plugin_configs, effective_provider)\n",
        replace="",
        test="TestAnEmptyAnswer::test_a_declared_empty_key_is_named_at_the_provider_boundary",
        because="a declared key that arrived empty reads as 'not set'",
    ),
    Reversion(
        target=_PLUGIN,
        find='            how = "inherited" if inherited_secret(raw) is not None else "resolved-here"\n',
        replace='            how = "inherited"\n',
        test="TestThePlugin::test_a_spawn_says_the_runner_had_to_resolve_it",
        because="the spawn line cannot tell an inherited URI from one the "
                "runner resolved itself",
    ),
    Reversion(
        target=_PLUGIN,
        find=(
            "            self._log_spawn_binding(\n"
            "                agent_id, profile, provider, effective_plugin_configs)\n"
        ),
        replace="",
        test="TestThePlugin::test_a_spawn_names_the_file_it_bound",
        because="a spawn bound to the wrong file reads as a lost credential",
    ),
]

_URI = "pass://jaato/openrouter/api-key"

_FLAT = """\
name: documentalista
description: FLAT
plugins: [file_edit]
"""
_BASE = """\
name: _base_documentalista
description: base
plugins: [file_edit]
"""
_LEAF = f"""\
name: documentalista
description: SET
inherits: [_base_documentalista]
provider: openrouter
model: openai/gpt-4o-mini
plugins: [file_edit]
plugin_configs:
  openrouter:
    api_key: {_URI}
"""


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """The reported layout: a set leaf and a flat file of the same name."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for var in ("JAATO_PROFILE_SET", "JAATO_CONFIG_ROOT", "JAATO_WORKSPACE_ROOT"):
        monkeypatch.delenv(var, raising=False)
    config_token = set_config_root(None)
    workspace_token = set_workspace_root(None)
    record_seal.reset_cache()
    root = tmp_path / "ws"
    profiles = root / ".jaato" / "profiles"
    (profiles / "openrouter").mkdir(parents=True)
    (profiles / "documentalista.yaml").write_text(_FLAT)
    (profiles / "_base_documentalista.yaml").write_text(_BASE)
    (profiles / "openrouter" / "documentalista.yaml").write_text(_LEAF)
    yield root
    record_seal.reset_cache()
    reset_config_root(config_token)
    reset_workspace_root(workspace_token)


def _plugin(config: dict) -> SubagentPlugin:
    plugin = SubagentPlugin()
    plugin.initialize(config)
    return plugin


class TestThePlugin:
    def test_the_set_file_binds_the_name(self, ws):
        plugin = _plugin({
            "profiles_dir": str(ws / ".jaato" / "profiles"),
            "profile_set": "openrouter",
        })
        profile = plugin._config.get_profile("documentalista")
        assert profile.description == "SET"
        assert profile.provider == "openrouter"
        assert profile.plugin_configs["openrouter"]["api_key"] == _URI

    def test_with_no_set_the_flat_file_binds(self, ws):
        # Control: the set is what decides, nothing else changed.
        plugin = _plugin({"profiles_dir": str(ws / ".jaato" / "profiles")})
        assert plugin._config.get_profile("documentalista").description == "FLAT"

    def test_a_later_discovery_replaces_what_the_first_found(self, ws, tmp_path):
        # The first discovery runs before the config root is known and
        # finds a flat ``documentalista`` elsewhere; the one that knows the
        # config root and the set must decide.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "documentalista.yaml").write_text(_FLAT)
        plugin = _plugin({
            "profiles_dir": str(elsewhere), "profile_set": "openrouter",
        })
        assert plugin._config.get_profile("documentalista").description == "FLAT"
        plugin.set_config_root(str(ws / ".jaato"))
        assert plugin._config.get_profile("documentalista").description == "SET"

    def test_an_explicit_profile_still_wins(self, ws):
        plugin = _plugin({
            "profiles_dir": str(ws / ".jaato" / "profiles"),
            "profile_set": "openrouter",
            "profiles": {"documentalista": {
                "description": "EXPLICIT", "plugins": []}},
        })
        plugin.set_config_root(str(ws / ".jaato"))
        assert plugin._config.get_profile("documentalista").description == "EXPLICIT"

    def _spawn_line(self, ws, caplog, monkeypatch) -> str:
        plugin = _plugin({
            "profiles_dir": str(ws / ".jaato" / "profiles"),
            "profile_set": "openrouter",
        })
        profile = plugin._config.get_profile("documentalista")
        plugin._parent_session = None
        # The async path chdirs and exports the workspace; restore both.
        monkeypatch.chdir(ws)
        monkeypatch.setenv("JAATO_WORKSPACE_ROOT", str(ws))
        with caplog.at_level(logging.INFO, logger=plugin_logger_name()):
            _drive_async_spawn(plugin, profile, str(ws))
        lines = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("spawn_subagent subagent_1")]
        assert lines, "the spawn never logged what it bound"
        line = lines[0]
        assert _URI not in line and "api-key" not in line
        assert _SECRET not in line
        return line

    def test_a_spawn_names_the_file_it_bound(self, ws, caplog, monkeypatch):
        line = self._spawn_line(ws, caplog, monkeypatch)
        assert "openrouter/documentalista.yaml" in line
        assert "plugin_configs.openrouter keys=" in line

    def test_a_spawn_says_the_daemon_answered_its_uri(
            self, ws, caplog, monkeypatch, inherited):
        inherited({_URI: _SECRET})
        line = self._spawn_line(ws, caplog, monkeypatch)
        assert "'api_key=pass:// inherited'" in line

    def test_a_spawn_says_the_runner_had_to_resolve_it(
            self, ws, caplog, monkeypatch, inherited):
        # The reporter's open question: a URI the inherited map does not
        # hold goes to the runner's own resolver.  The line must say so.
        inherited({"pass://some/other/uri": _SECRET})
        line = self._spawn_line(ws, caplog, monkeypatch)
        assert "'api_key=pass:// resolved-here'" in line


_SECRET = "sk-or-v1-resolved-on-the-daemon-0123456789"


@pytest.fixture
def inherited():
    """Install an inherited-secrets map for one test, then clear it."""
    from jaato_server.shared.plugins.subagent import config as cfg
    yield cfg.set_inherited_secrets
    cfg.set_inherited_secrets(None)


class TestAnEmptyAnswer:
    """A declared key that comes out empty is refused by name, never
    reported as "not set" (#1731)."""

    def test_an_empty_inherited_value_is_refused(self, inherited):
        from jaato_server.shared.plugins.subagent import config as cfg
        inherited({_URI: ""})
        with pytest.raises(cfg.SecretResolutionError, match="daemon resolved"):
            cfg.expand_plugin_configs({"openrouter": {"api_key": _URI}})

    def test_an_empty_resolver_answer_is_refused(self, inherited):
        from jaato_server.shared.plugins.subagent import config as cfg
        inherited(None)

        class _Silent:
            def resolve(self, scheme, path, key):
                return ""

        with patch.object(cfg, "_discover_secret_resolvers",
                          lambda: {"pass": _Silent()}):
            with pytest.raises(cfg.SecretResolutionError,
                               match="resolver 'pass' answered with an empty"):
                cfg.expand_plugin_configs({"openrouter": {"api_key": _URI}})

    def test_a_real_answer_still_resolves(self, inherited):
        from jaato_server.shared.plugins.subagent import config as cfg
        inherited({_URI: _SECRET})
        out = cfg.expand_plugin_configs({"openrouter": {"api_key": _URI}})
        assert out["openrouter"]["api_key"] == _SECRET

    @staticmethod
    def _runtime() -> Any:
        from jaato_server.shared.jaato_runtime import JaatoRuntime
        from jaato_server.shared.plugins.registry import PluginRegistry
        rt = JaatoRuntime(provider_name="openrouter")
        rt.connect("", "")
        rt.configure_plugins(PluginRegistry())
        return rt

    def test_a_declared_empty_key_is_named_at_the_provider_boundary(self):
        from jaato_server.shared.plugins.subagent.config import (
            SecretResolutionError)
        with pytest.raises(SecretResolutionError,
                           match="declares an api_key .* empty"):
            self._runtime().create_provider(
                "openai/gpt-4o-mini", provider_name="openrouter",
                plugin_configs={"openrouter": {"api_key": ""}})

    def test_an_undeclared_key_reaches_the_provider_as_before(self):
        # Control: no api_key in the profile is the provider's own
        # "not set", not a resolution failure.
        from jaato_server.shared.plugins.model_provider.openrouter.errors import (
            APIKeyNotFoundError)
        with patch.dict("os.environ", {}, clear=False) as env:
            env.pop("JAATO_OPENROUTER_API_KEY", None)
            with patch("jaato_server.shared.plugins.model_provider.openrouter"
                       ".provider.resolve_api_key", lambda: None):
                with pytest.raises(APIKeyNotFoundError):
                    self._runtime().create_provider(
                        "openai/gpt-4o-mini", provider_name="openrouter",
                        plugin_configs={"openrouter": {"api_params": {}}})


def plugin_logger_name() -> str:
    from jaato_server.shared.plugins.subagent import plugin as mod
    return mod.logger.name


def _drive_async_spawn(plugin: SubagentPlugin, profile: Any, ws: str) -> None:
    """Run ``_run_subagent_async`` up to ``create_session`` and stop there."""
    class _Stop(Exception):
        pass

    def _refuse(**_kwargs: Any) -> Any:
        raise _Stop()

    plugin._runtime = SimpleNamespace(create_session=_refuse, registry=None)
    with patch("jaato_server.shared.runtime_limits.assert_inprocess_can_honor",
               lambda _profile: None):
        try:
            plugin._run_subagent_async(
                agent_id="subagent_1", profile=profile, prompt="do it",
                parent_cwd=ws,
            )
        except _Stop:
            pass


class TestTheEnvelope:
    @staticmethod
    def _envelope(profile_set: Any, plugin_configs: Any = None) -> Any:
        profile = SimpleNamespace(
            provider="openrouter", model="m", plugins=[],
            plugin_configs=plugin_configs or {}, preloaded_plugins=set(),
            system_instructions=None, gc=None, env={},
        )
        server = SimpleNamespace(
            _profile=profile, config_root=None, _profile_set=profile_set)
        return build_session_envelope(
            server=server, session_id="s", workspace_path="/tmp/ws",
            profile_name="p")

    def test_the_set_reaches_the_subagent_plugin(self):
        env = self._envelope("openrouter")
        assert env.plugin_configs["subagent"]["profile_set"] == "openrouter"

    def test_no_set_adds_nothing(self):
        assert "subagent" not in self._envelope(None).plugin_configs

    def test_a_profiles_own_value_is_kept(self):
        env = self._envelope(
            "openrouter", {"subagent": {"profile_set": "dumb"}})
        assert env.plugin_configs["subagent"]["profile_set"] == "dumb"


class TestTheDaemon:
    @staticmethod
    def _server(profile_ref: Any) -> Any:
        from jaato_server.server.tests.test_construct_and_initialize_server import (
            _FakeJaatoServer, _FakeSessionManager, _bind_subhelper,
        )
        with patch("jaato_server.server.session_manager.JaatoServer",
                   _FakeJaatoServer):
            helper = _bind_subhelper(_FakeSessionManager())
            server, _ = helper(BootstrapEnvelope(
                session_id="s", workspace_path="/tmp/ws", name="x",
                profile_ref=profile_ref))
        return server

    def test_a_qualified_ref_names_its_set(self):
        assert self._server("openrouter/documentalista")._profile_set == "openrouter"

    def test_a_bare_name_names_no_set(self):
        assert self._server("documentalista")._profile_set is None
        assert self._server(None)._profile_set is None

    def test_a_revive_carries_the_ref(self, ws, tmp_path):
        from jaato_server.server.tests.test_a_qualified_profile_binds_that_file_1588 import (
            _capturing_construct, _manager, _record, SID,
        )
        sm = _manager(tmp_path)
        _record(sm, ws, seal=True, ref="openrouter/documentalista")
        envelopes: List[Any] = []
        with patch.object(SessionManager, "_construct_and_initialize_server",
                          _capturing_construct(envelopes)), \
                patch.object(SessionManager, "_start_workspace_monitor",
                             lambda *a, **k: None):
            sm._load_session_impl(SID, None, str(ws))
        assert envelopes and envelopes[0].profile_ref == "openrouter/documentalista"
