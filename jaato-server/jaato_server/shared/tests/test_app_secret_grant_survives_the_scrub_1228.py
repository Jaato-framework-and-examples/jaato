"""A credential the daemon resolved from ``app://`` survives the secret scrub (#1228).

``GH_TOKEN`` is in the default ``scrub_secret_env`` set (#863), and a web-coder
workspace has no profile in which to write ``!GH_TOKEN``.  So a workspace owner
who bound a GitHub account (``GH_TOKEN=app://github``, resolved by the daemon
at every spawn, #1226) got a token that was stripped before ``gh`` ran.

The fix is a GRANT that follows the resolution, hop by hop:

1. ``JaatoServer`` records the names it resolved in this pass
   (``granted_env_names()``), and only the ones that resolved;
2. the names ride beside the env: ``SessionInitEnvelope.granted_env_names``
   at bootstrap, the ``session.reload_env`` payload on every reload;
3. the runner's ``apply_session_env`` records them, only for names present
   in the env it applied, and replaces them on every call;
4. ``cli`` (both execution paths) and ``interactive_shell`` keep those names
   through the scrub; ``mcp`` does not.

Every hop is driven here through the real function, because a grant that is
recorded and never read (or read and never shipped) is the #735 shape:
configured, rendered, and inert.

The one security property is pinned at the scrub itself: ``keep`` is an EXACT,
case-sensitive match.  The obvious alternative, turning grants into ``!NAME``
exemptions, goes through ``matches_secret``, which is case-insensitive, so a
model that writes ``anthropic_api_key=app://github`` into ``.env`` would keep
the real ``ANTHROPIC_API_KEY`` too.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from jaato_server.server.runner import session as runner_session
from jaato_server.shared.secret_scrub import (
    DEFAULT_SECRET_ENV_PATTERNS, granted_env_names, scrub_env,
    set_granted_env_names,
)
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion

_SCRUB = "jaato-server/jaato_server/shared/secret_scrub.py"
_CLI = "jaato-server/jaato_server/shared/plugins/cli/plugin.py"
_SHELL = "jaato-server/jaato_server/shared/plugins/interactive_shell/plugin.py"
_MCP = "jaato-server/jaato_server/shared/mcp_context_manager.py"
_RUNNER_SESSION = "jaato-server/jaato_server/server/runner/session.py"
_RPC = "jaato-server/jaato_server/server/runner/rpc.py"
_ENVELOPE = "jaato-server/jaato_server/shared/session_envelope.py"
_SPAWN = "jaato-server/jaato_server/server/runner_spawn.py"
_CORE = "jaato-server/jaato_server/server/core.py"

REVERSIONS = [
    Reversion(
        target=_SCRUB,
        find="""        if k in kept or not matches_secret(k, patterns)""",
        replace="""        if not matches_secret(k, list(patterns) + ["!" + n for n in kept])""",
        test="test_keep_is_exact_and_case_sensitive",
        because="grants routed through case-insensitive !NAME exemptions",
    ),
    Reversion(
        target=_CLI,
        find="""                scrub_keep=granted_env_names(),
            )""",
        replace="""            )""",
        test="test_cli_command_keeps_a_granted_name",
        because="cli's run_command path ignores the grant",
    ),
    Reversion(
        target=_CLI,
        find="""                env, self._scrub_secret_env, keep=granted_env_names(),""",
        replace="""                env, self._scrub_secret_env,""",
        test="test_cli_streaming_env_keeps_a_granted_name",
        because="cli's streaming path ignores the grant",
    ),
    Reversion(
        target=_SHELL,
        find="""                scrub_keep=granted_env_names(),
                # Same value as cwd""",
        replace="""                # Same value as cwd""",
        test="test_interactive_shell_spawn_keeps_a_granted_name",
        because="interactive_shell ignores the grant",
    ),
    Reversion(
        target=_MCP,
        find="""        inherited = scrub_env(os.environ, self.scrub_secret_env)""",
        replace="""        inherited = scrub_env(
            os.environ, self.scrub_secret_env,
            keep=__import__("jaato_server.shared.secret_scrub",
                            fromlist=["x"]).granted_env_names(),
        )""",
        test="test_mcp_never_receives_the_grant",
        because="the grant widened to MCP servers",
    ),
    Reversion(
        target=_RUNNER_SESSION,
        find="""        n for n in (granted_env_names or ()) if applied.get(n) is not None""",
        replace="""        n for n in (granted_env_names or ())""",
        test="test_only_names_in_the_applied_env_are_granted",
        because="a grant for a name this session was not given",
    ),
    Reversion(
        target=_RUNNER_SESSION,
        find="""    set_granted_env_names(
        n for n in""",
        replace="""    if granted_env_names: set_granted_env_names(
        n for n in""",
        test="test_a_reload_without_grants_clears_the_previous_ones",
        because="an unbind leaves the old grant in force",
    ),
    Reversion(
        target=_RUNNER_SESSION,
        find="""        envelope.session_env, getattr(envelope, "granted_env_names", None),""",
        replace="""        envelope.session_env,""",
        test="test_bootstrap_applies_the_envelope_grant",
        because="the envelope carries the grant and bootstrap drops it",
    ),
    Reversion(
        target=_RPC,
        find="""apply_session_env(
            dict(session_env), env_name_list(args.get("granted_env_names")),""",
        replace="""apply_session_env(
            dict(session_env),""",
        test="test_reload_handler_applies_the_payload_grant",
        because="session.reload_env carries the grant and the runner drops it",
    ),
    Reversion(
        target=_ENVELOPE,
        find="""            granted_env_names=env_name_list(d.get("granted_env_names")),""",
        replace="""            granted_env_names=[],""",
        test="test_envelope_round_trips_the_grant",
        because="the grant is lost crossing the wire",
    ),
    Reversion(
        target=_SPAWN,
        find="""        granted_env_names=_granted_env_names_of(server),""",
        replace="",
        test="test_build_session_envelope_carries_the_servers_grant",
        because="the producer never fills the envelope field",
    ),
    Reversion(
        target=_CORE,
        find="""            self._app_secret_names.add(key)""",
        replace="""            pass""",
        test="test_the_daemon_records_only_what_resolved",
        because="the daemon never records what it resolved",
    ),
    Reversion(
        target=_CORE,
        find="""            granted_env_names=self.granted_env_names(),""",
        replace="",
        test="test_reload_session_env_sends_the_grant",
        because="a live bind never reaches the runner",
    ),
]


_ECHO = 'sh -c \'echo "${MYAPP_TOKEN:-EMPTY}"\''

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="sh -c")


@pytest.fixture(autouse=True)
def _no_grant():
    """Every test starts and ends with no grant in this process."""
    set_granted_env_names(())
    yield
    set_granted_env_names(())


@pytest.fixture
def pristine(monkeypatch):
    """Isolate ``apply_session_env``'s snapshot and restore ``os.environ``."""
    saved = dict(os.environ)
    monkeypatch.setattr(runner_session, "_PRISTINE_ENVIRON", None)
    yield
    os.environ.clear()
    os.environ.update(saved)
    monkeypatch.setattr(runner_session, "_PRISTINE_ENVIRON", None)


# --------------------------------------------------------------------------
# the scrub
# --------------------------------------------------------------------------

def test_keep_is_exact_and_case_sensitive():
    env = {"GH_TOKEN": "gh", "ANTHROPIC_API_KEY": "real",
           "anthropic_api_key": "planted", "GITHUB_TOKEN": "other"}
    out = scrub_env(env, DEFAULT_SECRET_ENV_PATTERNS,
                    keep={"GH_TOKEN", "anthropic_api_key"})
    assert out.get("GH_TOKEN") == "gh"
    assert out.get("anthropic_api_key") == "planted"
    # the lowercase grant does NOT keep the real provider key
    assert "ANTHROPIC_API_KEY" not in out
    # and a grant is not a glob: GH_TOKEN does not keep GITHUB_TOKEN
    assert "GITHUB_TOKEN" not in out


def test_the_registry_holds_only_non_empty_strings():
    set_granted_env_names(["GH_TOKEN", "", None, 3])
    assert granted_env_names() == frozenset({"GH_TOKEN"})


# --------------------------------------------------------------------------
# the surfaces
# --------------------------------------------------------------------------

def _cli(monkeypatch):
    from jaato_server.shared.plugins.cli.plugin import CLIToolPlugin
    monkeypatch.setenv("MYAPP_TOKEN", "tok-granted")
    plugin = CLIToolPlugin()
    plugin.initialize({})
    return plugin


@posix_only
def test_cli_command_keeps_a_granted_name(monkeypatch):
    plugin = _cli(monkeypatch)
    # control: with no grant, the default set strips it
    before = plugin._execute({"command": _ECHO})
    assert "tok-granted" not in before["stdout"] and "EMPTY" in before["stdout"]

    set_granted_env_names(["MYAPP_TOKEN"])
    after = plugin._execute({"command": _ECHO})
    assert "error" not in after, after
    assert "tok-granted" in after["stdout"]


def test_cli_streaming_env_keeps_a_granted_name(monkeypatch):
    plugin = _cli(monkeypatch)
    env, _ = plugin._build_subprocess_env()
    assert "MYAPP_TOKEN" not in env

    set_granted_env_names(["MYAPP_TOKEN"])
    env, _ = plugin._build_subprocess_env()
    assert env.get("MYAPP_TOKEN") == "tok-granted"


@posix_only
def test_interactive_shell_spawn_keeps_a_granted_name(monkeypatch):
    from jaato_server.shared.plugins.interactive_shell.plugin import (
        InteractiveShellPlugin,
    )
    monkeypatch.setenv("MYAPP_TOKEN", "tok-granted")
    plugin = InteractiveShellPlugin()
    plugin._start_reaper = lambda: None
    plugin.initialize({})
    set_granted_env_names(["MYAPP_TOKEN"])
    try:
        result = plugin._exec_spawn({"command": _ECHO,
                                     "session_name": "grant"})
        assert "tok-granted" in result["output"]
    finally:
        for s in list(plugin._sessions.values()):
            s.close()


def test_mcp_never_receives_the_grant(monkeypatch):
    from jaato_server.shared.mcp_context_manager import ServerConfig
    monkeypatch.setenv("MYAPP_TOKEN", "tok-granted")
    set_granted_env_names(["MYAPP_TOKEN"])
    cfg = ServerConfig(name="s", command="x",
                       scrub_secret_env=list(DEFAULT_SECRET_ENV_PATTERNS))
    assert "MYAPP_TOKEN" not in (cfg.to_stdio_params().env or {})


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------

def test_only_names_in_the_applied_env_are_granted(pristine):
    runner_session.apply_session_env(
        {"GH_TOKEN": "tok"}, ["GH_TOKEN", "ANTHROPIC_API_KEY"])
    assert granted_env_names() == frozenset({"GH_TOKEN"})


def test_a_reload_without_grants_clears_the_previous_ones(pristine):
    runner_session.apply_session_env({"GH_TOKEN": "tok"}, ["GH_TOKEN"])
    assert granted_env_names() == frozenset({"GH_TOKEN"})
    # the account was unbound: the daemon resolved nothing this pass
    runner_session.apply_session_env({"PLAIN": "x"}, [])
    assert granted_env_names() == frozenset()


def test_bootstrap_applies_the_envelope_grant(pristine):
    envelope = SessionInitEnvelope(
        session_id="s", workspace_path="/ws", profile_name="p",
        provider_name="x", model_name="m", plugins=[],
        session_env={"GH_TOKEN": "tok"}, granted_env_names=["GH_TOKEN"],
    )
    runner_session._apply_envelope_session_env(envelope)
    assert granted_env_names() == frozenset({"GH_TOKEN"})


class _ReloadSession:
    is_running = False

    def __init__(self):
        self._session_env = {}

    def reload_provider(self):
        return {"provider": "x", "model": "m", "auth_info": ""}


def test_reload_handler_applies_the_payload_grant(pristine):
    import socket

    from jaato_server.server.runner.rpc import RunnerRPC
    from jaato_server.server.runner.session import RunnerSessionHost

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.close()
    rpc = RunnerRPC(a, lambda name, args: (False, {"error": "none"}))
    rpc._session_host = RunnerSessionHost(
        envelope=SessionInitEnvelope(
            session_id="s", workspace_path="/ws", profile_name="p",
            provider_name="x", model_name="m", plugins=[],
        ),
        runtime=None, session=_ReloadSession(),
    )
    try:
        ok, result = rpc._handle_session_reload_env({
            "session_env": {"GH_TOKEN": "tok"},
            "granted_env_names": ["GH_TOKEN"],
        })
    finally:
        a.close()
    assert ok is True, result
    assert granted_env_names() == frozenset({"GH_TOKEN"})


# --------------------------------------------------------------------------
# the wire
# --------------------------------------------------------------------------

def test_envelope_round_trips_the_grant():
    env = SessionInitEnvelope(
        session_id="s", workspace_path="/ws", profile_name="p",
        provider_name="x", model_name="m", plugins=[],
        granted_env_names=["GH_TOKEN"],
    )
    back = SessionInitEnvelope.from_dict(env.to_dict())
    assert back.granted_env_names == ["GH_TOKEN"]


def test_an_envelope_without_the_field_grants_nothing():
    d = SessionInitEnvelope(
        session_id="s", workspace_path="/ws", profile_name="p",
        provider_name="x", model_name="m", plugins=[],
    ).to_dict()
    d.pop("granted_env_names")
    assert SessionInitEnvelope.from_dict(d).granted_env_names == []
    d["granted_env_names"] = "GH_TOKEN"  # not a list: nothing, not characters
    assert SessionInitEnvelope.from_dict(d).granted_env_names == []


def _profile():
    return SimpleNamespace(
        name="p", description="d", provider="openrouter", model="m",
        plugins=[], preloaded_plugins=set(), plugin_configs={},
        tool_scopes={}, model_tiers={}, gc=None, runtime_limits=None,
        env={}, completion_payload_schema=None, spawn_payload_schema=None,
        completion_processors=[], budget_control=None, quirks={},
        apparmor=False, apparmor_fragments=None,
        system_instructions=None, agent_params={},
        suppress_base_instructions=False, config_root=None, inherits=None,
        icon=None, description_for_model=None,
    )


def test_build_session_envelope_carries_the_servers_grant():
    from jaato_server.server.runner_spawn import build_session_envelope
    server = SimpleNamespace(
        _profile=_profile(), config_root=None,
        _main_agent_id="main", _cascade_driver_id=None,
        granted_env_names=lambda: ["GH_TOKEN"],
    )
    env = build_session_envelope(server=server, session_id="s1",
                                 workspace_path="/tmp/ws", profile_name="p")
    assert env.granted_env_names == ["GH_TOKEN"]


def test_a_server_without_the_accessor_grants_nothing():
    from jaato_server.server.runner_spawn import build_session_envelope
    server = SimpleNamespace(
        _profile=_profile(), config_root=None,
        _main_agent_id="main", _cascade_driver_id=None,
    )
    env = build_session_envelope(server=server, session_id="s2",
                                 workspace_path="/tmp/ws", profile_name="p")
    assert env.granted_env_names == []


# --------------------------------------------------------------------------
# the daemon
# --------------------------------------------------------------------------

def _server(tmp_path, env_lines, answers):
    from jaato_server.server.app_secret import AppSecretAnswer, AppSecretResolver
    from jaato_server.server.core import JaatoServer

    envf = tmp_path / ".env"
    envf.write_text(env_lines)

    def transport(app_id, user, workspace, name, timeout):
        status, value = answers[name]
        return AppSecretAnswer(status=status, value=value, detail="test")

    resolver = AppSecretResolver(owner_of=lambda p: "acme:alice",
                                 transport=transport)
    srv = JaatoServer(env_file=str(envf), provider=None,
                      workspace_path=str(tmp_path), session_id="s1")
    srv.set_app_secret_resolver(resolver)
    return srv


def test_the_daemon_records_only_what_resolved(tmp_path):
    srv = _server(
        tmp_path,
        "GH_TOKEN=app://github\nGL_TOKEN=app://gitlab\nPLAIN_TOKEN=literal\n",
        {"github": ("ok", "tok"), "gitlab": ("unreachable", None)},
    )
    srv._resolve_session_env()
    # the unresolved one was dropped, and the literal is not a grant
    assert srv.granted_env_names() == ["GH_TOKEN"]


def test_reload_session_env_sends_the_grant(tmp_path):
    srv = _server(tmp_path, "GH_TOKEN=app://github\n",
                  {"github": ("ok", "tok")})
    calls = []

    def reload(env, **kwargs):
        calls.append((env, kwargs))
        return {"applied": len(env)}

    srv._runner_rpc = SimpleNamespace(session_reload_env_threadsafe=reload)
    srv.reload_session_env()
    [(env, kwargs)] = calls
    assert env["GH_TOKEN"] == "tok"
    assert kwargs.get("granted_env_names") == ["GH_TOKEN"]
