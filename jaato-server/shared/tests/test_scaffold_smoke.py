"""Smoke test for the jaato-scaffold engine (introspect / explain / validate / new).

Exercises the one introspection core and all three verbs end-to-end against
tmp workspaces — the same paths the CLI drives.  Heavier than the AST-only
contract guards (it imports the registry + resolves profiles), so it lives in
the full test suite rather than the minimal contract-guards job.
"""

import argparse
import py_compile

import pytest

from shared.scaffold import introspect, build, explain
from shared.scaffold import validate as V


# ----------------------------------------------------------- introspect core

def test_providers_all_declare_knobs_and_caps():
    P = introspect.providers()
    assert len(P) >= 15
    for name, info in P.items():
        assert info.knobs is not None, f"{name} missing knobs"
        assert info.capabilities is not None, f"{name} missing capabilities"


def test_resolve_provider_hyphen_underscore():
    info = introspect.resolve_provider("zhipuai-openai")
    assert info is not None and info.dir_name == "zhipuai_openai"


def test_accepts_rejects_typo_but_allows_opaque():
    k = introspect.providers()["openrouter"].knobs
    assert k.accepts("api_params", "temperature")
    assert not k.accepts("api_params", "temprature")   # the silent-ignore typo
    assert k.accepts("routing", "any_key_at_all")       # opaque pass-through


def test_gc_strategies_present():
    assert set(introspect.gc_strategies()) >= {"gc_budget", "gc_truncate"}


# ---------------------------------------------------------------- new verb

def _args(**kw):
    ns = argparse.Namespace()
    defaults = dict(archetype=None, workspace=None, provider=None, model=None,
                    set=None, agents=None, force=True, json=False,
                    recoverable=False)
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(ns, k, v)
    return ns


def test_new_profile_set_is_valid_by_construction(tmp_path):
    rc = build.run(_args(
        archetype="profile-set", workspace=str(tmp_path), set="nebius_test",
        provider="nebius", model="deepseek-ai/DeepSeek-V4-Pro",
        agents="codegen,context"))
    assert rc == 0  # emit-then-validate returned clean


@pytest.mark.parametrize("arch", ["client", "fire", "cascade", "observer"])
def test_new_client_archetypes_compile(tmp_path, arch):
    ws = tmp_path / arch
    rc = build.run(_args(
        archetype=arch, workspace=str(ws), provider="nebius", model="m"))
    assert rc == 0
    py_compile.compile(str(ws / f"run_{arch}.py"), doraise=True)


@pytest.mark.parametrize("arch", ["client", "fire", "cascade", "observer"])
def test_new_client_recoverable_emits_recovery_client(tmp_path, arch):
    # --recoverable swaps IPCClient → IPCRecoveryClient for every archetype.
    ws = tmp_path / f"{arch}_recov"
    rc = build.run(_args(archetype=arch, workspace=str(ws), provider="nebius",
                         model="m", recoverable=True))
    assert rc == 0
    src = (ws / f"run_{arch}.py").read_text()
    assert "IPCRecoveryClient(" in src           # the recoverable client
    assert "on_status_change=_on_status" in src  # wired the status callback
    assert "def _on_status" in src               # and defined it
    py_compile.compile(str(ws / f"run_{arch}.py"), doraise=True)


def test_new_client_default_is_plain_ipcclient(tmp_path):
    ws = tmp_path / "plain"
    build.run(_args(archetype="client", workspace=str(ws), provider="nebius",
                    model="m"))
    src = (ws / "run_client.py").read_text()
    assert "IPCClient(" in src and "IPCRecoveryClient" not in src
    assert "_on_status" not in src


def test_new_rejects_unknown_provider(tmp_path):
    rc = build.run(_args(
        archetype="profile-set", workspace=str(tmp_path), set="s",
        provider="notaprovider", model="m", agents="a"))
    assert rc == 2  # fail-loud, no guessed fallback


# ------------------------------------------------------------- validate verb

def test_validate_catches_silent_ignore_failures(tmp_path):
    pdir = tmp_path / ".jaato" / "profiles"
    (pdir / "setX").mkdir(parents=True)
    (pdir / "_base_x.yaml").write_text(
        "name: _base_x\ndescription: b\nplugins: []\n")
    (pdir / "setX" / "x.yaml").write_text(
        "name: x\ninherits: [_base_x]\nprovider: openrouter\nmodel: m\n"
        "plugins: [cli, nonexistent_plugin]\n"
        "plugin_configs:\n  openrouter:\n    api_params:\n"
        "      temprature: 0.0\n    bogus_top: 1\n")
    diags = V.validate_workspace(str(tmp_path), profile_set="setX", only="x")
    codes = {d.code for d in diags}
    assert "unknown_knob" in codes      # temprature + bogus_top
    assert "unknown_plugin" in codes    # nonexistent_plugin


def test_validate_env_catches_bad_provider_and_set(tmp_path):
    (tmp_path / ".jaato" / "profiles" / "realset").mkdir(parents=True)
    (tmp_path / ".env").write_text(
        "JAATO_PROVIDER=nope\nJAATO_PROFILE_SET=missing_set\nMODEL_NAME=m\n")
    codes = {d.code for d in V.validate_env(str(tmp_path))}
    assert "unknown_provider" in codes
    assert "unknown_profile_set" in codes


def test_validate_env_clean_when_refs_resolve(tmp_path):
    (tmp_path / ".jaato" / "profiles" / "realset").mkdir(parents=True)
    (tmp_path / ".env").write_text(
        "JAATO_PROVIDER=nebius\nJAATO_PROFILE_SET=realset\n")
    assert V.validate_env(str(tmp_path)) == []


def test_new_profile_set_emits_runnable_env(tmp_path):
    build.run(_args(
        archetype="profile-set", workspace=str(tmp_path), set="nebius_x",
        provider="nebius", model="m", agents="a"))
    assert "JAATO_PROFILE_SET=nebius_x" in (tmp_path / ".env").read_text()


# --------------------------------------------------------------- env vars

def test_env_scan_resolves_constant_indirected_keys():
    EV = introspect.env_vars()
    # const-indirected (ENV_X = "JAATO_NEBIUS_API_KEY"; os.getenv(ENV_X))
    assert "JAATO_NEBIUS_API_KEY" in EV
    assert EV["JAATO_NEBIUS_API_KEY"].category == "provider:nebius"
    assert "JAATO_GC_THRESHOLD" in EV  # plugin:gc knob


def test_composed_env_has_provider_and_framework_knobs(tmp_path):
    build.run(_args(
        archetype="client", workspace=str(tmp_path), provider="nebius", model="m"))
    env = (tmp_path / ".env").read_text()
    assert "MODEL_NAME=m" in env
    assert env.count("MODEL_NAME") == 1            # not active+commented dup
    assert "# JAATO_NEBIUS_API_KEY=" in env        # provider block
    assert "# JAATO_GC_THRESHOLD=" in env          # plugin:gc knob included


def test_provider_auth_resolution_is_introspected_and_explained():
    P = introspect.providers()
    nebius = P["nebius"]
    assert nebius.auth, "nebius should declare an auth resolution chain"
    kinds = [a.kind for a in nebius.auth]
    assert kinds[0] == "api_key_param"   # profile api_key wins
    assert "stored" in kinds
    data, text = explain.provider("nebius")
    assert "credentials" in text and "resolution order" in text
    assert any(a["kind"] == "env" for a in data["auth"])


def test_validate_unread_env_var_is_info_never_error(tmp_path):
    (tmp_path / ".jaato" / "profiles").mkdir(parents=True)
    (tmp_path / ".env").write_text("JAATO_TOTALLY_MADE_UP=1\n")
    diags = V.validate_env(str(tmp_path))
    hints = [d for d in diags if d.code == "unread_env_var"]
    assert hints and hints[0].severity == "info"
    assert all(d.severity != "error" for d in diags)


def test_validate_clean_profile_has_no_errors(tmp_path):
    pdir = tmp_path / ".jaato" / "profiles"
    (pdir / "setY").mkdir(parents=True)
    (pdir / "_base_y.yaml").write_text(
        "name: _base_y\ndescription: b\nplugins: []\n")
    (pdir / "setY" / "y.yaml").write_text(
        "name: y\ninherits: [_base_y]\nprovider: openrouter\nmodel: m\n"
        "plugins: []\n"
        "plugin_configs:\n  openrouter:\n    api_key: pass://x\n"
        "    api_params:\n      temperature: 0.0\n    routing:\n      sort: price\n")
    diags = V.validate_workspace(str(tmp_path), profile_set="setY", only="y")
    errors = [d for d in diags if d.severity == "error"]
    assert errors == [], [d.message for d in errors]
