"""The six findings from the PR #1040 review, each pinned against its cause.

Three of them were the NEW checks being wrong about the tree, which is the
one failure mode a validator can least afford: a false finding spends the
credibility the true ones need.  Two of those told an author to delete
config that was working — the exact inversion of the silent-ignore family
this work joins.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from shared.plugins.subagent.config import SubagentProfile
from shared.scaffold import build, introspect, validate


@pytest.fixture(scope="module")
def env():
    return introspect.providers(), introspect.plugins()


def _run(env, **kw):
    providers, plugins = env
    kw.setdefault("plugins", ["cli"])
    profile = SubagentProfile(name="t", description="d", **kw)
    return validate.validate_profile(profile, providers=providers,
                                     plugins=plugins, gc_names=[])


def _codes(diags, *codes):
    return [(d.code, d.where) for d in diags if d.code in codes]


def _policy(**policy):
    return {"permission": {"policy": policy}}


# --- 1. the nested descent must not call working config a typo -------------

#: Keys ``PermissionPolicy.from_config`` demonstrably reads and the plugin's
#: schema did not declare when the descent shipped.  Each is a real knob an
#: author could have been told to delete.
_READ_BUT_UNDECLARED = (
    ("policy.cwd", dict(cwd="/srv/work")),
    ("policy.sanitization.custom_blocked_commands",
     dict(sanitization={"custom_blocked_commands": ["rm"]})),
    ("policy.sanitization.path_scope.resolve_symlinks",
     dict(sanitization={"path_scope": {"resolve_symlinks": False}})),
)


@pytest.mark.parametrize("path,policy", _READ_BUT_UNDECLARED)
def test_a_nested_knob_the_plugin_reads_is_never_called_a_typo(env, path, policy):
    diags = _run(env, plugin_configs=_policy(**policy))
    assert _codes(diags, "unknown_knob") == [], (
        f"{path} is read by policy.py and was reported as unknown")


@pytest.mark.parametrize("path,policy", _READ_BUT_UNDECLARED)
def test_those_three_are_now_declared_so_explain_shows_them(env, path, policy):
    # The other half: the schema WAS incomplete, and `explain plugin
    # permission` is where an author would look for these.
    _data, text = __import__("shared.scaffold.explain", fromlist=["x"]).plugin(
        "permission")
    assert path.rsplit(".", 1)[-1] in text
    assert _codes(_run(env, plugin_configs=_policy(**policy)),
                  "unknown_knob", "undeclared_knob") == []


def test_a_nested_name_nothing_reads_is_still_reported(env):
    # The check must still do its job: evidence-aware is not evidence-free.
    diags = _run(env, plugin_configs=_policy(
        sanitization={"path_scope": {"utter_nonsense": 1}}))
    assert _codes(diags, "unknown_knob") == [
        ("unknown_knob",
         "plugin_configs.permission.policy.sanitization.path_scope.utter_nonsense")]


def test_nested_evidence_sees_a_read_off_a_local():
    # A nested key is read off `ps_cfg`/`san_cfg`, which the TOP-LEVEL
    # receiver set excludes on purpose — so the nested scanner is a third
    # scanner, not a widening of that one.
    nested = introspect.plugin_nested_config_read_sites("permission")
    top = introspect.plugin_config_read_sites("permission")
    assert nested["resolve_symlinks"].startswith("policy.py:")
    assert "resolve_symlinks" not in top
    assert nested.get("utterly_invented_key") is None


def test_an_out_of_tree_plugin_reports_not_checked():
    # None is "not scanned", which callers must not render as "not read".
    assert introspect.plugin_nested_config_read_sites("no_such_plugin") is None


# --- 2. the exemption its sibling applies ----------------------------------

def test_always_initialized_plugins_are_not_missing_plugins(env):
    # introspection's tools are CORE and reach every wire whatever plugins:
    # says — `plugin_config_without_plugin` already exempts the set.
    diags = _run(env, plugin_configs=_policy(
        whitelist={"tools": ["list_tools", "get_tool_schemas"]}))
    assert _codes(diags, "permission_rule_without_plugin", "unknown_tool") == []


def test_the_exemption_is_read_from_the_registry_not_respelled():
    from shared.plugins.registry import PluginRegistry
    assert (validate._always_initialized_plugins()
            == frozenset(PluginRegistry._ALWAYS_INITIALIZE_PLUGINS))


def test_a_genuinely_absent_plugin_is_still_reported(env):
    diags = _run(env, plugin_configs=_policy(
        whitelist={"tools": ["writeNewFile"]}))
    assert [c for c, _ in _codes(diags, "permission_rule_without_plugin")] == [
        "permission_rule_without_plugin"]


# --- 3. a flag accepted where it cannot be honoured ------------------------

def test_only_archetypes_with_a_binding_placeholder_take_a_profile():
    from shared.scaffold._client_templates import TEMPLATES
    for name, entry in TEMPLATES.items():
        expected = "__SESSION_BINDING__" in entry[1]
        assert build._archetype_takes_a_session_binding(name) is expected, name


def test_the_answer_is_derived_from_the_template_not_a_list():
    # cascade/sweep name a profile PER STAGE; client/host-tools bind one.
    assert build._archetype_takes_a_session_binding("client") is True
    assert build._archetype_takes_a_session_binding("cascade") is False


def test_a_cascade_profile_is_refused_rather_than_discarded(tmp_path, capsys):
    args = SimpleNamespace(workspace=str(tmp_path), profile="worker",
                           provider=None, model=None, set=None)
    code, _, _ = build._resolve_client_binding(args, "cascade", "ipc")
    assert code == 2
    assert "--profile" in capsys.readouterr().out


# --- 4. the reproducibility banner carries the binding --------------------

def test_provenance_names_the_profile_it_was_given(tmp_path):
    args = SimpleNamespace(workspace=str(tmp_path), profile="solo",
                           recoverable=False, url=None, ca=None, set=None,
                           agents=None)
    line = build._provenance(args, "client", "ipc", None, None)
    assert "--profile solo" in line
    # Re-running the printed command must not hit the missing-binding path.
    assert "--provider" not in line


# --- 5. the set that made resolution succeed is persisted -----------------

def test_the_forced_set_reaches_the_generated_env(tmp_path):
    args = SimpleNamespace(profile="scoped", set="myset")
    assert build._profile_set_env_lines(args) == ["JAATO_PROFILE_SET=myset"]


def test_a_set_without_a_profile_is_not_this_archetypes_business(tmp_path):
    assert build._profile_set_env_lines(
        SimpleNamespace(profile=None, set="myset")) == []
    assert build._profile_set_env_lines(
        SimpleNamespace(profile="scoped", set=None)) == []


def test_an_existing_selector_is_never_retargeted(tmp_path):
    envf = tmp_path / ".env"
    envf.write_text("JAATO_PROFILE_SET=already\n", encoding="utf-8")
    written: list = []
    plan = SimpleNamespace(write=lambda *a, **k: written.append(a))
    build._append_profile_set(plan, envf,
                              SimpleNamespace(profile="scoped", set="myset"))
    assert written == []


def test_a_selector_is_added_to_an_env_that_lacks_it(tmp_path):
    envf = tmp_path / ".env"
    envf.write_text("ANTHROPIC_API_KEY=\n", encoding="utf-8")
    written: list = []
    plan = SimpleNamespace(write=lambda *a, **k: written.append(a))
    build._append_profile_set(plan, envf,
                              SimpleNamespace(profile="scoped", set="myset"))
    assert written and "JAATO_PROFILE_SET=myset" in written[0][1]


# --- 6. the dependency finding claims only what it knows ------------------

#: The phrasing the first wording carried — a runtime consequence a static
#: import closure cannot support.  `azure_identity_available()` wraps its
#: import in try/except on a path only `auth: aad` takes.
_OVERCLAIM = "will fail at connect()"


def test_the_dependency_message_asserts_no_runtime_consequence(env, monkeypatch):
    from shared.scaffold import dependencies
    monkeypatch.setattr(dependencies, "provider_import_gaps",
                        lambda n: (("azure",), ("pip install x",))
                        if n == "echo" else ((), ()))
    diags = _run(env, provider="echo", model="m")
    msgs = [d.message for d in diags if d.code == "provider_dependency_missing"]
    assert len(msgs) == 1
    assert _OVERCLAIM not in msgs[0]
    assert "not a certain failure" in msgs[0]
    assert "pip install x" in msgs[0]      # still imperative


# --- re-review: the five block-level keys, and the surfaces below them ----

#: Read straight off ``initialize(config)`` (``plugin.py:508-544``) — the
#: surface an author writes and ``explain plugin`` publishes.  Declaring
#: them is this PR's own thesis applied to the plugin it audited: someone
#: configuring a webhook approval channel had no route to the shape but the
#: source.
_BLOCK_LEVEL = ("agent_name", "config_path", "workspace_path",
                "channel_type", "channel_config")


@pytest.mark.parametrize("key", _BLOCK_LEVEL)
def test_the_block_an_author_writes_is_declared(key):
    from shared.scaffold import explain

    data, text = explain.plugin("permission")
    assert key in {c["name"] for c in data["config"]}, key
    assert key in text


def test_channel_type_publishes_the_set_it_accepts(env):
    # An enum is worth more than a type here: `console`/`webhook`/`queue`/
    # `file` is what config_loader.validate_config enforces.
    diags = _run(env, plugin_configs={"permission": {
        "channel_type": "carrier_pigeon"}})
    assert [c for c, _ in _codes(diags, "invalid_knob_value")] == [
        "invalid_knob_value"]


def test_channel_config_is_an_open_key_set(env):
    # Surface 3: the names belong to the CHANNEL, not to this plugin, so
    # nothing here judges them.  additionalProperties is the honest marker.
    diags = _run(env, plugin_configs={"permission": {"channel_config": {
        "endpoint": "https://x", "headers": {}, "auth_token": "t",
        "timeout": 5, "base_path": "/tmp", "poll_interval": 1,
        "whatever_the_channel_wants": True}}})
    assert _codes(diags, "unknown_knob", "undeclared_knob") == []


def test_a_typo_beside_the_new_keys_is_still_caught(env):
    diags = _run(env, plugin_configs={"permission": {"channel_typ": "webhook"}})
    assert [c for c, _ in _codes(diags, "unknown_knob")] == ["unknown_knob"]


# --- the census is a measurement, and says so -----------------------------

def test_the_census_separates_the_framework_surface():
    # A wrong method name in the first draft returned [] and silently
    # reclassified ~20 framework reads as schema gaps — the census's own
    # headline number, wrong in the alarming direction.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "plugin_schema_census", Path("scripts/plugin_schema_census.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    keys = mod.framework_injected_keys()
    assert set(keys) == {"agent_name", "config_root", "session_id",
                         "workspace_path"}, keys


def test_the_census_is_not_wired_as_a_guard():
    # It is a measurement that has to precede a ratchet, not one itself:
    # "every key a plugin reads is declared" is four questions in this tree.
    ci = Path(".github/workflows/ci-tests.yml").read_text(encoding="utf-8")
    assert "plugin_schema_census" not in ci
