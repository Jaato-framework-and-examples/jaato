"""``jaato-scaffold explain`` answers from a snapshot when jaato-server is absent.

#1267 moved ``jaato-scaffold`` into jaato-sdk and shipped a snapshot of the
facts ``new`` reads, so an SDK-only install could author.  ``explain`` was left
answering only through a running daemon, although nearly every topic is a
deterministic render of the installed code.  jaato-server now renders those
topics ahead of time (:mod:`jaato_server.shared.scaffold.explain_snapshot`)
into ``jaato-sdk/jaato_sdk/scaffold/explain_snapshot.json``, and the SDK shell
answers from it (:mod:`jaato_sdk.scaffold.explain_snapshot`).

What this guards:

- the checked-in file is what the tree renders now (the drift guard);
- an SDK-only ``explain`` answers from it, with the same ``--json`` the live
  renderer gives and a line naming the snapshot's jaato-server version, and an
  accepted alias finds its canonical name;
- what the snapshot cannot answer still falls through to the refusal, which
  now says why (``live_only``, the workspace);
- the generator refuses an environment that would put something other than
  jaato-server in the file: a foreign plugin, a contributed topic, an in-tree
  plugin that did not load, a path of the generating machine;
- the render runs with a fresh ``$HOME``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from jaato_sdk.scaffold import explain_snapshot as reader
from jaato_server.shared.scaffold import explain_snapshot as gen
from jaato_server.shared.scaffold import introspection_verbs as iv
from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.tests.test_scaffold_ships_with_the_sdk_1267 import _sdk_only

_CLI = "jaato-sdk/jaato_sdk/scaffold/cli.py"
_READER = "jaato-sdk/jaato_sdk/scaffold/explain_snapshot.py"
_GEN = "jaato-server/jaato_server/shared/scaffold/explain_snapshot.py"

REVERSIONS = [
    Reversion(
        target=_CLI,
        find="        rc = _explain_from_snapshot(args)\n",
        replace="        rc = None\n",
        because="an SDK-only `explain` goes back to refusing every topic "
                "although the answer ships in the wheel",
        test="test_sdk_only_explain_answers_from_the_snapshot",
    ),
    Reversion(
        target=_READER,
        find='        canonical = data["aliases"].get(scope or "", {}).get(name)\n',
        replace="        canonical = name\n",
        because="a spelling the live lookup accepts (zhipuai-openai) would miss "
                "in the snapshot, so the two installs answer differently",
        test="test_an_accepted_spelling_finds_the_canonical_render",
    ),
    Reversion(
        target=_GEN,
        find='    env.update(HOME=str(home), JAATO_RELEASE_CHECK="off",\n',
        replace='    env.update(JAATO_RELEASE_CHECK="off",\n',
        because="the render would read the generating developer's home "
                "(prompt_library lists ~/.claude/skills) and ship it",
        test="test_the_render_runs_in_a_fresh_home",
    ),
    Reversion(
        target=_GEN,
        find="                     if not getattr(i, \"builtin\", True))\n",
        replace="                     if False)\n",
        because="a plugin from another distribution would be claimed for "
                "every jaato-sdk install",
        test="test_a_foreign_plugin_is_refused",
    ),
    Reversion(
        target=_GEN,
        find="    missing = sorted(set(_in_tree_plugins()) - set(plugins))\n",
        replace="    missing = []\n",
        because="an environment without an extra (pexpect) would ship "
                "listings that silently lack interactive_shell",
        test="test_an_in_tree_plugin_that_did_not_load_is_refused",
    ),
    Reversion(
        target=_GEN,
        find="        hit = next((p for p in paths if p in blob), None)\n",
        replace="        hit = None\n",
        because="a topic that renders this machine's paths would ship them as "
                "facts about the framework",
        test="test_a_rendering_naming_the_machine_is_refused",
    ),
]


@pytest.fixture
def no_contributions(monkeypatch):
    """No contributed topics, whatever this environment has installed."""
    monkeypatch.setattr(iv, "_discover_external_topics", lambda: [])


def _plugin(builtin=True):
    return SimpleNamespace(builtin=builtin, source="dist-x (x.plugin)")


# ------------------------------------------------------------------ drift


def test_the_checked_in_snapshot_is_what_the_tree_renders():
    assert gen.main(["--check"]) == 0, (
        f"{gen.SNAPSHOT_FILE.name} is stale; run `{gen.REGENERATE}`")


def test_every_snapshottable_topic_is_in_it():
    data = reader.load()
    assert data is not None
    want = [s for s, spec in iv._SCOPES.items()
            if not spec.live_only and spec.kind != "workspace"]
    assert data["scopes"] == want
    assert set(data["live_only"]) == {s for s, spec in iv._SCOPES.items()
                                      if spec.live_only}
    for scope, spec in iv._SCOPES.items():
        if spec.kind == "named" and not spec.live_only:
            assert spec.names is not None, scope


# --------------------------------------------------------------- SDK-only


def test_sdk_only_explain_answers_from_the_snapshot(tmp_path):
    rc, out, err, leaked = _sdk_only("explain", "providers", cwd=tmp_path)
    assert rc == 0, err
    assert "snapshot shipped with jaato-sdk" in out
    assert reader.server_version() in out
    assert leaked == "[]"


def test_sdk_only_json_is_the_live_renderers(tmp_path):
    rc, out, err, _ = _sdk_only("explain", "plugin", "cli", "--json",
                                cwd=tmp_path)
    assert rc == 0, err
    ok, data, _text, _ = iv.render_topic("plugin", "cli")
    assert ok
    assert json.loads(out) == json.loads(json.dumps(data, default=str))


def test_an_accepted_spelling_finds_the_canonical_render():
    assert reader.resolve("provider", "zhipuai-openai") == "provider zhipuai_openai"
    assert reader.answer("event", "agent.output") is not None
    assert reader.answer("provider", "no-such-provider") is None


@pytest.mark.parametrize("argv,why", [
    (("explain", "releases"), "PyPI"),
    (("explain", "sets"), "reads your workspace"),
    (("explain", "no-such-topic"), "answers:"),
])
def test_what_the_snapshot_lacks_still_refuses_and_says_why(tmp_path, argv, why):
    rc, out, err, leaked = _sdk_only(*argv, cwd=tmp_path)
    assert rc == 2
    assert "pip install jaato-server" in err
    assert why in err, err
    assert leaked == "[]"


# -------------------------------------------------------------- generator


def test_the_render_runs_in_a_fresh_home(tmp_path, monkeypatch):
    monkeypatch.setenv("JAATO_PROFILE_SET", "leaked")
    env = gen._clean_env(tmp_path)
    assert env["HOME"] == str(tmp_path)
    assert not any(k.startswith("JAATO_") and k != "JAATO_RELEASE_CHECK"
                   for k in env)


def test_a_foreign_plugin_is_refused(no_contributions):
    plugins = {n: _plugin() for n in gen._in_tree_plugins()}
    gen._refuse_foreign(plugins)                       # all in-tree: fine
    plugins["moon_phase"] = _plugin(builtin=False)
    with pytest.raises(gen.SnapshotRefused, match="moon_phase"):
        gen._refuse_foreign(plugins)


def test_a_contributed_topic_is_refused(monkeypatch):
    monkeypatch.setattr(iv, "_discover_external_topics",
                        lambda: [SimpleNamespace(name="reactors")])
    plugins = {n: _plugin() for n in gen._in_tree_plugins()}
    with pytest.raises(gen.SnapshotRefused, match="reactors"):
        gen._refuse_foreign(plugins)


def test_an_in_tree_plugin_that_did_not_load_is_refused(no_contributions):
    names = gen._in_tree_plugins()
    assert "interactive_shell" in names and "cli" in names
    plugins = {n: _plugin() for n in names if n != "interactive_shell"}
    with pytest.raises(gen.SnapshotRefused, match="interactive_shell"):
        gen._refuse_foreign(plugins)


def test_a_rendering_naming_the_machine_is_refused():
    clean = {"plugins": {"data": {}, "text": "nothing local"}}
    gen._refuse_machine_paths(clean)
    leaky = {"plugins": {"data": {}, "text": f"see {Path.home()}/x"},
             "pool": {"data": {"cwd": os.getcwd()}, "text": ""}}
    with pytest.raises(gen.SnapshotRefused, match="live_only"):
        gen._refuse_machine_paths(leaky)
