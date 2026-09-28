"""The authoring half of ``jaato-scaffold`` stands without the introspection half (#1267).

#1267 proposes shipping ``jaato-scaffold new`` / ``integration`` in jaato-sdk
while ``explain`` / ``validate`` stay with jaato-server.  A measurement on the
issue found three things in the way, all fixed here and pinned below:

A. ``build`` (the ``new`` verb) imported ``explain``, ``introspect`` and
   ``validate`` at module level, so every archetype loaded the introspection
   surface, including those that use none of it.  It now imports only
   ``archetypes``, ``authoring_facts`` and ``authoring_contracts``.
B. ``new`` read provider contracts and env vars by parsing jaato-server's
   SOURCE with ``ast``: invisible in ``sys.modules``, and absent from an
   install without jaato-server.  ``authoring_contracts`` reads the live tree
   when it is there and a checked-in snapshot when it is not.  The snapshot
   must equal the live projection (C) and must make ``new`` write the same
   text the live tree does (D).
E. ``subagent.config`` (the ``SubagentProfile`` schema) pulled in the whole
   subagent plugin through an eager package ``__init__``, and, when looking
   for premium profiles, the whole runtime.  Both are cut: the ``__init__``
   is lazy, and the premium-path helper lives in ``premium_content``.

Import footprints are measured in a fresh interpreter, because a module this
process already imported says nothing about what a cold import loads.  The
child inherits ``PYTHONPATH``, so under the reversion meta-guard it imports
the sandbox's copy.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from jaato_server.shared.scaffold import authoring_contracts as contracts
from jaato_server.shared.scaffold import build
from jaato_server.shared.tests.reversion import Reversion

_BUILD = "jaato-server/jaato_server/shared/scaffold/build.py"
_CONTRACTS = "jaato-server/jaato_server/shared/scaffold/authoring_contracts.py"
_SUBAGENT_INIT = "jaato-server/jaato_server/shared/plugins/subagent/__init__.py"
_CONFIG = "jaato-server/jaato_server/shared/plugins/subagent/config.py"

REVERSIONS = [
    Reversion(
        target=_BUILD,
        find="from . import authoring_facts as _facts\n",
        replace="from . import authoring_facts as _facts\nfrom . import introspect\n",
        because="a module-level introspection import puts it back on every archetype",
        test="test_build_loads_no_introspection",
    ),
    Reversion(
        target=_SUBAGENT_INIT,
        find='PLUGIN_TIER = "runner"\n',
        replace='PLUGIN_TIER = "runner"\nfrom .plugin import SubagentPlugin, create_plugin\n',
        because="an eager package import makes the profile schema load the plugin",
        test="test_subagent_config_loads_neither_the_plugin_nor_the_runtime",
    ),
    Reversion(
        target=_CONFIG,
        find="from jaato_server.shared.premium_content import _get_premium_content_path",
        replace="from jaato_server.shared.jaato_runtime import _get_premium_content_path",
        because="asking for a premium directory must not import the runtime",
        test="test_subagent_config_loads_neither_the_plugin_nor_the_runtime",
    ),
    Reversion(
        target=_CONTRACTS,
        find='    return category.startswith("provider:") or default not in (None, "")\n',
        replace="    return True\n",
        because="a projection change not followed by --write leaves a stale snapshot",
        test="test_the_snapshot_is_the_live_projection",
    ),
    Reversion(
        target=_CONTRACTS,
        find="                return True if opaque else key in keys\n",
        replace="                return key in keys\n",
        because="the snapshot must accept a key exactly where the live contract does",
        test="test_the_snapshot_answers_what_the_live_tree_answers",
    ),
]


# ------------------------------------------------------------------ helpers


def _cold_modules(code: str) -> set:
    """``jaato_server.*`` modules loaded by *code* in a fresh interpreter."""
    probe = (
        "import sys, json\n"
        f"{code}\n"
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.startswith('jaato_server'))))\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe],
                          capture_output=True, text=True, check=True)
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


@pytest.fixture
def snapshot_mode(monkeypatch):
    """Force the snapshot branch on a tree where the live one exists."""
    assert contracts.source() == "live", "the live tree must be here to compare"
    monkeypatch.setattr(contracts, "_FORCE_SNAPSHOT", True)
    assert contracts.source() == "snapshot"
    yield


# ---------------------------------------------------------------------- A, E


def test_build_loads_no_introspection():
    loaded = _cold_modules("import jaato_server.shared.scaffold.build")
    leaked = {m for m in loaded if m.rsplit(".", 1)[-1]
              in ("introspect", "explain", "validate", "dossier")}
    assert not leaked, (
        f"importing the `new` verb loaded {sorted(leaked)}; the authoring "
        "commands must stand without the introspection modules (#1267)")


def test_subagent_config_loads_neither_the_plugin_nor_the_runtime():
    loaded = _cold_modules(
        "from jaato_server.shared.plugins.subagent import config\n"
        "config._discover_premium_profiles()")
    assert "jaato_server.shared.plugins.subagent.plugin" not in loaded
    assert "jaato_server.shared.jaato_runtime" not in loaded


def test_the_subagent_package_still_exports_its_names():
    import jaato_server.shared.plugins.subagent as pkg

    for name in pkg._LAZY_IMPORTS:
        assert getattr(pkg, name) is not None, name
        assert name in dir(pkg), name
    assert set(pkg._LAZY_IMPORTS) == set(pkg.__all__)
    plugin = pkg.create_plugin()
    assert type(plugin).__name__ == "SubagentPlugin"
    with pytest.raises(AttributeError):
        getattr(pkg, "no_such_name")


def test_the_runtime_still_reexports_the_premium_helper():
    """``jaato_runtime`` calls it by its own global name, and tests patch it there."""
    from jaato_server.shared import jaato_runtime, premium_content

    assert jaato_runtime._get_premium_content_path is premium_content._get_premium_content_path
    assert jaato_runtime._premium_content_cache is premium_content._premium_content_cache


# ---------------------------------------------------------------------- C, D


def test_the_snapshot_is_the_live_projection():
    expected = contracts.render_snapshot(contracts.build_snapshot())
    current = contracts.SNAPSHOT_FILE.read_text(encoding="utf-8")
    assert current == expected, (
        f"{contracts.SNAPSHOT_FILE.name} differs from the tree.  Regenerate it: "
        "python -m jaato_server.shared.scaffold.authoring_contracts --write "
        "(the repo's pre-commit hook does this for you: "
        "git config core.hooksPath .githooks)")


def test_the_snapshot_answers_what_the_live_tree_answers(monkeypatch):
    live = contracts.providers()
    names = sorted(live)
    assert names, "no providers found"

    def answers():
        out = {}
        for name in names:
            info = contracts.resolve_provider(name)
            hyphen = contracts.resolve_provider(name.replace("_", "-"))
            accepts = {}
            if info.knobs is not None:
                for lyr in live[name].knobs.layers:
                    for key in sorted(lyr.keys) + ["__not_a_declared_knob__"]:
                        accepts[(lyr.layer, key)] = info.knobs.accepts(lyr.layer, key)
                accepts[("__no_such_layer__", "x")] = info.knobs.accepts("__no_such_layer__", "x")
            out[name] = (
                info.dir_name,
                hyphen.dir_name,
                build._primary_key_env_var(info, name),
                accepts,
                build._compose_env(name, [f"MODEL_NAME=m"]),
                build._set_profile_yaml("worker", name, "m"),
            )
        out[None] = (build._compose_env(None, []),
                     contracts.resolve_provider("__no_such_provider__"))
        return out

    live_answers = answers()
    monkeypatch.setattr(contracts, "_FORCE_SNAPSHOT", True)
    assert contracts.source() == "snapshot"
    snap_answers = answers()
    for key in live_answers:
        assert snap_answers[key] == live_answers[key], (
            f"`new` writes different text for provider {key!r} from the "
            "snapshot than from the live tree")


def test_a_snapshot_of_another_shape_is_refused(monkeypatch, tmp_path):
    bad = tmp_path / "snap.json"
    bad.write_text(json.dumps({"snapshot_version": 999}), encoding="utf-8")
    monkeypatch.setattr(contracts, "SNAPSHOT_FILE", bad)
    monkeypatch.setattr(contracts, "_SNAPSHOT_CACHE", None)
    monkeypatch.setattr(contracts, "_FORCE_SNAPSHOT", True)
    with pytest.raises(RuntimeError, match="snapshot_version 999"):
        contracts.providers()


def test_the_snapshot_is_never_generated_from_itself(snapshot_mode):
    with pytest.raises(RuntimeError, match="source tree is not available"):
        contracts.build_snapshot()
