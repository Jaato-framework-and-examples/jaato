"""``jaato-scaffold`` stops answering "yes, it's wired" for a plugin the
runner will discard (issue #917).

The tooling split runs straight through the diagnostic.
``introspect.plugins()`` discovers with **no** tier filter, and so does
the daemon-side registry; the runner, the runner's ``__main__`` and the
embedded client all pass ``tier_filter="runner"``, which excludes any
plugin whose package declares no ``PLUGIN_TIER``.

So the surface an author consults to answer *is my plugin wired* was
answering **yes** — complete with a provenance line — for a plugin the
session would come up without.  Since ``plugins()`` must keep
discovering unfiltered (it is an inventory, not a session), the fix is
that the row says so, and that a profile naming such a plugin fails
validation rather than passing it.
"""
from types import SimpleNamespace

import pytest

from shared.scaffold import explain, introspect
from shared.scaffold.validate import validate_profile


def _info(name: str, *, tier, tier_missing: bool) -> introspect.PluginInfo:
    return introspect.PluginInfo(
        name=name, kind="tool", tier=tier, tier_missing=tier_missing,
        source=f"jaato-{name} ({name}.plugin)", builtin=False,
    )


# ---- introspect: every in-tree plugin is annotated --------------------------

def test_no_builtin_plugin_is_reported_as_missing_a_tier():
    """The marker must be inert in a healthy tree.

    ``test_plugin_tier_partition`` already fails the build on an
    unannotated in-tree plugin, so any hit here means the introspect
    walk reads the annotation from the wrong place — which would print
    an alarming, wrong warning on every ``explain plugins``.
    """
    offenders = [
        name for name, pi in introspect.plugins().items()
        if pi.builtin and pi.tier_missing
    ]
    assert offenders == [], (
        f"built-in plugins reported as un-annotated: {offenders}. "
        "Either the build gate is broken or introspect reads PLUGIN_TIER "
        "from a different module than _lookup_module_tier does."
    )


# ---- explain plugins: the row carries the warning ---------------------------

def test_the_row_says_the_runner_will_not_load_it(monkeypatch):
    monkeypatch.setattr(
        introspect, "plugins",
        lambda: {"m365": _info("m365", tier=None, tier_missing=True)},
    )
    data, text = explain.plugins()

    assert data["m365"]["tier_missing"] is True, "--json needs the same fact"
    assert "no PLUGIN_TIER" in text
    assert "will not load in the runner" in text
    # The footer must name the fix, not merely flag the row.
    assert 'PLUGIN_TIER = "runner"' in text
    assert "__init__.py" in text


def test_an_annotated_plugin_adds_no_marker_and_no_footer(monkeypatch):
    """A healthy inventory renders exactly as before — the footer is
    printed only when something in the table needs it."""
    monkeypatch.setattr(
        introspect, "plugins",
        lambda: {"m365": _info("m365", tier="runner", tier_missing=False)},
    )
    data, text = explain.plugins()

    assert data["m365"]["tier_missing"] is False
    assert "PLUGIN_TIER" not in text
    assert "will not load" not in text


def test_the_real_inventory_prints_no_footer():
    """End-to-end on the installed tree: no built-in triggers it."""
    _, text = explain.plugins()
    assert "will not load in the runner" not in text


# ---- validate: a profile naming such a plugin fails -------------------------

def _diags(plugin_names, plugins):
    prof = SimpleNamespace(provider=None, model="m", plugins=plugin_names)
    return validate_profile(
        prof, providers={}, plugins=plugins, gc_names=[],
    )


def test_profile_naming_an_unannotated_plugin_is_an_error():
    """It is an error rather than a warning because the session is
    already broken: every tool the profile asked for is absent, and the
    profile is otherwise valid, so nothing else in the pipeline will
    say a word."""
    diags = _diags(
        ["m365"], {"m365": _info("m365", tier=None, tier_missing=True)},
    )
    hits = [d for d in diags if d.code == "plugin_missing_tier"]
    assert len(hits) == 1
    assert hits[0].severity == "error"
    assert hits[0].where == "plugins.m365"
    assert "PLUGIN_TIER" in hits[0].message
    assert "without its tools" in hits[0].message


def test_an_annotated_plugin_validates_clean():
    diags = _diags(
        ["m365"], {"m365": _info("m365", tier="runner", tier_missing=False)},
    )
    assert [d for d in diags if d.code == "plugin_missing_tier"] == []


def test_an_uninstalled_plugin_is_still_reported_as_unknown_only():
    """The two findings are different problems and must not double-fire:
    an absent plugin has no tier to be missing."""
    diags = _diags(["nope"], {})
    codes = {d.code for d in diags}
    assert "unknown_plugin" in codes
    assert "plugin_missing_tier" not in codes
