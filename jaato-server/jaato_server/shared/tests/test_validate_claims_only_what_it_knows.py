"""Guard: a ``validate`` finding may state only what the validator knows.

WHY THIS EXISTS (jaato #910, #937).  ``jaato-scaffold validate`` is the tool
the README tells an agent to trust INSTEAD of reading the source, so a finding
that states a false consequence is worse than a vaguer one — the reader acts
on it.  Three messages asserted runtime behaviour the validator cannot observe,
and each was demonstrably false for a real plugin:

* ``unknown_knob`` said an undeclared knob was "silently ignored at runtime".
  ``memory.global_storage_path`` and ``references.exclude_tools`` are both
  READ — ``config.get(...)`` in each plugin's own ``initialize`` — and both
  work.  Told a working knob is ignored, the reasonable response is to delete
  it, which on ``global_storage_path`` silently moves where memories are
  stored.  ``introspect`` already held the other half of the answer.
* ``invalid_knob_value`` said the value "is not rejected at runtime, it is
  silently replaced by a fallback".  ``jaato-m365``, out of tree, ``raise``\\s
  a ``ValueError`` on an unknown ``cloud``.
* ``knob_type_mismatch`` said the value is "assigned without coercion at
  runtime and carried downstream".  The same plugin coerces
  (``int(config.get(...))``) — and ``_check_knob_value``'s OWN docstring said
  "a plugin may coerce", so the message contradicted the reasoning its
  severity was chosen on.

The severities are unchanged and are not what this guards: an enum violation
is an error whether the plugin raises or falls back, because either way the
profile asks for something the plugin does not offer.  What is guarded is the
EVIDENCE each message claims.

A message correction is exactly the kind of thing a later refactor silently
reverts — the string is the deliverable, and no test read it before now (both
old strings survived every one of the 354 scaffold tests).  Hence REVERSIONS
below: put each old claim back, and the matching test must go red.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jaato_server.shared.scaffold import introspect
from jaato_server.shared.scaffold.validate import validate_profile
from jaato_server.shared.tests.reversion import Reversion

#: Phrasings that assert a runtime consequence a JSON Schema cannot support.
#: Each is the literal tail one of the three findings used to carry.
_OVERCLAIMS = (
    "silently ignored at runtime",
    "silently replaced by a fallback",
    "assigned without coercion",
)


REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/validate.py",
        find='f"\'{key}\' is not a declared {cfg_name} config knob — {tail} "\n'
             '        f"(known: {valid})", where=where)',
        replace='f"\'{key}\' is not a declared {cfg_name} config knob "\n'
                '        f"(silently ignored at runtime; known: {valid})",\n'
                '        where=where)',
        test="test_unknown_knob_does_not_claim_the_runtime_ignores_it",
        because="the tail #910 reported — a knob the plugin demonstrably "
                "reads, described to the author as dead",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/validate.py",
        find='f"plugin declares ({valid}) — what happens next is the plugin\'s "\n'
             '            f"choice: it may raise, or fall back silently", where=where)',
        replace='f"plugin declares ({valid}) — the value is not rejected at "\n'
                '            f"runtime, it is silently replaced by a fallback", where=where)',
        test="test_invalid_knob_value_hands_the_consequence_back_to_the_plugin",
        because="the enum tail #937 reported, false for any plugin that "
                "validates its own config and raises",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/validate.py",
        find='f"match the declared type \'{setting.type}\' — the plugin may "\n'
             '            f"coerce it, reject it, or carry it downstream as-is", where=where)',
        replace='f"match the declared type \'{setting.type}\' — assigned without "\n'
                '            f"coercion at runtime and carried downstream", where=where)',
        test="test_knob_type_mismatch_message_agrees_with_its_own_docstring",
        because="the type tail #937 reported, which denied the 'a plugin may "
                "coerce' reasoning its own severity was chosen on",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/validate.py",
        find="    sites = introspect.plugin_config_read_sites(cfg_name)\n",
        replace="    sites = None\n",
        test="test_a_knob_the_plugin_reads_is_reported_as_live_not_ignored",
        because="the read-site evidence #910 asked for, without which every "
                "undeclared knob falls back to the unknown_knob wording",
    ),
]


@pytest.fixture(scope="module")
def env():
    return (introspect.providers(), introspect.plugins(),
            list(introspect.gc_strategies().keys()))


def _validate(env, plugin_configs):
    providers, plugins, gc_names = env
    prof = SimpleNamespace(provider=None, model="m",
                           plugin_configs=plugin_configs)
    return validate_profile(prof, providers=providers, plugins=plugins,
                            gc_names=gc_names)


def _find(diags, code, needle):
    return [d for d in diags if d.code == code and needle in (d.where or "")]


# ---- #910: the knob the plugin actually reads -------------------------------

@pytest.mark.parametrize("plugin,knob", [
    ("memory", "global_storage_path"),      # plugin.py — the issue's case 1
    ("references", "exclude_tools"),        # plugin.py — the issue's case 2
])
def test_a_knob_the_plugin_reads_is_reported_as_live_not_ignored(
        env, plugin, knob):
    """Both knobs work.  The old finding told the author they did nothing."""
    diags = _validate(env, {plugin: {knob: "x"}})
    assert _find(diags, "unknown_knob", knob) == []
    live = _find(diags, "undeclared_knob", knob)
    assert [d.severity for d in live] == ["warn"], diags
    assert "reads a config key by that name" in live[0].message
    # The evidence is quoted, not asserted: a file:line the reader can check.
    assert "plugin.py:" in live[0].message


def test_the_read_site_scan_is_narrower_than_the_coarse_key_union():
    """``plugin_config_keys`` unions in every ``"properties"`` block, so for
    ``memory`` it carries the ``store_memory`` TOOL parameters.  Telling an
    author ``content`` is a live CONFIG knob would be a new false claim in
    place of the one #910 removed."""
    sites = introspect.plugin_config_read_sites("memory")
    assert sites is not None
    assert "global_storage_path" in sites
    for tool_param in ("content", "tags", "confidence"):
        assert tool_param in introspect.plugin_config_keys("memory")
        assert tool_param not in sites


def test_unknown_knob_does_not_claim_the_runtime_ignores_it(env):
    """A genuine typo still warns — and says what was actually established:
    the schema does not declare it and no read site names it."""
    diags = _find(_validate(env, {"todo": {"storage_typo": "file"}}),
                  "unknown_knob", "storage_typo")
    assert [d.severity for d in diags] == ["warn"]
    msg = diags[0].message
    assert "silently ignored at runtime" not in msg
    assert "no config read site for it appears in the plugin's source" in msg
    # The known-knob list names the near-miss (todo declares more than one
    # knob since #1195's ``initial_plan_name``, so no fixed position).
    assert "storage_type" in msg.split("known:", 1)[1]


def test_an_unscannable_plugin_reports_absence_of_evidence_as_such(env):
    """An out-of-tree plugin's source is not in the scanned tree, so "no read
    site was found" is vacuous there and must not be dressed up as evidence of
    absence.  This is the shape #925's out-of-tree reach put in play."""
    providers, plugins, gc_names = env
    fake = introspect.PluginInfo(name="acme")
    fake.config_settings = [introspect.ConfigSetting(name="region", type="str")]
    fake.config_keys = ["region"]
    prof = SimpleNamespace(provider=None, model="m",
                           plugin_configs={"acme": {"regoin": "x"}})
    diags = _find(validate_profile(prof, providers=providers,
                                   plugins=dict(plugins, acme=fake),
                                   gc_names=gc_names),
                  "unknown_knob", "regoin")
    assert [d.severity for d in diags] == ["warn"]
    assert "not in the scanned tree" in diags[0].message
    assert "most likely a typo" not in diags[0].message


def test_a_declared_knob_is_still_clean(env):
    """The check must not have become a blanket warning."""
    assert [d for d in _validate(env, {"todo": {"storage_type": "file"}})
            if d.code in ("unknown_knob", "undeclared_knob")] == []


# ---- #937: the consequence belongs to the plugin ----------------------------

def test_invalid_knob_value_hands_the_consequence_back_to_the_plugin(env):
    """Still an **error** — the profile asks for something the plugin does not
    offer, however the plugin then behaves."""
    diags = _find(_validate(env, {"todo": {"storage_type": "sqlite"}}),
                  "invalid_knob_value", "storage_type")
    assert [d.severity for d in diags] == ["error"]
    msg = diags[0].message
    assert "silently replaced by a fallback" not in msg
    assert "what happens next is the plugin's choice" in msg
    assert "'memory'" in msg and "'hybrid'" in msg     # the set is still named


def test_knob_type_mismatch_message_agrees_with_its_own_docstring(env):
    """``_check_knob_value``'s docstring reasons "a plugin may coerce"; the
    emitted string denied it.  The docstring was right — that reasoning is
    what the **warn** severity was chosen on — so the message moved."""
    from jaato_server.shared.scaffold.validate import _check_knob_value

    diags = _find(_validate(env, {"cli": {"max_output_chars": "lots"}}),
                  "knob_type_mismatch", "max_output_chars")
    assert [d.severity for d in diags] == ["warn"]
    msg = diags[0].message
    assert "assigned without coercion" not in msg
    assert "may coerce it, reject it, or carry it downstream as-is" in msg
    assert "a plugin may coerce" in _check_knob_value.__doc__


def test_no_finding_message_asserts_an_unobservable_runtime_consequence(env):
    """The whole family at once, over every shape the two issues named."""
    diags = _validate(env, {
        "memory": {"global_storage_path": "/tmp/m"},
        "references": {"exclude_tools": ["x"]},
        "todo": {"storage_type": "sqlite", "storage_typo": "file"},
        "cli": {"max_output_chars": "lots"},
        "web_fetch": {"timeout": True},
    })
    offenders = [(d.code, d.message) for d in diags
                 for claim in _OVERCLAIMS if claim in d.message]
    assert offenders == []


# ---- what must stay unjudged whatever the wording -------------------------

@pytest.mark.parametrize("value", ["${TODO_STORAGE}", "pass://jaato/storage",
                                   "vault://secret/x#k", None])
def test_a_deferred_or_unset_value_is_still_not_judged(env, value):
    """``${VAR}`` and secret URIs resolve later, against an environment the
    validator does not have, and their literal form is a ``str`` whatever the
    knob declares.  ``None`` is "unset", not "wrongly typed"."""
    diags = _validate(env, {"todo": {"storage_type": value},
                            "web_fetch": {"timeout": value}})
    assert [d for d in diags
            if d.code in ("invalid_knob_value", "knob_type_mismatch")] == []


def test_true_is_still_not_an_integer(env):
    """Python makes ``bool`` a subclass of ``int``.  A knob declared
    ``integer`` must not accept ``timeout: true`` on that technicality — the
    silent shape the check exists to catch (``int(True) == 1``)."""
    assert len(_find(_validate(env, {"web_fetch": {"timeout": True}}),
                     "knob_type_mismatch", "timeout")) == 1


def test_every_reversion_here_names_a_test_in_this_module():
    """A reversion naming a renamed test is BLOCKED in the meta-guard, which
    it reports; this makes the rename fail here first, where it is edited."""
    names = set(globals())
    for rev in REVERSIONS:
        assert rev.test in names, rev.test
