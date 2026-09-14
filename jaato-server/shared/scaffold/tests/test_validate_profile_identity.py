"""Two things a profile says about ITSELF that nothing checked.

**`description`** has no default on ``SubagentProfile`` and ``explain profile``
prints ``(required)`` beside it — but the YAML loader turns a missing key into
``""`` and inheritance does not rescue it (the merge takes
``description=child.description``, so a tier-2 set profile that omits it
OVERRIDES its base's with the empty string).  Nothing fails.  What breaks is
the one line the subagent plugin advertises to the model, ``- worker:  (tools:
cli)`` — the prose a delegate is chosen from.  Every profile
``jaato-scaffold new profile-set`` emitted was in exactly that state.

**`system_instructions`** is deprecated in favour of a persona under
``.jaato/agents/``.  ``explain profile`` said so; ``validate`` did not, and the
field keeps working, so nothing corrected an author who had not found the
agents directory.

Both WARN.  Either profile loads and runs, and an error would fail existing
workspaces wholesale — the posture ``unknown_knob`` and
``budget_control_absent`` already take.
"""

import types

import pytest

from shared.scaffold import introspect
from shared.scaffold.validate import validate_profile


def _profile(**kw):
    base = dict(name="worker", description="does the work", provider=None,
                model=None, plugins=[], plugin_configs={},
                system_instructions=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _codes(profile):
    return [d.code for d in validate_profile(
        profile, providers=introspect.providers(),
        plugins=introspect.plugins(),
        gc_names=list(introspect.gc_strategies()))]


def _find(profile, code):
    return next(d for d in validate_profile(
        profile, providers=introspect.providers(),
        plugins=introspect.plugins(),
        gc_names=list(introspect.gc_strategies())) if d.code == code)


# ---------------------------------------------------------------- description

@pytest.mark.parametrize("value", ["", "   ", None])
def test_absent_description_is_reported(value):
    d = _find(_profile(description=value), "missing_description")
    assert d.severity == "warn"
    assert d.where == "description"


def test_a_real_description_is_silent():
    assert "missing_description" not in _codes(_profile())


def test_message_says_inheritance_will_not_supply_it():
    """The trap: `inherits:` looks like it covers this, and it does not."""
    d = _find(_profile(description=""), "missing_description")
    assert "inheritance does not supply it" in d.message
    assert "REPLACES" in d.message


def test_message_says_the_model_reads_it():
    d = _find(_profile(description=""), "missing_description")
    assert "advertises it to the model" in d.message


# -------------------------------------------------------- system_instructions

def test_deprecated_system_instructions_is_reported():
    d = _find(_profile(system_instructions="You are a worker."),
              "deprecated_system_instructions")
    assert d.severity == "warn"
    assert d.where == "system_instructions"


def test_deprecation_message_names_the_replacement_route():
    d = _find(_profile(system_instructions="x"),
              "deprecated_system_instructions")
    assert ".jaato/agents/" in d.message
    assert "default_agent" in d.message
    assert "explain agents" in d.message


def test_deprecation_warns_only_when_the_field_is_set():
    assert "deprecated_system_instructions" not in _codes(_profile())
    assert "deprecated_system_instructions" not in _codes(
        _profile(system_instructions=""))


def test_deprecation_message_admits_it_may_be_inherited():
    """`system_instructions` concatenates down the chain, so the profile the
    finding is attributed to may not be the file that set it."""
    d = _find(_profile(system_instructions="x"),
              "deprecated_system_instructions")
    assert "Inherited" in d.message


# ------------------------------------------------------------------ integration

def test_a_clean_profile_reports_neither():
    codes = _codes(_profile())
    assert "missing_description" not in codes
    assert "deprecated_system_instructions" not in codes
