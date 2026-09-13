"""Every jaato distribution's extras are reported, including ones shipped apart.

WHAT WENT WRONG (#966).  ``jaato-premium`` is a SEPARATE distribution.  It was
added to ``FIRST_PARTY`` and to ``JAATO_DISTS``, and to neither of the two
places that enumerate extras — ``EXTRA_DISTS`` and an inline
``("jaato-sdk", "jaato-server")`` literal inside ``framework_picture`` that had
drifted from the constant beside it.  So ``jaato-scaffold explain
dependencies`` listed the distribution and none of its nine extras, and a
missing ``presidio_analyzer`` import could not be traced back to
``pip install 'jaato-premium[pseudonymization]'`` the way ``pexpect`` is traced
to ``jaato-server[interactive]``.  An operator was told what to install and the
list was incomplete — the one thing a diagnostic must not do.

WHY THE TEST FAKES A DISTRIBUTION.  This repository cannot depend on
jaato-premium, and the environment a test runs in usually does not have it, so
the only way to assert the behaviour is to fake installed metadata and reset
the module's caches around it.  ``reset_metadata_caches`` exists for that.

WHY ONE OF THESE ASSERTS ABOUT A DISTRIBUTION THAT DOES NOT EXIST.  Adding
``"jaato-premium"`` to a hardcoded tuple would satisfy every premium-named test
here and leave the NEXT separately-shipped distribution equally invisible — the
same defect wearing the fix as a disguise.  ``test_a_jaato_distribution_this_
repo_never_heard_of_is_enumerated`` uses an invented name precisely because no
literal in the tree can name it: it passes only if the set is MEASURED from
installed metadata.
"""
from __future__ import annotations

import importlib.metadata

import pytest

from shared.scaffold import dependencies as D
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion

REVERSIONS = [
    Reversion(
        target="jaato-server/shared/scaffold/dependencies.py",
        find="""    for dist_name in framework_dists():
        for req in _requires(dist_name):
            extra = _extra_marker(req)
            pkg = _requirement_name(req)""",
        replace="""    for dist_name in ("jaato-sdk", "jaato-server"):
        for req in _requires(dist_name):
            extra = _extra_marker(req)
            pkg = _requirement_name(req)""",
        test="test_a_missing_import_names_the_extra_of_a_separately_shipped_dist",
        because="the missing-import -> extra index scanning only sdk and server",
    ),
    Reversion(
        target="jaato-server/shared/scaffold/dependencies.py",
        find="""    for dist_name in framework_dists():
        for req in _requires(dist_name):
            extra = _extra_marker(req)
            if not extra:""",
        replace="""    for dist_name in ("jaato-sdk", "jaato-server"):
        for req in _requires(dist_name):
            extra = _extra_marker(req)
            if not extra:""",
        test="test_explain_dependencies_lists_a_separately_shipped_dists_extras",
        because="the rendered extras listing scanning only sdk and server",
    ),
    Reversion(
        target="jaato-server/shared/scaffold/dependencies.py",
        find="""    out: List[str] = list(JAATO_DISTS)
    seen = {_norm_dist(n) for n in out}
    for name in installed_jaato_dists():
        if _norm_dist(name) not in seen:
            out.append(name)
            seen.add(_norm_dist(name))
    return tuple(out)""",
        replace="""    return tuple(JAATO_DISTS)""",
        test="test_a_jaato_distribution_this_repo_never_heard_of_is_enumerated",
        because="the distribution set being a hardcoded tuple rather than measured",
    ),
]


class _FakeDist:
    """The one attribute :func:`installed_jaato_dists` reads off a distribution."""

    def __init__(self, name: str) -> None:
        self.metadata = {"Name": name}


def _install_fakes(monkeypatch, extra_dists):
    """Make ``extra_dists`` look installed, with their own requirements.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        extra_dists: ``{distribution name: [requirement string, ...]}``.  The
            requirement strings are the raw ``Requires-Dist`` lines a real
            ``METADATA`` file carries, markers included.
    """
    real_requires = D.requires

    def fake_distributions():
        return [_FakeDist(n) for n in ("jaato-sdk", "jaato-server", *extra_dists)]

    def fake_requires(name):
        if name in extra_dists:
            return list(extra_dists[name])
        return real_requires(name)

    monkeypatch.setattr(importlib.metadata, "distributions", fake_distributions)
    monkeypatch.setattr(D, "requires", fake_requires)
    # Force the name-match tier of _provided_import_names: a package nobody
    # installed cannot say which import names it would have provided.
    monkeypatch.setattr(D, "packages_distributions", dict)
    D.reset_metadata_caches()


_PREMIUM = {
    "jaato-premium": [
        'presidio-analyzer>=2.2; extra == "pseudonymization"',
        'pynacl>=1.5; extra == "pseudonymization"',
        'hvac>=1.0; extra == "vault"',
    ],
}


@pytest.fixture(autouse=True)
def _clean_caches():
    """Metadata caches are process-lifetime; a faking test must not leak."""
    D.reset_metadata_caches()
    yield
    D.reset_metadata_caches()


def test_explain_dependencies_lists_a_separately_shipped_dists_extras(monkeypatch):
    """#966's own reproduction: the extras listing names the premium extras."""
    _install_fakes(monkeypatch, _PREMIUM)
    extras = D.framework_picture()["extras"]
    assert "jaato-premium[pseudonymization]" in extras, (
        "an installed jaato distribution's extras must be listed; #966 is "
        f"exactly their absence. Got: {sorted(extras)}"
    )
    assert any("presidio-analyzer" in r
               for r in extras["jaato-premium[pseudonymization]"]), \
        "the extra must name what it pulls in, read from its own metadata"
    assert "jaato-premium[vault]" in extras, "every extra, not just the one reported"


def test_a_missing_import_names_the_extra_of_a_separately_shipped_dist(monkeypatch):
    """The pexpect->jaato-server[interactive] mapping, for premium's packages.

    This is the index ``PluginRegistry._install_advice`` and
    ``_install_hint`` both read, so it is what turns a missing
    ``presidio_analyzer`` into a pip command rather than a guess.
    """
    _install_fakes(monkeypatch, _PREMIUM)
    assert D._extras_index().get("presidio_analyzer") == \
        ("jaato-premium[pseudonymization]",), (
            "a missing import declared by a premium extra must resolve to that "
            f"extra. Got: {D._extras_index().get('presidio_analyzer')!r}"
        )
    hint = D._install_hint(["presidio_analyzer"], "pseudonymization")
    assert "pip install 'jaato-premium[pseudonymization]'" in hint["commands"], \
        f"the hint must name the install target. Got: {hint['commands']!r}"


def test_a_jaato_distribution_this_repo_never_heard_of_is_enumerated(monkeypatch):
    """The anti-hardcode assertion: an invented name no literal here can carry.

    ``jaato-premium`` could be satisfied by appending one string to a tuple.
    This one cannot: it passes only while the distribution set is measured
    from installed metadata.
    """
    invented = {"jaato-thing-nobody-wrote-down": [
        'zzz-widget>=9; extra == "frobnicate"']}
    _install_fakes(monkeypatch, invented)
    extras = D.framework_picture()["extras"]
    assert "jaato-thing-nobody-wrote-down[frobnicate]" in extras, (
        "the extras scan must range over the jaato distributions actually "
        f"installed, not a list written here. Got: {sorted(extras)}"
    )
    assert D._extras_index().get("zzz_widget") == \
        ("jaato-thing-nobody-wrote-down[frobnicate]",)


def test_a_non_jaato_distribution_is_not_swept_in(monkeypatch):
    """Only jaato's own distributions; a venv full of packages is not the scan."""
    _install_fakes(monkeypatch, {"some-unrelated-lib": [
        'requests; extra == "http"']})
    extras = D.framework_picture()["extras"]
    assert not any(k.startswith("some-unrelated-lib") for k in extras), \
        f"the scan must stay first-party. Got: {sorted(extras)}"


def test_premium_absent_is_silence_not_a_finding(monkeypatch):
    """Not installed is not a missing extra — it is nothing to say.

    A workspace that never bought premium must not be told about an extra of
    a package it does not have; ``_requires`` answers ``[]`` for an
    uninstalled distribution, so the whole family simply does not appear.
    """
    _install_fakes(monkeypatch, {})
    picture = D.framework_picture()
    assert not any(k.startswith("jaato-premium") for k in picture["extras"]), \
        "an uninstalled distribution must contribute no extras"
    assert D._extras_index().get("presidio_analyzer") is None
    rendered = D._render_framework(picture)
    assert "pseudonymization" not in rendered
    assert "jaato-premium" in picture["missing"], (
        "it is still a distribution this framework expects, so 'not installed' "
        "stays sayable — that is the honest report, not a warning about extras"
    )


def test_the_two_enumerations_cannot_drift_apart():
    """One source: what is rendered is what the missing-import index inverts.

    The defect was two copies of one list — a constant and an inline literal —
    that stopped agreeing.  Assert the relationship rather than the literal:
    every extra label the index can name is a label the listing carries.
    """
    labels = set(D._extras_by_label())
    indexed = {lab for labs in D._extras_index().values() for lab in labs}
    assert indexed <= labels, (
        "the missing-import index named extras the dependency listing does "
        f"not show: {sorted(indexed - labels)}"
    )
