"""#1188: secret-resolver discovery is observable.

Two silences the reporter hit, both closed here:

  * a resolver whose entry point FAILS to load (e.g. ``PassResolver.__init__``
    raising ``ImportError`` on a minimal-PATH daemon with no ``pass`` /
    ``gpg-connect-agent``) is warned once per process, NAMING the entry point;
  * an EMPTY discovery result — cached for the process lifetime — is warned
    once, so an operator sees it rather than only downstream "no resolver"
    errors.

Both are exercised against the real ``_discover_secret_resolvers_uncached``
(the function the cached wrapper calls exactly once) with a fabricated
``entry_points()`` result, so no test depends on jaato-premium being present
or absent on the machine.
"""

import logging
from typing import FrozenSet, Optional
from unittest.mock import patch

import pytest

from ..config import (
    _discover_secret_resolvers_uncached,
    reset_secret_resolvers,
)


class _FakeResolver:
    @property
    def schemes(self) -> FrozenSet[str]:
        return frozenset({"pass"})

    def resolve(self, scheme: str, path: str, key: Optional[str] = None) -> str:
        return "value"


class _FakeEP:
    """A minimal entry point whose ``load()`` returns ``factory``."""

    def __init__(self, name, value, factory):
        self.name = name
        self.value = value
        self._factory = factory

    def load(self):
        return self._factory


class _FakeEPs:
    """Stands in for ``importlib.metadata.entry_points()`` (3.12 shape)."""

    def __init__(self, eps):
        self._eps = eps

    def select(self, group=None, name=None):
        return [
            ep for ep in self._eps
            if name is None or ep.name == name
        ]


def _patch_entry_points(eps):
    return patch(
        "importlib.metadata.entry_points",
        return_value=_FakeEPs(eps),
    )


@pytest.fixture(autouse=True)
def _reset():
    reset_secret_resolvers()
    yield
    reset_secret_resolvers()


def test_empty_discovery_warns_once_naming_reset(caplog):
    with _patch_entry_points([]), caplog.at_level(logging.WARNING):
        result = _discover_secret_resolvers_uncached()
    assert result == {}
    empty_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "No secret resolvers were discovered" in r.getMessage()
    ]
    assert len(empty_warnings) == 1
    assert "reset_secret_resolvers" in empty_warnings[0].getMessage()


def test_populated_discovery_does_not_warn_empty(caplog):
    ep = _FakeEP("secret_resolvers", "premium:factory",
                 lambda: [_FakeResolver()])
    with _patch_entry_points([ep]), caplog.at_level(logging.WARNING):
        result = _discover_secret_resolvers_uncached()
    assert "pass" in result
    assert not [
        r for r in caplog.records
        if "No secret resolvers were discovered" in r.getMessage()
    ]


def test_entry_point_load_failure_warns_naming_the_entry_point(caplog):
    def _boom():
        raise ImportError("no `pass` / `gpg-connect-agent` on PATH")

    ep = _FakeEP("secret_resolvers", "premium.pass:make", _boom)
    with _patch_entry_points([ep]), caplog.at_level(logging.WARNING):
        result = _discover_secret_resolvers_uncached()
    # The factory blew up, so nothing was registered.
    assert result == {}
    msgs = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]
    # The load-failure warning names the entry point that was skipped...
    load_fail = [m for m in msgs if "failed to load" in m]
    assert len(load_fail) == 1
    assert "premium.pass:make" in load_fail[0]
    # ...and the empty-result warning still fires, because the failure left
    # the registry empty.
    assert any("No secret resolvers were discovered" in m for m in msgs)
