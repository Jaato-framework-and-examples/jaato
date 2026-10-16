"""``validate`` parses the provider tree once, and the env tree only when asked (#1712).

Two costs made every ``scaffold.validate`` slow:

- ``introspect.providers()`` was not memoised, and one validate called it
  ~109 times (through ``resolve_provider``), re-reading and AST-evaluating
  every provider ``__init__.py`` each time: ~1.5 s of every warm run.
- ``validate_env``'s ``unread_env_var`` INFO hint called ``env_vars()``,
  whose first call per process AST-scans ~1100 files (5-8 s), on every
  workspace with a ``.env`` -- so the first validate after a daemon restart
  paid it whatever the ``.env`` said.

Now ``providers()`` is computed once per process over frozen
``ProviderInfo`` entries, and the scan runs only for a ``JAATO_`` key the
declared catalog (``shared/env_scope.CATALOG``, held equal to the scan by
``test_env_scope_catalog``) does not name.
"""

from __future__ import annotations

import dataclasses

import pytest

from jaato_server.shared.scaffold import introspect, validate
from jaato_server.shared.tests.reversion import Reversion

_INTRO = "jaato-server/jaato_server/shared/scaffold/introspect.py"
_VALIDATE = "jaato-server/jaato_server/shared/scaffold/validate.py"

REVERSIONS = [
    Reversion(
        target=_INTRO,
        find="    if _PROVIDERS_CACHE is None:\n",
        replace="    if True:\n",
        because="every providers() call would re-parse every provider again",
        test="test_a_validate_parses_the_providers_once",
    ),
    Reversion(
        target=_INTRO,
        find="@dataclass(frozen=True)\nclass ProviderInfo:",
        replace="@dataclass\nclass ProviderInfo:",
        because="one caller could change the memoised entry every later caller reads",
        test="test_a_shared_provider_entry_cannot_be_changed",
    ),
    Reversion(
        target=_VALIDATE,
        find="    known = set(introspect.env_vars()) if candidates else set()\n",
        replace="    known = set(introspect.env_vars())\n",
        because="every .env would pay the cold tree scan for an INFO hint",
        test="test_a_catalogued_env_does_not_scan_the_tree",
    ),
]


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    (ws / ".jaato" / "profiles").mkdir(parents=True)
    for i in range(3):
        (ws / ".jaato" / "profiles" / f"p{i}.yaml").write_text(
            f"name: p{i}\ndescription: profile {i}\n"
            "provider: anthropic\nmodel: claude-sonnet-4-5\n"
            "plugins: [cli]\n")
    (ws / ".env").write_text(
        "JAATO_PROVIDER=anthropic\nMODEL_NAME=claude-sonnet-4-5\n")
    return ws


def test_a_validate_parses_the_providers_once(workspace, monkeypatch):
    calls = []
    real = introspect._read_provider_consts
    monkeypatch.setattr(introspect, "_PROVIDERS_CACHE", None)
    monkeypatch.setattr(introspect, "_read_provider_consts",
                        lambda name: calls.append(name) or real(name))
    validate.validate_workspace(str(workspace))
    validate.validate_workspace(str(workspace))
    assert calls, "the run never asked for provider metadata"
    assert len(calls) == len(set(calls)), (
        f"{len(calls)} provider parses for {len(set(calls))} providers")


def test_a_shared_provider_entry_cannot_be_changed():
    info = introspect.resolve_provider("anthropic")
    assert info is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.notes = ("changed",)  # type: ignore[misc]


def test_the_providers_dict_is_the_callers(monkeypatch):
    first = introspect.providers()
    first.pop("anthropic")
    assert "anthropic" in introspect.providers()


def test_a_catalogued_env_does_not_scan_the_tree(workspace, monkeypatch):
    (workspace / ".env").write_text(
        "JAATO_PROVIDER=anthropic\nJAATO_PARALLEL_TOOLS=false\n"
        "JAATO_PROFILE_SET=\n")

    def refuse():
        raise AssertionError("env_vars() scanned for catalogued keys only")

    monkeypatch.setattr(introspect, "env_vars", refuse)
    out = validate.validate_env(str(workspace))
    assert not [d for d in out if d.code == "unread_env_var"]


def test_an_unknown_key_is_still_reported(workspace):
    (workspace / ".env").write_text(
        "JAATO_PROVIDER=anthropic\nJAATO_PARALEL_TOOLS=false\n")
    out = validate.validate_env(str(workspace))
    assert [d.where for d in out if d.code == "unread_env_var"] == [
        "JAATO_PARALEL_TOOLS"]
