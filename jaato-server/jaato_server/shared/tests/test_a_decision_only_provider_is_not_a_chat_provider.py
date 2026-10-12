"""A decision-only provider is never a session's chat provider.

``laya`` serves ``decide()`` and nothing else: its ``complete()`` raises
``DecisionModelOnlyError``.  It declares that as
``ProviderCapabilities.chat=False``, and the authoring surface reads the
flag:

- ``jaato-scaffold new profile-set`` / ``new client --provider laya``
  refuse before writing anything, naming ``decision_models:``.  Before, the
  set was scaffolded and reported "valid by construction", and every turn
  of every session built from it failed.
- ``jaato-scaffold validate`` reports ``decision_only_provider`` (error)
  for a profile's ``provider:`` or a tier's ``provider:`` naming one.
- With jaato-server absent, ``new`` answers from the authoring snapshot,
  which records the flag.

The conformance half (``chat=False`` providers refuse a turn) lives in
``test_provider_capability_conformance.py``.
"""

from __future__ import annotations

import pytest

from jaato_sdk.scaffold import authoring_contracts as contracts
from jaato_sdk.scaffold import cli
from jaato_server.shared.plugins.subagent.config import SubagentProfile
from jaato_server.shared.tests.reversion import Reversion

_BUILD = "jaato-sdk/jaato_sdk/scaffold/build.py"
_VALIDATE = "jaato-server/jaato_server/shared/scaffold/validate.py"
_LAYA = "jaato-server/jaato_server/shared/plugins/model_provider/laya/__init__.py"
_CONTRACTS = "jaato-sdk/jaato_sdk/scaffold/authoring_contracts.py"

REVERSIONS = [
    Reversion(
        target=_BUILD,
        find="    return _refuse_decision_only(archetype, provider, model)\n",
        replace="    return False\n",
        test="test_new_profile_set_refuses_laya",
        because="new profile-set scaffolding a decision-only provider as the "
                "set's chat provider",
    ),
    Reversion(
        target=_BUILD,
        find="    if provider and _refuse_decision_only(archetype, provider, model):\n        return 2, None, None\n",
        replace="",
        test="test_new_client_refuses_laya",
        because="new client binding a session to a decision-only provider",
    ),
    Reversion(
        target=_VALIDATE,
        find="        if info is None or info.chat:\n            continue\n",
        replace="        continue\n",
        test="test_validate_flags_laya_as_the_profile_provider",
        because="validate passing a profile whose provider cannot chat",
    ),
    Reversion(
        target=_LAYA,
        find="    chat=False,",
        replace="    chat=True, ",
        test="test_laya_declares_it_does_not_chat",
        because="laya claiming a chat wire it does not have",
    ),
    Reversion(
        target=_CONTRACTS,
        find="            chat=bool(rec.get(\"chat\", True)),\n",
        replace="",
        test="test_the_snapshot_says_laya_does_not_chat",
        because="the SDK-only snapshot path losing the flag, so new accepts "
                "laya without jaato-server installed",
    ),
]


def test_laya_declares_it_does_not_chat():
    from jaato_server.shared.plugins.model_provider import laya
    assert laya.PROVIDER_CAPABILITIES.chat is False
    assert laya.PROVIDER_CAPABILITIES.decisions is True


def test_new_profile_set_refuses_laya(tmp_path, capsys):
    ws = tmp_path / "ws"
    rc = cli.main(["new", "profile-set", "--workspace", str(ws), "--set", "s",
                   "--provider", "laya", "--model", "convaiinnovations/laya",
                   "--agents", "a"])
    out = capsys.readouterr().out
    assert rc == 2, out
    assert "decision_models:" in out and "laya" in out
    assert not ws.exists(), "nothing may be written for a refused binding"


def test_new_client_refuses_laya(tmp_path, capsys):
    ws = tmp_path / "ws"
    rc = cli.main(["new", "client", "--workspace", str(ws),
                   "--provider", "laya", "--model", "convaiinnovations/laya"])
    out = capsys.readouterr().out
    assert rc == 2, out
    assert "decision models only" in out
    assert not ws.exists()


def test_new_profile_set_still_accepts_a_chat_provider(tmp_path, capsys):
    rc = cli.main(["new", "profile-set", "--workspace", str(tmp_path / "ws"),
                   "--set", "s", "--provider", "openrouter", "--model", "m",
                   "--agents", "a"])
    assert rc == 0, capsys.readouterr().out


def test_the_snapshot_says_laya_does_not_chat(monkeypatch):
    monkeypatch.setattr(contracts, "_live", lambda: None)
    assert contracts.resolve_provider("laya").chat is False
    assert contracts.resolve_provider("openrouter").chat is True


@pytest.fixture(scope="module")
def introspected():
    from jaato_server.shared.scaffold import introspect
    return introspect.providers(), introspect.plugins()


def _codes(introspected, **kw):
    from jaato_server.shared.scaffold import validate
    providers, plugins = introspected
    profile = SubagentProfile(name="t", description="d", plugins=[], **kw)
    return [(d.severity, d.code, d.where) for d in validate.validate_profile(
        profile, providers=providers, plugins=plugins, gc_names=[])]


def test_validate_flags_laya_as_the_profile_provider(introspected):
    found = _codes(introspected, provider="laya", model="convaiinnovations/laya")
    assert ("error", "decision_only_provider", "provider") in found


def test_validate_flags_laya_as_a_tier_provider(introspected):
    found = _codes(introspected, provider="openrouter", model_tiers={
        "judge": {"provider": "laya", "model": "convaiinnovations/laya",
                  "description": "d"},
    })
    assert ("error", "decision_only_provider",
            "model_tiers.judge.provider") in found


def test_validate_accepts_laya_under_decision_models(introspected):
    found = _codes(introspected, provider="openrouter", model="m",
                   decision_models={"local": {"provider": "laya",
                                              "model": "convaiinnovations/laya"}})
    assert not [c for c in found if c[1].startswith("decision")], found


def test_an_abstract_base_naming_laya_is_left_to_its_children(introspected):
    found = _codes(introspected, provider="laya")
    assert not [c for c in found if c[1] == "decision_only_provider"]
