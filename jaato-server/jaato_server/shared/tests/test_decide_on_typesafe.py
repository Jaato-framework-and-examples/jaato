"""``decide()`` on ``typesafe``: step 7 of docs/design/decision-models.md.

TypeSafe's own API (``POST /v1/systemone``) driven against the local
decision stand-in: the URL and bearer token, where the key comes from
(the profile knob FIRST, then ``JAATO_TYPESAFE_API_KEY``,
``TYPESAFE_API_KEY`` and the ``typesafe-auth`` store), retries and
refusals, the window, the chat refusal, a ``decision_models:`` binding,
and the ``typesafe-auth`` command against the same stand-in.

No test touches the network.
"""

from __future__ import annotations

import json

import pytest

from jaato_sdk.plugins.model_provider.decisions import (
    DecisionModelOnlyError,
    DecisionQuestion,
    DecisionQuestionError,
)
from jaato_server.shared.plugins.model_provider import load_provider
from jaato_server.shared.plugins.model_provider.base import ProviderConfig
from jaato_server.shared.plugins.model_provider.typesafe import TypeSafeProvider
from jaato_server.shared.plugins.model_provider.typesafe.errors import (
    AuthenticationError,
    CredentialsNotFoundError,
    DecisionRequestRejectedError,
)
from jaato_server.shared.tests.decision_standin import (
    DecisionStandIn,
    typesafe_against,
)
from jaato_server.shared.tests.reversion import Reversion

_PROVIDER = "jaato-server/jaato_server/shared/plugins/model_provider/typesafe/provider.py"

REVERSIONS = [
    Reversion(
        target=_PROVIDER,
        find="    if knob:\n        return str(knob), \"plugin_configs.typesafe.api_key\"\n",
        replace="",
        test="test_the_profile_knob_outranks_the_environment",
        because="a profile's key would be ignored in favour of whatever the "
                "daemon's environment holds",
    ),
    Reversion(
        target=_PROVIDER,
        find="        headers = {\"Authorization\": f\"Bearer {self._api_key}\"}\n",
        replace="        headers = {}\n",
        test="test_the_request_reaches_systemone_with_the_key",
        because="every request would reach TypeSafe without a key and be refused",
    ),
    Reversion(
        target=_PROVIDER,
        find="        if not self._api_key:\n            raise CredentialsNotFoundError()\n",
        replace="",
        test="test_no_key_anywhere_is_refused_at_initialize",
        because="a missing key would surface as a 401 on the first decision, "
                "naming none of the places to put one",
    ),
    Reversion(
        target=_PROVIDER,
        find="        if isinstance(exc, InfrastructureError):\n",
        replace="        if False:\n",
        test="test_a_server_failure_is_retried",
        because="a 502 or 529 from TypeSafe would fail the decision outright",
    ),
    Reversion(
        target=_PROVIDER,
        find="        if estimate > limit:\n",
        replace="        if False:\n",
        test="test_a_configured_window_refuses_before_sending",
        because="a state over the configured window would be sent anyway",
    ),
    Reversion(
        target=_PROVIDER,
        find="        raise DecisionModelOnlyError(self._model_name or DEFAULT_MODEL, self.name)\n",
        replace="        return None\n",
        test="test_complete_refuses_a_chat_turn",
        because="a chat turn would be handed to a model that writes no text",
    ),
]

QUESTIONS = {
    "urgent": DecisionQuestion("noul", "Is this urgent?"),
    "team": DecisionQuestion("choice", "Which team?",
                             {"billing": "payments", "tech": None}),
    "mood": DecisionQuestion("score", "How upset?", ["Calm", "Upset", "Angry"]),
}
URGENT = {"urgent": QUESTIONS["urgent"]}

_ENV = ("JAATO_TYPESAFE_API_KEY", "TYPESAFE_API_KEY", "JAATO_TYPESAFE_BASE_URL",
        "JAATO_TYPESAFE_CONTEXT_LENGTH", "JAATO_CONFIG_ROOT", "JAATO_WORKSPACE_ROOT")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """No ambient key, and a store that resolves under ``tmp_path``."""
    for name in _ENV + ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        monkeypatch.delenv(name, raising=False)   # the stand-in is local
    home = tmp_path / "home"
    (home / ".jaato").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return home


@pytest.fixture
def standin():
    with DecisionStandIn() as server:
        yield server


@pytest.fixture
def fast_retries(monkeypatch):
    monkeypatch.setenv("AI_RETRY_BASE_DELAY", "0.01")
    monkeypatch.setenv("AI_RETRY_MAX_DELAY", "0.02")
    monkeypatch.setenv("AI_RETRY_LOG_SILENT", "1")


def _bearer(server):
    (_path, headers, _body), = server.posts()
    return headers.get("Authorization")


def _store(home, key):
    (home / ".jaato" / "typesafe_auth.json").write_text(
        json.dumps({"api_key": key, "created_at": 0}))


# ------------------------------------------------------------------- wire

def test_the_request_reaches_systemone_with_the_key(standin):
    result = typesafe_against(standin, api_key="ts-knob").decide(
        "Payouts failing for 3 days", QUESTIONS)
    (path, headers, body), = standin.posts()
    assert path == "/v1/systemone"
    assert headers.get("Authorization") == "Bearer ts-knob"
    assert body["model"] == "jev-latest"
    assert body["state"] == "Payouts failing for 3 days"
    assert "TypeSafe-Danger-Zone-Images" not in headers   # text only (§12)
    assert result.answers["team"].value == "billing"


# --------------------------------------------------------------- the key

def test_the_profile_knob_outranks_the_environment(standin, monkeypatch, _isolated):
    monkeypatch.setenv("JAATO_TYPESAFE_API_KEY", "ts-jaato-env")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-vendor-env")
    _store(_isolated, "ts-stored")
    provider = typesafe_against(standin, api_key="ts-knob")
    provider.decide("x", URGENT)
    assert _bearer(standin) == "Bearer ts-knob"
    assert provider.get_auth_info() == "API key from plugin_configs.typesafe.api_key"


@pytest.mark.parametrize("present, expected", [
    (("JAATO_TYPESAFE_API_KEY", "TYPESAFE_API_KEY", "stored"), "ts-jaato-env"),
    (("TYPESAFE_API_KEY", "stored"), "ts-vendor-env"),
    (("stored",), "ts-stored"),
])
def test_the_fallbacks_beneath_the_knob_come_in_order(
        standin, monkeypatch, _isolated, present, expected):
    values = {"JAATO_TYPESAFE_API_KEY": "ts-jaato-env",
              "TYPESAFE_API_KEY": "ts-vendor-env"}
    for name in present:
        if name == "stored":
            _store(_isolated, "ts-stored")
        else:
            monkeypatch.setenv(name, values[name])
    typesafe_against(standin, api_key=None).decide("x", URGENT)
    assert _bearer(standin) == f"Bearer {expected}"


def test_no_key_anywhere_is_refused_at_initialize():
    with pytest.raises(CredentialsNotFoundError) as info:
        TypeSafeProvider().initialize(ProviderConfig(extra={}))
    message = str(info.value)
    for place in ("plugin_configs.typesafe.api_key", "JAATO_TYPESAFE_API_KEY",
                  "TYPESAFE_API_KEY", "typesafe-auth"):
        assert place in message


def test_verify_auth_sees_the_env_and_the_store(monkeypatch, _isolated):
    assert TypeSafeProvider().verify_auth() is False
    _store(_isolated, "ts-stored")
    assert TypeSafeProvider().verify_auth() is True


def test_a_401_names_where_the_key_came_from(standin, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-stale")
    standin.queue.append((401, {"error": "invalid key"}, None))
    with pytest.raises(AuthenticationError, match="TYPESAFE_API_KEY"):
        typesafe_against(standin, api_key=None).decide("x", URGENT)
    assert len(standin.posts()) == 1


# ------------------------------------------------------------ other knobs

def test_the_base_url_knob_outranks_its_env(standin, monkeypatch):
    monkeypatch.setenv("JAATO_TYPESAFE_BASE_URL", "http://127.0.0.1:9")   # nothing there
    typesafe_against(standin).decide("x", URGENT)        # knob: the stand-in
    assert len(standin.posts()) == 1


def test_the_base_url_env_is_the_fallback(standin, monkeypatch):
    monkeypatch.setenv("JAATO_TYPESAFE_BASE_URL", standin.host)
    provider = TypeSafeProvider()
    provider.initialize(ProviderConfig(api_key="ts-test"))
    provider.connect("jev-latest")
    provider.decide("x", URGENT)
    assert len(standin.posts()) == 1


def test_a_server_failure_is_retried(standin, fast_retries):
    standin.queue += [(502, {"error": "bad gateway"}, None),
                      (529, {"error": "overloaded"}, None)]
    assert typesafe_against(standin).decide("x", URGENT).answers
    assert len(standin.posts()) == 3


def test_a_422_is_not_retried(standin, fast_retries):
    standin.queue.append((422, {"detail": "criteria must be a list"}, None))
    with pytest.raises(DecisionRequestRejectedError, match="criteria must be a list"):
        typesafe_against(standin).decide("x", {"mood": QUESTIONS["mood"]})
    assert len(standin.posts()) == 1


def test_no_window_configured_is_not_checked(standin):
    provider = typesafe_against(standin)
    assert provider.get_context_limit() == 0
    provider.decide("word " * 50_000, URGENT)
    assert len(standin.posts()) == 1


def test_a_configured_window_refuses_before_sending(standin, monkeypatch):
    with pytest.raises(DecisionQuestionError, match="1000-token window"):
        typesafe_against(standin, context_length=1000).decide("word " * 2000, URGENT)
    monkeypatch.setenv("JAATO_TYPESAFE_CONTEXT_LENGTH", "1000")
    with pytest.raises(DecisionQuestionError):
        typesafe_against(standin).decide("word " * 2000, URGENT)
    assert standin.posts() == []


# ------------------------------------------------------------------ shape

def test_complete_refuses_a_chat_turn(standin):
    provider = typesafe_against(standin)
    with pytest.raises(DecisionModelOnlyError, match="decide"):
        provider.complete([])
    assert provider.is_decisions_only()
    assert provider.output_modalities() == {"decisions"}


def test_discovery_finds_it():
    provider = load_provider("typesafe", ProviderConfig(api_key="ts-test"))
    assert provider.name == "typesafe"


def test_a_decision_models_entry_is_served_by_typesafe(standin):
    from unittest.mock import MagicMock

    from jaato_server.shared.jaato_session import JaatoSession

    runtime = MagicMock()
    runtime.provider_name = "anthropic"
    runtime.create_provider.side_effect = (
        lambda model=None, provider_name=None, **_kw:
        typesafe_against(standin, model=model) if provider_name == "typesafe" else None)
    runtime.get_tool_schemas.return_value = []
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.reliability_plugin = None
    runtime.registry = MagicMock()
    runtime.registry.get_exposed_tool_schemas.return_value = []
    session = JaatoSession(runtime, "claude-sonnet-4-6", provider_name="anthropic")
    session.configure(decision_models={
        "triage": {"provider": "typesafe", "model": "jev-1.13.0"}})
    provider = session.decision_provider("triage")
    assert isinstance(provider, TypeSafeProvider)
    provider.decide("x", URGENT)
    (_path, _headers, body), = standin.posts()
    assert body["model"] == "jev-1.13.0"


def test_explain_lists_the_key_order():
    from jaato_server.shared.scaffold.introspection_verbs import render_topic

    ok, _data, text, error = render_topic("provider", "typesafe")
    assert ok, error
    line, = [ln for ln in text.splitlines() if ln.strip().startswith("credentials")]
    order = [line.index(s) for s in ("api_key_param:api_key",
                                     "env:JAATO_TYPESAFE_API_KEY",
                                     "env:TYPESAFE_API_KEY", "stored:typesafe-auth")]
    assert order == sorted(order), line
    assert "decision_models" in text


# ---------------------------------------------------------- typesafe-auth

def _auth(output):
    from jaato_server.shared.plugins.typesafe_auth import create_plugin

    plugin = create_plugin()
    plugin.initialize({})
    plugin.set_output_callback(lambda _src, text, _mode: output.append(text))
    return plugin


def test_typesafe_auth_validates_stores_and_clears(standin, monkeypatch, _isolated):
    monkeypatch.setenv("JAATO_TYPESAFE_BASE_URL", standin.host)
    out = []
    plugin = _auth(out)
    plugin.execute_user_command("typesafe-auth", {"action": "key ts-new"})
    (path, headers, body), = standin.posts()
    assert path == "/v1/systemone" and headers.get("Authorization") == "Bearer ts-new"
    stored = _isolated / ".jaato" / "typesafe_auth.json"
    assert json.loads(stored.read_text())["api_key"] == "ts-new"
    assert stored.stat().st_mode & 0o077 == 0
    plugin.execute_user_command("typesafe-auth", {"action": "status"})
    assert "4. stored:" in out[-1] and "ts-new" not in out[-1]
    plugin.execute_user_command("typesafe-auth", {"action": "logout"})
    assert not stored.exists()


def test_typesafe_auth_does_not_store_a_refused_key(standin, monkeypatch, _isolated):
    monkeypatch.setenv("JAATO_TYPESAFE_BASE_URL", standin.host)
    standin.queue.append((401, {"error": "invalid key"}, None))
    out = []
    _auth(out).execute_user_command("typesafe-auth", {"action": "key ts-bad"})
    assert len(standin.posts()) == 1             # it was asked, and refused
    assert not (_isolated / ".jaato" / "typesafe_auth.json").exists()
    assert "NOT saved" in "".join(out)
