"""SessionInitEnvelope.client_tools — the wire field carrying client-provided
("host") tool schemas to the runner so the runner-tier model SEES them
(the #344-sibling daemon-vs-runner split; client tools registered only on the
daemon registry pre-fix)."""

from shared.session_envelope import SessionInitEnvelope


def _env(**kw):
    base = dict(session_id="s", workspace_path="/ws", profile_name=None,
                provider_name="p", model_name="m")
    base.update(kw)
    return SessionInitEnvelope(**base)


def test_client_tools_round_trip():
    tools = [{"name": "send_to_telegram", "description": "d", "parameters": {}}]
    d = _env(client_tools=tools).to_dict()
    assert d["client_tools"] == tools
    assert SessionInitEnvelope.from_dict(d).client_tools == tools


def test_client_tools_default_empty():
    assert _env().client_tools == []


def test_backward_compat_old_wire_without_field():
    d = _env().to_dict()
    del d["client_tools"]                       # an old daemon's envelope
    assert SessionInitEnvelope.from_dict(d).client_tools == []


def test_non_dict_entries_dropped():
    d = _env().to_dict()
    d["client_tools"] = [{"name": "ok"}, "garbage", 42]
    assert SessionInitEnvelope.from_dict(d).client_tools == [{"name": "ok"}]
