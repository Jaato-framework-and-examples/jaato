"""The ``decide`` tool sends images and asks ``box`` questions.

Step 9 of docs/design/decision-models.md (§12.5).  An image is named by a
workspace ``path`` or a history ``attachment_id``, never by bytes; the
schema offers ``images`` and ``box`` only to a session with an entry whose
provider declares ``decision_images``; a ``box`` answer carries
``box_px``; the ledger keeps each image's digest, never its bytes.

These tests drive the real plugin, a real ``JaatoSession`` and the real
``TypeSafeProvider`` against the local decision stand-in.  No test touches
the network.
"""

from __future__ import annotations

import base64
import hashlib
from unittest.mock import MagicMock

import pytest

from jaato_sdk.media_identity import ATTACHMENT_ID_KEY, mint_attachment_id
from jaato_sdk.plugins.model_provider.types import Message, Part, Role
from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.plugins.decision.plugin import TOOL_NAME, DecisionPlugin
from jaato_server.shared.session_context import (
    isolated_current_session,
    reset_workspace_root,
    set_current_session,
    set_workspace_root,
)
from jaato_server.shared.tests.decision_standin import (
    IMAGES_HEADER,
    DecisionStandIn,
    openrouter_against,
    png_bytes,
    typesafe_against,
)
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/decision/plugin.py"
_IMAGES = "jaato-server/jaato_server/shared/plugins/decision/images.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find="        if with_images:\n",
        replace="        if False:\n",
        test="test_images_and_box_are_offered_where_an_entry_takes_images",
        because="the model is never offered the images parameter, so the "
                "feature cannot be reached",
    ),
    Reversion(
        target=_IMAGES,
        find='    if name == ".env":\n        return True\n',
        replace="",
        test="test_a_credential_file_is_never_sent",
        because="a .env file holding PNG bytes would be sent to a third party",
    ),
    Reversion(
        target=_IMAGES,
        find="        return check_path_with_jaato_containment(\n",
        replace="        return True or check_path_with_jaato_containment(\n",
        test="test_a_path_the_session_may_not_read_is_refused",
        because="the tool would send a file readFile refuses",
    ),
    Reversion(
        target=_IMAGES,
        find="    if _was_evicted(history, attachment_id):\n",
        replace="    if False:\n",
        test="test_an_evicted_attachment_is_refused_by_name",
        because="the model is told the id is unknown when the bytes were "
                "evicted, and hunts for another id",
    ),
    Reversion(
        target=_PLUGIN,
        find='        answers[qid]["box_px"] = answer.value.to_pixels(\n'
             "            source.width, source.height)\n",
        replace="        pass\n",
        test="test_a_box_answer_carries_pixel_edges",
        because="the model gets only fractions and must guess the image size",
    ),
    Reversion(
        target=_PLUGIN,
        find='        entry["images"] = [image_record(r) for r in images]\n',
        replace="        pass\n",
        test="test_the_ledger_records_digests_not_bytes",
        because="the ledger cannot say which images a decision was made on",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/explain.py",
        find='    if name == "decision":\n        return _decision_image_lines()\n',
        replace="",
        test="test_explain_states_how_images_reach_a_session",
        because="explain shows only the text-only schema, so an author "
                "never learns that decide takes images",
    ),
    Reversion(
        target=_PLUGIN,
        find="        if str(name) not in _image_names(session):\n",
        replace="        if False:\n",
        test="test_a_text_only_entry_refuses_images_before_reading",
        because="a file is read for an entry that can never send it",
    ),
]

Q_HERE = {"id": "here", "type": "noul",
          "instructions": "Is there a person in {{image:door}}?"}
Q_WHERE = {"id": "where", "type": "box", "instructions": "the person"}


@pytest.fixture
def standin():
    with DecisionStandIn() as server:
        yield server


@pytest.fixture(autouse=True)
def _restore_session():
    with isolated_current_session():
        yield


@pytest.fixture
def workspace(tmp_path):
    token = set_workspace_root(str(tmp_path))
    try:
        yield tmp_path
    finally:
        reset_workspace_root(token)


def _session(decision_models, provider=None, provider_name="typesafe"):
    ledger = MagicMock()
    runtime = MagicMock()
    runtime.provider_name = provider_name
    runtime.ledger = ledger
    runtime.create_provider.side_effect = (
        lambda model=None, provider_name=None, **_kw: provider)
    runtime.get_tool_schemas.return_value = []
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.permission_plugin = None
    runtime.reliability_plugin = None
    runtime.registry = MagicMock()
    runtime.registry.get_exposed_tool_schemas.return_value = []
    session = JaatoSession(runtime, "chat-model", provider_name=provider_name)
    session.configure(decision_models=decision_models)
    set_current_session(session)
    return session, ledger


def _vision(server, **kw):
    return _session({"eyes": {"provider": "typesafe", "model": "jev-1.14.0"}},
                    typesafe_against(server, model="jev-1.14.0"), **kw)


def _decide(images, questions=None, model="eyes", state="{{image:door}}"):
    return DecisionPlugin().get_executors()[TOOL_NAME]({
        "model": model, "state": state, "images": images,
        "questions": questions or [Q_HERE, Q_WHERE]})


def _write_png(root, rel, **kw):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    data = png_bytes(**kw)
    path.write_bytes(data)
    return data


def _schema():
    plugin = DecisionPlugin()
    return plugin.narrow_tool_schema(plugin.get_tool_schemas()[0])


# ============================================================ the schema


def test_images_and_box_are_offered_where_an_entry_takes_images(standin):
    _vision(standin)
    schema = _schema()
    assert "images" in schema.parameters["properties"]
    qtype = schema.parameters["properties"]["questions"]["items"]["properties"]["type"]
    assert "box" in qtype["enum"]
    assert "ALWAYS returned" in qtype["description"]
    assert "eyes" in schema.description


def test_a_text_only_session_is_offered_neither(standin):
    # Through openrouter only TypeSafe Jev is sent images; another
    # decision model there is text-only.
    _session({"triage": {"provider": "openrouter", "model": "acme/decide-1"}},
             openrouter_against(standin), provider_name="openrouter")
    schema = _schema()
    assert "images" not in schema.parameters["properties"]
    qtype = schema.parameters["properties"]["questions"]["items"]["properties"]["type"]
    assert "box" not in qtype["enum"]


def test_the_base_schema_carries_no_images():
    schema = DecisionPlugin().get_tool_schemas()[0]
    assert "images" not in schema.parameters["properties"]


# ============================================================ paths


def test_a_workspace_image_is_sent_with_the_header(standin, workspace):
    data = _write_png(workspace, "captures/door.png", width=400, height=300)
    _vision(standin)
    result = _decide([{"name": "door", "path": "captures/door.png"}])
    assert not isinstance(result, tuple), result
    (_path, headers, body), = standin.posts()
    assert headers.get(IMAGES_HEADER) == "true"
    (asset,) = body["assets"].values()
    assert base64.b64decode(asset["data_base64"]) == data
    assert result["answers"]["here"]["value"] == 0.95


def test_a_credential_file_is_never_sent(standin, workspace):
    _write_png(workspace, "sub/.env")
    _vision(standin)
    ok, payload = _decide([{"name": "door", "path": "sub/.env"}])
    assert ok is False and "credentials" in payload["error"]
    assert standin.posts() == []


def test_a_path_the_session_may_not_read_is_refused(standin, workspace):
    _write_png(workspace, ".jaato/shot.png")
    _vision(standin)
    ok, payload = _decide([{"name": "door", "path": ".jaato/shot.png"}])
    assert ok is False and "may read" in payload["error"]
    assert standin.posts() == []


def test_a_file_that_is_not_an_image_is_refused_naming_it(standin, workspace):
    (workspace / "notes.png").write_text("not a picture")
    _vision(standin)
    ok, payload = _decide([{"name": "door", "path": "notes.png"}])
    assert ok is False and "image 'door'" in payload["error"]
    assert standin.posts() == []


def test_an_entry_needs_exactly_one_reference(standin, workspace):
    _vision(standin)
    ok, payload = _decide([{"name": "door"}])
    assert ok is False and "exactly one of" in payload["error"]


# ============================================================ attachments


def test_an_image_in_history_is_sent_by_its_id(standin, workspace):
    data = png_bytes(width=500, height=400)
    aid = mint_attachment_id(data)
    session, _ledger = _vision(standin)
    session._history.append(Message(role=Role.USER, parts=[Part(inline_data={
        "mime_type": "image/png", "data": data, ATTACHMENT_ID_KEY: aid})]))
    result = _decide([{"name": "door", "attachment_id": aid}])
    assert not isinstance(result, tuple), result
    (_path, _headers, body), = standin.posts()
    (asset,) = body["assets"].values()
    assert base64.b64decode(asset["data_base64"]) == data


def test_an_evicted_attachment_is_refused_by_name(standin, workspace):
    aid = "att_0123456789abcdef"
    session, _ledger = _vision(standin)
    session._history.append(Message(role=Role.USER, parts=[Part.from_text(
        f"[Media evicted after the turn that consumed it — image/png, ref {aid}.]")]))
    ok, payload = _decide([{"name": "door", "attachment_id": aid}])
    assert ok is False and "evicted" in payload["error"]


def test_an_unknown_attachment_is_refused(standin, workspace):
    _vision(standin)
    ok, payload = _decide([{"name": "door", "attachment_id": "att_ffffffffffffffff"}])
    assert ok is False and "no attachment" in payload["error"]


# ============================================================ answers and records


def test_a_box_answer_carries_pixel_edges(standin, workspace):
    _write_png(workspace, "door.png", width=1000, height=500)
    _vision(standin)
    result = _decide([{"name": "door", "path": "door.png"}])
    where = result["answers"]["where"]
    assert where["value"] == {"top": 0.25, "bottom": 0.82,
                              "left": 0.44, "right": 0.84}
    assert where["image"] == "door"
    assert where["box_px"] == {"top": 125, "bottom": 410,
                               "left": 440, "right": 840}


def test_no_box_px_when_the_image_is_ambiguous(standin, workspace):
    _write_png(workspace, "a.png")
    _write_png(workspace, "b.png")
    _vision(standin)
    result = _decide([{"name": "door", "path": "a.png"},
                      {"name": "yard", "path": "b.png"}],
                     state="{{image:door}} {{image:yard}}")
    assert "box_px" not in result["answers"]["where"]


def test_the_ledger_records_digests_not_bytes(standin, workspace):
    data = _write_png(workspace, "door.png", width=400, height=300)
    _session_obj, ledger = _vision(standin)
    _decide([{"name": "door", "path": "door.png"}])
    kind, record = ledger._record.call_args.args
    assert kind == "decision"
    (image,) = record["images"]
    assert image == {"name": "door", "source": "path", "ref": "door.png",
                     "sha256": hashlib.sha256(data).hexdigest(),
                     "bytes": len(data), "format": "png",
                     "width": 400, "height": 300}


def test_a_text_only_entry_refuses_images_before_reading(standin, workspace):
    # Through openrouter only TypeSafe Jev is sent images; another
    # decision model there is text-only.
    _session({"triage": {"provider": "openrouter", "model": "acme/decide-1"}},
             openrouter_against(standin), provider_name="openrouter")
    ok, payload = _decide([{"name": "door", "path": "missing.png"}],
                          questions=[Q_HERE], model="triage")
    assert ok is False and "does not take images" in payload["error"]
    assert standin.posts() == []


# ============================================================ jaato-scaffold


def test_explain_states_how_images_reach_a_session():
    from jaato_server.shared.plugins.decision.plugin import IMAGE_RULES
    from jaato_server.shared.scaffold import explain

    data, text = explain.plugin("decision")
    assert "images and box" in text
    for rule in IMAGE_RULES:
        assert rule in text
    assert data["images"]["providers"] == ["openrouter", "typesafe"]
    assert data["images"]["parameter"]["items"]["required"] == ["name"]
