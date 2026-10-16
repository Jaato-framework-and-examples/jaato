"""Images and ``box`` on OpenRouter: step 10 of docs/design/decision-models.md.

The same TypeSafe Jev is reached two ways.  TypeSafe's own API takes an
image as a top-level ``assets`` map the text refers to; OpenRouter's
decision endpoint takes it as an ``image_url`` part inside the text, with
the string that held ``{{image:NAME}}`` split around it.  Both are built by
one substitution walk in ``jaato_sdk.plugins.model_provider.decisions``
(``image_layout``), so every local check is the same for both; only the
layout differs.

OpenRouter reaches other decision models too, each with its own image
limits; the local checks are Jev's, so jaato sends images through
OpenRouter to Jev only, and ``box`` only to Jev 1.14 and later.

The SDK half is checked on its own; the provider half is driven against the
local stand-in, which holds a request to OpenRouter's documented image
rules.  No test touches the network;
``examples/provider_smoke_decisions.py --live-openrouter --image PATH`` is
the check against the real endpoint.
"""

from __future__ import annotations

import base64
import os
import struct
import zlib
from unittest.mock import MagicMock

import pytest

from jaato_sdk.plugins.model_provider.decisions import (
    DecisionBox,
    DecisionImage,
    DecisionImagesUnsupportedError,
    DecisionQuestion,
    DecisionQuestionError,
    build_decision_request,
)
from jaato_server.shared.jaato_session import JaatoSession
from jaato_server.shared.plugins.model_provider._decision_capability import (
    model_declares_decision_images,
)
from jaato_server.shared.plugins.model_provider.openrouter.decisions import (
    answers_box,
    takes_images,
)
from jaato_server.shared.session_context import isolated_current_session
from jaato_server.shared.tests.decision_standin import (
    IMAGES_HEADER,
    DecisionStandIn,
    jpeg_bytes,
    openrouter_against,
    png_bytes,
)
from jaato_server.shared.tests.reversion import Reversion

_SDK = "jaato-sdk/jaato_sdk/plugins/model_provider/decisions.py"
_OR = "jaato-server/jaato_server/shared/plugins/model_provider/openrouter/provider.py"
_CAP = "jaato-server/jaato_server/shared/plugins/model_provider/_decision_capability.py"

REVERSIONS = [
    Reversion(
        target=_SDK,
        find="            if self.parts and IMAGE_TOKEN_RE.search(value):\n",
        replace="            if False:\n",
        test="test_a_token_in_a_string_becomes_an_image_part",
        because="OpenRouter would be sent the literal {{image:NAME}} text "
                "and answer about it with a plausible probability",
    ),
    Reversion(
        target=_OR,
        find="        if images and not takes_images(model):\n",
        replace="        if False:\n",
        test="test_an_image_for_another_decision_model_is_refused_before_sending",
        because="an image would go to a model whose limits nothing checked",
    ),
    Reversion(
        target=_OR,
        find='            if question.type == "box" and not answers_box(model):\n',
        replace="            if False:\n",
        test="test_box_on_jev_1_13_is_refused_before_sending",
        because="a box question would be sent to a model that answers 400",
    ),
    Reversion(
        target=_OR,
        find="json.dumps(_without_image_data(body), ensure_ascii=False",
        replace="json.dumps(body, ensure_ascii=False",
        test="test_image_bytes_do_not_count_against_the_window",
        because="an image's base64 would be counted as text, refusing a "
                "request that fits",
    ),
    Reversion(
        target=_CAP,
        find='    per_model = getattr(module, "decision_images_for_model", None)\n',
        replace="    per_model = None\n",
        test="test_only_a_jev_entry_on_openrouter_is_offered_images",
        because="every openrouter decision entry would be offered images "
                "that only Jev is sent",
    ),
]

Q_HERE = DecisionQuestion("noul", "Is there a person in {{image:door}}?")
Q_WHERE = DecisionQuestion("box", "the person")


@pytest.fixture
def standin():
    with DecisionStandIn() as server:
        yield server


@pytest.fixture(autouse=True)
def _restore_session():
    with isolated_current_session():
        yield


def _part(data: bytes, mime: str = "image/png") -> dict:
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _noisy_png(width: int, height: int) -> bytes:
    """A greyscale PNG of random pixels, which zlib cannot shrink."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))
    rows = b"".join(b"\x00" + os.urandom(width) for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 0)) + chunk(b"IEND", b""))


# ============================================================ the parts layout


def test_a_token_in_a_string_becomes_an_image_part():
    data = png_bytes()
    body = build_decision_request(
        "Photo: {{image:door}} taken at noon", {"here": Q_HERE}, "typesafe/jev-1.14",
        [DecisionImage("image/png", data, "door")], image_layout="parts")
    assert body["state"] == ["Photo: ", _part(data), " taken at noon"]
    assert body["questions"]["here"]["instructions"] == [
        "Is there a person in ", _part(data), "?"]
    assert "assets" not in body
    assert "jaato-image-" not in str(body)


def test_a_state_array_stays_one_flat_array():
    data = png_bytes()
    body = build_decision_request(
        ["Listing photo:", "{{image:door}}", {"caption": "front {{image:door}}"}],
        {"here": Q_HERE}, "typesafe/jev-1.14",
        [DecisionImage("image/png", data, "door")], image_layout="parts")
    assert body["state"] == ["Listing photo:", _part(data),
                             {"caption": ["front ", _part(data)]}]


def test_criteria_take_image_parts_too():
    data = png_bytes()
    question = DecisionQuestion("choice", "Which is the door in {{image:door}}?",
                                {"left": "the left one in {{image:door}}", "right": None})
    body = build_decision_request(
        "x", {"pick": question}, "typesafe/jev-1.14",
        [DecisionImage("image/png", data, "door")], image_layout="parts")
    assert body["questions"]["pick"]["criteria"]["left"] == [
        "the left one in ", _part(data)]


def test_a_jpeg_is_sent_as_image_jpeg():
    data = jpeg_bytes()
    body = build_decision_request(
        "{{image:door}}", {"here": Q_HERE}, "typesafe/jev-1.14",
        [DecisionImage("image/jpeg", data, "door")], image_layout="parts")
    assert body["state"] == [_part(data, "image/jpeg")]


@pytest.mark.parametrize("state, reason", [
    ("{{image:other}}", "names no image"),
    ("no token here", "is not referenced"),
    ({"{{image:door}}": "x"}, "object key"),
])
def test_the_parts_layout_is_held_to_the_same_checks(state, reason):
    question = DecisionQuestion("noul", "Is it open?")
    with pytest.raises(DecisionQuestionError, match=reason):
        build_decision_request(state, {"q": question}, "typesafe/jev-1.14",
                               [DecisionImage("image/png", png_bytes(), "door")],
                               image_layout="parts")


def test_an_unknown_layout_is_refused():
    with pytest.raises(ValueError, match="image_layout"):
        build_decision_request("x", {"here": Q_HERE}, "m", image_layout="inline")


# ============================================================ which models


@pytest.mark.parametrize("model, images, box", [
    ("typesafe/jev-1.14", True, True),
    ("~typesafe/jev-latest", True, True),
    ("typesafe/jev-1.14.0", True, True),
    ("typesafe/jev-1.13", True, False),
    ("acme/decide-1", False, False),
    ("", False, False),
])
def test_images_go_to_jev_and_box_to_jev_1_14(model, images, box):
    assert takes_images(model) is images
    assert answers_box(model) is box


def _session(decision_models):
    runtime = MagicMock()
    runtime.provider_name = "openrouter"
    runtime.get_tool_schemas.return_value = []
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.permission_plugin = None
    runtime.reliability_plugin = None
    runtime.registry = MagicMock()
    runtime.registry.get_exposed_tool_schemas.return_value = []
    session = JaatoSession(runtime, "chat-model", provider_name="openrouter")
    session.configure(decision_models=decision_models)
    return session


def test_only_a_jev_entry_on_openrouter_is_offered_images():
    session = _session({
        "eyes": {"provider": "openrouter", "model": "typesafe/jev-1.14"},
        "other": {"provider": "openrouter", "model": "acme/decide-1"},
    })
    assert session.decision_image_model_names() == ["eyes"]


def test_a_per_model_rule_that_raises_answers_no(monkeypatch):
    from jaato_server.shared.plugins.model_provider import openrouter

    def broken(model):
        raise RuntimeError("boom")
    monkeypatch.setattr(openrouter, "decision_images_for_model", broken)
    assert model_declares_decision_images("openrouter", "typesafe/jev-1.14") is False


# ============================================================ on the wire


def test_an_image_reaches_openrouter_as_a_part_and_box_is_answered(standin):
    data = png_bytes(400, 300)
    result = openrouter_against(standin).decide(
        "{{image:door}}", {"here": Q_HERE, "where": Q_WHERE},
        images=[DecisionImage("image/png", data, "door")], model="typesafe/jev-1.14")
    (path, headers, body), = standin.posts()
    assert path == "/api/alpha/decisions"
    assert body["state"] == [_part(data)]
    assert body["questions"]["where"] == {"type": "box", "instructions": "the person"}
    assert IMAGES_HEADER.lower() not in {k.lower() for k in headers}
    assert isinstance(result.answers["where"].value, DecisionBox)
    assert result.answers["here"].value == pytest.approx(0.95)


def test_an_image_for_another_decision_model_is_refused_before_sending(standin):
    with pytest.raises(DecisionImagesUnsupportedError, match="acme/decide-1"):
        openrouter_against(standin).decide(
            "{{image:door}}", {"here": Q_HERE},
            images=[DecisionImage("image/png", png_bytes(), "door")],
            model="acme/decide-1")
    assert standin.posts() == []


def test_box_on_jev_1_13_is_refused_before_sending(standin):
    with pytest.raises(DecisionQuestionError) as excinfo:
        openrouter_against(standin).decide(
            "{{image:door}}", {"here": Q_HERE, "where": Q_WHERE},
            images=[DecisionImage("image/png", png_bytes(), "door")],
            model="typesafe/jev-1.13")
    assert excinfo.value.question_id == "where"
    assert standin.posts() == []


def test_image_bytes_do_not_count_against_the_window(standin):
    # Jev 1.13's window on the stand-in is 32,000 tokens; this image's
    # base64 alone is ~53,000 tokens' worth of characters.
    data = _noisy_png(400, 400)
    assert len(base64.b64encode(data)) // 4 > 32_000
    result = openrouter_against(standin).decide(
        "{{image:door}}", {"here": Q_HERE},
        images=[DecisionImage("image/png", data, "door")])
    assert len(standin.posts()) == 1
    assert result.answers["here"].value == pytest.approx(0.95)


def test_text_over_the_window_is_still_refused(standin):
    with pytest.raises(DecisionQuestionError, match="window"):
        openrouter_against(standin).decide(
            "word " * 40_000 + "{{image:door}}", {"here": Q_HERE},
            images=[DecisionImage("image/png", png_bytes(), "door")])
    assert standin.posts() == []
