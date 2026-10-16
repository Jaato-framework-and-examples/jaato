"""Images and ``box`` in a decision: step 8 of docs/design/decision-models.md.

The contract (§12.2-§12.4): ``{{image:NAME}}`` substituted by a minted
reference, ``assets`` built, the local image checks (format and size from
the bytes, with no Pillow), ``box`` questions and answers, and the opt-in
header.  Then the ``typesafe`` adapter, which declares ``decision_images``,
driven against the decision stand-in, which holds a request to TypeSafe's
image rules; ``openrouter`` and ``laya`` refusing images is the
conformance guard's (``test_undeclared_decision_images_are_refused``).

No test touches the network.
"""

from __future__ import annotations

import base64
import json

import pytest

from jaato_sdk.plugins.model_provider.decision_images import (
    MAX_SIDE,
    read_image_info,
)
from jaato_sdk.plugins.model_provider.decisions import (
    IMAGE_REFERENCE_PREFIX,
    MAX_IMAGES,
    DecisionBox,
    DecisionImage,
    DecisionQuestion,
    DecisionQuestionError,
    DecisionResponseError,
    build_decision_request,
    needs_image_header,
    parse_decision_response,
)
from jaato_server.shared.tests.decision_standin import (
    BOX_ANSWER,
    IMAGES_HEADER,
    DecisionStandIn,
    jpeg_bytes,
    png_bytes,
    typesafe_against,
)
from jaato_server.shared.tests.reversion import Reversion

_CONTRACT = "jaato-sdk/jaato_sdk/plugins/model_provider/decisions.py"
_IMAGES = "jaato-sdk/jaato_sdk/plugins/model_provider/decision_images.py"
_TYPESAFE = "jaato-server/jaato_server/shared/plugins/model_provider/typesafe/provider.py"

REVERSIONS = [
    Reversion(
        target=_CONTRACT,
        find="    unused = [name for name, count in sub.uses.items() if not count]\n    if unused:\n",
        replace="    unused = []\n    if unused:\n",
        test="test_an_image_no_token_names_is_refused",
        because="an image the questions never mention would reach the vendor, "
                "which answers 400",
    ),
    Reversion(
        target=_CONTRACT,
        find="                if isinstance(key, str) and IMAGE_TOKEN_RE.search(key):\n",
        replace="                if False:\n",
        test="test_a_token_in_an_object_key_is_refused",
        because="a token in a key is never replaced, so the model would read "
                "{{image:x}} as literal text",
    ),
    Reversion(
        target=_CONTRACT,
        find="        if info.format != declared:\n",
        replace="        if False:\n",
        test="test_a_mime_that_disagrees_with_the_bytes_is_refused",
        because="a JPEG declared as PNG would be relabelled and sent (#829)",
    ),
    Reversion(
        target=_IMAGES,
        find="        if kind == b\"acTL\":\n",
        replace="        if False:\n",
        test="test_an_animated_png_is_refused",
        because="an APNG would be sent and rejected by the vendor",
    ),
    Reversion(
        target=_IMAGES,
        find="        if not MIN_SIDE <= value <= MAX_SIDE:\n",
        replace="        if False:\n",
        test="test_an_image_out_of_the_size_limits_is_refused_naming_it",
        because="an oversize screenshot would cost a round trip to be refused, "
                "with the vendor's message instead of the limit",
    ),
    Reversion(
        target=_CONTRACT,
        find="    if edges[\"left\"] >= edges[\"right\"] or edges[\"top\"] >= edges[\"bottom\"]:\n",
        replace="    if False:\n",
        test="test_an_inverted_box_is_refused_naming_the_question",
        because="a box with left past right would be handed to the caller as an answer",
    ),
    Reversion(
        target=_TYPESAFE,
        find="        if needs_image_header(body):\n            headers[IMAGES_HEADER] = \"true\"\n",
        replace="",
        test="test_typesafe_sends_the_image_with_its_header",
        because="every image request to TypeSafe would be refused for the "
                "missing opt-in header",
    ),
]

Q_HERE = DecisionQuestion("noul", "Is there a person in `photo`?")
Q_WHERE = DecisionQuestion("box", "the person in `photo`")


def _png(name: str = "door", **kw) -> DecisionImage:
    return DecisionImage("image/png", png_bytes(**kw), name)


# ---------------------------------------------------------------- contract


def test_a_text_request_is_unchanged():
    body = build_decision_request("Payouts failing", {"u": Q_HERE}, "jev-latest")
    assert body == {"state": "Payouts failing", "model": "jev-latest",
                    "questions": {"u": {"type": "noul",
                                        "instructions": Q_HERE.instructions}}}
    assert not needs_image_header(body)


def test_tokens_become_one_minted_reference_per_image():
    body = build_decision_request(
        {"photo": "{{image:door}}", "note": "see {{image:door}} and {{image:yard}}"},
        {"here": DecisionQuestion("noul", "Person in {{image:door}}?"),
         "pick": DecisionQuestion("choice", "Which?",
                                  {"door": "the {{image:door}} one", "yard": None})},
        "jev-1.14.0",
        [_png("door"), DecisionImage("image/jpeg", jpeg_bytes(), "yard")],
    )
    door, yard = body["assets"]
    assert door.startswith(IMAGE_REFERENCE_PREFIX) and door != yard
    assert body["state"] == {"photo": door, "note": f"see {door} and {yard}"}
    assert body["questions"]["here"]["instructions"] == f"Person in {door}?"
    assert body["questions"]["pick"]["criteria"]["door"] == f"the {door} one"
    assert body["assets"][door] == {
        "type": "image", "format": "png",
        "data_base64": base64.b64encode(png_bytes()).decode()}
    assert body["assets"][yard]["format"] == "jpeg"
    assert "{{image:" not in json.dumps({k: v for k, v in body.items() if k != "assets"})
    assert needs_image_header(body)


def test_a_token_naming_no_image_is_refused():
    with pytest.raises(DecisionQuestionError, match="names no image") as info:
        build_decision_request({"photo": "{{image:door}}"},
                               {"here": DecisionQuestion("noul", "In {{image:gate}}?")},
                               "jev-latest", [_png("door")])
    assert info.value.question_id == "here"


def test_an_image_no_token_names_is_refused():
    with pytest.raises(DecisionQuestionError, match="'yard' is not referenced"):
        build_decision_request({"photo": "{{image:door}}"}, {"here": Q_HERE},
                               "jev-latest", [_png("door"), _png("yard")])


def test_a_token_in_an_object_key_is_refused():
    with pytest.raises(DecisionQuestionError, match="object key"):
        build_decision_request({"{{image:door}}": "photo"}, {"here": Q_HERE},
                               "jev-latest", [_png("door")])


def test_a_mime_that_disagrees_with_the_bytes_is_refused():
    with pytest.raises(DecisionQuestionError, match="bytes are jpeg"):
        build_decision_request(
            {"photo": "{{image:door}}"}, {"here": Q_HERE}, "jev-latest",
            [DecisionImage("image/png", jpeg_bytes(), "door")])


def test_an_animated_png_is_refused():
    with pytest.raises(DecisionQuestionError, match="animated PNG"):
        build_decision_request({"photo": "{{image:door}}"}, {"here": Q_HERE},
                               "jev-latest", [_png("door", animated=True)])


def test_formats_other_than_png_and_jpeg_are_refused():
    gif = DecisionImage("image/gif", b"GIF89a" + b"\x00" * 20, "door")
    with pytest.raises(DecisionQuestionError, match="image/gif"):
        build_decision_request({"photo": "{{image:door}}"}, {"here": Q_HERE},
                               "jev-latest", [gif])
    webp = DecisionImage("image/png", b"RIFF\x00\x00\x00\x00WEBPVP8 ", "door")
    with pytest.raises(DecisionQuestionError, match="not a PNG or JPEG"):
        build_decision_request({"photo": "{{image:door}}"}, {"here": Q_HERE},
                               "jev-latest", [webp])


def test_an_image_out_of_the_size_limits_is_refused_naming_it():
    wide = _png("shot", width=MAX_SIDE + 1, height=300)
    with pytest.raises(DecisionQuestionError, match=r"image 'shot': width 1569 px"):
        build_decision_request({"photo": "{{image:shot}}"}, {"here": Q_HERE},
                               "jev-latest", [wide])
    small = DecisionImage("image/jpeg", jpeg_bytes(200, 400), "tiny")
    with pytest.raises(DecisionQuestionError, match="width 200 px"):
        build_decision_request({"photo": "{{image:tiny}}"}, {"here": Q_HERE},
                               "jev-latest", [small])


def test_the_header_is_read_from_the_bytes():
    assert read_image_info(png_bytes(640, 480)).width == 640
    assert (read_image_info(jpeg_bytes(800, 600)).width,
            read_image_info(jpeg_bytes(800, 600)).height) == (800, 600)


def test_too_many_images_and_duplicate_names_are_refused():
    many = [_png(f"i{n}") for n in range(MAX_IMAGES + 1)]
    with pytest.raises(DecisionQuestionError, match=f"at most {MAX_IMAGES}"):
        build_decision_request("x", {"here": Q_HERE}, "jev-latest", many)
    with pytest.raises(DecisionQuestionError, match="two images are named"):
        build_decision_request("{{image:a}}", {"here": Q_HERE}, "jev-latest",
                               [_png("a"), _png("a")])


def test_a_box_question_needs_an_image_and_no_criteria():
    with pytest.raises(DecisionQuestionError, match="needs an image") as info:
        build_decision_request("a photo", {"where": Q_WHERE}, "jev-1.14.0")
    assert info.value.question_id == "where"
    with pytest.raises(DecisionQuestionError, match="box takes no criteria"):
        build_decision_request("{{image:door}}",
                               {"where": DecisionQuestion("box", "it", {"a": None})},
                               "jev-1.14.0", [_png("door")])


def test_a_box_answer_is_read_and_converts_to_pixels():
    result = parse_decision_response(
        {"model": "jev-1.14.0", "answers": {"where": BOX_ANSWER}}, {"where": Q_WHERE})
    box = result.answers["where"].value
    assert box == DecisionBox(top=0.25, bottom=0.82, left=0.44, right=0.84)
    assert result.answers["where"].confidence == 0.9
    assert result.answers["where"].probabilities is None
    assert box.to_pixels(1000, 500) == {"top": 125, "bottom": 410,
                                        "left": 440, "right": 840}


def test_an_inverted_box_is_refused_naming_the_question():
    bad = {"type": "box", "box": {"top": 0.5, "bottom": 0.4, "left": 0.1, "right": 0.2}}
    with pytest.raises(DecisionResponseError, match="left < right") as info:
        parse_decision_response({"answers": {"where": bad}}, {"where": Q_WHERE})
    assert info.value.question_id == "where"
    out = {"type": "box", "box": {"top": 0.1, "bottom": 1.2, "left": 0.1, "right": 0.2}}
    with pytest.raises(DecisionResponseError, match="bottom"):
        parse_decision_response({"answers": {"where": out}}, {"where": Q_WHERE})


# ---------------------------------------------------------------- typesafe


def test_typesafe_sends_the_image_with_its_header():
    with DecisionStandIn() as server:
        result = typesafe_against(server, model="jev-1.14.0").decide(
            {"photo": "{{image:door}}"}, {"here": Q_HERE, "where": Q_WHERE},
            images=[_png("door")])
        (path, headers, body), = server.posts()
    assert path == "/v1/systemone"
    assert headers.get(IMAGES_HEADER) == "true"
    assert list(body["assets"]) == [body["state"]["photo"]]
    assert result.answers["where"].value.left == 0.44


def test_typesafe_sends_no_header_for_text():
    with DecisionStandIn() as server:
        typesafe_against(server).decide("Payouts failing", {"here": Q_HERE})
        (_path, headers, body), = server.posts()
    assert IMAGES_HEADER not in headers and "assets" not in body


def test_typesafe_refuses_a_bad_image_before_sending():
    with DecisionStandIn() as server:
        with pytest.raises(DecisionQuestionError, match="animated"):
            typesafe_against(server).decide(
                "{{image:door}}", {"here": Q_HERE}, images=[_png("door", animated=True)])
        assert server.posts() == []


def test_typesafe_reads_images_as_an_input_modality():
    with DecisionStandIn() as server:
        provider = typesafe_against(server)
    assert provider.supports_modality("image")
