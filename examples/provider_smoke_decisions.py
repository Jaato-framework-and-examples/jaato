#!/usr/bin/env python3
"""Smoke-test ``decide()`` on the ``openrouter``, ``typesafe`` and ``laya`` providers.

Six modes, the same checks in each:

    # (1) no key, no money: drive the provider against the local stand-in
    #     the test suite uses, and print what reached it
    python examples/provider_smoke_decisions.py

    # (2) the real endpoint: one request, one question of each type
    JAATO_OPENROUTER_API_KEY=sk-or-... \\
        python examples/provider_smoke_decisions.py --live-openrouter
    # another decision model:  --model typesafe/jev-1.13

    # (3) laya against the same stand-in (its /v1/systemone path)
    python examples/provider_smoke_decisions.py --laya

    # (4) a running laya-serve (pip install "laya[serve]"; laya-serve)
    python examples/provider_smoke_decisions.py --live-laya http://localhost:8000
    # a token, when the server has LAYA_API_KEY:  JAATO_LAYA_API_KEY=... first

    # (5) typesafe against the same stand-in (TypeSafe's /v1/systemone path)
    python examples/provider_smoke_decisions.py --typesafe

    # (6) TypeSafe's own API: one request, the vendor's documented example
    JAATO_TYPESAFE_API_KEY=... python examples/provider_smoke_decisions.py --live-typesafe
    # (or TYPESAFE_API_KEY, or a key stored with 'typesafe-auth key')

    # (7) an image and a box question, on either typesafe mode: one more
    #     request, to jev-1.14.0 unless --image-model names another
    python examples/provider_smoke_decisions.py --live-typesafe --image photo.jpg

``--image PATH`` takes a still PNG or JPEG, 280-1568 px a side (it is
checked before sending and never resized).  It asks whether the image shows
a person (``noul``) and where its most prominent object is (``box``, Jev
1.14 only), and prints the box as fractions and in pixels.

The live run sends ONE request to ``POST /api/alpha/decisions`` with three
questions (``noul``, ``choice``, ``score``) about a short support ticket.
At Jev 1.13's price ($0.042 per million input tokens, output free) that
is a fraction of a cent.  ``--live-laya`` sends the same request to
``POST {host}/v1/systemone``, at no cost; ``--live-typesafe`` sends it to
``POST https://api.typesafe.ai/v1/systemone`` at TypeSafe's price.  It prints:

* what the catalog says about the model (output modalities, window);
* the request body that was sent;
* the RAW response body, exactly as received;
* what jaato parsed from it.

The raw body is the point: ``/api/alpha/`` is not a stable route, and the
parser was written from the vendor's documentation.  If the live response
differs from the documented shape, the raw body shows how, and the parse
step fails naming the question it could not read.

Exit status is 0 only when every check passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Runnable straight from a checkout, not only from an installed venv.
_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("jaato-server", "jaato-sdk"):
    _path = str(_ROOT / _pkg)
    if (_ROOT / _pkg).is_dir() and _path not in sys.path:
        sys.path.insert(0, _path)

from jaato_sdk.plugins.model_provider.decisions import (  # noqa: E402
    DecisionModelOnlyError,
    DecisionQuestion,
    DecisionResponseError,
    DecisionResult,
    build_decision_request,
)
from jaato_server.shared.plugins.model_provider.base import ProviderConfig  # noqa: E402
from jaato_server.shared.plugins.model_provider.openrouter import (  # noqa: E402
    OpenRouterProvider,
)

DEFAULT_MODEL = "typesafe/jev-1.13"
LAYA_MODEL = "convaiinnovations/laya"
TYPESAFE_MODEL = "jev-latest"
#: ``box`` is Jev 1.14 only.
IMAGE_MODEL = "jev-1.14.0"

IMAGE_STATE = {"photo": "{{image:photo}}"}
IMAGE_QUESTIONS: Dict[str, DecisionQuestion] = {
    "person": DecisionQuestion("noul", "Does `photo` show a person?"),
    "subject": DecisionQuestion("box", "the most prominent object in `photo`"),
}

STATE = (
    "Subject: payouts failing\n\n"
    "Help! My payouts have been failing for 3 days and my suppliers are "
    "threatening to stop shipping. I've emailed twice with no answer."
)

QUESTIONS: Dict[str, DecisionQuestion] = {
    "is_urgent": DecisionQuestion(
        "noul", "Does this message convey urgency?",
        {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
    ),
    "department": DecisionQuestion(
        "choice", "Which team should handle this?",
        {"billing": "Payments, invoicing, refunds, payouts",
         "technical": "Bugs, outages, integrations",
         "sales": None},
    ),
    "frustration": DecisionQuestion(
        "score", "How frustrated is the customer?",
        ["Calm", "Frustrated", "Very angry"],
    ),
}


def _section(title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 66 - len(title)))


def _dump(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _describe(result: DecisionResult) -> None:
    print(f"model answered: {result.model}")
    for qid, answer in result.answers.items():
        extra = []
        if answer.confidence is not None:
            extra.append(f"confidence={answer.confidence:.2f}")
        if answer.probabilities:
            extra.append("p=" + json.dumps(answer.probabilities))
        print(f"  {qid:12} {answer.type:6} -> {answer.value!r}  {'  '.join(extra)}")
    usage = result.usage
    if usage.reported:
        cost = f", cost ${usage.cost_usd:.8f}" if usage.cost_usd is not None else ""
        print(f"usage: {usage.prompt_tokens} in, {usage.output_tokens} out{cost}")
    else:
        print("usage: not reported by the endpoint")


def _check_answers(result: DecisionResult, failures: List[str]) -> None:
    expected = {qid: q.type for qid, q in QUESTIONS.items()}
    got = {qid: a.type for qid, a in result.answers.items()}
    if got != expected:
        failures.append(f"answers {got} do not match the questions {expected}")
    if result.answers.get("is_urgent") and result.answers["is_urgent"].confidence is not None:
        failures.append("a noul answer carried a confidence; the docs say it has none")


def check_catalog(provider: Any, model: str, failures: List[str]) -> None:
    _section("catalog")
    outputs = sorted(provider.output_modalities(model))
    print(f"output modalities: {outputs}")
    print(f"decisions only:    {provider.is_decisions_only(model)}")
    print(f"context window:    {provider.get_context_limit() or 'unknown'}")
    if not provider.is_decisions_only(model):
        failures.append(f"{model} is not reported as decisions-only by the catalog")


def check_complete_refuses(provider: Any, failures: List[str]) -> None:
    _section("complete() on a decision model")
    try:
        provider.complete([])
    except DecisionModelOnlyError as exc:
        print(f"refused, as it should be: {exc}")
        return
    except Exception as exc:   # noqa: BLE001 - reported, not hidden
        failures.append(f"complete() raised {type(exc).__name__}, not DecisionModelOnlyError")
        return
    failures.append("complete() did not refuse a decisions-only model")


def check_decide(provider: Any, model: str, failures: List[str]) -> None:
    _section("request sent")
    _dump(build_decision_request(STATE, QUESTIONS, model))
    try:
        result = provider.decide(STATE, QUESTIONS, model=model)
    except DecisionResponseError as exc:
        _section("raw response (could not be parsed)")
        _dump(exc.body)
        failures.append(f"parse: {exc}")
        return
    except Exception as exc:   # noqa: BLE001 - the whole point is to see it
        _section("request failed")
        print(f"{type(exc).__name__}: {exc}")
        failures.append(f"decide() raised {type(exc).__name__}")
        return
    _section("raw response")
    _dump(result.raw)
    _section("parsed")
    _describe(result)
    _check_answers(result, failures)


def check_image(provider: Any, model: str, path: str, failures: List[str]) -> None:
    """One request with an image: a ``noul`` and a ``box`` question."""
    from jaato_sdk.plugins.model_provider.decision_images import read_image_info
    from jaato_sdk.plugins.model_provider.decisions import (
        DecisionBox, DecisionImage, DecisionQuestionError,
    )

    _section(f"image request ({path}, {model})")
    data = open(path, "rb").read()
    try:
        info = read_image_info(data)
    except ValueError as exc:
        failures.append(f"image: {exc}")
        print(exc)
        return
    print(f"{info.format}, {info.width}x{info.height}, {len(data)} bytes")
    image = DecisionImage(f"image/{info.format}", data, "photo")
    try:
        result = provider.decide(IMAGE_STATE, IMAGE_QUESTIONS, images=[image], model=model)
    except DecisionQuestionError as exc:
        failures.append(f"image refused before sending: {exc}")
        print(exc)
        return
    except DecisionResponseError as exc:
        _section("raw response (could not be parsed)")
        _dump(exc.body)
        failures.append(f"image parse: {exc}")
        return
    except Exception as exc:   # noqa: BLE001 - the whole point is to see it
        print(f"{type(exc).__name__}: {exc}")
        failures.append(f"image decide() raised {type(exc).__name__}")
        return
    _section("raw response")
    _dump(result.raw)
    _section("parsed")
    _describe(result)
    box = result.answers["subject"].value
    if isinstance(box, DecisionBox):
        print(f"box in pixels: {box.to_pixels(info.width, info.height)}")
    else:
        failures.append(f"box answer is {box!r}, not a DecisionBox")


def run(provider: Any, model: str, image: Optional[str] = None,
        image_model: str = IMAGE_MODEL) -> List[str]:
    failures: List[str] = []
    check_catalog(provider, model, failures)
    check_complete_refuses(provider, failures)
    check_decide(provider, model, failures)
    if image:
        check_image(provider, image_model, image, failures)
    return failures


def run_mock(model: str, laya: bool = False, typesafe: bool = False,
             image: Optional[str] = None, image_model: str = IMAGE_MODEL) -> List[str]:
    from jaato_server.shared.tests.decision_standin import (
        DecisionStandIn,
        laya_against,
        openrouter_against,
        typesafe_against,
    )

    against = (laya_against if laya else
               typesafe_against if typesafe else openrouter_against)
    with DecisionStandIn() as server:
        print(f"stand-in at {server.host} (answers anything, proves only our side)")
        failures = run(against(server, model=model), model, image, image_model)
        _section("requests the stand-in received")
        for method, path, _headers, _body in server.seen:
            print(f"  {method} {path}")
        return failures


def run_live(model: str) -> List[str]:
    provider = OpenRouterProvider()
    try:
        provider.initialize(ProviderConfig())   # env key, then stored credentials
    except Exception as exc:   # noqa: BLE001
        print(f"cannot initialize the provider: {exc}")
        return ["initialize() failed"]
    provider.connect(model)
    return run(provider, model)


def run_live_laya(host: str, model: str) -> List[str]:
    from jaato_server.shared.plugins.model_provider.laya import LayaProvider

    provider = LayaProvider()
    provider.initialize(ProviderConfig(extra={"host": host}))   # token from env
    provider.connect(model)
    print(f"laya-serve at {host}")
    return run(provider, model)


def run_live_typesafe(model: str, image: Optional[str] = None,
                      image_model: str = IMAGE_MODEL) -> List[str]:
    from jaato_server.shared.plugins.model_provider.typesafe import TypeSafeProvider

    provider = TypeSafeProvider()
    try:
        # The knob is not set here, so the key comes from
        # JAATO_TYPESAFE_API_KEY, TYPESAFE_API_KEY or typesafe-auth.
        provider.initialize(ProviderConfig())
    except Exception as exc:   # noqa: BLE001
        print(f"cannot initialize the provider: {exc}")
        return ["initialize() failed"]
    provider.connect(model)
    print(f"TypeSafe, {provider.get_auth_info()}")
    return run(provider, model, image, image_model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--live-openrouter", action="store_true",
                        help="send one request to the real OpenRouter endpoint")
    parser.add_argument("--laya", action="store_true",
                        help="drive the laya provider against the stand-in")
    parser.add_argument("--live-laya", metavar="HOST",
                        help="send one request to a running laya-serve")
    parser.add_argument("--typesafe", action="store_true",
                        help="drive the typesafe provider against the stand-in")
    parser.add_argument("--live-typesafe", action="store_true",
                        help="send one request to TypeSafe's own API")
    parser.add_argument("--model", default=None,
                        help=f"decision model (default {DEFAULT_MODEL}; "
                             f"{LAYA_MODEL} for laya, {TYPESAFE_MODEL} for typesafe)")
    parser.add_argument("--image", metavar="PATH",
                        help="also ask about this PNG/JPEG (typesafe modes only)")
    parser.add_argument("--image-model", default=IMAGE_MODEL,
                        help=f"model for the image request (default {IMAGE_MODEL})")
    args = parser.parse_args()
    if args.image and not (args.typesafe or args.live_typesafe):
        parser.error("--image needs --typesafe or --live-typesafe: only the "
                     "typesafe provider sends images")

    laya = bool(args.laya or args.live_laya)
    typesafe = bool(args.typesafe or args.live_typesafe)
    model = args.model or (LAYA_MODEL if laya else
                           TYPESAFE_MODEL if typesafe else DEFAULT_MODEL)
    live = bool(args.live_openrouter or args.live_laya or args.live_typesafe)
    if args.live_laya:
        failures = run_live_laya(args.live_laya, model)
    elif args.live_typesafe:
        failures = run_live_typesafe(model, args.image, args.image_model)
    elif args.live_openrouter:
        failures = run_live(model)
    else:
        failures = run_mock(model, laya=laya, typesafe=typesafe,
                            image=args.image, image_model=args.image_model)
    _section("result")
    if failures:
        for failure in failures:
            print(f"FAIL  {failure}")
        return 1
    print("PASS" + ("" if live else
                    "  (mock only: run --live-openrouter, --live-typesafe or "
                    "--live-laya HOST to check a real endpoint)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
