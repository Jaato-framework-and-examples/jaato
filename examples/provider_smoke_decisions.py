#!/usr/bin/env python3
"""Smoke-test ``decide()`` on the ``openrouter`` and ``laya`` providers.

Four modes, the same checks in each:

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

The live run sends ONE request to ``POST /api/alpha/decisions`` with three
questions (``noul``, ``choice``, ``score``) about a short support ticket.
At Jev 1.13's price ($0.042 per million input tokens, output free) that
is a fraction of a cent.  ``--live-laya`` sends the same request to
``POST {host}/v1/systemone``, at no cost.  It prints:

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
from typing import Any, Dict, List

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


def run(provider: Any, model: str) -> List[str]:
    failures: List[str] = []
    check_catalog(provider, model, failures)
    check_complete_refuses(provider, failures)
    check_decide(provider, model, failures)
    return failures


def run_mock(model: str, laya: bool = False) -> List[str]:
    from jaato_server.shared.tests.decision_standin import (
        DecisionStandIn,
        laya_against,
        openrouter_against,
    )

    against = laya_against if laya else openrouter_against
    with DecisionStandIn() as server:
        print(f"stand-in at {server.host} (answers anything, proves only our side)")
        failures = run(against(server, model=model), model)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--live-openrouter", action="store_true",
                        help="send one request to the real OpenRouter endpoint")
    parser.add_argument("--laya", action="store_true",
                        help="drive the laya provider against the stand-in")
    parser.add_argument("--live-laya", metavar="HOST",
                        help="send one request to a running laya-serve")
    parser.add_argument("--model", default=None,
                        help=f"decision model (default {DEFAULT_MODEL}, "
                             f"or {LAYA_MODEL} for laya)")
    args = parser.parse_args()

    laya = bool(args.laya or args.live_laya)
    model = args.model or (LAYA_MODEL if laya else DEFAULT_MODEL)
    live = bool(args.live_openrouter or args.live_laya)
    if args.live_laya:
        failures = run_live_laya(args.live_laya, model)
    elif args.live_openrouter:
        failures = run_live(model)
    else:
        failures = run_mock(model, laya=laya)
    _section("result")
    if failures:
        for failure in failures:
            print(f"FAIL  {failure}")
        return 1
    print("PASS" + ("" if live else
                    "  (mock only: run --live-openrouter or --live-laya HOST "
                    "to check a real endpoint)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
