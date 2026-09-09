#!/usr/bin/env python3
"""Smoke-test the native ``openai`` and ``azure_openai`` providers (#508).

Answers the question "is this thing wired up correctly?" in two modes, and
the SAME assertions run in both — which is the point, because the mock mode
costs nothing and the live mode is the one that proves the vendor agrees.

    # (1) no account, no key, no money: drive both providers against a
    #     local stand-in and print what reached the wire
    python examples/provider_smoke_openai_azure.py

    # (2) the real thing
    OPENAI_API_KEY=sk-...            python examples/provider_smoke_openai_azure.py --live-openai
    AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com \
    AZURE_OPENAI_API_KEY=...         python examples/provider_smoke_openai_azure.py \
                                         --live-azure --deployment my-gpt4o-deployment

Mock mode is worth running on its own: the providers' *interesting* work is
deciding what to put on the wire, and that decision is fully observable
without a vendor.  The run prints the requests the mock received, so
deployment-name routing and the ``api-version`` query string are visible as
facts rather than claims::

    POST /v1/chat/completions
    POST /v1/responses
    POST /openai/deployments/my-gpt4o-deployment/chat/completions?api-version=2024-10-21

**What mock mode does NOT prove.**  That the vendor accepts any of it.  The
mock says yes to everything; a live run is the only way to learn that a
field was renamed, a model was retired, or a key lacks a scope.  Treat a
green mock run as "the framework is not the problem" and nothing more.

Exit status is 0 only if every selected check passed, so this is usable in
CI or as a post-install sanity check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Runnable straight from a checkout, not only from an installed venv.
_ROOT = Path(__file__).resolve().parents[1]
for _pkg in ("jaato-server", "jaato-sdk"):
    _path = str(_ROOT / _pkg)
    if (_ROOT / _pkg).is_dir() and _path not in sys.path:
        sys.path.insert(0, _path)

from jaato_sdk.plugins.model_provider.types import (  # noqa: E402
    FinishReason, Message, Part, Role,
)
from shared.plugins.model_provider.base import ProviderConfig  # noqa: E402

MOCK_HOST, MOCK_PORT = "127.0.0.1", 8123
MOCK_BASE = f"http://{MOCK_HOST}:{MOCK_PORT}"

#: Requests the mock received, as ``(method, path, body)``.  The record the
#: run prints at the end, and what the wire-shape checks assert against.
SEEN: List[Tuple[str, str, Optional[dict]]] = []


# ====================================================================
# The stand-in endpoint
# ====================================================================

def _sse(events: List[dict]) -> bytes:
    """Server-sent events in the shape both OpenAI wires stream.

    The chat wire terminates with the literal ``[DONE]`` sentinel; the
    Responses wire does not use one (its terminal ``response.completed``
    event is what ends the turn), but sending it is harmless and keeps one
    encoder for both.
    """
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return (body + "data: [DONE]\n\n").encode()


def _chat_stream(model: str) -> List[dict]:
    """A minimal streamed chat completion: two text deltas, then a finish."""
    def frame(delta: dict, finish: Optional[str] = None) -> dict:
        return {
            "id": "chatcmpl_1", "object": "chat.completion.chunk",
            "created": 0, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    return [
        frame({"role": "assistant", "content": "hello from "}),
        frame({"content": "the chat wire"}),
        frame({}, finish="stop"),
        # The usage frame arrives last, after the finish — this is what
        # ``stream_options: {include_usage: true}`` asks for, and the
        # provider reads it as evidence the stream really ended.
        {"id": "chatcmpl_1", "object": "chat.completion.chunk", "created": 0,
         "model": model, "choices": [],
         "usage": {"prompt_tokens": 7, "completion_tokens": 5,
                   "total_tokens": 12}},
    ]


def _responses_stream() -> List[dict]:
    """A minimal streamed Responses turn.

    Deltas for the UX, then the terminal ``response.completed`` carrying the
    authoritative ``output`` — the split the transport is built around.
    """
    text = "hello from the Responses wire"
    return [
        {"type": "response.output_text.delta", "output_index": 0,
         "delta": "hello from "},
        {"type": "response.output_text.delta", "output_index": 0,
         "delta": "the Responses wire"},
        {"type": "response.completed", "response": {
            "id": "resp_1", "object": "response", "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 7, "output_tokens": 5,
                      "total_tokens": 12},
        }},
    ]


class _Handler(BaseHTTPRequestHandler):
    """Serves the three shapes the two providers address.

    ``GET /v1/models`` (catalog), ``POST /v1/chat/completions`` (native chat
    and, via a different path, Azure), and ``POST /v1/responses``.  Azure's
    deployment URL is matched by suffix, so the deployment name and the
    ``api-version`` query string ride through untouched and land in
    :data:`SEEN`.
    """

    def log_message(self, *_a):  # keep the run's output to the point
        pass

    def _send_json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, events: List[dict]) -> None:
        body = _sse(events)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        SEEN.append(("GET", self.path, None))
        self._send_json({"object": "list", "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "openai"}
            for m in ("gpt-4.1", "gpt-5.1")
        ]})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        SEEN.append(("POST", self.path, body))
        streaming = bool(body.get("stream"))

        if self.path.startswith("/v1/responses"):
            if streaming:
                return self._send_sse(_responses_stream())
            return self._send_json(_responses_stream()[-1]["response"])

        model = body.get("model", "?")
        if streaming:
            return self._send_sse(_chat_stream(model))
        self._send_json({
            "id": "chatcmpl_1", "object": "chat.completion", "created": 0,
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "hello from the chat wire"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5,
                      "total_tokens": 12},
        })


def start_mock() -> HTTPServer:
    """Start the stand-in on a daemon thread and return the server.

    Also pins ``NO_PROXY`` for loopback: on a machine configured with a
    corporate ``HTTPS_PROXY`` — exactly the machines these two providers
    exist for — the SDK would otherwise try to reach 127.0.0.1 through the
    proxy and fail in a way that looks like a provider bug.
    """
    for var in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(var, "")
        os.environ[var] = f"{existing},{MOCK_HOST},localhost".lstrip(",")
    server = HTTPServer((MOCK_HOST, MOCK_PORT), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ====================================================================
# The checks — identical against the mock and against the vendor
# ====================================================================

def _one_turn(provider, *, stream: bool) -> Tuple[str, int, FinishReason]:
    """Run one turn and return ``(text, total_tokens, finish_reason)``.

    ``stream=True`` passes an ``on_chunk`` callback, which is what selects
    the streaming transport — the path a real session almost always takes,
    and the one carrying the Responses accumulator.
    """
    chunks: List[str] = []
    result = provider.complete(
        [Message(role=Role.USER, parts=[Part(text="say hello")])],
        system_instruction="Answer in one short sentence.",
        on_chunk=(lambda c: chunks.append(c)) if stream else None,
    )
    text = "".join(chunks) if (stream and chunks) else result.text
    usage = result.response.usage if result.response else None
    return text, (usage.total_tokens if usage else 0), result.finish_reason


def check_openai(*, base_url: Optional[str], api_mode: str,
                 model: str, context_length: int) -> List[str]:
    """Connect the native provider on one wire and run a turn each way."""
    from shared.plugins.model_provider.openai.provider import OpenAIProvider

    extra: Dict[str, Any] = {"context_length": context_length,
                             "api": api_mode}
    if base_url:
        extra["base_url"] = base_url
    provider = OpenAIProvider()
    provider.initialize(ProviderConfig(extra=extra))
    provider.connect(model)

    notes = [
        f"credential  : {provider.get_auth_info()}",
        f"wire        : {provider.api_mode}",
        f"context     : {provider.get_context_limit():,} tokens",
        f"modalities  : {', '.join(sorted(provider.modalities()))}",
        f"thinking    : {provider.supports_thinking()}",
    ]
    for stream in (False, True):
        text, tokens, finish = _one_turn(provider, stream=stream)
        label = "streamed" if stream else "batched "
        assert text.strip(), f"{api_mode} {label}: empty answer"
        assert finish in (FinishReason.STOP, FinishReason.UNKNOWN), \
            f"{api_mode} {label}: unexpected finish {finish}"
        notes.append(f"{label}    : {text.strip()[:60]!r} ({tokens} tokens)")
    provider.shutdown()
    return notes


def check_azure(*, endpoint: str, api_version: str, deployment: str,
                context_length: int, model_name: Optional[str]) -> List[str]:
    """Connect the Azure provider and run a turn each way.

    ``deployment`` is what jaato's ``model:`` field carries on this
    provider — the name the subscription chose, which the SDK turns into
    the URL path.
    """
    from shared.plugins.model_provider.azure_openai.provider import (
        AzureOpenAIProvider,
    )

    extra: Dict[str, Any] = {
        "endpoint": endpoint, "api_version": api_version,
        "context_length": context_length,
    }
    if model_name:
        extra["model_name"] = model_name
    provider = AzureOpenAIProvider()
    provider.initialize(ProviderConfig(extra=extra))
    provider.connect(deployment)

    notes = [
        f"credential  : {provider.get_auth_info()} ({provider.auth_method})",
        f"deployment  : {provider.model_name}",
        f"api-version : {api_version}",
        f"context     : {provider.get_context_limit():,} tokens",
        f"modalities  : {', '.join(sorted(provider.modalities()))}",
    ]
    for stream in (False, True):
        text, tokens, finish = _one_turn(provider, stream=stream)
        label = "streamed" if stream else "batched "
        assert text.strip(), f"azure {label}: empty answer"
        assert finish in (FinishReason.STOP, FinishReason.UNKNOWN), \
            f"azure {label}: unexpected finish {finish}"
        notes.append(f"{label}    : {text.strip()[:60]!r} ({tokens} tokens)")
    provider.shutdown()
    return notes


def check_wire_shapes() -> List[str]:
    """Assert the mock saw the URLs the providers are supposed to build.

    Mock mode only: it is the half of the contract a live vendor confirms
    by answering at all, and that only the stand-in can show directly.
    """
    paths = [path for method, path, _ in SEEN if method == "POST"]
    problems = []
    if not any(p.startswith("/v1/chat/completions") for p in paths):
        problems.append("native chat never reached /v1/chat/completions")
    if not any(p.startswith("/v1/responses") for p in paths):
        problems.append("the responses wire never reached /v1/responses")
    azure = [p for p in paths if p.startswith("/openai/deployments/")]
    if not azure:
        problems.append("azure never used deployment routing")
    elif not all("api-version=" in p for p in azure):
        problems.append("an azure request carried no api-version")
    if problems:
        raise AssertionError("; ".join(problems))
    return [f"{len(paths)} requests, all correctly addressed"]


# ====================================================================

def _section(title: str, notes: List[str]) -> None:
    print(f"\n  {title}")
    for note in notes:
        print(f"    {note}")


def _run_openai(args: argparse.Namespace, failures: List[str]) -> None:
    """Run the native provider's checks on both wires, recording failures.

    A missing key under ``--live-openai`` is reported once rather than
    twice: the second wire would fail for the identical reason, and two
    copies of one cause read as two problems.
    """
    for api_mode in ("chat", "responses"):
        if args.live_openai and not os.environ.get("OPENAI_API_KEY"):
            failures.append("openai: OPENAI_API_KEY is not set")
            return
        try:
            notes = check_openai(
                base_url=None if args.live_openai else f"{MOCK_BASE}/v1",
                api_mode=api_mode,
                model=args.model or "gpt-4.1",
                context_length=args.context_length,
            )
            _section(f"openai ({api_mode} wire)  OK", notes)
        except Exception as exc:  # noqa: BLE001 - report, don't abort the run
            failures.append(f"openai/{api_mode}: {type(exc).__name__}: {exc}")
            _section(f"openai ({api_mode} wire)  FAILED",
                     [f"{type(exc).__name__}: {exc}"])


def _azure_settings(args: argparse.Namespace) -> Tuple[str, str, str]:
    """Resolve ``(endpoint, deployment, api_version)`` for the Azure check.

    Live mode reads the vendor's own variables; mock mode substitutes the
    stand-in and a deployment name chosen to be obviously not a model id,
    so the routing in the printed URL is unmistakable.
    """
    if args.live_azure:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        deployment = args.deployment or os.environ.get(
            "AZURE_OPENAI_DEPLOYMENT", "")
        os.environ.setdefault("AZURE_OPENAI_API_KEY", "")
    else:
        endpoint = MOCK_BASE
        deployment = args.deployment or "my-gpt4o-deployment"
        os.environ.setdefault("AZURE_OPENAI_API_KEY", "azure-mock-key")
    return (endpoint, deployment,
            os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"))


def _run_azure(args: argparse.Namespace, failures: List[str]) -> None:
    """Run the Azure provider's checks, recording any failure.

    The two missing-configuration cases are raised by name here rather than
    left to the provider, because at this layer we know which *flag* the
    user reached for and can say what to pass.
    """
    endpoint, deployment, api_version = _azure_settings(args)
    try:
        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set")
        if not deployment:
            raise RuntimeError(
                "no deployment name — pass --deployment, or set "
                "AZURE_OPENAI_DEPLOYMENT")
        notes = check_azure(
            endpoint=endpoint, api_version=api_version,
            deployment=deployment, context_length=args.context_length,
            model_name=None if args.live_azure else "gpt-4o",
        )
        _section("azure_openai            OK", notes)
    except Exception as exc:  # noqa: BLE001 - report, don't abort the run
        failures.append(f"azure_openai: {type(exc).__name__}: {exc}")
        _section("azure_openai            FAILED",
                 [f"{type(exc).__name__}: {exc}"])


def _report(failures: List[str], fully_mocked: bool) -> int:
    """Print the verdict and return the process exit status."""
    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed."
          + ("  Mock mode proves the framework's half only — a live run is "
             "what proves the vendor's." if fully_mocked else ""))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the native OpenAI and Azure OpenAI providers.")
    parser.add_argument("--live-openai", action="store_true",
                        help="use the real api.openai.com (needs OPENAI_API_KEY)")
    parser.add_argument("--live-azure", action="store_true",
                        help="use a real Azure resource (needs AZURE_OPENAI_*)")
    parser.add_argument("--model", default=None,
                        help="OpenAI model id (live mode; default gpt-4.1)")
    parser.add_argument("--deployment", default=None,
                        help="Azure DEPLOYMENT name (live mode)")
    parser.add_argument("--context-length", type=int, default=128000,
                        help="context window to declare; required by both "
                             "providers, which refuse to guess one")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # The two providers choose mock-or-live INDEPENDENTLY, so --live-azure
    # on its own still exercises the OpenAI half against the stand-in.  The
    # mock therefore starts unless BOTH halves are live; anything else left
    # one provider pointed at an endpoint nobody had started.
    if not (args.live_openai and args.live_azure):
        start_mock()
        print(f"mock endpoint on {MOCK_BASE} — no account, no key, no spend")
        if not args.live_openai:
            os.environ.setdefault("OPENAI_API_KEY", "sk-mock-not-a-real-key")

    failures: List[str] = []
    _run_openai(args, failures)
    _run_azure(args, failures)

    # The wire-shape assertions are only meaningful when NEITHER half was
    # live: otherwise the record is half the conversation, and the missing
    # half would read as a provider that never addressed its endpoint.
    fully_mocked = not args.live_openai and not args.live_azure
    if fully_mocked:
        try:
            _section("wire shapes             OK", check_wire_shapes())
        except AssertionError as exc:
            failures.append(f"wire shapes: {exc}")
            _section("wire shapes             FAILED", [str(exc)])
        print("\n  what actually hit the wire")
        for method, path, _body in SEEN:
            print(f"    {method} {path}")

    return _report(failures, fully_mocked)


if __name__ == "__main__":
    raise SystemExit(main())
