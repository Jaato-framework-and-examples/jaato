"""A local stand-in for OpenRouter's decision endpoint, for tests.

Serves, on ``127.0.0.1`` and an ephemeral port:

* ``POST /api/alpha/decisions``: answers every question it is asked, in
  the vendor's documented shape (``noul`` 0.95; ``choice`` the first
  option; ``score`` 1.0), unless a scripted reply is queued;
* ``GET /api/v1/models``: a listing with ``openai/gpt-4o`` only;
* ``GET /api/v1/models/typesafe/jev-1.13/endpoints``: the document the
  real API returned for Jev on 2026-10-06 (trimmed);
* anything else: ``404``.

Every request is recorded on ``server.seen`` as ``(method, path,
headers, body)``.  ``server.queue`` holds scripted ``(status, body,
headers)`` replies for the decision endpoint, consumed in order;
``server.hang`` (an ``Event``) makes the endpoint wait for it, up to
``HANG_SECONDS``, before answering, and ``server.arrived`` is set the
moment a POST is received; together they let a test cancel a request
that is provably in flight.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

JEV = "typesafe/jev-1.13"
HANG_SECONDS = 5.0

JEV_ENDPOINTS_DOC: Dict[str, Any] = {
    "data": {
        "id": JEV,
        "architecture": {
            "modality": "text->decisions",
            "input_modalities": ["text"],
            "output_modalities": ["decisions"],
        },
        "endpoints": [{"context_length": 32000, "supported_parameters": []}],
    },
}

LISTING: Dict[str, Any] = {
    "data": [{
        "id": "openai/gpt-4o",
        "context_length": 128000,
        "architecture": {"input_modalities": ["text", "image"],
                         "output_modalities": ["text"]},
    }],
}


def default_answer(question: Dict[str, Any]) -> Dict[str, Any]:
    """A valid answer to ``question`` in the vendor's shape."""
    qtype = question.get("type")
    if qtype == "noul":
        return {"type": "noul", "noul": 0.95}
    if qtype == "choice":
        options = list(question.get("criteria") or {})
        probs = {o: (1.0 if i == 0 else 0.0) for i, o in enumerate(options)}
        return {"type": "choice", "choice": options[0],
                "probabilities": probs, "confidence": 0.9}
    levels = question.get("criteria") or []
    legend = {str(i): str(level) for i, level in enumerate(levels)}
    probs = {str(i): (1.0 if i == 1 else 0.0) for i in range(len(levels))}
    return {"type": "score", "score": 1.0, "legend": legend,
            "probabilities": probs, "confidence": 0.9}


def default_reply(request: Dict[str, Any]) -> Dict[str, Any]:
    questions = request.get("questions") or {}
    return {
        "model": "jev-1.13.0",
        "answers": {qid: default_answer(q) for qid, q in questions.items()},
        "usage": {"input_tokens": 296, "output_tokens": 20},
    }


class _Handler(BaseHTTPRequestHandler):
    server: "DecisionStandIn"

    def log_message(self, *_args: Any) -> None:   # keep test output quiet
        pass

    def _reply(self, status: int, body: Any,
               headers: Optional[Dict[str, str]] = None) -> None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self.server.seen.append(("GET", self.path, dict(self.headers), None))
        if self.path == "/api/v1/models":
            self._reply(200, LISTING)
        elif self.path == f"/api/v1/models/{JEV}/endpoints":
            self._reply(200, JEV_ENDPOINTS_DOC)
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"null")
        except ValueError:
            body = None
        self.server.seen.append(("POST", self.path, dict(self.headers), body))
        self.server.arrived.set()
        if self.path != "/api/alpha/decisions":
            self._reply(404, {"error": "not found"})
            return
        if self.server.hang is not None:
            self.server.hang.wait(HANG_SECONDS)
        if self.server.queue:
            status, reply, headers = self.server.queue.pop(0)
            self._reply(status, reply, headers)
            return
        self._reply(200, default_reply(body or {}))


class DecisionStandIn(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.seen: List[Tuple[str, str, Dict[str, str], Any]] = []
        self.queue: List[Tuple[int, Any, Optional[Dict[str, str]]]] = []
        self.hang: Optional[threading.Event] = None
        self.arrived = threading.Event()   # set when a POST is received
        self._thread = threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    @property
    def base_url(self) -> str:
        """The chat ``base_url`` a provider is pointed at."""
        return f"http://127.0.0.1:{self.server_address[1]}/api/v1"

    def posts(self) -> List[Tuple[str, Dict[str, str], Any]]:
        return [(p, h, b) for m, p, h, b in self.seen if m == "POST"]

    def __enter__(self) -> "DecisionStandIn":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.hang is not None:
            self.hang.set()
        self.shutdown()
        self.server_close()


def openrouter_against(server: DecisionStandIn, model: str = JEV, **extra: Any):
    """An initialized, connected ``OpenRouterProvider`` aimed at ``server``."""
    from jaato_server.shared.plugins.model_provider.base import ProviderConfig
    from jaato_server.shared.plugins.model_provider.openrouter import (
        OpenRouterProvider,
    )

    provider = OpenRouterProvider()
    provider.initialize(ProviderConfig(
        api_key="sk-or-test",
        extra={"framework_overrides": {"base_url": server.base_url}, **extra},
    ))
    provider.connect(model)
    return provider
