"""A local stand-in for OpenRouter's decision endpoint, for tests.

Serves, on ``127.0.0.1`` and an ephemeral port:

* ``POST /api/alpha/decisions`` (OpenRouter) and ``POST /v1/systemone``
  (``laya-serve`` and TypeSafe's own API): answers every question it is asked, in
  the vendor's documented shape (``noul`` 0.95; ``choice`` the first
  option; ``score`` 1.0; ``box`` TypeSafe's documented example), unless a
  scripted reply is queued.  A request carrying ``assets`` or a ``box``
  question is held to TypeSafe's rules first: without
  ``TypeSafe-Danger-Zone-Images: true`` it is a ``400``, and so is an
  asset whose reference appears nowhere else in the body;
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

#: OpenRouter's decision endpoint, and laya-serve's (TypeSafe's path).
DECISION_PATHS = ("/api/alpha/decisions", "/v1/systemone")
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


#: TypeSafe's header for a request with an image or a ``box`` question.
IMAGES_HEADER = "TypeSafe-Danger-Zone-Images"

#: The ``box`` answer in TypeSafe's docs (docs.typesafe.ai/go/box).
BOX_ANSWER: Dict[str, Any] = {
    "type": "box",
    "box": {"top": 0.25, "bottom": 0.82, "left": 0.44, "right": 0.84},
    "confidence": 0.9,
}


def image_rule_problem(headers: Dict[str, str], body: Any) -> Optional[str]:
    """Why TypeSafe would ``400`` this body's images, or ``None``."""
    if not isinstance(body, dict):
        return None
    assets = body.get("assets") or {}
    has_box = any(isinstance(q, dict) and q.get("type") == "box"
                  for q in (body.get("questions") or {}).values())
    if not assets and not has_box:
        return None
    sent = {k.lower(): v for k, v in headers.items()}
    if sent.get(IMAGES_HEADER.lower()) != "true":
        return f"{IMAGES_HEADER}: true is required for images and box"
    return _asset_problem(body, assets)


def _asset_problem(body: Dict[str, Any], assets: Dict[str, Any]) -> Optional[str]:
    """An asset referenced nowhere, or a ``box`` with no image."""
    if not assets:
        return "box needs an image"
    rest = json.dumps({k: v for k, v in body.items() if k != "assets"})
    for reference in assets:
        if reference not in rest:
            return f"asset {reference!r} is not referenced"
    return None


def default_answer(question: Dict[str, Any]) -> Dict[str, Any]:
    """A valid answer to ``question`` in the vendor's shape."""
    qtype = question.get("type")
    if qtype == "box":
        return json.loads(json.dumps(BOX_ANSWER))
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
        if self.path not in DECISION_PATHS:
            self._reply(404, {"error": "not found"})
            return
        problem = image_rule_problem(dict(self.headers), body)
        if problem:
            self._reply(400, {"error": problem})
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
    def host(self) -> str:
        """The server root, as ``plugin_configs.laya.host`` names it."""
        return f"http://127.0.0.1:{self.server_address[1]}"

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


#: Placeholder chat client; truthy so ``complete()`` gets past "not connected".
_NO_CHAT_CLIENT = object()


def openrouter_against(server: DecisionStandIn, model: str = JEV, **extra: Any):
    """An initialized, connected ``OpenRouterProvider`` aimed at ``server``."""
    from jaato_server.shared.plugins.model_provider.base import ProviderConfig
    from jaato_server.shared.plugins.model_provider.openrouter import (
        OpenRouterProvider,
    )

    provider = OpenRouterProvider()
    try:
        import openai  # noqa: F401
    except ImportError:
        # decide() never uses the chat client, but initialize() builds
        # one.  Where the SDK is absent (the contract-guards and
        # reversion-guard jobs) a placeholder stands in, so these tests
        # run there instead of failing on the import whatever the code
        # does, which the reversion meta-guard would count as detection.
        provider._create_client = lambda: _NO_CHAT_CLIENT
    provider.initialize(ProviderConfig(
        api_key="sk-or-test",
        extra={"framework_overrides": {"base_url": server.base_url}, **extra},
    ))
    provider.connect(model)
    return provider


def typesafe_against(server: DecisionStandIn, model: str = "jev-latest",
                     **extra: Any):
    """An initialized, connected ``TypeSafeProvider`` aimed at ``server``.

    The key is ``ts-test`` unless ``extra`` names one (or ``api_key=None``
    to leave the knob unset and fall back to the environment).
    """
    from jaato_server.shared.plugins.model_provider.base import ProviderConfig
    from jaato_server.shared.plugins.model_provider.typesafe import TypeSafeProvider

    extra.setdefault("api_key", "ts-test")
    if extra["api_key"] is None:
        del extra["api_key"]
    provider = TypeSafeProvider()
    provider.initialize(ProviderConfig(extra={"base_url": server.host, **extra}))
    provider.connect(model)
    return provider


def laya_against(server: DecisionStandIn, model: str = "convaiinnovations/laya",
                 **extra: Any):
    """An initialized, connected ``LayaProvider`` aimed at ``server``."""
    from jaato_server.shared.plugins.model_provider.base import ProviderConfig
    from jaato_server.shared.plugins.model_provider.laya import LayaProvider

    provider = LayaProvider()
    provider.initialize(ProviderConfig(extra={"host": server.host, **extra}))
    provider.connect(model)
    return provider


# ------------------------------------------------------------------ images
#
# Byte fixtures for image tests, built with the standard library so no
# test needs Pillow.

def png_bytes(width: int = 300, height: int = 300, *, animated: bool = False) -> bytes:
    """A real still (or, with ``animated``, an APNG-marked) greyscale PNG."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    rows = b"".join(b"\x00" + b"\x80" * width for _ in range(height))
    out = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    if animated:
        out += chunk(b"acTL", struct.pack(">II", 1, 0))
    return out + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def jpeg_bytes(width: int = 400, height: int = 300) -> bytes:
    """A baseline JPEG header (SOI, APP0, SOF0, EOI): enough for the size
    check, which reads headers only, and not a decodable picture."""
    import struct

    app0 = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = struct.pack(">BHHB", 8, height, width, 1) + b"\x01\x11\x00"
    return (b"\xff\xd8"
            + b"\xff\xe0" + struct.pack(">H", len(app0) + 2) + app0
            + b"\xff\xc0" + struct.pack(">H", len(sof0) + 2) + sof0
            + b"\xff\xd9")
