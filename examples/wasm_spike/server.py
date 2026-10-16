"""One origin for the spike: static files and a mock OpenAI chat endpoint.

Same origin avoids CORS.  ``/v1/*`` answers like OpenAI (streamed SSE when the
request asks for it); everything else is a file under ``web/`` or ``build/``.
Usage: python server.py [PORT]  (default 8766)."""
import json, http.server, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = {".js": "text/javascript", ".wasm": "application/wasm", ".html": "text/html",
         ".json": "application/json", ".mjs": "text/javascript"}


def resolve(path):
    path = path.split("?")[0].lstrip("/")
    if path.startswith("web/"):
        return os.path.join(HERE, path)
    if path.startswith("node_modules/"):
        return os.path.join(HERE, "build", path)
    if path.startswith("wheels/"):
        return os.path.join(HERE, "build", path)
    if path.startswith("build/"):
        return os.path.join(HERE, path)
    return None


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._send(200, json.dumps({"object": "list", "data": [
                {"id": "gpt-mock", "object": "model"}]}).encode(), "application/json")
        fp = resolve(self.path)
        if not fp or not os.path.isfile(fp):
            return self._send(404, b"", "text/plain")
        self._send(200, open(fp, "rb").read(),
                   TYPES.get(os.path.splitext(fp)[1], "application/octet-stream"))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        print("POST", self.path, "stream=", body.get("stream"), flush=True)
        usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
        if not body.get("stream"):
            return self._send(200, json.dumps({
                "id": "c1", "object": "chat.completion", "model": "gpt-mock",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": "Hello from the mock."}}],
                "usage": usage}).encode(), "application/json")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for word in ["Hello ", "from ", "the ", "mock."]:
            chunk = {"id": "c1", "object": "chat.completion.chunk", "model": "gpt-mock",
                     "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        chunk = {"id": "c1", "object": "chat.completion.chunk", "model": "gpt-mock",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage}
        self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
