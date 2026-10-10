"""A real streamable-HTTP MCP server for the #1580 guard.

Run as ``python mcp_http_fixture_server.py PORT HEADER_LOG [sse]``.  It
serves one tool, ``echo``, at ``http://127.0.0.1:PORT/mcp`` (streamable HTTP)
or, with ``sse``, at ``http://127.0.0.1:PORT/sse``, and appends the
``Authorization`` header of every request it receives to ``HEADER_LOG`` (one
line per request, ``-`` when absent), so a test can see what reached it.

An optional fourth argument names a TOKEN FILE: when given, a request
whose ``Authorization`` header differs from the file's (stripped) content
is answered **401** before it reaches the MCP app, so a test can rotate
the credential the server accepts by rewriting the file (#1609).

Built on whichever mcp SDK generation is installed: 2.x ``MCPServer``,
else 1.x ``FastMCP``.
"""

import sys

import uvicorn


def _build_app(transport):
    try:
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server
    server = Server("fixture")

    @server.tool()
    def echo(text: str) -> str:
        """Return the text it was given."""
        return f"echo: {text}"

    if transport == "sse":
        return server.sse_app()
    return server.streamable_http_app()


def _recording(app, log_path, token_path=None):
    async def wrapped(scope, receive, send):
        if scope.get("type") == "http":
            value = "-"
            for key, raw in scope.get("headers", []):
                if key.lower() == b"authorization":
                    value = raw.decode("latin-1")
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(value + "\n")
            if token_path is not None:
                with open(token_path, encoding="utf-8") as fh:
                    accepted = fh.read().strip()
                if value != accepted:
                    await send({"type": "http.response.start", "status": 401,
                                "headers": [(b"content-type", b"text/plain")]})
                    await send({"type": "http.response.body",
                                "body": b"unauthorized"})
                    return
        await app(scope, receive, send)
    return wrapped


def main() -> None:
    port, log_path = int(sys.argv[1]), sys.argv[2]
    app = _build_app(sys.argv[3] if len(sys.argv) > 3 else "http")
    token_path = sys.argv[4] if len(sys.argv) > 4 else None
    uvicorn.run(_recording(app, log_path, token_path), host="127.0.0.1", port=port,
                log_level="warning")


if __name__ == "__main__":
    main()
