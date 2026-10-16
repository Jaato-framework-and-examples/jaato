"""WASM spike shim: Pyodide's httpx XHR transport yields memoryview chunks;
the openai SDK's SSE decoder calls bytes methods on them."""
import httpx
_orig = httpx.Response.iter_bytes
def iter_bytes(self, chunk_size=None):
    for c in _orig(self, chunk_size):
        yield bytes(c)
httpx.Response.iter_bytes = iter_bytes
