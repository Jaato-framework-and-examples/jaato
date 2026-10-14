"""Runs inside Pyodide in a Web Worker: install wheels, then drive two real
jaato sessions through jaato_embedded.InProcessClient, unmodified source.

STUB_STALLS (from ?stalls=keep in the page URL) controls the two measured
startup stalls; see docs/design/wasm-embedding.md for what each is."""
import os, sys, time, traceback

import js
import micropip

for wheel in WHEELS:  # noqa: F821 - set by worker.js
    await micropip.install(wheel, deps=False)  # noqa: F704 - top-level await (runPythonAsync)
sys.path[:0] = ["/shims", "/src"]
os.chdir("/ws")

import wasm_http  # noqa: E402,F401 - patches httpx on import
import wasm_threads  # noqa: E402 - patches threading on import

if "stalls=keep" not in str(js.location.search):
    # Stall 1: the registry builds a metadata-wide extras index to word one
    # "missing dependency" hint (pexpect); in Pyodide that walk is ~80% of
    # connect().  Stall 2: MCP waits 10 s for a loop thread that cannot start.
    import jaato_server.shared.plugins.registry as _registry
    from jaato_server.shared.plugins.mcp.plugin import MCPToolPlugin
    _registry._install_advice = lambda *a, **k: ""
    MCPToolPlugin._ensure_thread = lambda self: None

from jaato_embedded.client import InProcessClient  # noqa: E402

BASE = str(js.location.origin)


async def run(provider):
    kw = dict(model="echo-1", provider="echo")
    if provider == "openai":
        kw = dict(model="gpt-mock", provider="openai", plugin_configs={"openai": {
            "api_key": "sk-mock", "base_url": f"{BASE}/v1", "context_length": 32000}})
    async with InProcessClient.session(plugins=[], workspace_path="/ws",
                                       config_root="/ws/.jaato", env_file=None, **kw) as s:
        return await s.ask("hello")


for provider in ("echo", "openai"):
    t = time.time()
    try:
        reply = await run(provider)  # noqa: F704
        print(f"[session {provider}] OK {time.time() - t:.1f}s reply={reply!r}")
    except BaseException as e:  # report, keep going
        print(f"[session {provider}] FAIL {type(e).__name__}: {e}")
        print("".join(traceback.format_exception(e)[-12:]))
print("[threads not started]", wasm_threads.SKIPPED)
print("[jaato modules loaded]", len([m for m in sys.modules if m.startswith("jaato")]))
