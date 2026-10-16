"""Runs inside Pyodide in a Web Worker: drive two real jaato sessions through
jaato_embedded.InProcessClient, unmodified source.  worker.js has already
unpacked the precompiled bundle (pure wheels into site-packages, jaato under
/src).  Open the page with ``?profile`` to print cProfile output for the
import and for an echo, echo, openai sequence of sessions before the timed
run."""
import os, sys, time, traceback

import js

PROFILE = "profile" in str(js.location.search)
T0 = time.time()


def mark(label):
    print(f"[{label}] +{time.time() - T0:.2f}s")


def report(pr, title, pattern):
    import io, pstats
    out = io.StringIO()
    pstats.Stats(pr, stream=out).sort_stats("cumulative").print_stats(pattern, 20)
    print(f"===== profile: {title} =====\n{out.getvalue()}")


sys.path[:0] = ["/shims", "/src"]
os.chdir("/ws")

import wasm_http  # noqa: E402,F401 - patches httpx on import
import wasm_threads  # noqa: E402 - patches threading on import

if PROFILE:
    import cProfile
    _pr = cProfile.Profile()
    _pr.enable()
from jaato_embedded.client import InProcessClient  # noqa: E402
if PROFILE:
    _pr.disable()
    report(_pr, "import jaato_embedded.client", r".")
mark("import jaato_embedded.client")

BASE = str(js.location.origin)


async def run(provider):
    kw = dict(model="echo-1", provider="echo")
    if provider == "openai":
        kw = dict(model="gpt-mock", provider="openai", plugin_configs={"openai": {
            "api_key": "sk-mock", "base_url": f"{BASE}/v1", "context_length": 32000}})
    async with InProcessClient.session(plugins=[], workspace_path="/ws",
                                       config_root="/ws/.jaato", env_file=None, **kw) as s:
        return await s.ask("hello")


if PROFILE:
    for label in ("echo", "echo", "openai"):
        pr = cProfile.Profile()
        pr.enable()
        await run(label)  # noqa: F704 - top-level await (runPythonAsync)
        pr.disable()
        report(pr, f"{label} session", r"(jaato|openai/__init__|importlib/__init__)")

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
tops = {}
for name in sys.modules:
    tops[name.split(".")[0]] = tops.get(name.split(".")[0], 0) + 1
print("[modules by package]", sorted(tops.items(), key=lambda kv: -kv[1])[:8],
      "total", len(sys.modules))
