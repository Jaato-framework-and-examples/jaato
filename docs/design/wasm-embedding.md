# jaato in a Web Page (WASM): Spike Findings

**Status: spike.** The three startup costs it found were fixed in the
framework by #1693 (#1689, #1690, #1691); the harness now runs on `main`
with no stubs. The harness is
[`examples/wasm_spike/`](../../examples/wasm_spike/README.md); every number
below comes from running it in headless Chromium (Playwright's build 1194).

## The question

Can the minimum of jaato needed for one profiled session (or one cascade) run
embedded in a web page, compiled to WASM?

Python is not compiled to WASM. What runs is CPython compiled to WASM
(Pyodide 0.28.3, Python 3.13) interpreting the jaato source unchanged. So the
real question is whether jaato's in-process path (`jaato_embedded`) runs
under Pyodide, in a Web Worker, and what it costs.

## Result

It runs. Two real sessions through `InProcessClient.session(...)` with
`plugins=[]`, from the unmodified source tree:

| Session | Result | Before #1693, stalls stubbed | Before #1693, unstubbed | `main`, source bundle | `main`, precompiled bundle |
|---|---|---|---|---|---|
| `echo` provider | reply `'hello'` | 6.8 s | 17.1 s | 3.9-4.2 s | 1.3-1.5 s |
| `openai` provider, mock at the page's origin | reply `'Hello from the mock.'`, a streamed `POST /v1/chat/completions` reached the server | 3.8 s | 13.9 s | 2.1-2.2 s | 1.1-1.4 s |

Page open to the second reply: ~43 s unstubbed before #1693, ~15 s on
`main` with the source bundle, **~7.7-8.2 s** with the precompiled bundle
(served locally, so download time is ~0; see below).

## Where the startup time goes

Measured per phase in the worker (`[boot]`, `[unpack]`, `[packages]`) and
the probe (`[import ...]`, `[session ...]`), with cProfile inside Pyodide
for the import and the sessions (`?profile`). Medians of three runs:

| Phase | Source bundle + micropip | Precompiled bundle | What changed |
|---|---|---|---|
| Pyodide boot | 1.8 s | 1.8 s | nothing (Pyodide's own) |
| `loadPackage` | 1.0 s | ~1.9 s for both, overlapped | the bundle unpacks while `loadPackage` downloads |
| unpack source zip | 0.3 s | (above) | |
| `micropip` install of pure wheels | 1.1 s | 0 | wheels unpacked into site-packages by the bundle; micropip not loaded |
| `import jaato_embedded.client` | 2.1 s | 1.4 s | pycs |
| first session (`echo`) | 3.9 s | 1.4 s | pycs |
| second session (`openai`) | 2.1 s | 1.2 s | pycs for Pyodide's packages too |

The profile pointed at one cause. Pyodide compiles every `.py` it imports
to bytecode on the spot, and nothing shipped any: not jaato, not the pure
wheels, not Pyodide's own wheels (`pydantic`, `requests`, `openai` ship
sources only). `builtins.compile` was 2.7 s of a 5.0 s first session.
`fetch_deps.py` now compiles everything with a host CPython 3.13 (same
bytecode magic number as Pyodide 0.28) as **unchecked-hash** pycs, so the
loader neither compiles nor stats the source for an mtime check. jaato and
the pure wheels go into one `bundle.zip` unpacked at `/`, and Pyodide's
packages get a pyc-only overlay for the files `loadPackage` installs. The
`anthropic` wheel was left out: neither session imports it
(`retry_utils` guards its import), and it was 55% of the bundle's files.

What is left, and what would remove it:

| Cost | Size | Lever |
|---|---|---|
| Pyodide boot | 1.8 s | none in 0.28. Its memory snapshot (`_makeSnapshot`) copies the main heap only; side modules (`pydantic_core`, `jiter`, `cryptography`, `_yaml`) are separate WASM instances and do not restore |
| package load + unpack | ~1.9 s | fewer files: a bundle trimmed to the import trace (next step 3). Storing the zip uncompressed saved ~0.5 s of unpack but tripled its size |
| `import jaato_embedded.client` | 1.4 s | about 1.2 s of it is pydantic building the 181 models in `jaato_sdk/events.py` at import. `defer_build=True` on them, tried as a local patch, brought the import to ~1.0 s and the page to ~7.2 s (natively, 0.33 s to ~0.2 s per client process). A framework change, not made here |
| first session, beyond the steady cost | ~1.1 s | the registry's directory scan imports every plugin module although the profile names none (next step 2) |
| steady cost per session | ~0.3 s | most of it is `_has_deferred_to_discover` calling `get_plugin_for_tool` 83 times, each reaching `prompt_library.get_executors`, whose mtime-cached discovery still stats every source. Negligible natively, visible here; also gone with step 2 |
| `import openai` | ~1 s | the SDK imports 532 modules, mostly typed models. A provider transport that does not use the SDK (step 4) |

Two more things decide what a real visitor sees. **Download:** Pyodide core
~12 MB, its packages ~7.5 MB, the bundle 22 MB (16 MB with `SOURCELESS=1`,
which ships `x.pyc` in place of `x.py`; tracebacks then name files and
lines without source text). The pycs more than doubled the bundle (it was
~9 MB of sources and wheels). On a first visit over a real network this
download dominates. **Repeat visits:** the HTTP cache (or a service worker)
removes the download; the boot, unpack and imports above are paid on every
page load.

## What it took

Every dependency the embedded path imports exists in Pyodide 0.28 or as a
pure wheel. `psutil`, `watchdog`, `websockets` and `pexpect` are not needed
on this path; `mcp` was left out on purpose and nothing broke. Four
workarounds were needed; one is gone on `main`, three remain (in
`examples/wasm_spike/shims/` and `fetch_deps.py`):

| Shim | Why | Where in jaato |
|---|---|---|
| ~~`readline` stub~~ | imported unconditionally, and Pyodide has no terminal. **Gone:** the import is guarded since #1693 (#1689) | `plugins/permission/channels.py` |
| `googleapis-common-protos<1.71` | newer protos are generated for a protobuf runtime newer than Pyodide's 6.31; `google-api-core` is imported by `token_accounting` for error classification | `shared/token_accounting.py` |
| `ThreadPoolExecutor` runs inline; `Thread.start` is recorded and skipped | Pyodide cannot start threads. `asyncio.to_thread` (`InProcessClient` builds the registry with it) and `expose_all` both submit to a pool. MCP and LSP no longer start loop threads with no server configured (#1691), so on `main` the `Thread.start` half records nothing | `jaato_embedded/client.py` (`_build_registry`), `plugins/registry.py` (`expose_all`) |
| httpx response chunks converted to `bytes` | Pyodide's httpx XHR transport yields `memoryview`, which the openai SDK's SSE decoder cannot split | outside jaato |

With the pool shimmed, executor work runs inline, so the turn blocks the
worker until it ends. That is acceptable in a Web Worker and impossible on
the page's main thread. (An earlier version of this note said
`asyncio.to_thread` needed no shim; removing the shim on `main` showed it
fails with `can't start new thread` in `_build_registry`.)

## Two stalls, and where the time went (fixed by #1693)

Profiled under Pyodide in Node, almost all of `connect()` was spent in two
places, neither of which runs a session. Both are fixed natively too:

1. **`registry._install_advice`** (~80% of `connect()`): the directory scan
   finds `interactive_shell` missing `pexpect` and, to word one "install
   this extra" hint, builds an index over every installed distribution's
   metadata (`scaffold/dependencies._metadata_index`). It is a diagnostic
   that dominated startup. Fixed in #1690: the metadata walk runs once per
   process.
2. **`MCPToolPlugin._ensure_thread`**: waits up to 10 s for a loop thread,
   once per session, even with no `.mcp.json`. The second session's 10 s gap
   between the first two columns above is this wait. Fixed in #1691: no
   loop thread without a configured server.

What remained after them was compiling sources on import, measured and
largely removed in [Where the startup time goes](#where-the-startup-time-goes).

## What the spike also showed

- **The minimum is not minimal yet.** `plugins=[]` still loads ~520 jaato
  modules (~920 total), because `InProcessClient` builds the full runner-tier
  registry. The source bundle is 5.7 MB / 912 files. `registry_factory`
  already exists as a seam for a registry holding only the profile's plugins.
- **Streaming is not live.** The request goes out as sync XHR, which hands
  back the whole response at once, so tokens arrive together at the end and a
  turn cannot be cancelled mid-flight. Live streaming needs either the
  `fetch` streaming API reached through JSPI (Chromium only today) or an
  async provider path.
- **No network without a browser.** In plain Node, Pyodide's httpx needs
  `XMLHttpRequest`, and a polyfill was not faithful enough. The browser
  worker is the real target and works.

## What does not come along

The daemon, runner, pool, AppArmor/SELinux/seccomp, IPC, `cli`,
`interactive_shell`, `notebook`, `lsp`, MCP over stdio, and concurrent
subagents or cascade fan-out (no threads). A sequential cascade is
expressible as a JS driver creating one in-process session per stage; that
was not tested.

## Not verified

- A real provider over the internet (CORS, key handling). OpenAI, Anthropic
  (with `anthropic-dangerous-direct-browser-access`) and OpenRouter allow
  browser calls; a key in a page is the deployer's problem, best solved by
  a thin proxy or a user-supplied key.
- `chrome_ai` driven directly through `LanguageModel` (no CDP), the
  configuration where in-page jaato needs no key and no network. Needs a
  provider variant that calls the page's JS rather than CDP.
- Tool plugins in the page (todo, template, memory) and `client_tools`
  handlers written in JS.
- Firefox and Safari.

## If this goes further

In order, each a separate change:

1. ~~Make the two stalls cheap natively too~~ and guard the `readline`
   import: done in #1693.
2. A `wasm` mode for `jaato_embedded`: a registry with only the profile's
   plugins, and registry build and plugin initialization without a thread
   pool (so the remaining threads shim goes away).
3. A bundle builder: record the import trace for a profile (as
   `template_preload.py` already does for the pool) and pack exactly those
   modules, precompiled.
4. `defer_build=True` on the `jaato_sdk.events` models (~0.5 s here,
   ~0.1 s per native client process).
5. A fetch-based provider transport with real streaming (JSPI), then a
   `chrome_ai` variant that calls `LanguageModel` in the page.
