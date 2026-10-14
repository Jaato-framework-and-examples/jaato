# jaato in a Web Page (WASM): Spike Findings

**Status: spike, nothing in the framework changed.** The harness is
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

| Session | Result | Time (stalls stubbed) | Time (stalls kept) |
|---|---|---|---|
| `echo` provider | reply `'hello'` | 6.8 s | 17.1 s |
| `openai` provider, mock at the page's origin | reply `'Hello from the mock.'`, a streamed `POST /v1/chat/completions` reached the server | 3.8 s | 13.9 s |

Fixed costs before the first session: Pyodide boot ~2.5 s, Pyodide packages
~1.7 s, `micropip` install of the pure wheels plus unpacking the source zip
~8 s. Page open to the second reply: ~23 s stubbed, ~43 s kept.

## What it took

Every dependency the embedded path imports exists in Pyodide 0.28 or as a
pure wheel. `psutil`, `watchdog`, `websockets` and `pexpect` are not needed
on this path; `mcp` was left out on purpose and nothing broke. Four shims,
all in `examples/wasm_spike/shims/`:

| Shim | Why | Where in jaato |
|---|---|---|
| `readline` stub | imported unconditionally, and Pyodide has no terminal | `plugins/permission/channels.py:11` |
| `googleapis-common-protos<1.71` | newer protos are generated for a protobuf runtime newer than Pyodide's 6.31; `google-api-core` is imported by `token_accounting` for error classification | `shared/token_accounting.py` |
| `ThreadPoolExecutor` runs inline; `Thread.start` is recorded and skipped | Pyodide cannot start threads. `expose_all` initializes plugins on a pool; MCP and LSP each start a loop thread | `plugins/registry.py` (`expose_all`), `mcp/plugin.py:2423`, `lsp/plugin.py:3160` |
| httpx response chunks converted to `bytes` | Pyodide's httpx XHR transport yields `memoryview`, which the openai SDK's SSE decoder cannot split | outside jaato |

`asyncio.to_thread` needed no shim: Pyodide's event loop runs executor work
inline, so the turn blocks the worker until it ends. That is acceptable in a
Web Worker and impossible on the page's main thread.

## Two stalls, and where the time goes

Profiled under Pyodide in Node, almost all of `connect()` is spent in two
places, neither of which runs a session:

1. **`registry._install_advice`** (~80% of `connect()`): the directory scan
   finds `interactive_shell` missing `pexpect` and, to word one "install
   this extra" hint, builds an index over every installed distribution's
   metadata (`scaffold/dependencies._metadata_index`). It is a diagnostic
   that dominates startup.
2. **`MCPToolPlugin._ensure_thread`**: waits up to 10 s for a loop thread,
   once per session, even with no `.mcp.json`. The second session's 10 s gap
   between the two columns above is this wait.

With both stubbed, what remains is ~2 s of imports and the registry
initializing every plugin.

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
- **Download size.** Pyodide core ~12 MB, Pyodide packages ~7.5 MB, pure
  wheels ~3 MB, jaato source 5.7 MB, before HTTP compression. Trimming the
  source to the import trace and dropping unused providers would cut the
  last two.
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

1. Make the two stalls cheap natively too: compute the install hint lazily
   (or cache the metadata index), and start the MCP thread only when a
   server is configured.
2. A `wasm` mode for `jaato_embedded`: a registry with only the profile's
   plugins, plugin initialization without a thread pool, no MCP/LSP threads,
   and a lazy `readline` import.
3. A bundle builder: record the import trace for a profile (as
   `template_preload.py` already does for the pool) and pack exactly those
   modules.
4. A fetch-based provider transport with real streaming (JSPI), then a
   `chrome_ai` variant that calls `LanguageModel` in the page.
