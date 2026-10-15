# jaato in a browser Web Worker (Pyodide spike)

Runs two real jaato sessions (`echo`, and the `openai` provider against a
mock at the page's own origin) from the unmodified source tree, inside
Pyodide in a Web Worker. Findings: [docs/design/wasm-embedding.md](../../docs/design/wasm-embedding.md).

```bash
cd examples/wasm_spike
python3 fetch_deps.py            # Pyodide, its packages, precompiled bundle.zip -> build/
python3 server.py 8766 &         # static files + mock OpenAI on one origin
node drive.mjs http://127.0.0.1:8766          # headless Chromium, prints the worker log
```

Or open `http://127.0.0.1:8766/web/index.html` in Chromium. `drive.mjs`
needs Playwright; set `PLAYWRIGHT_MODULE` and `CHROMIUM_PATH` if it is not
importable as `playwright`.

`fetch_deps.py` needs a CPython 3.13 (`python3.13`, or `PY313=...`): it
compiles every module the page imports to an unchecked-hash pyc, which is
what keeps Pyodide from compiling sources at import time. `SOURCELESS=1`
ships the pycs in place of the sources (16 MB instead of 22 MB). Open the
page with `?profile` (`QS=?profile node drive.mjs ...`) for cProfile output
of the import and of three sessions.

| File | Role |
|---|---|
| `fetch_deps.py` | downloads everything into `build/` (git-ignored) and builds `bundle.zip`: jaato and the pure wheels, plus a pyc overlay for Pyodide's packages |
| `server.py` | serves `web/` and `build/`, answers `/v1/*` like OpenAI |
| `web/worker.js` | boots Pyodide, loads packages, unpacks the bundle and shims; logs phase times |
| `web/probe.py` | runs the sessions, reports times and imported modules |
| `shims/` | the two runtime shims the findings doc explains |
