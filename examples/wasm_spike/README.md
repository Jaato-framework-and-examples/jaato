# jaato in a browser Web Worker (Pyodide spike)

Runs two real jaato sessions (`echo`, and the `openai` provider against a
mock at the page's own origin) from the unmodified source tree, inside
Pyodide in a Web Worker. Findings: [docs/design/wasm-embedding.md](../../docs/design/wasm-embedding.md).

```bash
cd examples/wasm_spike
python3 fetch_deps.py            # Pyodide, its packages, pure wheels, source zip -> build/
python3 server.py 8766 &         # static files + mock OpenAI on one origin
node drive.mjs http://127.0.0.1:8766          # headless Chromium, prints the worker log
```

Or open `http://127.0.0.1:8766/web/index.html` in Chromium. `drive.mjs`
needs Playwright; set `PLAYWRIGHT_MODULE` and `CHROMIUM_PATH` if it is not
importable as `playwright`.

| File | Role |
|---|---|
| `fetch_deps.py` | downloads everything into `build/` (git-ignored) |
| `server.py` | serves `web/` and `build/`, answers `/v1/*` like OpenAI |
| `web/worker.js` | boots Pyodide, loads packages, unpacks source and shims |
| `web/probe.py` | installs wheels, runs the sessions, reports |
| `shims/` | the two runtime shims the findings doc explains |
