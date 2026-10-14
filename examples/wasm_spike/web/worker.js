importScripts("/node_modules/pyodide/pyodide.js");
const log = (m) => postMessage({log: m});
(async () => {
  const t0 = performance.now();
  const py = await loadPyodide({ indexURL: "/node_modules/pyodide/", stdout: log, stderr: log });
  log(`[boot] ${Math.round(performance.now()-t0)} ms`);
  await py.loadPackage(["micropip","pydantic","pyyaml","requests","httpx","jiter","jsonschema","cryptography","rich","pygments","protobuf","typing-extensions","anyio","certifi","wcwidth","sniffio","distro","ssl","openai"], {messageCallback:()=>{}});
  log(`[packages] ${Math.round(performance.now()-t0)} ms`);
  for (const [url, dir] of [["/build/jaato-src.zip","/src"],["/build/shims.zip","/shims"]]) {
    const buf = await (await fetch(url)).arrayBuffer();
    py.unpackArchive(buf, "zip", {extractDir: dir});
  }
  const wheels = await (await fetch("/build/wheels.json")).json();
  py.globals.set("WHEELS", py.toPy(wheels));
  py.FS.mkdirTree("/ws");
  const code = await (await fetch("/web/probe.py")).text();
  try { await py.runPythonAsync(code); } catch (e) { log("[fatal] " + e.message); }
  log(`[done] ${Math.round(performance.now()-t0)} ms`);
  postMessage({done: true});
})();
