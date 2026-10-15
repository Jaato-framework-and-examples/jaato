importScripts("/node_modules/pyodide/pyodide.js");
const log = (m) => postMessage({log: m});
(async () => {
  const t0 = performance.now();
  // Start the downloads now; they overlap Pyodide's boot.
  const get = (url) => fetch(url).then((r) => r.arrayBuffer());
  const archives = [["/build/bundle.zip", "/"], ["/build/shims.zip", "/shims"]]
    .map(([url, dir]) => get(url).then((buf) => [buf, dir]));
  const py = await loadPyodide({ indexURL: "/node_modules/pyodide/", stdout: log, stderr: log });
  log(`[boot] ${Math.round(performance.now()-t0)} ms`);
  // Unpack while loadPackage downloads.  The bundle's pyc overlay lands in
  // site-packages directories loadPackage then fills; it leaves them alone.
  const packages = py.loadPackage(["pydantic","pyyaml","requests","httpx","jiter","jsonschema","cryptography","rich","pygments","protobuf","typing-extensions","anyio","certifi","wcwidth","sniffio","distro","ssl","openai"], {messageCallback:()=>{}});
  for (const [buf, dir] of await Promise.all(archives)) {
    py.unpackArchive(buf, "zip", {extractDir: dir});
  }
  log(`[unpack] ${Math.round(performance.now()-t0)} ms`);
  await packages;
  log(`[packages] ${Math.round(performance.now()-t0)} ms`);
  py.FS.mkdirTree("/ws");
  const code = await (await fetch("/web/probe.py")).text();
  try { await py.runPythonAsync(code); } catch (e) { log("[fatal] " + e.message); }
  log(`[done] ${Math.round(performance.now()-t0)} ms`);
  postMessage({done: true});
})();
