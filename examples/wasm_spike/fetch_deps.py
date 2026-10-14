"""Fetch everything the spike serves: Pyodide, its packages, pure wheels, and
a zip of the jaato source.  Run from anywhere; writes into this directory's
``build/`` (git-ignored).  Needs node/npm, curl and pip on PATH."""
import json, os, subprocess, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BUILD = os.path.join(HERE, "build")
PYODIDE = "0.28.3"
# Pyodide-built packages (and their deps, resolved from the lock file).
PYODIDE_PKGS = ["micropip", "pydantic", "pyyaml", "requests", "httpx", "jiter",
                "jsonschema", "cryptography", "rich", "pygments", "protobuf",
                "typing-extensions", "anyio", "certifi", "wcwidth", "sniffio",
                "distro", "ssl", "openai", "packaging"]
# Pure wheels Pyodide does not ship.  googleapis-common-protos is pinned below
# 1.71: newer ones are generated for a protobuf runtime newer than Pyodide's.
PURE_WHEELS = ["anthropic", "python-dotenv", "docstring-parser", "google-api-core",
               "googleapis-common-protos<1.71", "proto-plus", "google-auth"]


def run(*cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def main():
    os.makedirs(BUILD, exist_ok=True)
    if not os.path.isdir(os.path.join(BUILD, "node_modules", "pyodide")):
        run("npm", "init", "-y", cwd=BUILD, stdout=subprocess.DEVNULL)
        run("npm", "install", f"pyodide@{PYODIDE}", cwd=BUILD)
    pdir = os.path.join(BUILD, "node_modules", "pyodide")
    lock = json.load(open(os.path.join(pdir, "pyodide-lock.json")))["packages"]
    seen = set()

    def walk(name):
        name = name.lower().replace("_", "-")
        if name in seen or name not in lock:
            return
        seen.add(name)
        for dep in lock[name]["depends"]:
            walk(dep)

    for name in PYODIDE_PKGS:
        walk(name)
    for name in sorted(seen):
        fname = lock[name]["file_name"]
        dest = os.path.join(pdir, fname)
        if not os.path.exists(dest):
            run("curl", "-sSfL", "-o", dest,
                f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE}/full/{fname}")
    wheels = os.path.join(BUILD, "wheels")
    os.makedirs(wheels, exist_ok=True)
    run(sys.executable, "-m", "pip", "download", "-q", "--no-deps", "-d", wheels, *PURE_WHEELS)
    json.dump(["/wheels/" + w for w in sorted(os.listdir(wheels))],
              open(os.path.join(BUILD, "wheels.json"), "w"))
    # The jaato source, unmodified, minus tests.
    with zipfile.ZipFile(os.path.join(BUILD, "jaato-src.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for root, pkg in (("jaato-sdk", "jaato_sdk"), ("jaato-server", "jaato_server"),
                          ("jaato-server", "jaato_embedded")):
            for d, ds, fs in os.walk(os.path.join(REPO, root, pkg)):
                ds[:] = [x for x in ds if x not in ("tests", "__pycache__")]
                for f in fs:
                    if not f.endswith(".pyc"):
                        fp = os.path.join(d, f)
                        z.write(fp, os.path.relpath(fp, os.path.join(REPO, root)))
    with zipfile.ZipFile(os.path.join(BUILD, "shims.zip"), "w") as z:
        for f in os.listdir(os.path.join(HERE, "shims")):
            z.write(os.path.join(HERE, "shims", f), f)
    print("build ready:", BUILD)


if __name__ == "__main__":
    main()
