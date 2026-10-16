"""Fetch everything the spike serves: Pyodide, its packages, pure wheels, and
a zip of the jaato source.  Run from anywhere; writes into this directory's
``build/`` (git-ignored).  Needs node/npm, curl, pip and a CPython 3.13 on
PATH (``python3.13``, or ``PY313=/path/to/python3.13``)."""
import json, os, shutil, subprocess, sys, tempfile, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BUILD = os.path.join(HERE, "build")
PYODIDE = "0.28.3"
# Pyodide-built packages (and their deps, resolved from the lock file).
PYODIDE_PKGS = ["pydantic", "pyyaml", "requests", "httpx", "jiter",
                "jsonschema", "cryptography", "rich", "pygments", "protobuf",
                "typing-extensions", "anyio", "certifi", "wcwidth", "sniffio",
                "distro", "ssl", "openai"]
# Pure wheels Pyodide does not ship.  googleapis-common-protos is pinned below
# 1.71: newer ones are generated for a protobuf runtime newer than Pyodide's.
# anthropic is left out: these sessions never import it (retry_utils guards
# its import) and it was 55% of the bundle's files.  Add it back for the
# anthropic provider.
PURE_WHEELS = ["python-dotenv", "docstring-parser", "google-api-core",
               "googleapis-common-protos<1.71", "proto-plus", "google-auth"]


SITE = "lib/python3.13/site-packages"  # Pyodide's site-packages, relative to /

# Compiled by a host CPython 3.13 (same bytecode magic as Pyodide 0.28) with
# unchecked-hash pycs: the loader then neither compiles nor stats the source.
COMPILE = r"""
import py_compile, sys
for line in sys.stdin:
    src, dfile = line.rstrip("\n").split("\t")
    py_compile.compile(src, dfile=dfile, doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
"""


def run(*cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


# SOURCELESS=1 ships ``x.pyc`` in place of ``x.py`` (half the files and
# bytes; tracebacks then name files and lines but show no source text).
SOURCELESS = os.environ.get("SOURCELESS") == "1"


def add_tree(z, top, arc_root, skip=("tests", "__pycache__"), pyc_only=False):
    """Add every file under ``top`` at ``arc_root/<rel>``, with a pyc beside
    each .py (``__pycache__/<name>.cpython-313.pyc``).  ``pyc_only`` adds the
    pycs alone: an overlay for files another step installs."""
    py313 = os.environ.get("PY313", "python3.13")
    files = []
    for d, ds, fs in os.walk(top):
        ds[:] = [x for x in ds if x not in skip]
        files += [os.path.join(d, f) for f in fs if not f.endswith(".pyc")]
    pys = [f for f in files if f.endswith(".py")]
    feed = "".join(f"{f}\t/{arc_root}/{os.path.relpath(f, top)}\n" for f in pys)
    run(py313, "-c", COMPILE, input=feed.encode())
    for f in files:
        rel = os.path.relpath(f, top)
        sourceless = SOURCELESS and not pyc_only and f.endswith(".py")
        if not pyc_only and not sourceless:
            z.write(f, f"{arc_root}/{rel}")
        if sourceless:
            d, name = os.path.split(f)
            z.write(os.path.join(d, "__pycache__", name[:-3] + ".cpython-313.pyc"),
                    f"{arc_root}/{rel[:-3]}.pyc")
        elif f.endswith(".py"):
            d, name = os.path.split(f)
            pyc = os.path.join(d, "__pycache__", name[:-3] + ".cpython-313.pyc")
            rd, _ = os.path.split(rel)
            z.write(pyc, f"{arc_root}/{os.path.join(rd, '__pycache__', name[:-3] + '.cpython-313.pyc')}")


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
    # One bundle, unpacked at / : the pure wheels into site-packages and the
    # jaato source (unmodified, minus tests) under /src, all precompiled.
    with tempfile.TemporaryDirectory() as tmp, \
            zipfile.ZipFile(os.path.join(BUILD, "bundle.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        site = os.path.join(tmp, "site")
        for w in sorted(os.listdir(wheels)):
            zipfile.ZipFile(os.path.join(wheels, w)).extractall(site)
        add_tree(z, site, SITE)
        # Pyodide's own wheels ship sources only, so each import compiles
        # them; overlay pycs into what loadPackage installs.
        pyo = os.path.join(tmp, "pyodide-site")
        for name in sorted(seen):
            zipfile.ZipFile(os.path.join(pdir, lock[name]["file_name"])).extractall(pyo)
        add_tree(z, pyo, SITE, pyc_only=True)
        for root, pkg in (("jaato-sdk", "jaato_sdk"), ("jaato-server", "jaato_server"),
                          ("jaato-server", "jaato_embedded")):
            # Compile a copy: unchecked-hash pycs left in the checkout would
            # be trusted by a local 3.13 after the source changed.
            copy = os.path.join(tmp, "src", pkg)
            shutil.copytree(os.path.join(REPO, root, pkg), copy,
                            ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"))
            add_tree(z, copy, f"src/{pkg}")
    with zipfile.ZipFile(os.path.join(BUILD, "shims.zip"), "w") as z:
        for f in os.listdir(os.path.join(HERE, "shims")):
            z.write(os.path.join(HERE, "shims", f), f)
    print("build ready:", BUILD)


if __name__ == "__main__":
    main()
