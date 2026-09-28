"""Installing a toolchain and its language server, from inside the session's runner.

Everything lands under ``<ws>/.home``:

| What | Where |
|---|---|
| mise installs | ``.home/.local/share/mise/installs/<tool>/<ver>/`` |
| links to their binaries | ``.home/.local/bin/<name>`` (on every command's ``PATH``) |
| basedpyright | its own venv under ``.home/.local/share/jaato-lsp/`` |
| typescript-language-server | ``npm -g --prefix .home/.local/share/jaato-lsp/npm``, started through the linked ``node`` |
| gopls | ``GOBIN=.home/.local/bin`` |
| jdtls | a checksum-verified Eclipse milestone under ``.home/.local/share/jaato-lsp/jdtls/<v>/``, run by its own mise JDK |

**Every subprocess enters the session's ``//child`` AppArmor profile**
before it execs (the callback the framework hands every subprocess-spawning
plugin): an installer runs a vendor's code, and the runner's base profile
keeps rights ``//child`` exists to drop (#1323).  Each gets a CLEAN
environment: HOME and the XDG directories under the workspace, mise's
directories under them, ``PATH`` with the workspace's linked binaries first,
and only the proxy / CA variables of the runner.  ``MISE_CEILING_PATHS``
stops mise reading a project config above ``.home``: a cloned repository's
``mise.toml`` must not choose what gets downloaded.

jdtls is started with ``java -jar <launcher>``, heap capped (#806: nothing
reaps a language server at session end), ``-data`` at ``${jdtlsStateRoot}``,
the framework's sibling-of-the-workspace state directory that the lsp plugin
grants ``rw`` because it sees it in ``.lsp.json``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.request
from typing import Callable, Dict, List, Optional

from .catalog import (
    LOCAL_BIN, LSP_DIR, MISE_CACHE_DIR, MISE_CONFIG_DIR, MISE_CONFIG_PATH, MISE_DATA_DIR, MISE_STATE_DIR, TOOLCHAINS,
)

PASSTHROUGH_ENV = (
    "LANG", "LC_ALL", "TZ",
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "REQUESTS_CA_BUNDLE", "PIP_INDEX_URL", "PIP_CERT",
)
SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024


class InstallError(Exception):
    pass


class InstallCancelled(Exception):
    pass


def mise_binary() -> Optional[str]:
    """``JAATO_TOOLCHAINS_MISE``, else ``mise`` on the runner's ``PATH``."""
    explicit = os.environ.get("JAATO_TOOLCHAINS_MISE")
    if explicit:
        return explicit if os.access(explicit, os.X_OK) else None
    return shutil.which("mise")


def jdtls_config_dir() -> str:
    osname = {"Darwin": "mac", "Windows": "win"}.get(platform.system(), "linux")
    arm = platform.machine().lower() in ("arm64", "aarch64")
    return f"config_{osname}_arm" if arm and osname != "win" else f"config_{osname}"


def is_our_link(path: str, mise_root: str) -> bool:
    try:
        if not os.path.islink(path):
            return False
        target = os.path.realpath(path)
        return target.startswith(os.path.realpath(mise_root) + os.sep)
    except OSError:
        return False


def link_binaries(src_bin: str, dest_bin: str, mise_root: str, log: Callable[[str], None]) -> List[str]:
    """Relative symlinks for each executable in ``src_bin``; a foreign file in ``dest_bin`` is left alone."""
    linked: List[str] = []
    try:
        names = sorted(os.listdir(src_bin))
    except OSError:
        return linked
    os.makedirs(dest_bin, exist_ok=True)
    for name in names:
        src = os.path.join(src_bin, name)
        try:
            if not os.stat(src).st_mode & 0o111:
                continue
        except OSError:
            continue
        dest = os.path.join(dest_bin, name)
        if os.path.lexists(dest):
            if not is_our_link(dest, mise_root):
                log(f"kept .home/.local/bin/{name}: it is not a link this plugin made")
                continue
            os.unlink(dest)
        os.symlink(os.path.relpath(src, dest_bin), dest)
        linked.append(name)
    return linked


def unlink_binaries(dest_bin: str, names: List[str], mise_root: str) -> List[str]:
    removed = []
    for name in names:
        dest = os.path.join(dest_bin, name)
        if is_our_link(dest, mise_root):
            os.unlink(dest)
            removed.append(name)
    return removed


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_PROGRESS_RE = re.compile(r"\d+(?:\.\d+)?/\d+(?:\.\d+)?\b.*\b\d+(?:\.\d+)?s\b|\b\d+(?:\.\d+)?s\b.*\d+(?:\.\d+)?/\d+(?:\.\d+)?\b")
_VOLATILE_RE = re.compile(r"[\d.]+|[\u2580-\u259f\u2800-\u28ff#=>\-]+")


_ERROR_LINE_RE = re.compile(r"^\s*(?:\w+Error|Error|ERROR|error|fatal|FATAL|go: .*(?:requires|cannot|not found))\b")


def error_line(out: List[str]) -> Optional[str]:
    """The last line of ``out`` that names an error, stripped; ``None`` when none does.

    A failing step's last line is often a trace or a footer (Node ends with
    ``Node.js v24.21.0`` after the stack), and the line that says what went
    wrong (``Error: Cannot find module '...'``) is above it.
    """
    for line in reversed(out):
        if _ERROR_LINE_RE.match(line):
            return line.strip()
    return None


def _venv_has_pip(venv: str) -> bool:
    import glob
    return bool(glob.glob(os.path.join(venv, "lib", "python*", "site-packages", "pip", "__init__.py")))


def clean_line(line: str) -> str:
    """The last state of a line a progress bar redrew with ``\\r``, without ANSI codes."""
    return _ANSI_RE.sub("", line.rsplit("\r", 1)[-1])


def progress_key(line: str) -> Optional[str]:
    """What stays the same across snapshots of one progress bar; ``None`` for an ordinary line.

    mise draws its bars on a terminal and, with no terminal, prints a
    snapshot of each every few seconds (``maven@3.9.9 downloading 3.0s
    0.5/9.1 MB``, then ``mise 0/1 · 6.0s``).  Two snapshots of one bar differ
    only in their numbers and bar glyphs, so the key is the line without them.
    """
    if not _PROGRESS_RE.search(line):
        return None
    return " ".join(_VOLATILE_RE.sub("", line).split())


def append_log_line(log: List[str], line: str, lookback: int = 6) -> None:
    """Append ``line``, or replace a recent snapshot of the same progress bar in place."""
    key = progress_key(line)
    if key is not None:
        for i in range(len(log) - 1, max(-1, len(log) - 1 - lookback), -1):
            if progress_key(log[i]) == key:
                log[i] = line
                return
    log.append(line)


class Installer:
    """One install, in one workspace.  Not reusable across workspaces."""

    def __init__(self, workspace: str, *, mise: str, timeout: int, paranoid: bool,
                 preexec: Optional[Callable[[], None]], cancel: threading.Event,
                 log: Callable[[str], None]) -> None:
        self.ws = workspace
        self.mise = mise
        self.timeout = timeout
        self.paranoid = paranoid
        self.preexec = preexec
        self.cancel = cancel
        self.log = log
        self._proc: Optional[subprocess.Popen] = None

    def abs(self, rel: str) -> str:
        return os.path.join(self.ws, rel)

    def env(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        home = self.abs(".home")
        env = {k: os.environ[k] for k in PASSTHROUGH_ENV if k in os.environ}
        env.update({
            "HOME": home,
            "XDG_CONFIG_HOME": os.path.join(home, ".config"),
            "XDG_CACHE_HOME": os.path.join(home, ".cache"),
            "XDG_DATA_HOME": os.path.join(home, ".local/share"),
            "XDG_STATE_HOME": os.path.join(home, ".local/state"),
            "PATH": f"{self.abs(LOCAL_BIN)}:{SYSTEM_PATH}",
            "MISE_DATA_DIR": self.abs(MISE_DATA_DIR),
            "MISE_CONFIG_DIR": self.abs(MISE_CONFIG_DIR),
            "MISE_CACHE_DIR": self.abs(MISE_CACHE_DIR),
            "MISE_STATE_DIR": self.abs(MISE_STATE_DIR),
            "MISE_GLOBAL_CONFIG_FILE": self.abs(MISE_CONFIG_PATH),
            "MISE_CEILING_PATHS": home,
            "MISE_YES": "1",
            "TMPDIR": os.path.join(home, ".cache", "tmp"),
        })
        if self.paranoid:
            env["MISE_PARANOID"] = "1"
        env.update(extra or {})
        return env

    def step(self, what: str, argv: List[str], extra_env: Optional[Dict[str, str]] = None) -> List[str]:
        """Run one step to completion; a non-zero exit is an :class:`InstallError` naming it."""
        if self.cancel.is_set():
            raise InstallCancelled()
        os.makedirs(os.path.join(self.abs(".home"), ".cache", "tmp"), exist_ok=True)
        self.log("$ " + " ".join(argv))
        out: List[str] = []
        last = ""
        try:
            proc = subprocess.Popen(
                argv, cwd=self.abs(".home"), env=self.env(extra_env), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                preexec_fn=self.preexec, start_new_session=True,
            )
        except OSError as e:
            raise InstallError(f"{what}: could not run {argv[0]}: {e}") from e
        self._proc = proc
        timer = threading.Timer(self.timeout, self._kill)
        timer.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = clean_line(line.rstrip("\n"))
                out.append(line)
                if line.strip():
                    last = line.strip()
                    self.log(line)
            code = proc.wait()
        finally:
            timer.cancel()
            self._proc = None
        if self.cancel.is_set():
            raise InstallCancelled()
        if code != 0:
            cause = error_line(out) or last
            raise InstallError(f"{what} failed (exit {code})" + (f": {cause}" if cause else ""))
        return out

    def _kill(self) -> None:
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass

    def stop(self) -> None:
        """Cancel: set the flag and kill the running step's process group."""
        self.cancel.set()
        self._kill()

    def _inside_mise(self, ref: str, path: str) -> str:
        """``path`` resolved; an :class:`InstallError` when it is not under the workspace's mise directory."""
        d = os.path.realpath(path)
        if not d.startswith(os.path.realpath(self.abs(MISE_DATA_DIR)) + os.sep):
            raise InstallError(f"locating {ref}: {d} is outside the workspace's mise directory")
        return d

    def _mise_install(self, ref: str) -> str:
        self.step(f"installing {ref}", [self.mise, "install", ref])
        where = [ln.strip() for ln in self.step(f"locating {ref}", [self.mise, "where", ref]) if ln.strip()]
        if not where:
            raise InstallError(f"locating {ref}: mise named no directory")
        return self._inside_mise(ref, where[-1])

    def _bin_paths(self, ref: str, install_dir: str) -> List[str]:
        """The directories holding ``ref``'s executables, as mise reports them.

        Not ``<install dir>/bin``: Maven and Gradle unpack one level deeper
        (``maven/3.9.9/apache-maven-3.9.9/bin/mvn``), so that guess linked
        nothing for them.  ``mise bin-paths`` is where mise itself puts them
        on ``PATH``.  Falls back to ``<install dir>/bin`` when it names none.
        """
        lines = [ln.strip() for ln in self.step(f"locating {ref}'s executables", [self.mise, "bin-paths", ref]) if ln.strip()]
        paths = [self._inside_mise(ref, ln) for ln in lines if os.path.isabs(ln)]
        return paths or [os.path.join(install_dir, "bin")]

    def install_toolchain(self, tool: str, version: str) -> Dict[str, object]:
        spec = TOOLCHAINS[tool]
        os.makedirs(self.abs(LOCAL_BIN), exist_ok=True)
        if not spec.mise:
            return {"installDir": None, "bin": []}
        ref = f"{spec.mise}@{version}"
        d = self._mise_install(ref)
        linked: List[str] = []
        for src in self._bin_paths(ref, d):
            linked += [b for b in link_binaries(src, self.abs(LOCAL_BIN), self.abs(MISE_DATA_DIR), self.log) if b not in linked]
        return {"installDir": os.path.relpath(d, self.ws), "bin": linked}

    def install_server(self, sid: str, spec: Dict[str, str]) -> Dict[str, object]:
        """Install a pinned server; returns what ``.lsp.json`` needs (absolute paths)."""
        version = spec["version"]
        lsp = self.abs(LSP_DIR)
        os.makedirs(lsp, exist_ok=True)
        if sid == "jdtls":
            return self._install_jdtls(version, spec)
        if sid == "basedpyright":
            venv = os.path.join(lsp, "basedpyright")
            self.step(f"installing basedpyright {version}", self._pip_install(venv) + [
                      "--disable-pip-version-check", "--no-input", f"basedpyright=={version}"])
            return {"id": sid, "version": version, "language": "python",
                    "command": os.path.join(venv, "bin", "basedpyright-langserver"), "args": ["--stdio"]}
        if sid == "typescript-language-server":
            prefix = os.path.join(lsp, "npm")
            self.step(f"installing typescript-language-server {version}", self._npm() + ["install", "-g",
                      "--no-fund", "--no-audit", "--prefix", prefix,
                      f"typescript-language-server@{version}", f"typescript@{spec['typescript']}"])
            return {"id": sid, "version": version, "language": "typescript", "command": self.abs(LOCAL_BIN + "/node"),
                    "args": [os.path.join(prefix, "lib", "node_modules", "typescript-language-server", "lib", "cli.mjs"), "--stdio"]}
        # gopls, built with its own Go (a gopls release needs a newer Go than
        # many projects pin; at run time it uses the project's go on PATH);
        # GOTOOLCHAIN=local so go fetches no other toolchain behind that pin.
        home = self.abs(".home")
        go_home = self._mise_install(f"go@{spec.get('go', 'latest')}")
        self.step(f"installing gopls {version}", [os.path.join(go_home, "bin", "go"), "install", f"golang.org/x/tools/gopls@{version}"], {
            "GOBIN": self.abs(LOCAL_BIN), "GOPATH": os.path.join(home, "go"),
            "GOCACHE": os.path.join(home, ".cache", "go-build"), "GOTOOLCHAIN": "local",
        })
        return {"id": sid, "version": version, "language": "go", "command": self.abs(LOCAL_BIN + "/gopls"), "args": []}

    def _pip_install(self, venv: str) -> List[str]:
        """The argv prefix that ``pip install``s into ``venv``, creating it first.

        The venv is made ``--without-pip``: ``python -m venv`` bootstraps pip
        with ``ensurepip``, which a distribution Python may lack, and then
        fails AFTER writing ``bin/python``, so a retry that saw the
        interpreter skipped creating it and ran a venv with no pip.  Pip is
        added with ``ensurepip`` when this Python has it; otherwise the
        runner's own pip installs into the venv (``pip --python``).
        """
        py = os.path.join(venv, "bin", "python")
        if not os.path.exists(py):
            self.step("creating the basedpyright venv", [sys.executable, "-m", "venv", "--without-pip", venv])
        if _venv_has_pip(venv):
            return [py, "-m", "pip", "install"]
        try:
            self.step("adding pip to the basedpyright venv", [py, "-m", "ensurepip", "--default-pip"])
        except InstallError as e:
            if importlib.util.find_spec("pip") is None:
                raise InstallError(f"{e}; and the runner's Python has no pip to install with either") from e
            self.log("no ensurepip here; installing with the runner's pip instead")
            return [sys.executable, "-m", "pip", "--python", py, "install"]
        return [py, "-m", "pip", "install"]

    def _npm(self) -> List[str]:
        """npm, as the bound Node's own ``node .../npm-cli.js``.

        Not ``.home/.local/bin/npm``: that is a link to mise's ``bin/npm``,
        itself a link to ``npm-cli.js``, started through ``#!/usr/bin/env
        node``.  Running the script with the Node binary directly has no
        link to resolve and no interpreter to look up.
        """
        node = os.path.realpath(self.abs(LOCAL_BIN + "/node"))
        cli = os.path.join(os.path.dirname(os.path.dirname(node)), "lib", "node_modules", "npm", "bin", "npm-cli.js")
        if os.path.isfile(cli):
            self._inside_mise("node", cli)
            return [node, cli]
        return [self.abs(LOCAL_BIN + "/npm")]

    def _install_jdtls(self, version: str, spec: Dict[str, str]) -> Dict[str, object]:
        java_home = self._mise_install(f"java@{spec['java']}")
        d = os.path.join(self.abs(LSP_DIR), "jdtls", version)
        if not self._launcher(d):
            fetch_jdtls(spec["mirror"], version, d, self.log, self.cancel)
        jar = self._launcher(d)
        if not jar:
            raise InstallError(f"jdtls {version}: the archive has no equinox launcher under plugins/")
        tmp = self.abs(".home/.cache/jdtls")
        os.makedirs(tmp, exist_ok=True)
        return {
            "id": "jdtls", "version": version, "language": "java", "command": os.path.join(java_home, "bin", "java"),
            "args": [
                "-Declipse.application=org.eclipse.jdt.ls.core.id1", "-Dosgi.bundles.defaultStartLevel=4",
                "-Declipse.product=org.eclipse.jdt.ls.core.product", "-Dosgi.checkConfiguration=true",
                f"-Dosgi.sharedConfiguration.area={os.path.join(d, jdtls_config_dir())}",
                "-Dosgi.sharedConfiguration.area.readOnly=true", "-Dosgi.configuration.cascaded=true",
                f"-Djava.io.tmpdir={tmp}", "-Xms100m", f"-Xmx{spec['max_heap']}",
                "-XX:+UseParallelGC", "-XX:GCTimeRatio=4", "-XX:AdaptiveSizePolicyWeight=90", "-XX:-UsePerfData",
                "-Dsun.zip.disableMemoryMapping=true", "--add-modules=ALL-SYSTEM",
                "--add-opens", "java.base/java.util=ALL-UNNAMED", "--add-opens", "java.base/java.lang=ALL-UNNAMED",
                "-jar", os.path.join(d, "plugins", jar),
                "-configuration", "${jdtlsStateRoot}/.jaato-config", "-data", "${jdtlsStateRoot}",
            ],
            "runtimeDir": os.path.relpath(java_home, self.ws),
        }

    @staticmethod
    def _launcher(d: str) -> Optional[str]:
        try:
            jars = sorted(n for n in os.listdir(os.path.join(d, "plugins"))
                          if re.match(r"^org\.eclipse\.equinox\.launcher_.*\.jar$", n))
        except OSError:
            return None
        return jars[-1] if jars else None


def _get(url: str, cancel: threading.Event) -> bytes:
    if cancel.is_set():
        raise InstallCancelled()
    with urllib.request.urlopen(url, timeout=300) as r:
        data = r.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise InstallError(f"{url}: larger than {MAX_DOWNLOAD_BYTES} bytes")
    return data


def fetch_jdtls(mirror: str, version: str, dest: str, log: Callable[[str], None], cancel: threading.Event) -> None:
    """Download a jdtls milestone, verify its published sha256, extract it refusing escaping members."""
    base = mirror.rstrip("/") + "/" + version + "/"
    try:
        name = _get(base + "latest.txt", cancel).decode().strip()
    except (OSError, ValueError):
        listing = _get(base, cancel).decode("utf-8", "replace")
        names = sorted(set(re.findall(r"jdt-language-server-" + re.escape(version) + r"-\d+\.tar\.gz(?![.\w])", listing)))
        name = names[-1] if names else ""
    if not re.fullmatch(r"jdt-language-server-[\w.-]+\.tar\.gz", name or ""):
        raise InstallError(f"no jdtls {version} archive found under {base}")
    log("downloading " + base + name)
    data = _get(base + name, cancel)
    want = _get(base + name + ".sha256", cancel).decode().split()[0].lower()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise InstallError(f"checksum mismatch for {name}: got {got}, published {want}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".jdtls-", dir=os.path.dirname(dest))
    try:
        arc = os.path.join(tmp, name)
        with open(arc, "wb") as f:
            f.write(data)
        out = os.path.join(tmp, "x")
        root = os.path.realpath(out)
        with tarfile.open(arc) as t:
            for m in t.getmembers():
                p = os.path.realpath(os.path.join(out, m.name))
                if not (p == root or p.startswith(root + os.sep)) or m.issym() or m.islnk() or m.isdev():
                    raise InstallError(f"refusing archive member {m.name}")
            t.extractall(out, filter="data")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.replace(out, dest)
        log("extracted " + name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
