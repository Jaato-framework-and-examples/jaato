"""The files a binding writes, all from inside the session's runner.

| File | Marker | Read by |
|---|---|---|
| ``.jaato/environment.json`` | ``"_jaato_managed": "environment v1"`` | the page (via ``workspace.file.fetch``), ``get_environment(aspect="runtime")`` (#1346) |
| ``.home/.config/mise/config.toml`` | ``# jaato-managed: toolchains v1`` | mise, for a person who opens a shell |
| ``.lsp.json`` | ``"_jaato_managed": "lsp v1"`` | the ``lsp`` plugin (it reads only ``languageServers``) |
| ``.home/.mavenrc`` | ``# jaato-managed: mavenrc v1`` | Maven's ``mvn`` script (and ``./mvnw``), while Java or Maven is bound |
| ``.home/.config/go/env`` | ``# jaato-managed: goenv v1`` | the go command (``go env``), while Go is bound |
| each checkout's ``.git/info/exclude`` | a ``# >>> jaato-managed: toolchains v1`` ... ``# <<<`` block | git: the bound toolchains' caches and build output (:mod:`.ignores`) |

``environment.json`` is the record: the bound toolchains, the proposals the
last scan found, the repository guidance files, and the current or last job.
It is in the workspace, where the model can write, so nothing read back from
it is trusted beyond what it names: a link is removed only if it resolves
into the workspace's mise directory, and a name with a ``/`` is ignored.

The mise config, ``.lsp.json``, ``.mavenrc`` and the Go env file are managed files: written when absent or
still carrying their marker, never over a copy whose marker the user removed.
Every write is a temp file plus ``os.replace``.
"""

from __future__ import annotations

import json
import os
import secrets
from typing import Any, Dict, List, Optional

from .ignores import write_excludes
from .catalog import GOENV_PATH, GOTMP_DIR, LSP_CONFIG_PATH, MANIFEST_PATH, MAVENRC_PATH, MISE_CONFIG_PATH, TOOLCHAINS, VERSION_RE

MARKER_KEY = "_jaato_managed"
ENVIRONMENT_MARKER = "environment v1"
LSP_MARKER = "lsp v1"
MISE_MARKER = "# jaato-managed: toolchains v1 — delete this line to keep your own edits"
MAVENRC_MARKER = "# jaato-managed: mavenrc v1 — delete this line to keep your own edits"

#: Sourced by Maven's ``mvn`` script from ``$HOME/.mavenrc``; ``$HOME`` is the
#: workspace home (#1225).  Java takes ``user.home`` from the account, not
#: from ``$HOME``, so without it Maven uses the account's ``~/.m2``, which a
#: confined session cannot write.  ``java.io.tmpdir`` likewise ignores
#: ``$TMPDIR`` and defaults to ``/tmp``, which a confined session cannot
#: write either (#1361).
MAVENRC_BODY = "\n".join([
    MAVENRC_MARKER,
    "# Java reads user.home from the account, not $HOME: point it (and the temp dir) at this workspace.",
    'MAVEN_OPTS="-Duser.home=$HOME${TMPDIR:+ -Djava.io.tmpdir=$TMPDIR}${MAVEN_OPTS:+ $MAVEN_OPTS}"',
    "export MAVEN_OPTS",
]) + "\n"

GOENV_MARKER = "# jaato-managed: goenv v1 — delete this line to keep your own edits"


def goenv_body(workspace: str) -> str:
    """The Go env file: build and test binaries go under the workspace.

    ``go test`` and ``go run`` build into ``$GOTMPDIR``, which defaults to
    ``$TMPDIR`` (the session tmpdir, where the profile grants no exec); the
    workspace is exec-granted on a managed workspace, so they build there.
    The go command skips a line that does not start with a capital letter,
    so the marker is safe in this file.
    """
    return "\n".join([GOENV_MARKER, f"GOTMPDIR={os.path.join(os.path.realpath(workspace), GOTMP_DIR)}"]) + "\n"


#: The toolchains whose Maven (a bound one, or a repository's ``./mvnw``) reads ``.mavenrc``.
MAVENRC_TOOLS = ("java", "maven")


def atomic_write(path: str, body: str, mode: int = 0o644) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = os.path.join(os.path.dirname(path), f".{secrets.token_hex(6)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _safe_name(v: Any) -> bool:
    return isinstance(v, str) and 0 < len(v) <= 128 and "/" not in v and v not in (".", "..")


def empty_manifest() -> Dict[str, Any]:
    return {"toolchains": [], "proposals": [], "guidance": [], "job": None}


def read_manifest(workspace: str) -> Dict[str, Any]:
    """The manifest, cleaned; an absent or unreadable one is empty."""
    m = empty_manifest()
    try:
        with open(os.path.join(workspace, MANIFEST_PATH), encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return m
    if not isinstance(raw, dict):
        return m
    for t in raw.get("toolchains") or []:
        if not (isinstance(t, dict) and t.get("tool") in TOOLCHAINS and isinstance(t.get("version"), str) and VERSION_RE.match(t["version"])):
            continue
        m["toolchains"].append({
            "tool": t["tool"], "version": t["version"],
            "installDir": t.get("installDir") if isinstance(t.get("installDir"), str) else None,
            "bin": [b for b in t.get("bin") or [] if _safe_name(b)],
            "server": t.get("server") if isinstance(t.get("server"), dict) else None,
            "serverBin": [b for b in t.get("serverBin") or [] if _safe_name(b)],
            "boundAt": t.get("boundAt") if isinstance(t.get("boundAt"), str) else None,
        })
    for key in ("proposals", "guidance"):
        if isinstance(raw.get(key), list):
            m[key] = raw[key][:64]
    if isinstance(raw.get("job"), dict):
        m["job"] = raw["job"]
    if isinstance(raw.get("scannedAt"), str):
        m["scannedAt"] = raw["scannedAt"]
    return m


def write_manifest(workspace: str, m: Dict[str, Any]) -> None:
    body = {
        MARKER_KEY: ENVIRONMENT_MARKER,
        "note": "Written by the web coder's toolchain plugin. Bind and unbind toolchains from the web coder, not here.",
        **m,
        "toolchains": sorted(m.get("toolchains") or [], key=lambda t: t["tool"]),
    }
    atomic_write(os.path.join(workspace, MANIFEST_PATH), json.dumps(body, indent=2) + "\n")


def _ours_json(path: str, marker: str) -> Optional[bool]:
    """``True``: ours; ``False``: someone else's; ``None``: absent."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return False
    return isinstance(raw, dict) and raw.get(MARKER_KEY) == marker


def _ours_hash(path: str, prefix: str = "# jaato-managed: toolchains v") -> Optional[bool]:
    try:
        with open(path, encoding="utf-8") as f:
            first = f.readline().strip()
    except FileNotFoundError:
        return None
    except OSError:
        return False
    return first.startswith(prefix)


def _toml_string(v: str) -> str:
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_derived(workspace: str, m: Dict[str, Any]) -> List[str]:
    """Rewrite the mise config, ``.lsp.json``, ``.mavenrc``, the Go env and the git excludes from the manifest.

    Returns a note per file left alone.
    """
    notes: List[str] = []
    chains = sorted(m.get("toolchains") or [], key=lambda t: t["tool"])

    mise_path = os.path.join(workspace, MISE_CONFIG_PATH)
    owned = _ours_hash(mise_path)
    if owned is False:
        notes.append(f"kept your own {MISE_CONFIG_PATH} (its jaato-managed marker was removed)")
    else:
        lines = [MISE_MARKER, "# Toolchains bound by the web coder; bind or unbind them there.", "[tools]"]
        lines += [f"{TOOLCHAINS[t['tool']].mise} = {_toml_string(t['version'])}" for t in chains if TOOLCHAINS[t["tool"]].mise]
        atomic_write(mise_path, "\n".join(lines) + "\n")

    lsp_path = os.path.join(workspace, LSP_CONFIG_PATH)
    servers = {}
    for t in chains:
        s = t.get("server")
        if isinstance(s, dict) and isinstance(s.get("language"), str):
            servers[s["language"]] = {"command": s.get("command"), "args": s.get("args") or [], "languageId": s["language"]}
    owned = _ours_json(lsp_path, LSP_MARKER)
    if owned is False:
        notes.append(f"kept your own {LSP_CONFIG_PATH} (it has no jaato-managed marker), so it does not list the bound servers")
    elif servers:
        atomic_write(lsp_path, json.dumps({MARKER_KEY: LSP_MARKER, "languageServers": servers}, indent=2) + "\n")
    elif owned:
        os.unlink(lsp_path)

    go_path = os.path.join(workspace, GOENV_PATH)
    owned = _ours_hash(go_path, "# jaato-managed: goenv v")
    wanted = any(t["tool"] == "go" for t in chains)
    if owned is False:
        if wanted:
            notes.append(f"kept your own {GOENV_PATH} (its jaato-managed marker was removed); "
                         f"go test builds into $GOTMPDIR, which must be inside the workspace to run")
    elif wanted:
        os.makedirs(os.path.join(workspace, GOTMP_DIR), exist_ok=True)
        atomic_write(go_path, goenv_body(workspace))
    elif owned:
        os.unlink(go_path)

    rc_path = os.path.join(workspace, MAVENRC_PATH)
    owned = _ours_hash(rc_path, "# jaato-managed: mavenrc v")
    wanted = any(t["tool"] in MAVENRC_TOOLS for t in chains)
    if owned is False:
        if wanted:
            notes.append(f"kept your own {MAVENRC_PATH} (its jaato-managed marker was removed); "
                         "Maven needs -Duser.home=$HOME in it to use the workspace's ~/.m2")
    elif wanted:
        atomic_write(rc_path, MAVENRC_BODY)
    elif owned:
        os.unlink(rc_path)

    notes += write_excludes(workspace, m)
    return notes
