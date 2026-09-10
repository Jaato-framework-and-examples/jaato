"""The ``dependencies`` facet — an optional query word on any ``explain`` scope.

Dependencies are not a topic of their own; every topic has them. A provider
imports vendor packages, a plugin shells out to binaries, a transport needs an
extra, and the framework itself is two distributions that can drift apart. So
this is asked as a word appended to whatever you were already asking about:

    jaato-scaffold explain dependencies                     # the framework's own picture
    jaato-scaffold explain provider openrouter dependencies
    jaato-scaffold explain plugin cli dependencies
    jaato-scaffold explain transports dependencies

EVERYTHING HERE IS MEASURED, NOTHING IS DECLARED.  There is no table mapping a
provider to a package — such a table is wrong the moment someone adds an import,
and being wrong is worse than being absent because the reader stops checking.
Instead: distribution requirements come from installed metadata, third-party
imports from parsing the implementation, and import health from actually trying
it. Where a fact cannot be derived, this says so rather than guessing.

WHY THE VERSION SKEW MATTERS MOST.  `pip` records a version at install time; an
editable install keeps pointing at a working tree that moves. On the machine
this was written, `jaato-sdk` reported 0.15.0 while the checkout it points at
said 0.16.0 — so every version-derived answer in the system, including the
provenance stamp `jaato-scaffold install` writes, was quietly naming a build
that is not the one running.
"""
from __future__ import annotations

import ast
import json
import sys
from importlib.metadata import PackageNotFoundError, distribution, requires, version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FIRST_PARTY = {"shared", "server", "jaato_sdk", "jaato_embedded", "jaato_premium"}
JAATO_DISTS = ("jaato-sdk", "jaato-server", "jaato-tui", "jaato-eval", "jaato-premium")


# ----------------------------------------------------------------- distributions

def dist_state(name: str) -> Dict[str, Any]:
    """Installed version, whether it is editable, and the SOURCE version if so.

    The last is the whole point: an editable install's metadata is a snapshot
    taken at install time, and the tree underneath it keeps moving.
    """
    out: Dict[str, Any] = {"name": name, "installed": None, "editable": False,
                           "source": None, "source_version": None, "skew": False}
    try:
        out["installed"] = version(name)
        d = distribution(name)
    except PackageNotFoundError:
        return out
    raw = d.read_text("direct_url.json")
    if raw:
        try:
            info = json.loads(raw)
        except ValueError:
            info = {}
        if info.get("dir_info", {}).get("editable"):
            url = str(info.get("url", ""))
            src = Path(url[len("file://"):]) if url.startswith("file://") else None
            out["editable"] = True
            out["source"] = str(src) if src else None
            if src:
                pp = src / "pyproject.toml"
                if pp.is_file():
                    for line in pp.read_text(encoding="utf-8", errors="replace").splitlines():
                        t = line.strip()
                        if t.startswith("version") and "=" in t:
                            out["source_version"] = t.split("=", 1)[1].strip().strip('"\'')
                            break
    sv, iv = out["source_version"], out["installed"]
    out["skew"] = bool(sv and iv and sv != iv)
    return out


def environment() -> Dict[str, Any]:
    """Which interpreter answered, and whether the answer is being shadowed.

    This is not decoration.  `importlib.metadata` resolves a distribution by
    searching `sys.path`, so a `PYTHONPATH` pointing at a source checkout makes
    that checkout answer INSTEAD of the installed copy — the version reads as
    the source's, `editable` reads False, and a skew that exists for the daemon
    becomes invisible to whoever is asking.  Measured on the machine this was
    written: with PYTHONPATH set, jaato-sdk read 0.16.0 / not-editable; without
    it, 0.15.0 / editable / source 0.16.0 — the same environment, two answers.

    Hence doctor's standing rule that it must run in the daemon's environment.
    """
    import os
    pp = os.environ.get("PYTHONPATH", "")
    return {"executable": sys.executable,
            "pythonpath": [x for x in pp.split(os.pathsep) if x],
            "shadowed": bool(pp)}


def framework_picture() -> Dict[str, Any]:
    dists = [dist_state(n) for n in JAATO_DISTS]
    extras: Dict[str, List[str]] = {}
    for n in ("jaato-sdk", "jaato-server"):
        for req in (requires(n) or []):
            if 'extra ==' in req:
                mark = req.split('extra ==')[1].strip().strip('"\';')
                extras.setdefault(f"{n}[{mark}]", []).append(req.split(';')[0].strip())
    return {"environment": environment(),
            "distributions": [d for d in dists if d["installed"]],
            "missing": [d["name"] for d in dists if not d["installed"]],
            "extras": extras}


# ------------------------------------------------------------------- imports

def _module_file(dotted: str) -> Optional[Path]:
    try:
        mod = __import__(dotted, fromlist=["__file__"])
    except Exception:      # noqa: BLE001 — an unimportable module is a FINDING
        return None
    f = getattr(mod, "__file__", None)
    return Path(f) if f else None


def _imported_names(path: Path) -> set:
    """Every top-level module name one file imports, parsed not imported.

    Parsing rather than importing is the point: a package that is MISSING
    still shows up, which is exactly the case worth reporting.  A file that
    will not parse contributes nothing rather than failing the scan.
    """
    if not path or not path.is_file():
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            # node.level means a relative import — first-party by definition.
            names.add(node.module.split(".")[0])
    return names


def third_party_imports(paths: List[Path]) -> List[str]:
    """The non-stdlib, non-first-party subset of what ``paths`` import."""
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    found = set()
    for p in paths:
        found |= _imported_names(p)
    return sorted(n for n in found
                  if n and n not in stdlib and n not in FIRST_PARTY and not n.startswith("_"))


def _health(names: List[str]) -> Dict[str, str]:
    out = {}
    for n in names:
        try:
            __import__(n)
            out[n] = "importable"
        except ImportError:
            out[n] = "MISSING"
        except Exception as exc:      # noqa: BLE001 — importable but broken is not missing
            out[n] = f"import raised {type(exc).__name__}"
    return out


def _pkg_files(pkg_dir: Path) -> List[Path]:
    return sorted(pkg_dir.glob("*.py")) if pkg_dir.is_dir() else []


# -------------------------------------------------------------------- facets

def for_provider(name: str) -> Dict[str, Any]:
    base = Path(__file__).resolve().parents[1] / "plugins" / "model_provider" / name
    files = _pkg_files(base)
    imports = third_party_imports(files)
    return {"kind": "provider", "name": name,
            "implementation": str(base) if base.is_dir() else None,
            "files_parsed": len(files), "imports": imports, "health": _health(imports)}


def for_plugin(name: str, source: str) -> Dict[str, Any]:
    dotted = source.split("(")[-1].rstrip(")").strip() if "(" in source else source
    f = _module_file(dotted)
    files = _pkg_files(f.parent) if f and f.name == "__init__.py" else ([f] if f else [])
    imports = third_party_imports(files)
    return {"kind": "plugin", "name": name, "module": dotted,
            "implementation": str(f) if f else None,
            "files_parsed": len(files), "imports": imports, "health": _health(imports),
            "note": None if f else "module could not be imported — nothing to parse"}


# --------------------------------------------------------------------- render

def render(scope: Optional[str], name: Optional[str]) -> Tuple[Dict[str, Any], str]:
    """``(data, text)`` for the dependency facet of ``scope``/``name``."""
    if scope in ("provider", "providers") and name:
        d = for_provider(name)
        return d, _render_unit(d)
    if scope in ("plugin", "plugins") and name:
        from . import introspect
        info = introspect.plugins().get(name)
        if info is None:
            return {"error": f"unknown plugin '{name}'"}, f"unknown plugin '{name}'"
        d = for_plugin(name, getattr(info, "source", "") or "")
        return d, _render_unit(d)

    d = framework_picture()
    if scope:
        d["scope"] = scope
        d["note"] = (f"'{scope}' has no dependency facet of its own; "
                     f"showing the framework's. Ask a named provider or plugin "
                     f"for its imports.")
    return d, _render_framework(d)


def _render_unit(d: Dict[str, Any]) -> str:
    lines = [f"{d['kind']} '{d['name']}' — third-party imports, parsed from source", ""]
    if d.get("note"):
        lines += [f"  {d['note']}", ""]
    if d.get("implementation"):
        lines.append(f"  implementation : {d['implementation']}  ({d['files_parsed']} file(s))")
    if not d["imports"]:
        lines += ["", "  no third-party imports — stdlib and framework only."]
        return "\n".join(lines)
    lines.append("")
    for n in d["imports"]:
        state = d["health"].get(n, "?")
        mark = "  " if state == "importable" else "! "
        lines.append(f"  {mark}{n:24} {state}")
    if any(v != "importable" for v in d["health"].values()):
        lines += ["", "  A MISSING import is only a failure if the code path runs — some are",
                  "  optional or lazily imported.  It is reported because nothing else will."]
    return "\n".join(lines)


def _render_framework(d: Dict[str, Any]) -> str:
    lines = ["framework dependencies — what is installed, and whether it agrees with itself", ""]
    if d.get("note"):
        lines += [f"  {d['note']}", ""]
    env = d.get("environment") or {}
    lines.append(f"  answered by: {env.get('executable')}")
    if env.get("shadowed"):
        lines += ["  !! PYTHONPATH is set, so a source checkout may be answering INSTEAD",
                  "     of the installed distribution — versions below may be the tree's,",
                  "     not the daemon's.  Re-run without PYTHONPATH to see what the",
                  "     daemon sees: " + ", ".join(env.get("pythonpath", []))]
    lines.append("")
    for dist in d["distributions"]:
        tag = " (editable)" if dist["editable"] else ""
        lines.append(f"  {dist['name']:16} {dist['installed']}{tag}")
        if dist["editable"] and dist["source"]:
            lines.append(f"  {'':16} source: {dist['source']}")
        if dist["skew"]:
            lines.append(f"  {'':16} !! SKEW: metadata says {dist['installed']}, "
                         f"the source tree says {dist['source_version']}")
    if d["missing"]:
        lines += ["", f"  not installed: {', '.join(d['missing'])}"]
    if any(x["skew"] for x in d["distributions"]):
        lines += ["",
                  "  A skew means every version-derived answer in this environment names a",
                  "  build that is not the one running — including the provenance stamp",
                  "  `jaato-scaffold install` writes.  Reinstall the editable distribution",
                  "  (`pip install -e <source>`) to resync its metadata."]
    if d["extras"]:
        lines += ["", "  optional extras (install to enable):"]
        for k, v in sorted(d["extras"].items()):
            lines.append(f"    {k:22} {', '.join(v)}")
    return "\n".join(lines)
