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
import re
import sys
from functools import lru_cache
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    packages_distributions,
    requires,
    version,
)
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

FIRST_PARTY = {"shared", "server", "jaato_sdk", "jaato_embedded", "jaato_premium"}
JAATO_DISTS = ("jaato-sdk", "jaato-server", "jaato-tui", "jaato-eval", "jaato-premium")
EXTRA_DISTS = ("jaato-sdk", "jaato-server")


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
        for req in _requires(n):
            if 'extra ==' in req:
                mark = req.split('extra ==')[1].strip().strip('"\';')
                extras.setdefault(f"{n}[{mark}]", []).append(req.split(';')[0].strip())
    return {"environment": environment(),
            "distributions": [d for d in dists if d["installed"]],
            "missing": [d["name"] for d in dists if not d["installed"]],
            "extras": extras}


# ------------------------------------------------------------------- imports
#
# A UNIT IS NOT ONLY ITS OWN DIRECTORY.  Most of the model providers own no
# `import openai` at all — they inherit `_openai_compat.OpenAICompatProvider`,
# and the import that actually fails at runtime lives in its `_lazy.py`.  So a
# scan of `<provider>/*.py` reported `openai` for the handful that own their
# streaming loop and stayed silent for every other one, including the provider
# literally named `openai`.  Measured before the fix:
#
#     provider 'azure_openai' — third-party imports, parsed from source
#       ! azure    MISSING
#         httpx    importable            <- and the runtime died on `openai`
#
# The report therefore follows FIRST-PARTY imports too, and reports what the
# resulting closure imports.  Two bounds keep that from becoming a fiction:
#
#   * the walk is confined to the unit's FAMILY directory — `model_provider/`
#     for a provider, `plugins/` for a plugin — because that is where the
#     shared machinery a unit inherits lives.  Unbounded, `subagent` reached
#     84 files and claimed `anthropic` and `google` as its dependencies, which
#     is true of the framework and useless as an answer about the plugin.
#   * every import is reported WITH the file that imports it, so "this unit's
#     own source" and "the shared transport it inherits" stay distinguishable.


def _module_file(dotted: str) -> Optional[Path]:
    try:
        mod = __import__(dotted, fromlist=["__file__"])
    except Exception:      # noqa: BLE001 — an unimportable module is a FINDING
        return None
    f = getattr(mod, "__file__", None)
    return Path(f) if f else None


@lru_cache(maxsize=None)
def _parse(path: Path) -> Optional[ast.Module]:
    """The AST of one file, or ``None`` if it is absent or will not parse.

    Cached because the closure walk and the name scan both want it, and a
    file reached through two different imports must not be read twice.
    """
    if not path or not path.is_file():
        return None
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return None


@lru_cache(maxsize=None)
def _first_party_roots() -> Dict[str, Path]:
    """``{"shared": Path(...), "jaato_sdk": Path(...)}`` — found, not assumed.

    Resolved on the filesystem rather than with ``find_spec`` because
    importing a first-party package executes it, and this module's whole
    contract is that a package which cannot be imported still gets reported.
    Both layouts answer: a venv (every root a sibling in site-packages) and
    the source checkout (``jaato-server/shared`` beside ``jaato-server/server``,
    with ``jaato-sdk/jaato_sdk`` found through ``sys.path``).
    """
    roots: Dict[str, Path] = {}
    here = Path(__file__).resolve().parents[1]          # .../shared
    candidates = [here.parent] + [Path(p) for p in sys.path if p]
    for parent in candidates:
        for pkg in FIRST_PARTY:
            if pkg in roots:
                continue
            cand = parent / pkg
            if (cand / "__init__.py").is_file():
                roots[pkg] = cand.resolve()
    return roots


def _as_file(target: Path) -> Optional[Path]:
    """``foo`` → ``foo.py`` or ``foo/__init__.py``, whichever exists."""
    for cand in (target.with_suffix(".py"), target / "__init__.py"):
        if cand.is_file():
            return cand.resolve()
    return None


def _absolute_first_party(dotted: str) -> Optional[Path]:
    parts = dotted.split(".")
    root = _first_party_roots().get(parts[0])
    if root is None:
        return None
    return _as_file(root.joinpath(*parts[1:])) if len(parts) > 1 else _as_file(root)


def _relative_first_party(path: Path, level: int, module: Optional[str]) -> Optional[Path]:
    base = path.resolve().parent
    for _ in range(level - 1):
        base = base.parent
    return _as_file(base.joinpath(*module.split("."))) if module else _as_file(base)


def _from_targets(node: ast.ImportFrom, path: Path) -> Set[Path]:
    """First-party files one ``from ... import ...`` statement can reach.

    Both readings are resolved and whichever exists is kept: ``from .foo
    import Bar`` names the module ``foo``, while ``from . import foo`` names
    the same module through an attribute.  Trying both is cheaper than
    deciding which spelling was used, and a name that resolves to no file
    (a class, a function) simply contributes nothing.
    """
    out: Set[Path] = set()
    stems = [node.module] if node.module else []
    stems += [f"{node.module}.{a.name}" if node.module else a.name for a in node.names]
    for stem in stems:
        f = (_relative_first_party(path, node.level, stem) if node.level
             else _absolute_first_party(stem))
        if f:
            out.add(f)
    return out


def _is_third_party(top: str) -> bool:
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    return bool(top) and top not in stdlib and not top.startswith("_")


def _plain_import(node: ast.Import) -> Tuple[Set[str], Set[Path]]:
    third: Set[str] = set()
    first: Set[Path] = set()
    for alias in node.names:
        top = alias.name.split(".")[0]
        if top in FIRST_PARTY:
            f = _absolute_first_party(alias.name)
            if f:
                first.add(f)
        elif _is_third_party(top):
            third.add(top)
    return third, first


def _from_import(node: ast.ImportFrom, path: Path) -> Tuple[Set[str], Set[Path]]:
    top = (node.module or "").split(".")[0]
    if node.level or top in FIRST_PARTY:
        return set(), _from_targets(node, path)
    return ({top} if _is_third_party(top) else set()), set()


def _scan(path: Path) -> Tuple[Set[str], Set[Path]]:
    """``(third-party top-level names, first-party files)`` imported by ``path``.

    Parsed, never imported — a package that is MISSING is exactly the case
    worth reporting, and it cannot report itself.
    """
    tree = _parse(path)
    if tree is None:
        return set(), set()
    third: Set[str] = set()
    first: Set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names, files = _plain_import(node)
        elif isinstance(node, ast.ImportFrom):
            names, files = _from_import(node, path)
        else:
            continue
        third |= names
        first |= files
    return third, first


def _imported_names(path: Path) -> set:
    """Every top-level module name one file imports, parsed not imported."""
    third, _ = _scan(path)
    return third


def third_party_imports(paths: List[Path]) -> List[str]:
    """The non-stdlib, non-first-party subset of what ``paths`` import.

    One file at a time, following nothing — ``import_closure`` is the
    transitive form.  Kept because it is the honest answer to "what does THIS
    file import", which is what the framework picture and the tests ask.
    """
    found: Set[str] = set()
    for p in paths:
        found |= _imported_names(p)
    return sorted(n for n in found if n not in FIRST_PARTY)


def import_closure(files: List[Path],
                   scope_root: Optional[Path] = None) -> Tuple[Dict[str, List[Path]], List[Path]]:
    """``({import: [files importing it]}, files walked)`` over a first-party closure.

    ``scope_root`` bounds the walk: a first-party file outside it is not
    followed.  Passing ``None`` follows nothing, which makes this the
    single-file scan.  The importing files are carried because "your own
    source imports this" and "the base class you inherit imports this" are
    different findings, and the second one is why this function exists.
    """
    seen: Set[Path] = set()
    imports: Dict[str, Set[Path]] = {}
    frontier = [f.resolve() for f in files if f]
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        third, first = _scan(current)
        for name in third:
            imports.setdefault(name, set()).add(current)
        if scope_root is None:
            continue
        for target in first:
            if target not in seen and _within(target, scope_root):
                frontier.append(target)
    return ({k: sorted(v) for k, v in sorted(imports.items())}, sorted(seen))


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


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


# -------------------------------------------------------------------- extras
#
# "Why was this not installed automatically?" has an answer in the installed
# metadata, and printing MISSING without it leaves the reader to guess a pip
# command.  The mapping is measured where it can be — `packages_distributions`
# knows which import names an INSTALLED distribution provides — and matched by
# name where it cannot, since a package that is missing cannot say what it
# would have provided.  Both are labelled in the render, so a name match is
# never presented as a resolved fact.

_REQ_NAME = re.compile(r"^[A-Za-z0-9._-]+")


def _requirement_name(req: str) -> str:
    """``"openai>=1.66; extra == \"nim\""`` → ``"openai"``."""
    head = req.split(";")[0].strip().split("[")[0].strip()
    m = _REQ_NAME.match(head)
    return m.group(0) if m else ""


def _requires(dist_name: str) -> List[str]:
    """``requires()`` that answers ``[]`` for a distribution that is not installed.

    A checkout run straight from source has no `jaato-sdk` metadata, and a
    diagnostic that raises there is a diagnostic nobody can run at exactly
    the moment it is wanted.
    """
    try:
        return list(requires(dist_name) or [])
    except PackageNotFoundError:
        return []


def _extra_marker(req: str) -> Optional[str]:
    if "extra ==" not in req:
        return None
    return req.split("extra ==")[1].strip().strip('"\';')


@lru_cache(maxsize=None)
def _provided_import_names(dist_name: str) -> Tuple[str, ...]:
    """Import names ``dist_name`` provides — measured if installed, else guessed.

    The measured tier is `packages_distributions()`, which is a fact about
    what is on disk (``beautifulsoup4`` → ``bs4`` is only knowable this way).
    The fallback is the PyPI-name convention plus the first hyphen segment,
    which is what makes ``azure-identity`` → ``azure`` and ``google-genai`` →
    ``google`` land while a package nobody installed is still being asked
    about.
    """
    key = dist_name.lower().replace("_", "-")
    try:
        mapping = packages_distributions()
    except Exception:      # noqa: BLE001 — metadata is best-effort here
        mapping = {}
    measured = tuple(sorted(mod for mod, dists in mapping.items()
                            if any(d.lower().replace("_", "-") == key for d in dists)))
    if measured:
        return measured
    guesses = {key.replace("-", "_"), key.replace("-", ""), key.split("-")[0]}
    return tuple(sorted(g for g in guesses if g))


@lru_cache(maxsize=None)
def _extras_index() -> Dict[str, Tuple[str, ...]]:
    """``{import name: ("jaato-server[azure-openai]", ...)}``, from metadata."""
    return {k: tuple(sorted(v)) for k, v in _metadata_index()[0].items()}


@lru_cache(maxsize=None)
def _core_index() -> Dict[str, Tuple[str, ...]]:
    """``{import name: ("jaato-server", ...)}`` for UNCONDITIONAL requirements.

    A missing import declared here is a different finding from one declared in
    an extra: nobody has to select it, so the install is incomplete rather
    than merely narrow, and `pip install <package>` is the wrong advice.
    """
    return {k: tuple(sorted(v)) for k, v in _metadata_index()[1].items()}


@lru_cache(maxsize=None)
def _extra_sizes() -> Dict[str, int]:
    """How many packages each extra pulls in — the tie-break when several
    extras declare the same missing package and none is named after the unit.
    Recommending the narrowest is the one that installs least beside it."""
    return _metadata_index()[2]


@lru_cache(maxsize=None)
def _metadata_index() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, int]]:
    """One pass over installed metadata: extras, core requirements, extra sizes."""
    extras: Dict[str, Set[str]] = {}
    core: Dict[str, Set[str]] = {}
    sizes: Dict[str, int] = {}
    for dist_name in EXTRA_DISTS:
        for req in _requires(dist_name):
            extra = _extra_marker(req)
            pkg = _requirement_name(req)
            if not pkg or pkg.lower().startswith("jaato-"):
                continue
            label = f"{dist_name}[{extra}]" if extra else dist_name
            if extra:
                sizes[label] = sizes.get(label, 0) + 1
            for mod in _provided_import_names(pkg):
                (extras if extra else core).setdefault(mod, set()).add(label)
    return extras, core, sizes


def _install_hint(missing: List[str], unit_name: str) -> Dict[str, Any]:
    """What to run, and whether the answer is really "your install is broken".

    An extra named after the unit wins outright when it covers everything —
    `azure_openai`'s answer is `jaato-server[azure-openai]`, not the four
    other extras that also happen to declare `openai`.  Where no extra is
    named after the unit (ten OpenAI-compatible providers have none), the
    narrowest covering extra is offered and the report says plainly that the
    choice was arbitrary, rather than implying `minimax` belongs to `nim`.
    """
    index, core = _extras_index(), _core_index()
    covered = {n: set(index.get(n, ())) for n in missing}
    incomplete = sorted(n for n in missing if not covered[n] and n in core)
    orphans = sorted(n for n in missing if not covered[n] and n not in core)
    remaining = [n for n in missing if covered[n]]
    chosen, named = _cover(covered, remaining, unit_name)
    return {"commands": [f"pip install '{e}'" for e in chosen]
                        + ([f"pip install {' '.join(orphans)}"] if orphans else []),
            "named_after_unit": named,
            "incomplete_install": incomplete}


def _cover(covered: Dict[str, Set[str]], remaining: List[str],
           unit_name: str) -> Tuple[List[str], bool]:
    """Greedy set cover over extras; ``True`` if one is named after the unit."""
    for extra in sorted(set().union(*covered.values()) if covered else set()):
        tag = extra.split("[")[1].rstrip("]").replace("-", "_")
        if tag == unit_name and remaining and all(extra in covered[n] for n in remaining):
            return [extra], True
    chosen: List[str] = []
    sizes = _extra_sizes()
    while remaining:
        counts: Dict[str, int] = {}
        for name in remaining:
            for extra in covered[name]:
                counts[extra] = counts.get(extra, 0) + 1
        best = min(sorted(counts), key=lambda e: (-counts[e], sizes.get(e, 99)))
        chosen.append(best)
        remaining = [n for n in remaining if best not in covered[n]]
    return chosen, False


# -------------------------------------------------------------------- facets

def _unit(kind: str, name: str, files: List[Path],
          scope_root: Optional[Path], implementation: Optional[Path],
          note: Optional[str] = None) -> Dict[str, Any]:
    """The shape both named facets return, over the first-party closure.

    ``via`` is what makes the closure honest: for each import it names the
    file that actually imports it, relative to ``scope_root``, and an import
    the unit's own files never mention is marked as reached through shared
    machinery.  Without it the report would claim `azure_openai` imports
    `openai` — it does not; the base class it inherits does.
    """
    own = {f.resolve() for f in files}
    imports, walked = import_closure(files, scope_root)
    names = list(imports)
    via: Dict[str, List[str]] = {}
    for n, importers in imports.items():
        if importers and not any(f in own for f in importers):
            via[n] = [_display(f, scope_root) for f in importers]
    health = _health(names)
    missing = [n for n in names if health.get(n) != "importable"]
    return {"kind": kind, "name": name,
            "implementation": str(implementation) if implementation else None,
            "files_parsed": len(files), "files_reached": len(walked),
            "scope_root": str(scope_root) if scope_root else None,
            "imports": names, "health": health, "via": via,
            "extras": {n: list(_extras_index().get(n, ())) for n in names},
            "install": _install_hint(missing, name) if missing else {},
            "note": note}


def _display(path: Path, scope_root: Optional[Path]) -> str:
    if scope_root:
        try:
            return str(path.relative_to(scope_root))
        except ValueError:
            pass
    return str(path)


def for_provider(name: str) -> Dict[str, Any]:
    base = Path(__file__).resolve().parents[1] / "plugins" / "model_provider" / name
    files = _pkg_files(base)
    return _unit("provider", name, files,
                 base.parent if base.is_dir() else None,
                 base if base.is_dir() else None)


def for_plugin(name: str, source: str) -> Dict[str, Any]:
    """Dependencies of one plugin, located WITHOUT importing it where possible.

    ``_module_file`` imports to find a file, which fails precisely when a
    dependency is missing — the case this facet exists to explain.  An
    in-tree plugin is therefore resolved on the filesystem first, and the
    import is kept only as the fallback for an out-of-tree distribution
    whose layout this module cannot know.
    """
    dotted = source.split("(")[-1].rstrip(")").strip() if "(" in source else source
    f = _absolute_first_party(dotted) or _module_file(dotted)
    files = _pkg_files(f.parent) if f and f.name == "__init__.py" else ([f] if f else [])
    scope = f.resolve().parent.parent if f and f.name == "__init__.py" else (
        f.resolve().parent if f else None)
    d = _unit("plugin", name, files, scope, f,
              None if f else "module could not be imported — nothing to parse")
    d["module"] = dotted
    return d


# --------------------------------------------------------------------- render

def render(scope: Optional[str], name: Optional[str]) -> Tuple[Dict[str, Any], str]:
    """``(data, text)`` for the dependency facet of ``scope``/``name``."""
    if scope in ("provider", "providers") and name:
        d = for_provider(name)
        return d, _render_unit(d)
    if scope in ("plugin", "plugins") and name:
        d = _plugin_facet(name)
        if "error" in d:
            return d, d["error"]
        return d, _render_unit(d)

    d = framework_picture()
    if scope:
        d["scope"] = scope
        d["note"] = (f"'{scope}' has no dependency facet of its own; "
                     f"showing the framework's. Ask a named provider or plugin "
                     f"for its imports.")
    return d, _render_framework(d)


def _plugin_facet(name: str) -> Dict[str, Any]:
    """One plugin's dependencies, including when discovery could not load it.

    ``introspect.plugins()`` discovers by importing, so a plugin whose
    dependency is missing is SKIPPED — and that is exactly the plugin someone
    asks this facet about.  Answering "unknown plugin" there would withhold
    the report at the only moment it is wanted, so an in-tree package is
    located on the filesystem before the name is called unknown.
    """
    from . import introspect
    info = introspect.plugins().get(name)
    if info is not None:
        return for_plugin(name, getattr(info, "source", "") or "")
    dotted = f"shared.plugins.{name}"
    if _absolute_first_party(dotted) is None:
        return {"error": f"unknown plugin '{name}'"}
    d = for_plugin(name, dotted)
    d["note"] = ("discovery skipped this plugin in this environment — reported "
                 "from its source, which is what the imports below explain")
    return d


def _render_unit(d: Dict[str, Any]) -> str:
    lines = [f"{d['kind']} '{d['name']}' — third-party imports, parsed from source", ""]
    if d.get("note"):
        lines += [f"  {d['note']}", ""]
    if d.get("implementation"):
        lines.append(f"  implementation : {d['implementation']}  "
                     f"({d['files_parsed']} file(s))")
    reached = (d.get("files_reached") or 0) - (d.get("files_parsed") or 0)
    if reached > 0:
        lines.append(f"  also parsed    : {reached} shared file(s) this unit imports, "
                     f"under {d.get('scope_root')}")
    if not d["imports"]:
        lines += ["", "  no third-party imports — stdlib and framework only."]
        return "\n".join(lines)
    lines.append("")
    for n in d["imports"]:
        state = d["health"].get(n, "?")
        mark = "  " if state == "importable" else "! "
        origin = d.get("via", {}).get(n) or []
        shown = ", ".join(origin[:2]) + (f" (+{len(origin) - 2} more)" if len(origin) > 2 else "")
        tail = f"   via {shown}" if origin else ""
        lines.append(f"  {mark}{n:24} {state:16}{tail}".rstrip())
    if any(v != "importable" for v in d["health"].values()):
        lines += ["", "  A MISSING import is only a failure if the code path runs — some are",
                  "  optional or lazily imported.  It is reported because nothing else will.",
                  "  One reached only `via` a shared file is imported by machinery this unit",
                  "  inherits, which is where a lazily-imported vendor SDK usually sits."]
    lines += _render_extras(d)
    return "\n".join(lines)


def _render_extras(d: Dict[str, Any]) -> List[str]:
    """Which optional extra would have installed each import, and the command.

    This is the answer to "why is it not installed automatically": it is
    declared, but in an extra nobody selected.  Read from the installed
    metadata, so an extra added to `pyproject.toml` shows up here without
    anything in this file changing.
    """
    declared = {n: v for n, v in (d.get("extras") or {}).items() if v}
    lines: List[str] = []
    if declared:
        lines += ["", "  declared by (matched by name against installed metadata):"]
        for n, extras in sorted(declared.items()):
            lines.append(f"    {n:24} {', '.join(extras)}")
    install = d.get("install") or {}
    if install.get("commands"):
        lines += ["", "  to install what is missing here:"]
        lines += [f"      {cmd}" for cmd in install["commands"]]
        if not install.get("named_after_unit") and declared:
            lines += [f"  no extra is named after '{d['name']}' — the command above is simply",
                      "  the narrowest one declaring the missing package; any other extra",
                      "  listed above installs the same thing."]
    if install.get("incomplete_install"):
        lines += ["",
                  "  !! " + ", ".join(install["incomplete_install"]) +
                  " is an UNCONDITIONAL requirement of jaato-server, so this is",
                  "     not a missing extra — the installation itself is incomplete."]
    return lines


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
