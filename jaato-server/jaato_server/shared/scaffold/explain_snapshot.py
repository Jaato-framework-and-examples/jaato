"""Generate the ``explain`` snapshot jaato-sdk answers from without jaato-server.

The reader, and the account of what the snapshot holds and what it does not,
is :mod:`jaato_sdk.scaffold.explain_snapshot`.  This module produces the file
it reads, ``jaato-sdk/jaato_sdk/scaffold/explain_snapshot.json``::

    python -m jaato_server.shared.scaffold.explain_snapshot --write
    python -m jaato_server.shared.scaffold.explain_snapshot --check

Every entry is what :func:`introspection_verbs.render_topic` returned for that
topic, rendered by the same dispatch ``explain`` uses, so a snapshot answer is
by construction the answer a jaato-server install gives.  Which topics are
rendered is read from the topic table (``ExplainScope.kind``, ``.live_only``,
``.names``); nothing here lists topics.

**A snapshot mirrors jaato-server, and only jaato-server.**  It ships inside
jaato-sdk, so whatever else the generating environment happened to hold must
not reach it.  :func:`build` refuses, naming the cause, when:

- a plugin comes from another distribution, or a package contributes an
  ``explain`` topic or section (premium, an out-of-tree plugin): the snapshot
  would claim them for every jaato-sdk install;
- an in-tree plugin did not load (a missing optional dependency such as
  ``pexpect``): every listing would silently lack it.  Install the extras the
  generating command names;
- a rendering contains a path of the generating machine (its home, its
  working directory, its virtualenv, this checkout): that topic describes the
  machine, and belongs under ``live_only`` on the topic table.

It also renders in a clean subprocess (:func:`_render_in_clean_env`): an empty
``$HOME``, an empty working directory, no ``JAATO_*`` variables.  Plugins read
user-tier files while describing themselves (``prompt_library`` lists the
skills under ``~/.claude/skills``), and a snapshot generated from a
developer's home would ship that developer's skills.

The ``--check`` mode is what the drift guard runs; ``.githooks/pre-commit``
runs ``--write`` beside the authoring snapshot's.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from jaato_sdk.scaffold import explain_snapshot as _reader

#: Where the snapshot is written: beside the SDK reader that loads it.
SNAPSHOT_FILE = _reader.SNAPSHOT_FILE

#: This module's import name, for the clean subprocess (``__name__`` is
#: ``"__main__"`` when it runs as ``-m``).
_MODULE = "jaato_server.shared.scaffold.explain_snapshot"

#: The command that regenerates it, named by every refusal and by the guard.
REGENERATE = "python -m jaato_server.shared.scaffold.explain_snapshot --write"

#: Environment variables carried into the clean render; everything else is
#: dropped.  Interpreter and import resolution only: nothing that a renderer
#: could read as configuration.
_KEPT_ENV = ("PATH", "PYTHONPATH", "VIRTUAL_ENV", "SYSTEMROOT", "LANG",
             "LC_ALL", "TMPDIR")


class SnapshotRefused(RuntimeError):
    """The generating environment cannot produce a snapshot of jaato-server alone."""


# ------------------------------------------------------------------ checks


def _in_tree_plugins() -> List[str]:
    """The plugin packages jaato-server ships that ``introspect.plugins`` lists.

    Read from the source (``PLUGIN_KIND`` of ``tool`` or ``enrichment`` in a
    package ``__init__.py``) rather than from discovery, because the question
    is which plugins discovery failed to load.
    """
    from . import introspect

    out = []
    for init in sorted(introspect._PLUGIN_DIR.glob("*/__init__.py")):
        try:
            tree = ast.parse(init.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "PLUGIN_KIND"
                            for t in node.targets)
                    and isinstance(node.value, ast.Constant)
                    and node.value.value in ("tool", "enrichment")):
                out.append(init.parent.name)
    return out


def _refuse_foreign(plugins: Dict[str, Any]) -> None:
    """Refuse when anything but jaato-server would reach the snapshot."""
    from . import introspection_verbs as iv

    problems = []
    foreign = sorted(f"{n} ({getattr(i, 'source', '')})"
                     for n, i in plugins.items()
                     if not getattr(i, "builtin", True))
    if foreign:
        problems.append("plugins from other distributions: " + ", ".join(foreign))
    contributed = sorted(getattr(t, "name", "?")
                         for t in iv._discover_external_topics())
    if contributed:
        problems.append("contributed explain topics/sections: "
                        + ", ".join(contributed))
    missing = sorted(set(_in_tree_plugins()) - set(plugins))
    if missing:
        problems.append("in-tree plugins that did not load (install "
                        "jaato-server's extras, e.g. [interactive]): "
                        + ", ".join(missing))
    if problems:
        raise SnapshotRefused(
            "the explain snapshot mirrors jaato-server alone, and this "
            "environment would put something else in it:\n  - "
            + "\n  - ".join(problems))


def _machine_paths() -> List[str]:
    """Paths of the generating machine that no snapshot entry may contain."""
    import jaato_server

    paths = {
        str(Path.home()), os.getcwd(), sys.prefix,
        str(Path(jaato_server.__file__).resolve().parents[1]),
    }
    return sorted(p for p in paths if p and p not in ("/", ""))


def _refuse_machine_paths(topics: Dict[str, Any]) -> None:
    """Refuse a rendering that names a path of the generating machine."""
    paths = _machine_paths()
    leaks = []
    for key, entry in topics.items():
        blob = json.dumps(entry, default=str)
        hit = next((p for p in paths if p in blob), None)
        if hit:
            leaks.append(f"`explain {key}`".replace("explain `", "`")
                         + f" names {hit}")
    if leaks:
        raise SnapshotRefused(
            "these renderings describe the generating machine rather than "
            "the installed code; mark the topic `live_only` on the topic "
            "table in introspection_verbs.py:\n  - " + "\n  - ".join(leaks))


# ----------------------------------------------------------------- render


#: The argument-less ``introspect`` catalogs :func:`build` evaluates once.
_MEMOIZED = ("plugins", "providers", "events")


def _memoized(fn: Callable[[], Any]) -> Callable[[], Any]:
    """*fn* evaluated once.  Discovery is pure within one generation."""
    box: List[Any] = []

    def wrapper():
        if not box:
            box.append(fn())
        return box[0]
    return wrapper


def _render(scope, name=None) -> Dict[str, Any]:
    """One topic as the snapshot stores it, or :class:`SnapshotRefused`."""
    from . import introspection_verbs as iv

    ok, data, text, error = iv.render_topic(scope, name, ".")
    if not ok:
        raise SnapshotRefused(
            f"`explain {_reader.entry_key(scope, name)}`".replace(" `", "`")
            + f" failed to render: {error}")
    # A JSON round trip normalises tuples, Paths and enums the way the CLI's
    # own `--json` (``default=str``) does, so the stored data is what
    # `explain --json` prints.
    return {"data": json.loads(json.dumps(data, default=str)), "text": text}


def build() -> Dict[str, Any]:
    """The snapshot as this environment renders it, after every refusal check.

    Call it from :func:`_render_in_clean_env`'s subprocess; called anywhere
    else it renders whatever ``$HOME`` and working directory it finds.
    """
    from . import authoring_contracts, introspect
    from . import introspection_verbs as iv

    # Each named render re-asks introspect for the whole catalog; within one
    # generation the answer cannot change, so it is asked once per catalog.
    saved = {n: getattr(introspect, n) for n in _MEMOIZED}
    for n, fn in saved.items():
        setattr(introspect, n, _memoized(fn))
    try:
        _refuse_foreign(introspect.plugins())
        topics: Dict[str, Any] = {_reader.OVERVIEW_KEY: _render(None)}
        aliases: Dict[str, Dict[str, str]] = {}
        live_only: Dict[str, str] = {}
        scopes: List[str] = []
        for scope, spec in iv._SCOPES.items():
            if spec.live_only:
                live_only[scope] = spec.live_only
                continue
            if spec.kind == "workspace":
                continue
            if spec.kind == "named":
                if spec.names is None:
                    raise SnapshotRefused(
                        f"`explain {scope}` is a named topic with no `names` "
                        f"on the topic table, so the snapshot cannot "
                        f"enumerate it; declare one or mark it live_only")
                table = {}
                for canonical, spellings in sorted(spec.names().items()):
                    entry = _render(scope, canonical)
                    topics[_reader.entry_key(scope, canonical)] = entry
                    for spelling in {canonical, *spellings}:
                        table[spelling] = canonical
                aliases[scope] = dict(sorted(table.items()))
            else:
                topics[scope] = _render(scope)
            scopes.append(scope)
    finally:
        for n, fn in saved.items():
            setattr(introspect, n, fn)
    _refuse_machine_paths(topics)
    return {
        "snapshot_version": _reader.SNAPSHOT_VERSION,
        "jaato_server_version": authoring_contracts._live_server_version(
            introspect),
        "scopes": scopes,
        "live_only": live_only,
        "reads_workspace": sorted(
            row["scope"] for row in iv.scope_catalog()
            if row["reads_workspace"] and not row["contributed_by"]),
        "aliases": aliases,
        "topics": topics,
    }


def render_snapshot(data: Dict[str, Any]) -> str:
    """The on-disk text: sorted keys, one key per line, so diffs are readable."""
    return json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def _clean_env(home: Path) -> Dict[str, str]:
    """The environment the snapshot is rendered in: *home* as ``$HOME``.

    Only :data:`_KEPT_ENV` survives from the caller's environment, so no
    ``JAATO_*`` variable, profile set or provider key reaches a renderer.
    """
    env = {k: os.environ[k] for k in _KEPT_ENV if k in os.environ}
    env.update(HOME=str(home), JAATO_RELEASE_CHECK="off",
               PYTHONDONTWRITEBYTECODE="1")
    return env


def _render_in_clean_env() -> str:
    """:func:`build` run in a subprocess with an empty home and working dir.

    Returns:
        The snapshot text.

    Raises:
        SnapshotRefused: the subprocess refused, with its reason.
    """
    with tempfile.TemporaryDirectory(prefix="jaato-explain-snapshot-") as tmp:
        home = Path(tmp) / "home"
        cwd = Path(tmp) / "cwd"
        home.mkdir()
        cwd.mkdir()
        proc = subprocess.run(
            [sys.executable, "-m", _MODULE, "--emit"],
            cwd=str(cwd), env=_clean_env(home), capture_output=True,
            text=True)
    if proc.returncode != 0:
        reason = proc.stdout.strip() or proc.stderr.strip()[-2000:]
        raise SnapshotRefused(reason)
    return proc.stdout


def main(argv=None) -> int:
    """``--write`` regenerates the snapshot, ``--check`` compares it.

    ``--emit`` is the subprocess half: it prints the snapshot text, or the
    refusal and exit 1.
    """
    args = sys.argv[1:] if argv is None else argv
    if args == ["--emit"]:
        try:
            sys.stdout.write(render_snapshot(build()))
        except SnapshotRefused as exc:
            print(f"explain snapshot refused: {exc}")
            return 1
        return 0
    if args not in (["--write"], ["--check"]):
        print(f"usage: {REGENERATE} | --check", file=sys.stderr)
        return 2
    try:
        text = _render_in_clean_env()
    except SnapshotRefused as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args == ["--write"]:
        SNAPSHOT_FILE.write_text(text, encoding="utf-8")
        print(f"wrote {SNAPSHOT_FILE}")
        return 0
    current = (SNAPSHOT_FILE.read_text(encoding="utf-8")
               if SNAPSHOT_FILE.is_file() else "")
    if current == text:
        print(f"{SNAPSHOT_FILE.name} is current")
        return 0
    print(f"{SNAPSHOT_FILE.name} is stale; run `{REGENERATE}`")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
