"""Does this workspace hold jaato's own packages as source? (#1413)

A notebook kernel is launched with the DAEMON's jaato import dirs in its own
argv (``_kernel_argv``, kept deliberately by #1322: the kernel is the one
process that *is* jaato).  So code a cell runs in-process -- ``import
jaato_server``, ``pytest.main([...])``, ``importlib.reload`` -- resolves
``jaato_server`` / ``jaato_sdk`` / ``jaato_premium`` to the daemon's install,
whatever the workspace holds.  ``!python -m pytest`` and ``cli`` start a clean
interpreter and see the checkout.

That is correct and it was undisclosed: a session developing jaato inside
jaato ran ``pip install -e ./jaato-server``, got rc=0, and then spent several
rounds on "the edit didn't land" because its tests ran against the daemon's
code.  This module answers the one question every surface that discloses it
asks -- the notebook boundary notice, the first notebook result of a kernel,
and ``get_environment(aspect="runtime")`` -- so they cannot disagree.

Stdlib only, and it imports nothing: the daemon's location is found with
:func:`importlib.util.find_spec`, which is where a cell's ``import`` would
look in this process too.

What counts as "the workspace holds jaato":

- a package source at the workspace root or in an immediate child (a clone):
  ``jaato-server/jaato_server/__init__.py``,
  ``jaato-sdk/jaato_sdk/__init__.py``,
  ``jaato-premium/jaato_premium/__init__.py`` or ``jaato_premium/__init__.py``;
- an EDITABLE install of one of the three in a workspace venv (the configured
  ``workspace_venv``, ``.jaato/tool-venv`` or ``.venv``), read from the
  distribution's ``direct_url.json``.

A source that IS the daemon's own package (the daemon runs from this very
checkout) is not reported: cells then import the code the model edits.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
from typing import Dict, Iterable, List, Optional

#: The packages whose in-process import resolves to the daemon's install.
JAATO_PACKAGES = ("jaato_server", "jaato_sdk", "jaato_premium")

#: Where each package's source sits relative to a checkout root.
_SOURCE_LAYOUTS = {
    "jaato_server": ("jaato-server/jaato_server",),
    "jaato_sdk": ("jaato-sdk/jaato_sdk",),
    "jaato_premium": ("jaato-premium/jaato_premium", "jaato_premium"),
}

#: Workspace venvs looked at beside the configured one.
_DEFAULT_VENVS = (".jaato/tool-venv", ".venv")

#: The sentence every surface renders.  One place, so they say the same thing.
SHADOWING_NOTE = (
    "This workspace holds jaato's own source, but in-process notebook cells "
    "import the daemon's jaato from {daemon}, not this checkout; run tests "
    "with `!python -m pytest` or through `cli`."
)


def daemon_package_dirs() -> Dict[str, str]:
    """Where this process imports each jaato package from (absent = not found)."""
    out: Dict[str, str] = {}
    for name in JAATO_PACKAGES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        origin = getattr(spec, "origin", None) if spec else None
        if origin:
            out[name] = os.path.realpath(os.path.dirname(origin))
    return out


def _checkout_roots(workspace: str) -> List[str]:
    roots = [workspace]
    try:
        entries = sorted(os.listdir(workspace))
    except OSError:
        return roots
    for entry in entries:
        path = os.path.join(workspace, entry)
        if not entry.startswith(".") and os.path.isdir(path):
            roots.append(path)
    return roots


def _source_dirs(workspace: str) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {}
    for root in _checkout_roots(workspace):
        for name, layouts in _SOURCE_LAYOUTS.items():
            for rel in layouts:
                pkg = os.path.join(root, rel)
                if os.path.isfile(os.path.join(pkg, "__init__.py")):
                    found.setdefault(name, []).append(os.path.realpath(pkg))
    return found


def _editable_dirs(workspace: str, venvs: Iterable[Optional[str]]) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {}
    candidates = [v for v in venvs if v]
    candidates += [os.path.join(workspace, v) for v in _DEFAULT_VENVS]
    seen = set()
    for venv in candidates:
        venv = os.path.realpath(venv)
        if venv in seen:
            continue
        seen.add(venv)
        pattern = os.path.join(venv, "lib", "python*", "site-packages",
                               "jaato*.dist-info", "direct_url.json")
        for direct_url in sorted(glob.glob(pattern)):
            dist = os.path.basename(os.path.dirname(direct_url))
            name = dist.split("-", 1)[0].replace("-", "_").lower()
            if name not in JAATO_PACKAGES:
                continue
            try:
                with open(direct_url, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                continue
            if not (data.get("dir_info") or {}).get("editable"):
                continue
            url = str(data.get("url") or "")
            path = url[len("file://"):] if url.startswith("file://") else url
            if path:
                found.setdefault(name, []).append(os.path.realpath(path))
    return found


def workspace_jaato_shadowing(
    workspace: Optional[str],
    venvs: Iterable[Optional[str]] = (),
    *,
    daemon_dirs: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, object]]:
    """Report jaato sources in *workspace* that in-process cells will NOT import.

    Args:
        workspace: The session workspace root.  ``None`` reports nothing.
        venvs: Configured workspace venvs to check for editable installs,
            beside ``.jaato/tool-venv`` and ``.venv``.
        daemon_dirs: Override for :func:`daemon_package_dirs` (tests).

    Returns:
        ``None`` when nothing is shadowed -- callers then add nothing, so the
        prompt-cache prefix of an ordinary workspace is unchanged.  Otherwise
        ``{"daemon": {pkg: dir}, "workspace": {pkg: [dirs]}, "note": str}``.
    """
    if not workspace or not os.path.isdir(workspace):
        return None
    daemon = daemon_package_dirs() if daemon_dirs is None else daemon_dirs
    shadowed: Dict[str, List[str]] = {}
    for source in (_source_dirs(workspace), _editable_dirs(workspace, venvs)):
        for name, dirs in source.items():
            for path in dirs:
                if os.path.realpath(daemon.get(name, "")) == path:
                    continue  # the daemon runs this very checkout
                if path not in shadowed.get(name, []):
                    shadowed.setdefault(name, []).append(path)
    if not shadowed:
        return None
    where = sorted({daemon[n] for n in shadowed if n in daemon})
    return {
        "daemon": {n: daemon[n] for n in shadowed if n in daemon},
        "workspace": shadowed,
        "note": shadowing_note(where),
    }


def shadowing_note(daemon_paths: Iterable[str]) -> str:
    """The one sentence, naming where the daemon's jaato is imported from."""
    paths = list(daemon_paths)
    return SHADOWING_NOTE.format(
        daemon=", ".join(paths) if paths else "the daemon's install")
