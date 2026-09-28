"""The npm tarball carries every file the plugin reads, and a missing template fails soft.

``@jaato/web-coder-server`` ships this plugin, and its ``package.json``
``files`` list decides what reaches npm.  0.5.0 listed only ``*.py``, so the
vendored ``gitignore_templates/`` were left out and every bind raised
``FileNotFoundError`` on an npm install.  The first test checks every file of
the package against the ``files`` globs (npm semantics: ``*`` stays inside a
path segment, a directory entry covers everything beneath it).  The second
checks that an install missing a template leaves the checkout's excludes as
they were and says why, instead of failing the bind.
"""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path

from jaato_web_coder_toolchains import ignores, templates

_SERVER = Path(__file__).resolve().parents[2]
_PACKAGE = Path(templates.__file__).resolve().parent


def _covered(rel: str, globs: list) -> bool:
    parts = rel.split("/")
    for g in globs:
        gparts = g.rstrip("/").split("/")
        if len(gparts) > len(parts):
            continue
        if not all(fnmatch.fnmatchcase(p, gp) for p, gp in zip(parts, gparts)):
            continue
        if len(gparts) == len(parts) or "*" not in gparts[-1]:
            return True  # an exact match, or a directory entry covering its subtree
    return False


def test_every_plugin_file_is_in_the_npm_files_list():
    globs = json.loads((_SERVER / "package.json").read_text())["files"]
    missing = []
    for root, dirs, files in os.walk(_PACKAGE):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            if name.endswith((".pyc", ".pyo")):
                continue
            rel = Path(root, name).relative_to(_SERVER).as_posix()
            if not _covered(rel, globs):
                missing.append(rel)
    assert not missing, f"not published to npm by package.json 'files': {missing}"


def test_the_matcher_is_not_vacuous():
    assert not _covered("plugin/pkg/sub/x.gitignore", ["plugin/pkg/*.py"])
    assert not _covered("plugin/pkg/sub/x.gitignore", ["plugin/pkg/*"])
    assert _covered("plugin/pkg/sub/x.gitignore", ["plugin/pkg/sub/*"])
    assert _covered("plugin/pkg/sub/x.gitignore", ["plugin/pkg"])


def _checkout(ws: Path) -> Path:
    (ws / ".git" / "info").mkdir(parents=True)
    return ws


def test_a_missing_template_leaves_the_excludes_and_says_so(tmp_path, monkeypatch):
    ws = _checkout(tmp_path)
    exclude = ws / ".git" / "info" / "exclude"
    manifest = {"toolchains": [{"tool": "node"}]}
    templates._cache.clear()
    assert ignores.write_excludes(str(ws), manifest) == []
    written = exclude.read_text()
    assert "node_modules/" in written

    templates._cache.clear()
    monkeypatch.setattr(templates, "_TEMPLATES_DIR", str(tmp_path / "absent"))
    notes = ignores.write_excludes(str(ws), {"toolchains": [{"tool": "node"}, {"tool": "go"}]})
    assert exclude.read_text() == written  # not rewritten without the templates
    assert len(notes) == 1 and "template is missing" in notes[0]

    # Nothing cached from the failure: a repaired install is used at once.
    monkeypatch.undo()
    assert "node_modules/" in templates.patterns_for_tool("node")
