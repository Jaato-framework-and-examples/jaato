"""What a bound toolchain leaves in a repository, kept out of git.

A toolchain in use leaves caches and build output in the repository it runs
in (``__pycache__/``, ``node_modules/``, ``target/``, jdtls's
``.project``/``.settings/``, ...).  A project may not ignore them, and a
model running ``git add -A`` then commits them.  So while a toolchain is
bound, its :attr:`~.catalog.Toolchain.ignores` go into every checkout of the
workspace:

- the checkouts are the workspace root and its immediate, non-hidden
  subdirectories (where the web coder clones), each only if it holds ``.git``;
- the patterns go in that repository's ``.git/info/exclude``, never in its
  ``.gitignore``.  The ``.gitignore`` is the project's committed file: an
  edit there is a diff the next commit carries into somebody else's
  repository.  ``info/exclude`` follows the same rules, is local, and is
  never committed.  Like any ignore rule it hides nothing already tracked;
- they sit between two marker lines, and only that block is ever rewritten.
  Lines outside it are the user's.  With nothing bound the block is removed;
- a workspace-root checkout also gets the files this plugin writes at the
  root (``.lsp.json``, ``.jaato/environment.json``, the offer) and ``.home/``,
  where every install lands.  The daemon also gives ``.home/`` its own ``*``
  gitignore, but this block does not depend on that file being there;
- Java bound without Maven or Gradle still gets their output directories in a
  checkout that builds with them (``pom.xml``, ``build.gradle[.kts]``), since
  that is what ``./mvnw`` or ``./gradlew`` on the bound JDK produces.

The workspace is model-writable, so a ``.git`` file (a worktree's
``gitdir:``) or a planted symlink could point anywhere: an exclude file whose
real path is not inside the workspace is left alone and reported.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .catalog import HOME_DIR, LSP_CONFIG_PATH, MANIFEST_PATH, OFFER_PATH, TOOLCHAINS
from .detect import MAX_SCANNED_DIRS

BEGIN = "# >>> jaato-managed: toolchains v1 (bind or unbind toolchains in the web coder; edits inside this block are replaced)"
END = "# <<< jaato-managed: toolchains"

#: The root files this plugin writes, excluded in a workspace-root checkout.
ROOT_FILES = tuple("/" + p for p in (LSP_CONFIG_PATH, MANIFEST_PATH, OFFER_PATH)) + ("/" + HOME_DIR + "/",)

_MAVEN_FILES = ("pom.xml",)
_GRADLE_FILES = ("build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts")


def checkouts(workspace: str) -> List[str]:
    """The workspace root and its immediate subdirectories that hold ``.git``."""
    found: List[str] = []
    if os.path.lexists(os.path.join(workspace, ".git")):
        found.append(workspace)
    try:
        names = sorted(os.listdir(workspace))
    except OSError:
        return found
    for name in names[:MAX_SCANNED_DIRS]:
        path = os.path.join(workspace, name)
        if not name.startswith(".") and os.path.isdir(path) and os.path.lexists(os.path.join(path, ".git")):
            found.append(path)
    return found


def exclude_file(checkout: str, workspace: str) -> Optional[str]:
    """The checkout's ``info/exclude``, or ``None`` if it is not inside the workspace."""
    dot_git = os.path.join(checkout, ".git")
    git_dir = dot_git
    if os.path.isfile(dot_git):
        try:
            with open(dot_git, encoding="utf-8") as f:
                line = f.readline().strip()
        except OSError:
            return None
        if not line.startswith("gitdir:"):
            return None
        git_dir = os.path.join(checkout, line[len("gitdir:"):].strip())
        # A worktree keeps its info/ in the common directory.
        try:
            with open(os.path.join(git_dir, "commondir"), encoding="utf-8") as f:
                git_dir = os.path.join(git_dir, f.readline().strip())
        except OSError:
            pass
    target = os.path.realpath(os.path.join(git_dir, "info", "exclude"))
    root = os.path.realpath(workspace)
    if os.path.commonpath([root, target]) != root:
        return None
    return target


def patterns_for(checkout: str, workspace: str, tools: List[str]) -> List[str]:
    """The block's lines for one checkout, in a stable order."""
    wanted = list(tools)
    if "java" in wanted:
        if "maven" not in wanted and any(os.path.exists(os.path.join(checkout, f)) for f in _MAVEN_FILES):
            wanted.append("maven")
        if "gradle" not in wanted and any(os.path.exists(os.path.join(checkout, f)) for f in _GRADLE_FILES):
            wanted.append("gradle")
    lines: List[str] = []
    for tool in sorted(set(wanted)):
        for p in TOOLCHAINS[tool].ignores:
            if p not in lines:
                lines.append(p)
    if tools and os.path.realpath(checkout) == os.path.realpath(workspace):
        lines += [p for p in ROOT_FILES if p not in lines]
    return lines


def splice(text: str, lines: List[str]) -> str:
    """``text`` with the managed block replaced by ``lines`` (removed when empty)."""
    src = text.splitlines()
    kept: List[str] = []
    i = 0
    while i < len(src):
        if src[i] == BEGIN:
            end = next((j for j in range(i + 1, len(src)) if src[j] == END), None)
            # An end marker the user deleted: drop only the begin line, never what follows it.
            i = (end + 1) if end is not None else i + 1
            continue
        kept.append(src[i])
        i += 1
    while kept and not kept[-1].strip():
        kept.pop()
    if lines:
        kept += ([""] if kept else []) + [BEGIN, *lines, END]
    return "\n".join(kept) + "\n" if kept else ""


def write_excludes(workspace: str, m: Dict[str, Any]) -> List[str]:
    """Bring every checkout's managed block in line with the bound toolchains; a note per checkout skipped."""
    from .state import atomic_write

    tools = [t["tool"] for t in m.get("toolchains") or [] if t.get("tool") in TOOLCHAINS]
    notes: List[str] = []
    for checkout in checkouts(workspace):
        rel = os.path.relpath(checkout, workspace)
        rel = "the workspace root" if rel == "." else rel
        path = exclude_file(checkout, workspace)
        if path is None:
            notes.append(f"left {rel}'s git excludes alone: its git directory is not inside the workspace")
            continue
        try:
            with open(path, encoding="utf-8") as f:
                current = f.read()
        except FileNotFoundError:
            current = ""
        except OSError as e:
            notes.append(f"could not read {rel}'s .git/info/exclude ({e.strerror or e})")
            continue
        body = splice(current, patterns_for(checkout, workspace, tools))
        if body == current:
            continue
        try:
            atomic_write(path, body)
        except OSError as e:
            notes.append(f"could not write {rel}'s .git/info/exclude ({e.strerror or e})")
    return notes
