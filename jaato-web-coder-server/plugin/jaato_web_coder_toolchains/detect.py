"""What a workspace's files suggest it needs.  Proposes only; never installs.

Reads the workspace root and its immediate, non-hidden subdirectories (the
web coder clones each repository to ``<ws>/<name>``), never deeper.  Every
read is bounded and a file that cannot be read is skipped.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

MAX_SCANNED_DIRS = 64
MAX_MARKER_BYTES = 64 * 1024
_SKIP_DIRS = {"node_modules", "worktrees", "__pycache__", "venv", "target", "dist", "build"}


@dataclass
class Detection:
    tool: str
    #: The version the repository pins, verbatim; ``None`` when it names none.
    pin: Optional[str]
    #: Workspace-relative path of the file that said so.
    source: str

    def to_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


def read_bounded(path: str) -> Optional[str]:
    try:
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            return f.read(MAX_MARKER_BYTES).decode("utf-8", "replace")
    except OSError:
        return None


def scan_dirs(workspace: str) -> List[str]:
    """``""`` (the root), then each immediate non-hidden subdirectory, sorted."""
    try:
        names = sorted(
            e.name for e in os.scandir(workspace)
            if e.is_dir(follow_symlinks=False) and not e.name.startswith(".") and e.name not in _SKIP_DIRS
        )
    except OSError:
        names = []
    return [""] + names[:MAX_SCANNED_DIRS]


def _rel(d: str, name: str) -> str:
    return f"{d}/{name}" if d else name


def _parse_tool_versions(body: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in body.split("\n"):
        t = re.sub(r"#.*", "", line).strip()
        parts = t.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


_TOOL_VERSION_NAMES = [
    ("node", ["node", "nodejs"]), ("go", ["go", "golang"]), ("bun", ["bun"]),
    ("java", ["java"]), ("maven", ["maven"]), ("gradle", ["gradle"]),
]


def _pom_java_release(pom: str) -> Optional[str]:
    for tag in ("maven.compiler.release", "release", "java.version", "maven.compiler.source", "maven.compiler.target"):
        m = re.search(r"<" + re.escape(tag) + r">\s*([0-9][0-9.]*)\s*</", pom)
        if m:
            return m.group(1)
    return None


def _gradle_java_version(build: str) -> Optional[str]:
    for pattern in (
        r"JavaLanguageVersion\.of\(\s*[\"']?(\d+)[\"']?\s*\)",
        r"jvmToolchain\(\s*(\d+)\s*\)",
        r"sourceCompatibility\s*=\s*JavaVersion\.VERSION_(\d+(?:_\d+)?)",
        r"sourceCompatibility\s*=\s*[\"']?(\d+(?:\.\d+)?)[\"']?",
    ):
        m = re.search(pattern, build)
        if m:
            return m.group(1).replace("_", ".")
    return None


def _first_line(body: Optional[str]) -> Optional[str]:
    return body.split("\n")[0] if body is not None else None


def detect_dir(workspace: str, d: str) -> List[Detection]:
    base = os.path.join(workspace, d)
    found: List[Detection] = []
    seen: set = set()

    def add(tool: str, pin: Optional[str], source: str) -> None:
        if tool in seen:
            return
        seen.add(tool)
        found.append(Detection(tool, pin.strip() if pin and pin.strip() else None, source))

    def isfile(name: str) -> bool:
        return os.path.isfile(os.path.join(base, name))

    tv = read_bounded(os.path.join(base, ".tool-versions"))
    if tv is not None:
        pins = _parse_tool_versions(tv)
        for tool, names in _TOOL_VERSION_NAMES:
            name = next((n for n in names if n in pins), None)
            if name:
                add(tool, pins[name], _rel(d, ".tool-versions"))
    for f in (".nvmrc", ".node-version"):
        body = read_bounded(os.path.join(base, f))
        if body is not None:
            add("node", _first_line(body), _rel(d, f))
    pkg = read_bounded(os.path.join(base, "package.json"))
    if pkg is not None:
        engine = None
        try:
            parsed = json.loads(pkg)
            node = (parsed.get("engines") or {}).get("node") if isinstance(parsed, dict) else None
            engine = node if isinstance(node, str) else None
        except (ValueError, AttributeError):
            pass
        bun_lock = next((f for f in ("bun.lockb", "bun.lock", "bunfig.toml") if isfile(f)), None)
        if bun_lock:
            add("bun", None, _rel(d, bun_lock))
        add("node", engine, _rel(d, "package.json"))
    go_version = read_bounded(os.path.join(base, ".go-version"))
    if go_version is not None:
        add("go", _first_line(go_version), _rel(d, ".go-version"))
    gomod = read_bounded(os.path.join(base, "go.mod"))
    if gomod is not None:
        m = re.search(r"^go\s+(\S+)", gomod, re.M)
        add("go", m.group(1) if m else None, _rel(d, "go.mod"))
    java_version = read_bounded(os.path.join(base, ".java-version"))
    if java_version is not None:
        add("java", _first_line(java_version), _rel(d, ".java-version"))
    sdkman = read_bounded(os.path.join(base, ".sdkmanrc"))
    if sdkman is not None:
        m = re.search(r"^\s*java\s*=\s*(\S+)", sdkman, re.M)
        if m:
            add("java", m.group(1), _rel(d, ".sdkmanrc"))
    pom = read_bounded(os.path.join(base, "pom.xml"))
    if pom is not None:
        add("java", _pom_java_release(pom), _rel(d, "pom.xml"))
        if not isfile("mvnw"):
            add("maven", None, _rel(d, "pom.xml"))
    gradle_file = next((f for f in ("build.gradle.kts", "build.gradle", "settings.gradle.kts", "settings.gradle") if isfile(f)), None)
    if gradle_file:
        add("java", _gradle_java_version(read_bounded(os.path.join(base, gradle_file)) or ""), _rel(d, gradle_file))
        if not isfile("gradlew"):
            add("gradle", None, _rel(d, gradle_file))
    py = next((f for f in ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt") if isfile(f)), None)
    if py:
        add("python", None, _rel(d, py))
    return found


def detect_workspace(workspace: str) -> List[Detection]:
    """One suggestion per toolchain: the first directory decides, and a later one's pin fills in a missing one."""
    by_tool: Dict[str, Detection] = {}
    for d in scan_dirs(workspace):
        for det in detect_dir(workspace, d):
            prev = by_tool.get(det.tool)
            if prev is None or (prev.pin is None and det.pin is not None):
                by_tool[det.tool] = det
    return list(by_tool.values())


#: The files an agent-guidance pointer names, in the framework's #1347 order.
GUIDANCE_FILES = ("AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md", ".github/copilot-instructions.md", ".cursor/rules")


def find_repo_guidance(workspace: str) -> List[str]:
    """Workspace-relative guidance files in each immediate subdirectory (the root is the framework's)."""
    out: List[str] = []
    for d in scan_dirs(workspace)[1:]:
        for name in GUIDANCE_FILES:
            if os.path.exists(os.path.join(workspace, d, name)):
                out.append(f"{d}/{name}")
    return out
