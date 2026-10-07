"""What this plugin knows how to install, and where it puts it (#1344).

Two closed tables:

- :data:`TOOLCHAINS`: what mise installs and what gets linked into
  ``<ws>/.home/.local/bin``.  The operator's allow-list, which reaches this
  plugin through the offer the page stages, may name only these.  Rust is
  absent on purpose: rustup keeps its homes outside mise's data directory and
  its proxies need ``RUSTUP_HOME`` at run time.
- :data:`LANGUAGE_SERVERS`: one server per toolchain, installed only when the
  operator pins its version.

``python`` is a pseudo-toolchain: a managed workspace already has Python, so
binding it installs only basedpyright.

Every path is relative to the workspace and lands under ``.home``, where the
confinement template grants exec (``.home/.local/bin/*`` and
``.home/.local/share/**/bin/*``, #1273/#1274).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Toolchain:
    id: str
    #: mise's name; ``None`` for the python pseudo-toolchain.
    mise: Optional[str]
    #: ``.tool-versions`` names that pin it.
    tool_versions_names: Tuple[str, ...]
    #: The language server bound with it, if any.
    server: Optional[str]
    #: Names whose "command not found" suggests this toolchain.
    commands: Tuple[str, ...]


TOOLCHAINS: Dict[str, Toolchain] = {t.id: t for t in (
    Toolchain("python", None, (), "basedpyright", ("basedpyright", "basedpyright-langserver", "pyright")),
    Toolchain("node", "node", ("node", "nodejs"), "typescript-language-server", ("node", "npm", "npx", "corepack", "tsc")),
    Toolchain("go", "go", ("go", "golang"), "gopls", ("go", "gofmt", "gopls")),
    Toolchain("bun", "bun", ("bun",), None, ("bun", "bunx")),
    Toolchain("java", "java", ("java",), "jdtls", ("java", "javac", "jar", "jshell", "javadoc", "jlink", "jpackage", "keytool")),
    # Launcher scripts that run on the bound Java; a repository with ./mvnw or ./gradlew needs neither.
    Toolchain("maven", "maven", ("maven",), None, ("mvn",)),
    Toolchain("gradle", "gradle", ("gradle",), None, ("gradle",)),
)}

#: server id -> the ``.lsp.json`` language (also its ``languageId``).
LANGUAGE_SERVERS: Dict[str, str] = {
    "basedpyright": "python",
    "typescript-language-server": "typescript",
    "gopls": "go",
    "jdtls": "java",
}

JDTLS_MIRROR = "https://download.eclipse.org/jdtls/milestones"

HOME_DIR = ".home"
LSP_DIR = ".home/.local/share/jaato-lsp"
LOCAL_BIN = ".home/.local/bin"
MAVENRC_PATH = ".home/.mavenrc"
GOENV_PATH = ".home/.config/go/env"
GOTMP_DIR = ".home/.cache/go-tmp"
MISE_DATA_DIR = ".home/.local/share/mise"
MISE_CONFIG_DIR = ".home/.config/mise"
MISE_CACHE_DIR = ".home/.cache/mise"
MISE_STATE_DIR = ".home/.local/state/mise"
MISE_CONFIG_PATH = MISE_CONFIG_DIR + "/config.toml"

OFFER_PATH = ".jaato/toolchain-offer.json"
MANIFEST_PATH = ".jaato/environment.json"
LSP_CONFIG_PATH = ".lsp.json"
LOCK_PATH = ".home/.cache/jaato-toolchains.lock"

#: A version safe to hand to mise / pip / npm / go as ONE argument.
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")


def match_allowed_version(pin: str, allowed: List[str]) -> Optional[str]:
    """The allowed entry ``pin`` maps to, or ``None``.

    Equal, or extending it at a component boundary (``22.11.0`` matches ``22``,
    ``220`` does not); the most specific entry wins.  A leading ``v``, ``>=``,
    ``^``, ``~`` or ``=`` on the pin is ignored.
    """
    p = re.sub(r"^(?:>=|\^|~|=|v)+", "", pin.strip()).strip()
    if not p:
        return None
    best: Optional[str] = None
    for a in allowed:
        if (p == a or p.startswith(a + ".")) and (best is None or len(a) > len(best)):
            best = a
    return best


_JAVA_VENDORS = re.compile(
    r"^(?:temurin|openjdk|adoptopenjdk|corretto|zulu|liberica|microsoft|semeru|oracle|"
    r"graalvm(?:-community)?|sapmachine|dragonwell|kona|jetbrains)-")


def normalize_java_version(v: str) -> str:
    """``temurin-21.0.2`` -> ``21.0.2``, ``21.0.2-tem`` -> ``21.0.2``, ``1.8`` -> ``8``."""
    s = re.sub(r"-[a-z]+$", "", _JAVA_VENDORS.sub("", v.strip()))
    m = re.match(r"^1\.(\d+)(.*)$", s)
    if m and int(m.group(1)) <= 8:
        s = m.group(1) + m.group(2)
    return s


def match_tool_version(tool: str, pin: str, allowed: List[str]) -> Optional[str]:
    """:func:`match_allowed_version`, with Java spellings normalised; returns the entry as written."""
    if tool != "java":
        return match_allowed_version(pin, allowed)
    by_normal: Dict[str, str] = {}
    for a in allowed:
        by_normal.setdefault(normalize_java_version(a), a)
    hit = match_allowed_version(normalize_java_version(pin), list(by_normal))
    return None if hit is None else by_normal[hit]
