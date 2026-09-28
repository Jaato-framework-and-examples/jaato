"""The operator's policy, as the page stages it: ``.jaato/toolchain-offer.json``.

The web coder's backend holds the allow-list; it cannot write the workspace,
so the page stages the offer through the daemon.  This module is the only
reader.  The file is in the workspace: a confined session is denied writing
it (the ``jaato-web-coder-toolchains`` AppArmor fragment), an unconfined one
is not, so every field is checked and a bad one is dropped, never used.

Schema 2::

    {"schema": 2,
     "toolchains": [{"tool": "java", "label": "Java", "versions": ["21"]}],
     "servers": {"jdtls": {"version": "1.40.0", "java": "21", "max_heap": "1G"},
                 "typescript-language-server": {"version": "4.4.0", "typescript": "5.9.3"},
                 "gopls": {"version": "v0.20.0"}, "basedpyright": {"version": "1.31.0"}},
     "install": {"timeout_seconds": 900, "paranoid": false}}

A server is installed only when pinned here and its toolchain is allowed.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .catalog import JDTLS_MIRROR, LANGUAGE_SERVERS, OFFER_PATH, TOOLCHAINS, VERSION_RE

logger = logging.getLogger(__name__)

SCHEMA = 2
MAX_OFFER_BYTES = 64 * 1024
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .+()-]{0,47}$")
_HEAP_RE = re.compile(r"^[1-9][0-9]{0,5}[mMgG]$")


@dataclass
class Allowed:
    tool: str
    label: str
    versions: List[str]


@dataclass
class Offer:
    toolchains: Dict[str, Allowed] = field(default_factory=dict)
    #: server id -> its settings (``version`` always present).
    servers: Dict[str, Dict[str, str]] = field(default_factory=dict)
    timeout_seconds: int = 900
    paranoid: bool = False

    def server_for(self, tool: str) -> Optional[Tuple[str, Dict[str, str]]]:
        sid = TOOLCHAINS[tool].server
        if sid and sid in self.servers:
            return sid, self.servers[sid]
        return None

    def by_command(self) -> Dict[str, Allowed]:
        out: Dict[str, Allowed] = {}
        for a in self.toolchains.values():
            for c in TOOLCHAINS[a.tool].commands:
                out.setdefault(c, a)
        return out


def _version(v: Any) -> Optional[str]:
    return v if isinstance(v, str) and VERSION_RE.match(v) else None


def parse_offer(raw: Any) -> Optional[Offer]:
    """The offer, or ``None`` when ``raw`` is not a schema-2 offer."""
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA or not isinstance(raw.get("toolchains"), list):
        return None
    offer = Offer()
    for entry in raw["toolchains"][:32]:
        if not isinstance(entry, dict):
            continue
        tool, label = entry.get("tool"), entry.get("label")
        if tool not in TOOLCHAINS or not (isinstance(label, str) and _LABEL_RE.match(label)):
            continue
        versions = [v for v in (entry.get("versions") or [])[:16] if _version(v)]
        if versions:
            offer.toolchains[tool] = Allowed(tool, label, versions)
    servers = raw.get("servers")
    if isinstance(servers, dict):
        for sid, spec in servers.items():
            if sid not in LANGUAGE_SERVERS or not isinstance(spec, dict) or not _version(spec.get("version")):
                continue
            clean = {"version": spec["version"]}
            if sid == "typescript-language-server":
                if not _version(spec.get("typescript")):
                    continue
                clean["typescript"] = spec["typescript"]
            if sid == "gopls":
                go = spec.get("go", "latest")
                if not _version(go):
                    continue
                clean["go"] = go
            if sid == "jdtls":
                java = spec.get("java", "21")
                heap = spec.get("max_heap", "1G")
                mirror = spec.get("mirror") or JDTLS_MIRROR
                if not _version(java) or not (isinstance(heap, str) and _HEAP_RE.match(heap)):
                    continue
                if not (isinstance(mirror, str) and mirror.startswith("https://") and len(mirror) < 512 and not re.search(r"\s", mirror)):
                    continue
                clean.update({"java": java, "max_heap": heap, "mirror": mirror})
            offer.servers[sid] = clean
    install = raw.get("install")
    if isinstance(install, dict):
        t = install.get("timeout_seconds")
        if isinstance(t, int) and not isinstance(t, bool) and 30 <= t <= 7200:
            offer.timeout_seconds = t
        offer.paranoid = install.get("paranoid") is True
    return offer


class OfferReader:
    """The workspace's offer, re-read only when its size or modification time changed."""

    def __init__(self) -> None:
        self._workspace: Optional[str] = None
        self._key: Optional[Tuple[int, int]] = None
        self._offer: Optional[Offer] = None

    def set_workspace(self, workspace: Optional[str]) -> None:
        if workspace != self._workspace:
            self._workspace, self._key, self._offer = workspace, None, None

    def read(self) -> Optional[Offer]:
        if not self._workspace:
            return None
        path = os.path.join(self._workspace, OFFER_PATH)
        try:
            st = os.stat(path)
        except OSError:
            self._key, self._offer = None, None
            return None
        key = (st.st_mtime_ns, st.st_size)
        if key == self._key:
            return self._offer
        self._key, self._offer = key, None
        if st.st_size > MAX_OFFER_BYTES:
            logger.warning("web_coder_toolchains: %s is %d bytes (max %d); ignored", path, st.st_size, MAX_OFFER_BYTES)
            return None
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("web_coder_toolchains: cannot read %s: %s", path, e)
            return None
        self._offer = parse_offer(raw)
        if self._offer is None:
            logger.warning("web_coder_toolchains: %s is not a schema-%d offer; no hints, no binds", path, SCHEMA)
        return self._offer
