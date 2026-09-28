"""The gitignore patterns a bound toolchain's build output needs.

The patterns are NOT hand-written here; they are read from vendored copies
of the canonical ``github/gitignore`` templates under
``gitignore_templates/`` (CC0-1.0, pinned to a commit in ``PROVENANCE.json``
with a per-file sha256 the checksum test enforces).  So "what does Maven
leave behind" is answered by the maintained upstream list, and refreshing it
is bumping the pin — not a judgement call in this file.

One tool maps to one template (``bun`` reuses ``Node``).  The only patterns
this plugin adds on its own are for artifacts jaato's OWN tooling writes that
the project template would not carry: jdtls (the Java LSP jaato installs)
drops Eclipse ``.settings/`` and ``.factorypath`` into a project that has no
Eclipse setup.  Those are the plugin's to declare, the way it declares
``.home/.mavenrc``; everything else comes from upstream.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "gitignore_templates")

#: toolchain id -> vendored template basename (without ``.gitignore``).
TEMPLATE_BY_TOOL: Dict[str, str] = {
    "python": "Python",
    "node": "Node",
    "bun": "Node",
    "go": "Go",
    "java": "Java",
    "maven": "Maven",
    "gradle": "Gradle",
}

#: Artifacts jaato's own bound tooling writes that the upstream template omits.
#: jdtls writes these into a project without an Eclipse setup; the Maven and
#: Gradle templates already carry ``.project`` / ``.classpath``.
_JAATO_TOOL_EXTRAS: Dict[str, Tuple[str, ...]] = {
    "java": (".settings/", ".factorypath"),
}

_cache: Dict[str, List[str]] = {}


def _read_template(name: str) -> List[str]:
    path = os.path.join(_TEMPLATES_DIR, name + ".gitignore")
    lines: List[str] = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue  # skip comments (incl. the vendor provenance header) and blanks
            lines.append(stripped)
    return lines


def patterns_for_tool(tool: str) -> List[str]:
    """The gitignore lines for one toolchain, in upstream order, deduped.

    An unknown tool contributes nothing.  A negation line (``!x``) is kept in
    place — git reads the block top to bottom.
    """
    if tool in _cache:
        return _cache[tool]
    template = TEMPLATE_BY_TOOL.get(tool)
    lines: List[str] = list(_read_template(template)) if template else []
    for extra in _JAATO_TOOL_EXTRAS.get(tool, ()):  # jaato's own tooling
        if extra not in lines:
            lines.append(extra)
    # Dedup preserving order (a template has no dups, but be defensive).
    seen: set = set()
    result = [p for p in lines if not (p in seen or seen.add(p))]
    _cache[tool] = result
    return result
