"""Public helpers for external tooling that participates in the
template index lifecycle.

Some external tools (e.g. kb-enablement-2.0's
``.jaato/scripts/generate_kb_artifacts.py`` walker) emit
``TemplateIndexEntry`` JSON entries directly, bypassing the
framework's discoverer-side parser pass.  Without these helpers,
walkers either mirror the framework's parsing logic locally
(guaranteed to drift over time) or skip parser-derived fields
entirely (the framework's loader then defaults the field — silently
mis-classifying templates).

This module is the **single source of truth** for the parser-derived
classifications walkers need.  The server's template plugin
(``shared/plugins/template/plugin.py``) imports from here too, so
walker output matches the framework's discovery output byte-for-byte.

Server 0.6.58+.
"""

from __future__ import annotations

import re
from typing import FrozenSet

# ---------------------------------------------------------------------
# Mustache / Handlebars syntax markers
# ---------------------------------------------------------------------
# Same patterns the server's plugin uses for syntax detection.  Kept
# here so this module is self-contained — the SDK is not allowed to
# reach into ``shared/`` (server-only) packages.

_MUSTACHE_SECTION_PATTERN = re.compile(r'\{\{\s*#\s*\w+')
_MUSTACHE_END_SECTION_PATTERN = re.compile(r'\{\{\s*/\s*\w+')
_MUSTACHE_INVERTED_PATTERN = re.compile(r'\{\{\s*\^\s*\w+')
_MUSTACHE_CURRENT_ITEM_PATTERN = re.compile(r'\{\{\s*\.\s*\}\}')

# ---------------------------------------------------------------------
# Handlebars helper keywords
# ---------------------------------------------------------------------
# Recognised by ``classify_template_evaluation_kind``.  When this set
# expands (a new helper class lands), every walker that imports the
# helper picks up the change automatically on SDK upgrade — no walker
# code changes needed.

HELPER_KEYWORDS: FrozenSet[str] = frozenset({
    'if', 'unless', 'each', 'with', 'lookup',
})

# Pattern for the public classifier — matches ``{{#KW ARG}}`` and
# ``{{^KW ARG}}`` shapes where KW is one of HELPER_KEYWORDS and ARG
# is a non-empty token.  Mirrors the inner detection logic of the
# server's ``_parse_mustache_structure`` parser pass so the public
# helper and the framework's internal classifier agree byte-for-byte
# on the same content.
_HELPER_INVOCATION_RE = re.compile(
    r'\{\{\s*[#^]\s*(' + '|'.join(sorted(HELPER_KEYWORDS)) + r')\s+\S'
)


def classify_template_evaluation_kind(content: str) -> str:
    """Classify a template's evaluation kind from its raw content.

    Public helper for external walkers that emit
    ``TemplateIndexEntry`` JSON directly and therefore bypass the
    server's discoverer-side parser pass.

    Returns:
        ``"helpers"`` when the template contains any
        ``{{#KW ARG}}`` / ``{{^KW ARG}}`` invocation where KW is in
        :data:`HELPER_KEYWORDS` (``if``, ``unless``, ``each``,
        ``with``, ``lookup``) and ARG is a non-empty token; else
        ``"substitution"``.

    The server's template plugin imports this same function for its
    own internal classification — there is no second copy of the
    logic to drift from.

    Server 0.6.58+.
    """
    if not content:
        return "substitution"
    # Cheap mustache-syntax check first — non-mustache (jinja2, plain
    # text) templates can't carry Handlebars helpers by definition.
    has_mustache = (
        _MUSTACHE_SECTION_PATTERN.search(content)
        or _MUSTACHE_END_SECTION_PATTERN.search(content)
        or _MUSTACHE_INVERTED_PATTERN.search(content)
        or _MUSTACHE_CURRENT_ITEM_PATTERN.search(content)
    )
    if not has_mustache:
        return "substitution"
    if _HELPER_INVOCATION_RE.search(content):
        return "helpers"
    return "substitution"
