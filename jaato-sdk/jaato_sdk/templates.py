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

import hashlib
import re
from typing import FrozenSet, Tuple

# ---------------------------------------------------------------------
# Template ids (#1611)
# ---------------------------------------------------------------------
# ``listAvailableTemplates`` reports each template by an opaque id, and
# ``renderTemplateToFile`` / ``listTemplateVariables`` accept only that
# id.  An application that lists a workspace's templates itself (and
# lets a caller render one through jaato's tools) must hand out the same
# id, without opening a session.  This is the derivation, as a contract:
# the server's template plugin is guarded against it, and
# ``jaato-scaffold explain plugin template`` renders the lines below.

TEMPLATE_ID_PREFIX = "tpl"
"""Prefix of every template id: ``tpl_<8 lowercase hex>``."""

TEMPLATE_ID_DERIVATION: Tuple[str, ...] = (
    "template_id = \"tpl_\" + the first 8 hex digits (lowercase) of "
    "SHA-256 over the template's name, UTF-8 encoded",
    "the name is the template's index name: the `name` of its entry in "
    "the catalog's index.json, or the file's basename for a template "
    "discovered at runtime",
    "a pure function of the name: no session, workspace or configuration "
    "enters it, so it is stable across sessions and deployments",
    "outside a session: `from jaato_sdk.templates import template_id`",
)
"""The derivation, one rule per line, as ``explain plugin template`` shows it."""


def template_id(name: str) -> str:
    """Return the id jaato's template tools use for the template ``name``.

    ``"tpl_" + sha256(name.encode("utf-8")).hexdigest()[:8]``.  ``name``
    is the template's index name (the catalog ``index.json`` entry's
    ``name``, or a runtime-discovered file's basename), never a path.

    This is a stable contract (#1611): the server's template plugin
    produces exactly this value for ``listAvailableTemplates`` and
    resolves exactly this value in ``renderTemplateToFile`` and
    ``listTemplateVariables``, and a guard test holds the two together.

    Args:
        name: The template's index name.

    Returns:
        The template id, e.g. ``"tpl_9e8a395d"``.
    """
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"{TEMPLATE_ID_PREFIX}_{digest}"

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
