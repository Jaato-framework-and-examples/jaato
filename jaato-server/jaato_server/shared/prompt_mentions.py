"""Remove the ``@`` from the prompt mentions an enricher resolved (#1429).

Until #1429 the session ended every prompt with a blanket
``re.sub(r'@([\\w./\\-]+...)', r'\\1', ...)``: meant for ``@photo.png`` and
``@ref-id`` mentions the multimodal and references plugins had handled, it
removed the ``@`` from every word — ``@jaato/sdk`` became ``jaato/sdk``,
``dani@example.com`` became ``daniexample.com``, ``@dataclass`` became
``dataclass`` — before the model ever saw the prompt.

The rule now: a mention loses its ``@`` only when a prompt enricher reports,
under :data:`jaato_sdk.plugins.base.RESOLVED_MENTIONS_METADATA_KEY`, that it
resolved it.  Nothing here guesses which ``@word`` is a reference.

Stdlib only.
"""

import re
from typing import Any, Dict, Iterable, List

from jaato_sdk.plugins.base import RESOLVED_MENTIONS_METADATA_KEY


def resolved_mentions(metadata: Dict[str, Any]) -> List[str]:
    """Collect the mention tokens every enricher reported resolving.

    Args:
        metadata: The registry's combined enrichment metadata, keyed by
            plugin name (``PromptEnrichmentResult.metadata`` of
            ``PluginRegistry.enrich_prompt``).

    Returns:
        Distinct tokens (without ``@``), longest first so a token that is a
        prefix of another is never applied before the longer one.  A
        malformed report (not a list, a non-string, an empty string) is
        ignored rather than guessed at.
    """
    tokens = set()
    for meta in (metadata or {}).values():
        if not isinstance(meta, dict):
            continue
        reported = meta.get(RESOLVED_MENTIONS_METADATA_KEY)
        if not isinstance(reported, (list, tuple, set)):
            continue
        for token in reported:
            if isinstance(token, str) and token and not token.startswith('@'):
                tokens.add(token)
    return sorted(tokens, key=lambda t: (-len(t), t))


def strip_resolved_mentions(text: str, tokens: Iterable[str]) -> str:
    """Remove the ``@`` in front of each resolved mention, and nowhere else.

    A mention is ``@`` + the token, where the ``@`` does not follow a word
    character (so the ``@`` of ``dani@example.com`` is never a mention) and
    the token is not continued by further mention characters (so resolving
    ``ref`` leaves ``@ref-id`` and ``@ref/x`` alone).  A trailing ``.`` that
    ends a sentence does not continue a mention: ``@photo.png.`` is the
    mention ``photo.png``.

    Args:
        text: The enriched prompt.
        tokens: Mention tokens without their ``@``.

    Returns:
        ``text`` with exactly those mentions' ``@`` removed.
    """
    for token in tokens:
        pattern = re.compile(
            r'(?<![\w@])@(' + re.escape(token) + r')(?![\w\-]|[./][\w\-])'
        )
        text = pattern.sub(r'\1', text)
    return text
