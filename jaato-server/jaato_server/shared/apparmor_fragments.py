"""Where AppArmor ``.rules`` fragments are found, and what a rule in one denies.

Stdlib only, so both the daemon's renderer (``server/apparmor.py``) and
``jaato-scaffold validate`` read the same search path: two copies of the
tier list would be free to disagree about which file a name means.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

#: Basename suffix of a fragment file.
FRAGMENT_SUFFIX = ".rules"


def search_tiers(workspace_path: str) -> List[Tuple[str, Path]]:
    """The fragment directories, lowest precedence first.

    ``user`` (``~/.jaato/apparmor-fragments``), ``workspace``
    (``<ws>/.jaato/apparmor-fragments``) and ``cache``
    (``<ws>/.jaato/.cache/apparmor-fragments``, walker-generated).  A later
    tier wins a basename collision.
    """
    ws = Path(workspace_path)
    return [
        ("user", Path("~/.jaato/apparmor-fragments").expanduser().resolve()),
        ("workspace", (ws / ".jaato" / "apparmor-fragments").resolve()),
        ("cache", (ws / ".jaato" / ".cache" / "apparmor-fragments").resolve()),
    ]


def discover_fragments(
    workspace_path: str,
) -> Tuple[Dict[str, Tuple[str, Path]], Dict[str, List[str]]]:
    """Every fragment on the search path, keyed by basename (no suffix).

    Returns ``(discovered, found_in)``: name to the ``(tier, path)`` that
    wins, and name to every tier it was found in (for the ``shadows``
    record, #1326).  A tier that is not a directory is skipped.
    """
    discovered: Dict[str, Tuple[str, Path]] = {}
    found_in: Dict[str, List[str]] = {}
    for tier_label, fragments_dir in search_tiers(workspace_path):
        if not fragments_dir.is_dir():
            continue
        for path in sorted(fragments_dir.glob(f"*{FRAGMENT_SUFFIX}")):
            discovered[path.stem] = (tier_label, path)
            found_in.setdefault(path.stem, []).append(tier_label)
    return discovered, found_in


# ``deny network``, with or without a qualifier (``audit``) or a family, and
# never inside a comment.  ``deny network unix`` denies no outbound
# connection and is not counted.
_DENY_NETWORK_RE = re.compile(
    r"^\s*(?:audit\s+)?deny\s+network(?:\s+(?P<rest>[^,#]*))?\s*,")


def denies_network(body: str) -> List[str]:
    """The rule lines of *body* that deny IP networking.

    A runner's provider connection is an ``inet`` / ``inet6`` stream, so
    these are the lines that cut a model loop off when composed into the
    runner's body (#1675).  ``deny network unix,`` and the like are not.
    """
    hits: List[str] = []
    for line in body.splitlines():
        m = _DENY_NETWORK_RE.match(line.split("#", 1)[0])
        if not m:
            continue
        rest = (m.group("rest") or "").split()
        if not rest or rest[0] in ("inet", "inet6"):
            hits.append(line.strip())
    return hits
