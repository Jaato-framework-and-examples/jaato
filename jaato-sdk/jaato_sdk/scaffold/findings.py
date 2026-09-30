"""What ``jaato-scaffold validate`` is asked about, and how it prints findings.

Two things both sides of the ``validate`` split need, kept in one place so
they cannot disagree (#1267, tier 3):

* **Target resolution.**  ``validate`` takes a workspace directory or a
  profile file.  A file inside ``<ws>/.jaato/profiles[/<set>]/`` means "that
  profile of that workspace"; a file anywhere else is a standalone profile.
  jaato-server's local ``validate`` and the SDK shell's daemon route both
  read the target this way.
* **The text line.**  One finding prints the same whichever install produced
  it, so a reader cannot tell a local run from a daemon's by its format, only
  by the attribution line the daemon route adds.

Pure stdlib: the SDK shell imports this with no jaato-server present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


def is_canonical_profile_layout(p: Path) -> bool:
    """True if ``p`` lives under a real ``<ws>/.jaato/profiles[/<set>]/`` tree.

    Only such files can be resolved as part of their workspace (inherits +
    set overlay).  A file outside this layout (a docs example, an ad-hoc
    path) must be validated directly, or it silently resolves to a bogus
    workspace where profile discovery finds nothing and reports a false
    "valid".
    """
    par = p.parent
    if par.name == "profiles" and par.parent.name == ".jaato":
        return True  # <ws>/.jaato/profiles/<name>.yaml
    # <ws>/.jaato/profiles/<set>/<name>.yaml
    return par.parent.name == "profiles" and par.parent.parent.name == ".jaato"


def resolve_target(target: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Map a workspace dir OR a profile file to (workspace, set, profile_name).

    A profile file at ``<ws>/.jaato/profiles/<set>/<name>.yaml`` yields the
    set + profile name; a tier-1 file at ``.../profiles/<name>.yaml`` yields
    no set; a directory is taken as the workspace itself.
    """
    p = Path(target).resolve()
    if p.is_dir():
        return str(p), None, None
    name = p.stem
    parent = p.parent
    if parent.name == "profiles":
        return str(parent.parent.parent), None, name
    return str(parent.parent.parent.parent), parent.name, name


def format_finding(d: Mapping[str, Any]) -> str:
    """One finding (``Diagnostic.as_dict()`` shape) as the line ``validate`` prints.

    A contributed finding (#1306) ends ``(from <distribution>:<name>)``, so a
    reader knows which package to read or uninstall; the framework's own
    findings carry no ``source`` and print without it.
    """
    loc = f" @ {d.get('where')}" if d.get("where") else ""
    who = f"{d.get('profile')}: " if d.get("profile") else ""
    tier = f"[{d.get('tier')}] " if d.get("tier") else ""
    src = f"  (from {d.get('source')})" if d.get("source") else ""
    return (f"[{d.get('severity')}] {tier}{who}{d.get('code')}: "
            f"{d.get('message')}{loc}{src}")


def clean_line(scope: str, profile_set: Optional[str]) -> str:
    """The line a run with no findings prints."""
    sset = f" (set {profile_set})" if profile_set else ""
    return f"✓ {scope}{sset} valid — no findings"


def scope_label(only: Optional[str]) -> str:
    """How a workspace run names what it validated."""
    return f"profile '{only}'" if only else "all profiles"
