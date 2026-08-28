"""What code a run actually exercised, read from the live process.

NOT derivable from the checkout, which is why this is recorded rather
than assumed.  An editable install resolves ``jaato_sdk`` through a
MetaPathFinder to wherever it was installed FROM — for a git worktree
that is the original checkout, so a branch can ship its own
``jaato-sdk/`` directory and never run a line of it.  The recorded
version is no better: an editable install stamps the version at install
time, so a ``.pth`` reading 0.15.0 can be serving code whose pyproject
now says 0.16.0.

The only honest answer is the resolved path of the module that was
imported, so that is what goes into the results file beside every arm.
A sweep's numbers are evidence about the code that ran, and nothing in
the repository state establishes which code that was.
"""
from __future__ import annotations

from typing import Any, Dict


def provenance() -> Dict[str, Any]:
    """Resolved path and recorded version of the SDK this process imported."""
    out: Dict[str, Any] = {}
    try:
        import jaato_sdk
        out["jaato_sdk_path"] = getattr(jaato_sdk, "__file__", None)
    except ImportError as exc:
        out["jaato_sdk_path"] = f"unimportable: {exc}"
    try:
        from importlib.metadata import version
        out["jaato_sdk_version"] = version("jaato-sdk")
    except Exception:  # noqa: BLE001 — absent metadata is not a run failure
        out["jaato_sdk_version"] = None
    return out
