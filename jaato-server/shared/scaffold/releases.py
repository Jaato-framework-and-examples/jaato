"""``jaato-scaffold explain releases`` — what our two indexes carry.

The framework publishes to two channels and, until this, asked neither: a
production release on PyPI and a release candidate on TestPyPI could both be
sitting there with nothing in the tree that would ever mention it.  This is
the long-form rendering; :func:`jaato_sdk.doctor.check_package_releases` is
the one-line preflight form of the same report.

Both read :mod:`jaato_sdk.release_channels`, which does the asking and the
PEP 440 ordering.  This module only draws — the split the framework keeps
everywhere between producing a result and displaying one, and the reason a
version cannot read as "newest" on one surface and "behind" on the other.

It is the one ``explain`` topic that touches the NETWORK, which is why it is
its own topic rather than a section of ``explain dependencies``: that verb is
an offline introspection of the installed tree, and quietly giving it an
egress would change what running it means.  Asking is opt-out
(``JAATO_RELEASE_CHECK=off``), bounded by a short per-index deadline, and
cached, so the cost lands a few times a day rather than once per invocation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from jaato_sdk import release_channels as _rc


#: Marks in the same vocabulary ``explain integrations`` uses, so a reader
#: moving between the two topics does not relearn the glyphs.
_MARKS = {"update": "!", "current": "✔", "ahead": "^", "unknown": "?"}

_VERDICT_WORDS = {
    "update": "a newer build is published here",
    "current": "this is the newest build here",
    "ahead": "installed build is NEWER than anything published here",
    "unknown": "no verdict",
}


def _channel_line(status) -> str:
    """One channel's row: what it carries and how that compares."""
    latest = status.latest or "—"
    cached = " (cached)" if status.from_cache else ""
    return (f"    {_MARKS.get(status.verdict, '?')} {status.channel.label:20} "
            f"{latest:14} {_VERDICT_WORDS[status.verdict]}{cached}")


def releases(*, timeout: Optional[float] = None,
             refresh: bool = False) -> Tuple[Dict[str, Any], str]:
    """Render every channel's answer about every installed jaato package.

    Args:
        timeout: Per-index deadline in seconds; ``None`` uses the module
            default.
        refresh: Ignore cached answers and re-ask the indexes.

    Returns:
        ``(data, text)`` — the :meth:`ReleaseReport.to_dict` view and the
        human rendering, the shape every ``explain`` renderer returns.
    """
    report = _rc.check_releases(
        timeout=_rc.DEFAULT_TIMEOUT if timeout is None else timeout,
        refresh=refresh)
    data = report.to_dict()

    if not report.enabled:
        return data, ("release check — what our indexes carry\n\n"
                      f"  disabled by {_rc.ENV_SWITCH}. Unset it (or set it to "
                      "anything but off/0/no/false/none/never) to ask again.")

    lines: List[str] = [
        "releases — what our two distribution channels carry", "",
        "  PyPI      production release  — what `pip install -U <pkg>` gives you",
        "  TestPyPI  release candidate   — a staging build of a release not yet shipped",
        "",
    ]

    if not report.distributions:
        lines.append("  " + ("; ".join(report.errors)
                             or "no jaato distributions are installed here"))
        return data, "\n".join(lines)

    for dist in report.distributions:
        lines.append(f"  {dist.name}  {dist.installed}  (installed)")
        for status in dist.channels:
            lines.append(_channel_line(status))
            if status.error:
                lines.append(f"      {status.error}")
            if status.unparseable:
                # Never silently dropped: a release spelled in a form this
                # build cannot order is the one most worth naming.
                lines.append("      not ordered (unrecognised version): "
                             + ", ".join(status.unparseable))
        lines.append("")

    updates = report.updates
    if updates:
        lines.append("  to upgrade:")
        for channel in _rc.CHANNELS:
            names = [d.name for d, s in updates if s.channel is channel]
            if names:
                lines.append(f"    {channel.label}:")
                lines.append("      "
                             + channel.install_command(" ".join(names)))
    else:
        lines.append("  nothing newer is published on either channel.")

    lines += ["",
              f"  asked at most every {int(_rc.DEFAULT_MAX_AGE // 3600)}h and "
              f"cached at {_rc.default_cache_path()}",
              f"  switch it off with {_rc.ENV_SWITCH}=off; jaato-doctor reports "
              "the same thing as one preflight line."]
    return data, "\n".join(lines)
