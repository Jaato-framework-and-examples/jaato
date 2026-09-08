"""Scrub secret environment variables from model-driven subprocess environments.

The scrub half of the "secrets broker" (feature #10).  The confined runner
legitimately holds secrets in its own ``os.environ`` — the model-provider key,
tokens the framework itself uses (e.g. ``web_fetch`` host-bound header
expansion reads ``os.environ``).  But a shell command, PTY session or MCP
server the *model* drives inherits that environment too, so
``echo $GITHUB_TOKEN`` / ``env`` leaks raw credentials to model-controlled
code.

This module removes a declared set of secret variables from the environment
*handed to a subprocess*, without touching the runner's own ``os.environ`` — so
framework code keeps its secrets while model-driven subprocesses don't see them.

**Scrubbing is ON by default (#863).**  Until #863 this module was a policy
primitive only: it removed whatever a profile declared, and a profile that
declared nothing passed the daemon's full environment — provider keys
included — to every command the model ran.  The reason was real (developer
CLIs such as ``gh`` and cloud SDKs need their tokens), but the failure was
silent and the default was the unsafe one: a corporate harness author had
no signal that they had chosen the leaky posture, and #712 notes that
AppArmor gives this scrub no kernel-level backstop.  So the three surfaces
that spawn model-driven subprocesses (:data:`SCRUB_SURFACES`) now apply
:data:`DEFAULT_SECRET_ENV_PATTERNS` when nothing is declared, and opting
*out* is the explicit, WARNING-announced act — the same posture as
``--ws-unsafe-no-auth``.

**The value grammar** — one shape, accepted by the profile-level
``scrub_secret_env`` key and by ``plugin_configs.<surface>.scrub_secret_env``
alike (see :func:`normalize_scrub_patterns`):

============================  =================================================
``default`` (or absent)        the framework set, :data:`DEFAULT_SECRET_ENV_PATTERNS`
``none`` (also ``[]``)         scrub nothing — announced at WARNING
``"*_TOKEN"``                  one glob (a lone string is one pattern, never
                               split into characters)
``[glob, ...]``                an explicit list; the entry ``default`` expands
                               to the framework set in place
``"!NAME"`` (in a list)        an EXEMPTION: a variable matching it is never
                               scrubbed, whatever else matches.  This is the
                               developer-desktop answer — ``["default",
                               "!GH_TOKEN"]`` keeps ``gh`` working while the
                               provider key stays out of the shell.
============================  =================================================

Precedence, most specific first: ``plugin_configs.<surface>.scrub_secret_env``
→ the profile's ``scrub_secret_env`` → the framework default.  A malformed
value fails **closed** (the default set is applied and the defect logged),
because the only outcome worse than a broken workflow is a silently-leaked
credential.

Pairs with the egress proxy (feature #1): egress limits *where* a subprocess can
connect; this limits *what secrets* it can read to send.  The eventual
TLS-terminating broker (#505) moves credential handling out of the runner env
entirely (placeholders resolved proxy-side); until then this is the in-process
scrub.
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

logger = logging.getLogger(__name__)

# The framework's own set — applied when nothing is declared.  Case-insensitive
# fnmatch globs over env-var names.
DEFAULT_SECRET_ENV_PATTERNS = (
    "*_API_KEY", "*_APIKEY", "*_TOKEN", "*_SECRET", "*_SECRET_KEY",
    "*_PASSWORD", "*_PASSWD", "*_ACCESS_KEY", "*_ACCESS_KEY_ID",
    "*_SECRET_ACCESS_KEY", "*_PRIVATE_KEY", "*_CREDENTIALS",
    "ANTHROPIC_AUTH_TOKEN", "GH_TOKEN", "AWS_SESSION_TOKEN",
)

#: The shorthand that names :data:`DEFAULT_SECRET_ENV_PATTERNS` — as the whole
#: value (``scrub_secret_env: default``) or as one entry of a list.
SCRUB_DEFAULT = "default"

#: The explicit opt-out.  Resolves to no patterns and is announced at WARNING
#: by :func:`resolve_scrub_patterns`.
SCRUB_NONE = "none"

#: Prefix marking an exemption entry: ``"!GH_TOKEN"`` means a variable
#: matching ``GH_TOKEN`` survives the scrub even though ``*_TOKEN`` matches it.
EXEMPT_PREFIX = "!"

#: The plugins that spawn model-driven subprocesses, i.e. the surfaces the
#: scrub applies to and that ``jaato-scaffold validate`` / ``jaato-doctor``
#: audit.  A profile-level ``scrub_secret_env`` is folded into each of these
#: that the profile enables (see ``inject_scrub_secret_env`` in
#: ``shared/plugins/subagent/config.py``).
SCRUB_SURFACES = ("cli", "interactive_shell", "mcp")

#: The one-line fix the diagnostics point at.
SCRUB_HINT = (
    "set `scrub_secret_env: default` (the framework set), or keep the default "
    "and exempt only the names a tool needs, e.g. "
    "`scrub_secret_env: [default, '!GH_TOKEN']`"
)


def _split_patterns(patterns: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split a pattern list into ``(scrub_globs, exempt_globs)``.

    Exemptions are the entries carrying :data:`EXEMPT_PREFIX`; the prefix is
    stripped from the returned globs.  Blank entries are dropped.
    """
    scrub: List[str] = []
    exempt: List[str] = []
    for raw in patterns or ():
        p = str(raw).strip()
        if not p:
            continue
        if p.startswith(EXEMPT_PREFIX):
            body = p[len(EXEMPT_PREFIX):].strip()
            if body:
                exempt.append(body)
        else:
            scrub.append(p)
    return scrub, exempt


def matches_secret(name: str, patterns: Sequence[str]) -> bool:
    """True if env-var ``name`` is to be scrubbed under ``patterns``.

    Case-insensitive fnmatch over the variable NAME.  An exemption entry
    (``"!GLOB"``) wins over every scrub glob: a name matching any exemption
    is never scrubbed, whatever else matches it.
    """
    upper = (name or "").upper()
    scrub, exempt = _split_patterns(patterns)
    if any(fnmatch.fnmatchcase(upper, e.upper()) for e in exempt):
        return False
    return any(fnmatch.fnmatchcase(upper, p.upper()) for p in scrub)


def scrub_env(
    env: Mapping[str, str], patterns: Sequence[str],
) -> Dict[str, str]:
    """Return a copy of ``env`` with keys matching any ``patterns`` removed.

    Empty/None ``patterns`` returns a plain copy (no-op), as does a list of
    exemptions with no scrub glob.  Never mutates the input mapping.
    """
    if not patterns:
        return dict(env)
    return {k: v for k, v in env.items() if not matches_secret(k, patterns)}


def is_scrub_disabled(patterns: Sequence[str]) -> bool:
    """True when ``patterns`` scrubs nothing (no positive glob at all).

    The diagnostic predicate: ``()`` and ``["!GH_TOKEN"]`` are both disabled —
    an exemption with nothing to be exempt from removes no variable.
    """
    scrub, _ = _split_patterns(patterns)
    return not scrub


def _expand_entries(entries: Iterable[Any]) -> Tuple[str, ...]:
    """Expand a list value: ``default`` → the framework set; others verbatim."""
    out: List[str] = []
    for entry in entries:
        if isinstance(entry, bool) or not isinstance(entry, (str, int, float)):
            raise ValueError(
                f"scrub_secret_env entries must be strings (globs, "
                f"'default', or '!EXEMPT'); got {entry!r}"
            )
        text = str(entry).strip()
        if text.lower() == SCRUB_DEFAULT:
            out.extend(DEFAULT_SECRET_ENV_PATTERNS)
        elif text:
            out.append(text)
    return tuple(out)


def normalize_scrub_patterns(value: Any) -> Tuple[str, ...]:
    """Turn a ``scrub_secret_env`` value into the pattern tuple it means.

    The one grammar every ingress shares — profile key, plugin knob, snapshot
    — so ``default`` / ``none`` / a lone glob / a list mean the same thing
    everywhere (the module docstring has the table).  ``None`` (the key is
    absent) is the framework default, which is what makes the scrub ON by
    default: a caller that never mentions the knob gets
    :data:`DEFAULT_SECRET_ENV_PATTERNS`.

    Returns:
        The patterns to hand to :func:`scrub_env`; ``()`` for the opt-out.

    Raises:
        ValueError: For a value of an unsupported shape (a dict, a number, a
            list holding a non-string).  Callers that must not fail — the
            plugins — go through :func:`resolve_scrub_patterns`, which turns
            this into a fail-closed default; the validator surfaces it as an
            error instead.
    """
    if value is None or value is True:
        return tuple(DEFAULT_SECRET_ENV_PATTERNS)
    if value is False:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == SCRUB_NONE:
            return ()
        if text.lower() == SCRUB_DEFAULT:
            return tuple(DEFAULT_SECRET_ENV_PATTERNS)
        # A lone string is ONE pattern, never split into characters (which
        # would silently disable scrubbing — fail OPEN).
        return (text,)
    if isinstance(value, (list, tuple)):
        return _expand_entries(value)
    raise ValueError(
        f"scrub_secret_env must be 'default', 'none', a glob, or a list of "
        f"globs; got {type(value).__name__}: {value!r}"
    )


def resolve_scrub_patterns(value: Any, *, surface: str) -> Tuple[str, ...]:
    """The plugin-side resolver: normalize, fail closed, announce an opt-out.

    Args:
        value: The raw ``scrub_secret_env`` knob (``None`` when the plugin
            config never mentioned it).
        surface: The plugin applying the result (``cli`` / ``mcp`` /
            ``interactive_shell``), named in the log lines.

    Returns:
        The patterns to scrub with.  A malformed value yields the framework
        default and an ERROR log — fail closed, because the model-driven
        subprocess is what the defect would otherwise be handed to.  A value
        that scrubs nothing yields ``()`` and a WARNING naming the surface,
        so the leaky posture is a visible choice rather than a silent one.
    """
    try:
        patterns = normalize_scrub_patterns(value)
    except ValueError as exc:
        logger.error(
            "%s: invalid scrub_secret_env (%s) — applying the framework "
            "default set instead (fail closed)", surface, exc,
        )
        return tuple(DEFAULT_SECRET_ENV_PATTERNS)
    if is_scrub_disabled(patterns):
        logger.warning(
            "%s: secret env scrubbing is DISABLED (scrub_secret_env: none) — "
            "every model-driven %s subprocess inherits the runner's full "
            "environment, provider keys and tokens included.  This is the "
            "developer-desktop opt-out; %s.",
            surface, surface, SCRUB_HINT,
        )
    return patterns
