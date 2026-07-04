"""Scrub secret environment variables from model-driven subprocess environments.

The scrub half of the "secrets broker" (feature #10).  The confined runner
legitimately holds secrets in its own ``os.environ`` — the model-provider key,
tokens the framework itself uses (e.g. ``web_fetch`` host-bound header
expansion reads ``os.environ``).  But a shell command or MCP server the *model*
drives inherits that environment too, so ``echo $GITHUB_TOKEN`` / ``env`` leaks
raw credentials to model-controlled code.

This module removes a declared set of secret variables from the environment
*handed to a subprocess*, without touching the runner's own ``os.environ`` — so
framework code keeps its secrets while model-driven subprocesses don't see them.

It is a policy primitive: callers (the ``cli`` subprocess boundary, MCP spawn,
interactive_shell) pass the operator-declared patterns.  Deliberately no
implicit default set is applied — scrubbing is opt-in per surface so it never
silently breaks a workflow that legitimately needs a token in a subprocess
(``gh``, ``git push``, cloud CLIs).  ``DEFAULT_SECRET_ENV_PATTERNS`` is offered
as a documented starting point an operator can copy into config.

Pairs with the egress proxy (feature #1): egress limits *where* a subprocess can
connect; this limits *what secrets* it can read to send.  The eventual
TLS-terminating broker (v2) moves credential handling out of the runner env
entirely (placeholders resolved proxy-side); until then this is the in-process
scrub.
"""

from __future__ import annotations

import fnmatch
from typing import Dict, Mapping, Sequence

# A documented convenience set (NOT applied unless an operator opts in by
# referencing these). Case-insensitive fnmatch globs over env-var names.
DEFAULT_SECRET_ENV_PATTERNS = (
    "*_API_KEY", "*_APIKEY", "*_TOKEN", "*_SECRET", "*_SECRET_KEY",
    "*_PASSWORD", "*_PASSWD", "*_ACCESS_KEY", "*_ACCESS_KEY_ID",
    "*_SECRET_ACCESS_KEY", "*_PRIVATE_KEY", "*_CREDENTIALS",
    "ANTHROPIC_AUTH_TOKEN", "GH_TOKEN", "AWS_SESSION_TOKEN",
)


def matches_secret(name: str, patterns: Sequence[str]) -> bool:
    """True if env-var ``name`` matches any pattern (case-insensitive fnmatch)."""
    upper = (name or "").upper()
    return any(fnmatch.fnmatchcase(upper, p.upper()) for p in patterns)


def scrub_env(
    env: Mapping[str, str], patterns: Sequence[str],
) -> Dict[str, str]:
    """Return a copy of ``env`` with keys matching any ``patterns`` removed.

    Empty/None ``patterns`` returns a plain copy (no-op).  Never mutates the
    input mapping.
    """
    if not patterns:
        return dict(env)
    return {k: v for k, v in env.items() if not matches_secret(k, patterns)}
