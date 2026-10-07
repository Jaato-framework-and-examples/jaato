"""Facts the authoring commands (`new`) share with the `explain` pages.

The generator and the documentation must not disagree, so both render these
from one definition.  They used to live in :mod:`explain`, which made
:mod:`build` import the whole introspection surface (``explain`` ->
``introspect`` -> the provider tree) to read four constants and a ``.env``
line.  #1267 wants the authoring commands to be able to ship without
introspection, so the shared facts sit in this leaf instead.  ``explain``
re-exports every name, so ``explain.PROFILE_ENV_FACTS`` and friends still
resolve.

Stdlib only.  Keep it that way: this module is on the authoring import path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

#: Worked example wherever the profile ``env:`` block is documented -- in
#: ``explain env`` / ``explain profile`` and in the commented block ``new`` emits (``build._set_profile_yaml``).  It is a
#: path knob on purpose: the resolution fact below is the half that bites, and
#: this is the variable people reach for when a session misbehaves.
ENV_EXAMPLE_VAR = "JAATO_PROVIDER_TRACE"

#: The example's VALUE, shared for the same reason as its name.  Deliberately
#: RELATIVE: ``jaato_sdk.trace._resolve_trace_file`` joins a relative trace
#: path onto ``JAATO_WORKSPACE_ROOT``, which the runner seeds per session, so
#: this form gives every session its own trace in its own workspace.  The
#: absolute form is fixed at the PROFILE and every session sharing that
#: profile appends to one interleaved file -- the failure mode this example
#: exists to steer people away from, and the one an earlier draft of this note
#: recommended (jaato #752 review).
ENV_EXAMPLE_VALUE = "provider_trace.log"

#: The four load-bearing, non-obvious properties of the profile ``env:``
#: block, rendered by BOTH halves of its documentation from this ONE
#: definition.
#:
#: Sharing the strings is the anti-drift mechanism.  #716's
#: ``test_a_real_run_writes_only_documented_files`` asserts which FILES ``new``
#: writes, never their content, so a fact stated in the generated comment and
#: not in ``explain env`` (or reworded in one of them) would drift with nothing
#: failing -- "documentation about a generator rots", one level down.  Kept
#: short enough to render as a comment line inside a generated YAML file.
PROFILE_ENV_FACTS = (
    "outranks the workspace .env, per key",
    "takes ${VAR} expansion + secret URIs (pass://, vault://, ...)",
    "is applied verbatim — a relative path is resolved by its READER",
    "refuses a SWITCH (1/true/off) in a path var — #775, at profile load",
)


def workspace_profile_set(workspace: str) -> Optional[str]:
    """``JAATO_PROFILE_SET`` from a workspace's own ``.env``, if it names one.

    A profile inside ``profiles/<set>/`` is only in the effective set when
    that set is selected, and the selector a workspace runs under lives in
    its ``.env`` -- written there by ``new profile-set``.  Reading it is
    what makes ``explain oversight <name>`` resolve against the SAME set
    the workspace's own client will run under; without it, every profile a
    scaffolded workspace declares is invisible to these pages.

    One definition, shared by ``build`` and the ``explain`` pages, so the
    generator and the pages cannot disagree about which set a workspace is
    on.  It lives here rather than in ``explain`` so the authoring commands
    can read it without importing the introspection modules (#1267).
    """
    envf = Path(workspace).resolve() / ".env"
    if not envf.is_file():
        return None
    try:
        for line in envf.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "JAATO_PROFILE_SET":
                return value.strip() or None
    except OSError:             # pragma: no cover -- best-effort
        return None
    return None
