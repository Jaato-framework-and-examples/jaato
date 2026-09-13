"""Naming the BOUNDARY a runner wears, not the session that asked for it.

``aa_change_profile`` confines the calling TASK.  A pool slot that has
served one session therefore carries that session's profile on every
thread it created — the RPC lanes, the telemetry exporter, a plugin's
reaper — and #1023's per-thread verification is what finally made that
visible.  Those threads cannot be re-confined (the kernel enforces
``current != task -> -EACCES``) and only the two executor lanes can be
retired (#1026).  So a slot's profile is, in practice, **immutable for
the life of the slot**.

The profile name was ``jaato-ws-{session_id}``, and ``session_id`` is the
one property guaranteed to DIFFER on every reuse.  The slot key said
"reusable" and the profile name said "new boundary", and once confinement
became per-task both could not be true: every reused slot straddled two
profiles and the bootstrap was correctly refused (#1033's
``RunnerCallError`` on live session creation).

This module derives the name from what the boundary actually IS:

* the **workspace** the profile grants,
* the **config root** it grants,
* a digest of the **rendered profile body** — which folds in the env
  file, the composed AppArmor fragments (their contents, not just their
  names), the plugin-contributed rules, and the complain-mode flag.

Two sessions whose boundaries are identical get one name; two whose
boundaries differ get two.  That is what makes the reuse key and the
profile name agree by construction, so a reused slot never transitions
and there is no divergence to detect.

Including the rendered body is not belt-and-braces.  Without it, two
CONCURRENT sessions of one cascade — a narrow stage and a broad stage,
each on its own slot — would share a profile name, and provisioning the
broad one would reload the name the narrow one is already confined to.
A silent widening of a live session's boundary is a worse failure than
the one being fixed.

The id is human-readable on purpose (consequence 3 of the fix): an
operator greps ``jaato-ws-*`` in ``dmesg`` and in
``/proc/<pid>/attr/current``.  It leads with a slug of the workspace's
basename and ends in a short digest.  Session identity is not lost with
it — #812's ``runner_identity`` records which runner ran which session,
which is the mapping an operator actually needs.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Optional


#: Characters ``shared.session_id.is_safe_session_id`` accepts.  The id is
#: interpolated into the AppArmor profile grammar and into the on-disk
#: profile filename, so everything else is collapsed to ``-``.
_UNSAFE_RUN = re.compile(r"[^A-Za-z0-9._-]+")

#: Hex characters of the boundary digest carried in the id.  48 bits is
#: far more than the number of distinct boundaries any daemon holds, and
#: short enough to read back off a kernel log line.
DIGEST_CHARS = 12

#: Longest workspace slug carried in front of the digest.  The whole id
#: has to stay well inside ``session_id``'s 256-char ceiling once
#: ``jaato-ws-`` and a ``//child`` suffix are added.
MAX_SLUG_CHARS = 32

#: Stand-in identifier used when rendering a profile body purely to
#: digest it.  It must be a constant, because it appears IN the body
#: (the profile header, the ``/tmp/jaato-<id>-**`` grants, the refs
#: include glob) and the digest must not depend on it.
PROBE_ID = "confinement-probe"

#: Field separator for the canonical digest input.  A byte that cannot
#: occur in a path, so two different tuples cannot canonicalise alike.
_SEP = "\x1f"


def workspace_slug(workspace_root: Optional[str]) -> str:
    """A readable, path-safe stem for *workspace_root*.

    The basename, with every character AppArmor / the filesystem would
    object to collapsed to ``-`` and the result capped.  ``"ws"`` when
    there is nothing usable to read — a slug is a legibility aid, never
    an identity, so it is allowed to be generic.
    """
    if not workspace_root:
        return "ws"
    base = os.path.basename(os.path.normpath(str(workspace_root))) or "ws"
    slug = _UNSAFE_RUN.sub("-", base).strip("-.")
    if not slug:
        return "ws"
    return slug[:MAX_SLUG_CHARS]


def _canonical(path: Optional[str]) -> str:
    """Resolve *path* for comparison, tolerating one that does not exist.

    ``realpath`` answers with the input when nothing is there, which is
    the behaviour wanted: an id must be derivable before the directory
    is created, and two spellings of one existing path must land on one
    id.
    """
    if not path:
        return ""
    try:
        return os.path.realpath(str(path))
    except OSError:  # pragma: no cover — realpath is near-total
        return str(path)


def boundary_digest(
    *,
    workspace_root: Optional[str],
    config_root: Optional[str],
    rendered_body: Optional[str],
) -> str:
    """Hex digest of everything that makes this boundary what it is.

    *rendered_body* is the profile text rendered with :data:`PROBE_ID`
    in place of the identifier, so the digest is a function of the
    RULES and not of the name they will end up carrying.  ``None`` is
    accepted (a render that failed) and simply contributes nothing —
    the two paths still separate boundaries that differ in workspace or
    config root, which is the part that can be established without a
    render.
    """
    material = _SEP.join((
        _canonical(workspace_root),
        _canonical(config_root),
        rendered_body or "",
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def confinement_id(
    *,
    workspace_root: Optional[str],
    config_root: Optional[str],
    rendered_body: Optional[str],
) -> str:
    """The identifier that replaces ``session_id`` in the profile name.

    Returns e.g. ``my-repo-3f2a9c1b7d4e``, so the kernel-visible profile
    reads ``jaato-ws-my-repo-3f2a9c1b7d4e``.  Deterministic: the same
    boundary always produces the same id, in this process and in the
    next one, which is precisely the property a reused slot needs.
    """
    digest = boundary_digest(
        workspace_root=workspace_root,
        config_root=config_root,
        rendered_body=rendered_body,
    )
    return f"{workspace_slug(workspace_root)}-{digest}"
