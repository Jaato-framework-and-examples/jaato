"""Tests for ``RenderContext.session_id`` resolution (server 0.6.172+).

Closes the kb-side per-session-audit-ledger gap surfaced 2026-06-02:
kb-orchestrator's audit walker was reading a shared
``cascade_state/processor_audit.jsonl`` file, and an
``audit_offset`` reset at iter_start re-emitted iter-1 rejections
inside iter-2 panes (already patched at cascade.py:225).  The
structural fix — switch to per-session audit files at
``cascade_state/audit/<session_id>.jsonl`` — requires kb-side
code to read ``session_id`` from the processor context WITHOUT a
defensive fallback (Daniel's no-fallback rule).  Pre-0.6.172
RenderContext did not expose session_id; kb-side would have had
to reach into ``context.session._daemon_session_id`` (private
attribute, fragile) OR populate via getattr with a defensive
fallback (forbidden).

This file pins the resolution contract:
  - immediate session's ``_daemon_session_id`` wins
  - parent-walk fallback for subagent sessions (canonical pattern
    mirroring ``shared.jaato_session:621-628``)
  - returns None cleanly when no ancestor has it set
  - duck-typed via getattr so unit tests with bare objects don't
    raise AttributeError

Memory cross-ref:
``feedback_kernel_scoping_beats_persona_prose`` — the resolver is
the kernel-layer gate, replacing kb-side defensive ``getattr(context,
"session_id", None)`` with a populated-at-construction-time field.
"""

from __future__ import annotations

import pytest

from shared.dynamic_instructions import (
    RenderContext,
    _resolve_session_id,
    build_render_context,
)


class _FakeSession:
    """Minimal stand-in for ``JaatoSession`` exposing only the
    attributes ``_resolve_session_id`` reads — ``_daemon_session_id``
    and ``_parent_session``.

    Lets the test class exercise the parent-walk without standing up
    the full JaatoSession + JaatoRuntime machinery.  Mirrors the
    duck-typed contract the resolver uses via ``getattr``.
    """

    def __init__(
        self,
        daemon_session_id=None,
        parent_session=None,
    ):
        self._daemon_session_id = daemon_session_id
        self._parent_session = parent_session


class TestSessionIdResolution:
    """``RenderContext.session_id`` is populated at
    ``build_render_context`` time via ``_resolve_session_id``, which
    mirrors the canonical parent-walk in
    ``shared.jaato_session:621-628``.

    The walk is load-bearing: subagent sessions are spawned with a
    fresh ``JaatoSession`` that does NOT receive its own
    ``_daemon_session_id`` (the daemon-side session manager only
    registers the root agent's session).  Without the walk, every
    subagent's ``RenderContext.session_id`` would be None and any
    kb-side code relying on a non-None session_id (e.g. per-session
    audit ledger files at ``cascade_state/audit/<session_id>.jsonl``)
    would break — exactly what the per-session refactor needs to
    avoid.
    """

    def test_immediate_daemon_session_id_used(self):
        """When the immediate session has a daemon_session_id, the
        resolver returns it without walking the parent chain."""
        session = _FakeSession(daemon_session_id="20260602_153000")
        ctx = build_render_context(session=session)
        assert ctx.session_id == "20260602_153000"

    def test_parent_walk_when_immediate_session_id_absent(self):
        """Subagent shape: immediate session has no daemon_session_id;
        resolver walks up one level to find the parent's id.  Mirrors
        the empirical case where a top-level agent spawns one subagent
        and the subagent's RenderContext.session_id must match the
        parent's."""
        parent = _FakeSession(daemon_session_id="20260602_153000")
        child = _FakeSession(daemon_session_id=None, parent_session=parent)
        ctx = build_render_context(session=child)
        assert ctx.session_id == "20260602_153000"

    def test_multi_level_parent_walk(self):
        """Nested-subagent shape: A spawns B spawns C; only A has a
        daemon_session_id.  Resolver walks leaf -> middle -> root two
        levels up to find the root id.  Required because nested-
        cascade topologies have arbitrary depth — the walk must not
        short-circuit at the first None."""
        root = _FakeSession(daemon_session_id="20260602_153000")
        middle = _FakeSession(daemon_session_id=None, parent_session=root)
        leaf = _FakeSession(daemon_session_id=None, parent_session=middle)
        ctx = build_render_context(session=leaf)
        assert ctx.session_id == "20260602_153000"

    def test_no_ancestor_has_session_id_returns_none(self):
        """When no session in the parent chain has a daemon_session_id
        (e.g. bare JaatoSession in a unit test, or a session built
        before daemon-side registration runs), resolver returns None.
        Kb-side callers see None and either skip the artifact or
        fail loudly — they do NOT get a defensive substitute per
        Daniel's no-fallback rule."""
        parent = _FakeSession(daemon_session_id=None)
        child = _FakeSession(daemon_session_id=None, parent_session=parent)
        ctx = build_render_context(session=child)
        assert ctx.session_id is None

    def test_resolver_tolerates_non_session_input(self):
        """``_resolve_session_id`` is duck-typed via ``getattr`` so
        unit tests passing arbitrary objects as ``session`` see a
        clean None rather than an AttributeError.  Keeps test setup
        cheap and prevents the resolver from masking real bugs with
        try/except suppression."""
        assert _resolve_session_id(None) is None
        assert _resolve_session_id(object()) is None
        # Mid-walk: parent_session is set but is itself a non-session.
        broken_chain = _FakeSession(
            daemon_session_id=None,
            parent_session=object(),
        )
        assert _resolve_session_id(broken_chain) is None
