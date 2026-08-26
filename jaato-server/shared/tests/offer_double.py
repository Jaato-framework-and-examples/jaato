"""Test double for the runner's ATOMIC queue-or-report (``offer_message``).

Step 2 moved the queue-or-drive decision off the daemon's ``_model_running``
replica and onto the target session, which answers under ``_delivery_lock``
against its own ``_is_running``.  Every fixture that exercises
``deliver_prompt_to_session`` therefore needs a session that can ANSWER, not
just a flag the daemon can read.

Kept in one module because four test files need it and a fake that drifts
between them is how a suite starts passing for the wrong reason -- the same
argument that put the real decision in one place.
"""

from shared.message_queue import SourceType


class OfferRPC:
    """Mirrors ``session.offer_message`` closely enough to be worth trusting.

    In particular it reproduces the two behaviours the daemon depends on:

    * ``needs_turn`` enqueues NOTHING -- so a fixture cannot show a message
      both queued and driven, and "idle was driven" stays distinguishable
      from "nothing happened".
    * ``source_type`` arrives as its wire STRING and is converted back to the
      enum here, exactly as the real runner handler does, so assertions can
      keep comparing enums.
    """

    def __init__(self, server, session_id, sink):
        self._server = server
        self._session_id = session_id
        self._sink = sink

    def session_offer_message_threadsafe(
        self, text, *, source_id=None, source_type=None,
        require_idle=False, timeout=None,
    ):
        if not getattr(self._server, "_model_running", False):
            return "needs_turn"
        if require_idle:
            # Backpressure probe: still working, so take nothing.
            return "busy"
        st = SourceType(source_type) if source_type is not None else None
        self._sink.append((self._session_id, text, source_id, st))
        return "queued"


def wire_offer(session, sink):
    """Attach an :class:`OfferRPC` to *session*'s fake server.

    Also sets ``_terminal_reason = None``: ``deliver_prompt_to_session``
    checks it before delivering, and a double missing it would report every
    target as terminated.
    """
    server = session.server
    server._runner_rpc = OfferRPC(server, session.session_id, sink)
    server._terminal_reason = None
    return session
