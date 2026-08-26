"""#623's fix must leave a witness a default daemon can read.

#623 turned ``_pending_continuations`` from a single slot into a list, so N
sends landing in one wind-down window all survive instead of overwriting each
other.  It shipped **on inspection, with no live reproduction** — nobody had
ever observed the accumulate path run.

Both of its witnesses were ``server._trace`` — which is ``logger.debug``.  So
the only evidence that the fix works was invisible on a daemon at default log
level, and "no reproduction in my logs" was indistinguishable from "the
reproduction cannot reach my logs".  A peer went looking for it, found zero
occurrences, and nearly reported the first as the second.

That is the same defect #622 and #625 fixed elsewhere — a reason generated and
discarded — sitting on the line that decides whether a shipped fix can be
believed.
"""

from __future__ import annotations

import logging
import threading
from typing import List

from server.core import JaatoServer


def _server() -> JaatoServer:
    srv = JaatoServer.__new__(JaatoServer)
    srv._model_running = True
    srv._pending_continuations = []
    srv._pending_continuation_lock = threading.Lock()
    return srv


def _capture(level: int) -> tuple[logging.Handler, List[logging.LogRecord]]:
    records: List[logging.LogRecord] = []

    class _Cap(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Cap()
    logger = logging.getLogger("server.core")
    logger.addHandler(handler)
    logger.setLevel(level)
    return handler, records


def test_the_stash_is_visible_at_default_level():
    """A send landing in the wind-down window must say so at INFO."""
    srv = _server()
    handler, records = _capture(logging.INFO)
    logger = logging.getLogger("server.core")
    try:
        with srv._pending_continuation_lock:
            if srv._model_running:
                srv._pending_continuations.append("hello")
                import server.core as core_mod
                core_mod.logger.info(
                    "SEND_WHILE_UNWINDING: stashed %d chars for the model "
                    "thread's finally to pick up (%d now waiting)",
                    5, len(srv._pending_continuations),
                )
    finally:
        logger.removeHandler(handler)

    assert records, "the stash left no record at INFO"
    assert "SEND_WHILE_UNWINDING" in records[-1].getMessage()


def test_both_witnesses_are_info_not_debug():
    """Checked in SOURCE, because the failure is a level, not a behaviour.

    A runtime assertion cannot see a line that was never emitted; only the
    emitter's level says whether anyone downstream could have read it.
    """
    import inspect

    import server.core as core_mod

    src = inspect.getsource(core_mod)

    for marker in ("SEND_WHILE_UNWINDING:", "CONTINUATION: Processing"):
        idx = src.index(marker)
        # walk back to the emitting call
        head = src[max(0, idx - 400):idx]
        assert "logger.info(" in head, (
            f"{marker!r} is not emitted at INFO.  It is a witness to a fix "
            f"that has never been observed working; at debug it cannot be "
            f"read on a default daemon, and its absence from a log says "
            f"nothing about whether the path ran."
        )
        assert "_trace(" not in head.split("logger.info(")[-1], (
            f"{marker!r} still routes through _trace, which is logger.debug"
        )


def test_the_multiple_case_is_marked_in_the_line():
    """``count > 1`` is the case that used to lose messages.

    One grep must separate "the fix ran" from "the fix mattered", or a reader
    has to know the pre-#623 behaviour to interpret the count.
    """
    import inspect

    import server.core as core_mod

    src = inspect.getsource(core_mod)
    idx = src.index("CONTINUATION: Processing")
    window = src[idx:idx + 400]
    assert "len(stashed) > 1" in window, (
        "the multi-message case must be distinguishable in the line itself"
    )
    assert "pre-#623" in window
