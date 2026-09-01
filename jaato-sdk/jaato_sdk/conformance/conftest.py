"""One daemon per module, because starting one costs ~3s and asserting is free."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from jaato_sdk.conformance.daemon import ConformanceDaemon, echo_workspace

#: Every turn charges this.  Small enough that a ceiling of a few thousand
#: tokens is reached in one turn, which keeps budget invariants to a single
#: send rather than a loop whose length is itself a variable.
TURN_USAGE = {"prompt_tokens": 1000, "output_tokens": 200, "cost_usd": 0.0042}


#: A profile whose run ends at its DECLARED terminus.
#:
#: This is not a nicety.  A suite whose profiles all answer in prose settles
#: every scenario on TURN_COMPLETED and never exercises the path where a
#: session terminates INSIDE a tool-use turn -- and that terminus is the
#: condition under which a consumer measured event delivery silently stopping
#: for cid'd sessions.  Their first repro ended in prose and exonerated the
#: daemon; so did the first version of this suite, which went green on two
#: invariants a five-scenario matrix shows broken.
#:
#: echo is told to call ``signal_completion``, so the session reaches the
#: terminus deterministically, every run, with no model involved.
COMPLETION_SCHEMA = {
    "type": "object",
    "properties": {"status": {"type": "string"}},
    "required": ["status"],
}

#: THE SCHEMA *IS* THE PARAMETER LIST, so the payload's fields go at the TOP
#: LEVEL of ``args`` -- not nested under a ``payload`` key.
#: ``lifecycle_tools.py`` builds the tool as ``parameters =
#: self._payload_schema``, and its description says "call with the full
#: payload matching completion_payload_schema AS ARGS".
#:
#: Wrapping them cost 806 turns: ``signal_completion(payload={...})`` fails
#: validation because ``status`` is required at the top level and absent, the
#: tool errors, the agent retries, and the session never reaches its terminus
#: -- so the two invariants that depend on the terminus passed while never
#: arriving at the path they exist to test.
SIGNAL_COMPLETION_CALL = {
    "name": "signal_completion",
    "args": {"status": "done"},
}


@pytest.fixture(scope="module")
def daemon():
    """A daemon serving THREE profiles, one per ending a session can have.

    The first two are needed and neither substitutes for the other -- the
    defects that hide behind a prose ending are exactly the ones the terminus
    exposes, and a suite carrying only the second could not tell a general
    breakage from a terminus-specific one.

    ``conformance-nudged`` is the third ending, and it is a COMBINATION rather
    than a variant: a completion schema (so ``signal_completion`` is in the
    surface and the daemon expects it) with a model that answers in prose (so
    it never arrives).  That is the state in which the daemon RE-PROMPTS the
    session, and both defects in jaato #767 live only there -- an unbounded
    nudge loop, and a caller settling on the first of several turns.  Neither
    profile above reaches it: the prose one is not gated, and the terminus one
    signals on turn 1.
    """
    root = Path(tempfile.mkdtemp(prefix="jaato-conformance-ws-"))
    echo_workspace(root, usage=TURN_USAGE, response="conformance ok",
                   name="conformance")
    echo_workspace(root, usage=TURN_USAGE,
                   tool_call=SIGNAL_COMPLETION_CALL,
                   completion_schema=COMPLETION_SCHEMA,
                   name="conformance-terminus")
    echo_workspace(root, usage=TURN_USAGE, response="conformance ok",
                   completion_schema=COMPLETION_SCHEMA,
                   name="conformance-nudged")
    d = ConformanceDaemon(root)
    try:
        yield d.start()
    finally:
        d.stop()
