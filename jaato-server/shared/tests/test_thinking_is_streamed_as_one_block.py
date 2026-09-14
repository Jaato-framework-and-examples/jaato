"""Reasoning is emitted write-then-append, like text (#755).

Providers hand the session reasoning the way they hand it text — per delta
on the OpenAI-shaped wires, as one accumulated block on the others.  The
session's output contract is that the FIRST chunk of a block is a
``"write"`` (start a new block) and every later chunk an ``"append"``.
Every ``thinking_callback`` emitted ``"write"`` for every chunk, so a client
honouring the contract opened a new block per delta: the TUI rendered one
~20-character line per token, which #755 reported as a wrapping bug.
"""

from shared.jaato_session import JaatoSession


def _session():
    s = JaatoSession.__new__(JaatoSession)
    s._trace = lambda msg: None
    return s


def test_first_chunk_writes_then_appends():
    calls = []
    emit = _session()._make_thinking_emitter(lambda *a: calls.append(a), "T")
    for delta in ["), capture", " script (tcp", "dump wrappers"]:
        emit(delta)
    assert calls == [
        ("thinking", "), capture", "write"),
        ("thinking", " script (tcp", "append"),
        ("thinking", "dump wrappers", "append"),
    ]


def test_a_whole_block_is_one_write():
    """Anthropic / claude_cli / Bedrock call the callback once — unchanged."""
    calls = []
    emit = _session()._make_thinking_emitter(lambda *a: calls.append(a), "T")
    emit("all of the reasoning at once")
    assert calls == [("thinking", "all of the reasoning at once", "write")]


def test_each_provider_call_starts_a_fresh_block():
    """Interleaved reasoning between two tool rounds is two blocks."""
    calls = []
    session = _session()
    first = session._make_thinking_emitter(lambda *a: calls.append(a), "T")
    first("round one")
    first(" continues")
    second = session._make_thinking_emitter(lambda *a: calls.append(a), "T")
    second("round two")
    assert [c[2] for c in calls] == ["write", "append", "write"]


def test_no_output_callback_is_a_no_op():
    emit = _session()._make_thinking_emitter(None, "T")
    emit("ignored")  # must not raise


def test_trace_names_the_mode():
    traced = []
    session = _session()
    session._trace = traced.append
    emit = session._make_thinking_emitter(lambda *a: None, "SESSION_THINKING_CALLBACK")
    emit("a")
    emit("b")
    assert traced == [
        "SESSION_THINKING_CALLBACK mode=write len=1",
        "SESSION_THINKING_CALLBACK mode=append len=1",
    ]
