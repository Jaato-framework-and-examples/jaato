"""Attach race fix (b): the _runner_ready Event tracks _runner_rpc liveness so
the send path can await a (re)spawning runner instead of dereferencing None.

The Event set/clear runs at the top of set_runner_rpc (before the registry plumb,
which needs a real RPC handle) — so the bare-fake plumb failure is caught and we
assert the readiness signal, which is what this fix changed."""
import threading
import pytest
from server.core import JaatoServer


def _bare_server():
    s = JaatoServer.__new__(JaatoServer)        # skip heavy __init__
    s._runner_rpc = None
    s._spawned_runner = None
    s._runner_ready = threading.Event()
    s.registry = None
    return s


def _set(s, rpc):
    try:
        s.set_runner_rpc(rpc, None)
    except Exception:
        pass  # downstream registry/rpc_server plumb needs a real handle; the
              # readiness signal (this fix) already ran before it


def test_set_runner_rpc_signals_ready():
    s = _bare_server()
    assert not s._runner_ready.is_set()          # starts not-ready
    _set(s, object())                            # a live RPC handle
    assert s._runner_ready.is_set()              # signaled ready


def test_set_runner_rpc_none_clears_ready():
    s = _bare_server()
    s._runner_ready.set()                        # pretend ready
    _set(s, None)                                # torn down
    assert not s._runner_ready.is_set()          # cleared


def test_waiter_unblocks_when_runner_becomes_ready():
    s = _bare_server()
    got = {}

    def waiter():
        got["ready"] = s._runner_ready.wait(timeout=2.0)

    t = threading.Thread(target=waiter)
    t.start()
    _set(s, object())                            # async (re)spawn completes
    t.join(timeout=3.0)
    assert got.get("ready") is True              # the send-path await unblocks
