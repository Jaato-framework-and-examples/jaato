"""Re-attach client-tool-push stall fix: ``_runner_ready`` is BOOTSTRAP-complete,
not rpc-handle-live.

A reused warm pool slot's rpc handle is live the instant it's claimed, but the
slot can't service the session until its ``session.bootstrap`` completes — so
``set_runner_rpc`` CLEARS readiness and ``mark_runner_ready`` (called from
``dispatch_bootstrap_envelope`` after the bootstrap RPC settles) SETS it.  Both
the send path and the mid-session client-tool push gate on this Event.
"""
import threading
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
              # readiness clear (this fix) already ran before it


def test_set_runner_rpc_clears_ready_handle_alone_is_not_ready():
    # A wired rpc handle is NOT readiness — a reused warm pool slot's handle is
    # live before bootstrap.  Wiring it must clear any stale prior-session ready.
    s = _bare_server()
    s._runner_ready.set()                        # stale prior-session ready
    _set(s, object())                            # live RPC handle, pre-bootstrap
    assert not s._runner_ready.is_set()          # NOT ready until bootstrap


def test_set_runner_rpc_none_clears_ready():
    s = _bare_server()
    s._runner_ready.set()                        # pretend ready
    _set(s, None)                                # torn down
    assert not s._runner_ready.is_set()          # cleared


def test_mark_runner_ready_signals_after_bootstrap():
    s = _bare_server()
    _set(s, object())                            # handle wired (cleared)
    assert not s._runner_ready.is_set()
    s.mark_runner_ready()                        # dispatch_bootstrap_envelope settled
    assert s._runner_ready.is_set()              # NOW ready


def test_waiter_unblocks_when_bootstrap_completes():
    s = _bare_server()
    _set(s, object())                            # handle wired but not ready
    got = {}

    def waiter():
        got["ready"] = s._runner_ready.wait(timeout=2.0)

    t = threading.Thread(target=waiter)
    t.start()
    s.mark_runner_ready()                        # bootstrap completes -> ready
    t.join(timeout=3.0)
    assert got.get("ready") is True              # the send/push gate unblocks


# --- dispatch_bootstrap_envelope marks ready (the set-point) -----------------

class _FakeServer:
    def __init__(self, rpc):
        self.runner_rpc = rpc
        self.ready_calls = 0

    def mark_runner_ready(self):
        self.ready_calls += 1


def _dispatch(monkeypatch, rpc):
    from server import runner_spawn
    monkeypatch.setattr(runner_spawn, "build_session_envelope", lambda **k: object())
    monkeypatch.setattr(runner_spawn, "_emit_bootstrap_terminated", lambda **k: None)
    srv = _FakeServer(rpc)
    runner_spawn.dispatch_bootstrap_envelope(
        server=srv, session_id="sid", workspace_path="/ws", profile_name="prof")
    return srv


def test_dispatch_bootstrap_marks_ready_on_success(monkeypatch):
    class RPC:
        def bootstrap_session_threadsafe(self, env, timeout):
            return {"ok": True, "ready": True}
    assert _dispatch(monkeypatch, RPC()).ready_calls == 1


def test_dispatch_bootstrap_marks_ready_even_on_failure(monkeypatch):
    # The daemon-side session stays authoritative on bootstrap failure — the
    # finally must still mark ready so the push/send gate isn't stranded 30s.
    class RPC:
        def bootstrap_session_threadsafe(self, env, timeout):
            raise RuntimeError("boom")
    assert _dispatch(monkeypatch, RPC()).ready_calls == 1


def test_dispatch_bootstrap_skips_when_no_rpc(monkeypatch):
    # spawn-helper failed (rpc None) -> early return, NO mark_runner_ready, so
    # the gate correctly times out + raises (no runner to be ready).
    from server import runner_spawn
    monkeypatch.setattr(runner_spawn, "_emit_bootstrap_terminated", lambda **k: None)
    srv = _FakeServer(None)
    runner_spawn.dispatch_bootstrap_envelope(
        server=srv, session_id="sid", workspace_path="/ws", profile_name="prof")
    assert srv.ready_calls == 0
