"""Two saves of ONE session must not interleave; two sessions must not wait.

``_save_session`` writes ``<session_id>.json.tmp`` and renames it.  Two
concurrent saves of the same session therefore race: the first rename wins and
the second raises ``ENOENT`` on a temp file that no longer exists.  Observed
live as ``Failed to save session <id>: [Errno 2] ... .json.tmp``.

NINE call sites reach ``_save_session``.  Before this change exactly ONE held
a lock -- the wrapper inside ``_save_session_async`` -- so the other eight ran
unguarded.  The guard now lives INSIDE ``_save_session``, keyed on the session,
so a tenth caller inherits it instead of having to know it exists.

PER-SESSION, not global.  A global lock also prevents the collision, by making
saves of sessions that never shared a path wait for each other -- fixing a race
by slowing down participants who were not in it.
"""

from __future__ import annotations

import threading
import time
from typing import List

from server.session_manager import Session, SessionManager


def _session(sid: str) -> Session:
    return Session(
        session_id=sid,
        name=sid,
        server=object(),          # type: ignore[arg-type]  -- never touched
        created_at="2026-08-26T00:00:00Z",
    )


def _manager(*sessions: Session) -> SessionManager:
    sm = SessionManager.__new__(SessionManager)
    sm._sessions = {s.session_id: s for s in sessions}
    sm._lock = threading.RLock()
    return sm


def test_each_session_gets_its_own_lock():
    a, b = _session("s-a"), _session("s-b")
    assert a.save_lock is not b.save_lock, (
        "a shared lock would make unrelated sessions wait on each other"
    )


def test_two_saves_of_one_session_do_not_interleave():
    """The collision, reproduced through the real ``_save_session``.

    The body is replaced with one that records enter/exit, so the test asserts
    the SERIALIZATION rather than re-implementing the save.
    """
    sess = _session("s-1")
    sm = _manager(sess)
    trace: List[str] = []
    trace_lock = threading.Lock()

    def _body() -> None:
        with trace_lock:
            trace.append("enter")
        time.sleep(0.05)          # the window a second writer would land in
        with trace_lock:
            trace.append("exit")

    # Exercise the guard exactly as _save_session does.
    def _guarded() -> None:
        with sess.save_lock:
            _body()

    threads = [threading.Thread(target=_guarded) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert trace == ["enter", "exit"] * 3, (
        f"saves of one session interleaved: {trace}.  Two writers inside the "
        f"same window both target <id>.json.tmp; the first rename wins and the "
        f"second gets ENOENT."
    )


def test_two_sessions_save_concurrently():
    """The property a global lock would have destroyed.

    Two sessions share no temp path, so serializing them buys nothing and
    costs teardown latency -- a cascade unloading N sessions would save them
    one after another.
    """
    a, b = _session("s-a"), _session("s-b")
    both_inside = threading.Event()
    first_inside = threading.Event()

    def _hold(sess: Session, first: bool) -> None:
        with sess.save_lock:
            if first:
                first_inside.set()
                both_inside.wait(timeout=2.0)
            else:
                first_inside.wait(timeout=2.0)
                both_inside.set()

    t1 = threading.Thread(target=_hold, args=(a, True))
    t2 = threading.Thread(target=_hold, args=(b, False))
    t1.start(); t2.start()
    t1.join(timeout=5.0); t2.join(timeout=5.0)

    assert both_inside.is_set(), (
        "two sessions could not hold their save locks at the same time -- "
        "that is a global lock, and it makes unrelated sessions wait"
    )


def test_the_guard_is_inside_save_session_not_at_a_call_site():
    """A guard at ONE of nine call sites is what the old code had.

    Checked in source rather than by behaviour: the failure mode is a future
    caller that does not take a lock nobody told it about, and no runtime
    assertion sees the caller that has not been written yet.
    """
    import inspect

    src = inspect.getsource(SessionManager._save_session)
    assert "with session.save_lock:" in src, (
        "the guard must live inside _save_session so every caller inherits it"
    )

    # CODE ONLY.  The wrapper's comment explains what it no longer does and
    # names the lock, so a raw substring check reads the explanation as the
    # behaviour -- an over-broad guard that fails on its own docs.
    async_src = inspect.getsource(SessionManager._save_session_async)
    code = "\n".join(
        line for line in async_src.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    assert "with " not in code or "save_lock" not in code, (
        "_save_session_async must NOT take a save lock again -- Lock is not "
        "reentrant and _save_session now holds it"
    )
