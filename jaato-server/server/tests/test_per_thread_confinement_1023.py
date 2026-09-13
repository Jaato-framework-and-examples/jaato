"""Per-thread AppArmor confinement: the check, and the threads it checks (#1023).

WHAT WENT WRONG.  ``aa_change_profile`` confines the CALLING TASK.
``/proc/self/attr/current`` resolves to ``/proc/<pid>/attr/current``, which
reports the label of the task whose tid == pid -- the MAIN THREAD.  Every
confinement check in the tree reads that one path, so a worker thread
created before the transition keeps its ``unconfined`` cred for the life of
the pool slot while the post-transition readback, the idempotency check,
``sandbox_mode`` in the session record and an operator's
``cat /proc/<pid>/attr/current`` all report ``(enforce)``.

``ThreadPoolExecutor`` spawns workers lazily on first submit, and a pool
slot fields RPCs before its first ``session.bootstrap``, so the population
is routine rather than exotic -- confirmed live on 2 of 5 runners of a
fully enforcing host.

THE TWO HALVES GUARDED HERE.

1. ``RunnerRPC.recycle_worker_pools`` retires both lanes at the transition,
   so no thread predating it can serve a later RPC.  Asserted by THREAD
   IDENTITY, which needs no AppArmor: the property is "not the same
   thread", and that is true or false on any kernel.
2. ``bootstrap.verify_thread_confinement`` walks every thread's own
   ``attr/current`` and refuses the bootstrap on divergence.  Asserted
   against a FABRICATED ``<dir>/<tid>/attr/current`` tree, which reproduces
   the procfs shape exactly and lets a divergent thread be represented on a
   host with no AppArmor at all.

WHAT IS NOT EXERCISED HERE.  No test in this module confines anything.  The
kernel side -- that ``aa_change_profile`` on the reader thread leaves a
sibling thread behind, and that a recycled lane's threads come up inside
the profile -- requires an AppArmor-enforcing host and is not reproduced in
CI.  What is proven here is everything above the kernel: that the walk
classifies each shape correctly, that divergence is refused, that absence
of evidence is not, and that recycling replaces the threads.
"""
from __future__ import annotations

import ast
import inspect
import os
import socket
import threading
from pathlib import Path
from typing import List

import pytest

from server.runner.bootstrap import (
    DEFAULT_TASK_ATTR_DIR,
    ThreadConfinementDivergence,
    profile_name_of,
    scan_thread_profiles,
    verify_thread_confinement,
)
from server.runner.rpc import RunnerRPC
from server.runner.session import BootstrapError, _retire_and_verify_threads


# --------------------------------------------------------------------------
# Reversions: the one-line change that must make each guard go red.
# --------------------------------------------------------------------------
try:  # pragma: no cover - import shape differs per invocation
    from shared.tests.test_every_guard_detects_its_own_reversion import Reversion
except Exception:  # pragma: no cover
    Reversion = None  # type: ignore[assignment]

_SESSION = "jaato-server/server/runner/session.py"
_BOOTSTRAP = "jaato-server/server/runner/bootstrap.py"
_RPC = "jaato-server/server/runner/rpc.py"

REVERSIONS = [] if Reversion is None else [
    Reversion(
        target=_BOOTSTRAP,
        find="""    raise ThreadConfinementDivergence(
        expected_profile,
        scan.divergent,""",
        replace="""    return scan
    raise ThreadConfinementDivergence(
        expected_profile,
        scan.divergent,""",
        test="test_a_divergent_thread_is_refused",
        because="divergence stops being refused and reads as a clean scan",
    ),
    Reversion(
        target=_BOOTSTRAP,
        find="""        elif _label_is_inside(value, expected_profile):
            matched.append(tid)""",
        replace="""        elif True:
            matched.append(tid)""",
        test="test_a_divergent_thread_is_refused",
        because="every label is accepted, so no thread can ever diverge",
    ),
    Reversion(
        target=_BOOTSTRAP,
        find="""    except OSError as exc:
        return "unreadable", f"{type(exc).__name__}: {exc}\"""",
        replace="""    except OSError as exc:
        return "gone", f"{type(exc).__name__}: {exc}\"""",
        test="test_an_unreadable_thread_is_not_divergence_but_is_reported",
        because="an unreadable thread is silently counted as one that exited",
    ),
    Reversion(
        target=_RPC,
        find="""        old_work, old_ctl = self._pool, self._control_pool
        retired_work = self._lane_threads(old_work)""",
        replace="""        return {"work_threads_retired": 0, "ctl_threads_retired": 0}
        old_work, old_ctl = self._pool, self._control_pool
        retired_work = self._lane_threads(old_work)""",
        test="test_no_worker_predating_the_recycle_serves_later_work",
        because="the lanes are no longer recycled, so pre-transition workers survive",
    ),
    Reversion(
        target=_SESSION,
        # Anchored on the call and the except-block tail that precedes it,
        # NOT on the run from the call down to the next ``def``.  The old
        # anchor spanned that gap and went stale the moment a helper was
        # inserted between them (#1014's complain-mode announcement did
        # exactly that), which reports BLOCKED: the guard is then neither
        # known-good nor known-broken.  ``) from exc`` is the end of the
        # error handling this call follows, and it moves only when the code
        # this reversion is about moves.
        #
        # The 4-space indent is what distinguishes this call from the
        # IDENTICAL 8-space one on the idempotent path above it; that one
        # has its own reversion below.
        find="""        ) from exc

    _retire_and_verify_threads(target_profile, recycle_pools)""",
        replace="""        ) from exc""",
        test="test_the_transition_path_retires_and_verifies",
        because="a fresh transition stops recycling and stops being verified",
    ),
    Reversion(
        target=_SESSION,
        find="""        _retire_and_verify_threads(target_profile, recycle_pools)
        return""",
        replace="""        return""",
        test="test_the_idempotent_path_retires_and_verifies_too",
        because=(
            "a reused pool slot re-bootstrapping under the same profile stops "
            "being checked, which is the commonest #1023 case"
        ),
    ),
    Reversion(
        target=_SESSION,
        find="""    if recycle_pools is not None:
        try:
            recycle_pools(f"confined to {target_profile}")""",
        replace="""    if recycle_pools is None:
        try:
            recycle_pools(f"confined to {target_profile}")""",
        test="test_retire_and_verify_recycles_before_it_verifies",
        because="the recycle hook is never called, so nothing retires the threads",
    ),
]


# --------------------------------------------------------------------------
# A fabricated /proc/<pid>/task tree.
# --------------------------------------------------------------------------

def _fake_task_dir(tmp_path: Path, labels: dict) -> str:
    """Build ``<dir>/<tid>/attr/current`` for each ``{tid: label}``.

    ``label`` of ``None`` creates the tid directory with no ``attr/current``
    at all -- the thread-exited-mid-walk shape, which procfs produces as
    ENOENT.
    """
    root = tmp_path / "task"
    for tid, label in labels.items():
        attr = root / str(tid) / "attr"
        attr.mkdir(parents=True, exist_ok=True)
        if label is not None:
            (attr / "current").write_text(label)
    return str(root)


def test_a_divergent_thread_is_refused(tmp_path):
    """One unconfined thread beside confined ones raises, and names itself.

    This is the live shape: low-tid threads created before the transition
    read ``unconfined``, high-tid threads read the profile.
    """
    task_dir = _fake_task_dir(tmp_path, {
        101: "unconfined",                    # created pre-transition
        102: "jaato-ws-sess1 (enforce)\n",
        103: "jaato-ws-sess1 (enforce)\n",
    })
    with pytest.raises(ThreadConfinementDivergence) as excinfo:
        verify_thread_confinement(
            "jaato-ws-sess1", task_dir=task_dir,
            grace_seconds=0.0, sleep=lambda _s: None,
        )
    exc = excinfo.value
    assert exc.divergent == ((101, "unconfined"),)
    # The operator's next question is which thread and what it holds, so
    # both must be in the message rather than only in the attributes.
    assert "101" in str(exc) and "unconfined" in str(exc)


def test_a_uniformly_confined_process_passes(tmp_path):
    task_dir = _fake_task_dir(tmp_path, {
        201: "jaato-ws-sess1 (enforce)\n",
        202: "jaato-ws-sess1 (enforce)\n",
    })
    scan = verify_thread_confinement(
        "jaato-ws-sess1", task_dir=task_dir, grace_seconds=0.0,
    )
    assert scan.uniform
    assert scan.divergent == ()
    assert len(scan.matched) == 2
    assert scan.route == "task_dir"


def test_another_sessions_profile_is_divergence(tmp_path):
    """#1023 impact 3: a slot reused across sessions carries P1 into P2.

    Still ``(enforce)``, so no audit-hook symptom -- and the wrong
    workspace.  A check that only asked "is it confined" would pass this.
    """
    task_dir = _fake_task_dir(tmp_path, {
        301: "jaato-ws-sess1 (enforce)\n",     # previous session's profile
        302: "jaato-ws-sess2 (enforce)\n",
    })
    with pytest.raises(ThreadConfinementDivergence) as excinfo:
        verify_thread_confinement(
            "jaato-ws-sess2", task_dir=task_dir, grace_seconds=0.0,
        )
    assert excinfo.value.divergent == ((301, "jaato-ws-sess1 (enforce)"),)


def test_a_subprofile_or_complain_mode_is_not_divergence(tmp_path):
    """Narrower-or-same is inside the boundary; complain is a mode, not a profile.

    ``//child`` DROPS rules, so a thread wearing one is inside what the
    session claims.  ``JAATO_APPARMOR_COMPLAIN=1`` is a documented
    diagnostic that puts the whole chain in complain mode -- reading it as
    divergence would make the diagnostic unusable.
    """
    task_dir = _fake_task_dir(tmp_path, {
        401: "jaato-ws-sess1 (enforce)\n",
        402: "jaato-ws-sess1//child (enforce)\n",
        403: "jaato-ws-sess1 (complain)\n",
    })
    scan = verify_thread_confinement(
        "jaato-ws-sess1", task_dir=task_dir, grace_seconds=0.0,
    )
    assert scan.divergent == ()
    assert len(scan.matched) == 3


def test_an_unreadable_thread_is_not_divergence_but_is_reported(tmp_path):
    """Absence of evidence must not take a host down.

    A restricted ``/proc`` proves nothing about confinement, so the scan
    reports ``unreadable`` and the caller logs.  Represented here by making
    ``attr/current`` a directory, which raises ``IsADirectoryError`` (an
    ``OSError``) for any uid -- a permission bit would not, running as root.
    """
    task_dir = _fake_task_dir(tmp_path, {501: "jaato-ws-sess1 (enforce)\n"})
    attr = Path(task_dir) / "502" / "attr"
    attr.mkdir(parents=True)
    (attr / "current").mkdir()

    scan = verify_thread_confinement(
        "jaato-ws-sess1", task_dir=task_dir, grace_seconds=0.0,
    )
    assert scan.divergent == ()
    assert [tid for tid, _why in scan.unreadable] == [502]
    assert not scan.uniform


def test_a_thread_that_exited_mid_walk_is_benign(tmp_path):
    """Threads exit; ENOENT on the read is a race, not a finding."""
    task_dir = _fake_task_dir(tmp_path, {
        601: "jaato-ws-sess1 (enforce)\n",
        602: None,                             # tid dir, no attr/current
    })
    scan = verify_thread_confinement(
        "jaato-ws-sess1", task_dir=task_dir, grace_seconds=0.0,
    )
    assert scan.gone == (602,)
    assert scan.divergent == ()
    assert scan.uniform


def test_an_unlistable_task_dir_falls_back_rather_than_giving_up(tmp_path):
    """No task-dir grant (profile template < v32) still yields a walk.

    The fallback enumerates ``threading.enumerate()`` plus this process's
    own pid, needs no AppArmor grant, and names its route so the caller
    knows what it could NOT have seen.
    """
    scan = scan_thread_profiles(
        "jaato-ws-sess1", task_dir=str(tmp_path / "does-not-exist"),
    )
    assert scan.route == "threading"
    # It looked for this process's real threads and found their files
    # missing under the bogus root -- the point is that it enumerated at
    # all rather than returning nothing.
    assert scan.scanned + len(scan.gone) >= 1


def test_divergence_that_clears_inside_the_grace_window_is_not_reported(tmp_path):
    """A worker draining its last task is not the #1023 population.

    Recycling shuts the old lanes down without waiting (waiting could
    deadlock against a pre-bootstrap task blocked on an outgoing call the
    reader thread must answer), so a worker can be briefly alive holding
    the old cred.  It exits in milliseconds; a leaked thread does not.
    """
    task_dir = _fake_task_dir(tmp_path, {
        701: "unconfined",
        702: "jaato-ws-sess1 (enforce)\n",
    })
    draining = Path(task_dir) / "701"

    def _fake_sleep(_seconds: float) -> None:
        # The draining worker exits during the first poll interval.
        (draining / "attr" / "current").unlink(missing_ok=True)

    clock = iter([0.0, 0.0, 0.5, 1.0, 1.5])
    scan = verify_thread_confinement(
        "jaato-ws-sess1", task_dir=task_dir, grace_seconds=2.0,
        sleep=_fake_sleep, monotonic=lambda: next(clock),
    )
    assert scan.divergent == ()
    assert scan.gone == (701,)


def test_divergence_that_outlives_the_grace_window_is_reported(tmp_path):
    """The same window must not become a tolerance for the defect."""
    task_dir = _fake_task_dir(tmp_path, {
        801: "unconfined",
        802: "jaato-ws-sess1 (enforce)\n",
    })
    polls: List[float] = []
    clock = iter([0.0, 0.0, 1.0, 3.0])
    with pytest.raises(ThreadConfinementDivergence):
        verify_thread_confinement(
            "jaato-ws-sess1", task_dir=task_dir, grace_seconds=2.0,
            sleep=polls.append, monotonic=lambda: next(clock),
        )
    assert polls, "the window must actually re-scan before it gives up"


def test_profile_name_of_strips_mode_and_nothing_else():
    assert profile_name_of("jaato-ws-a (enforce)") == "jaato-ws-a"
    assert profile_name_of("jaato-ws-a (complain)") == "jaato-ws-a"
    assert profile_name_of("unconfined") == "unconfined"
    assert profile_name_of("jaato-ws-a//child (enforce)") == "jaato-ws-a//child"


def test_the_real_proc_shape_is_readable_on_this_host():
    """The walk runs against the live ``/proc``, whatever it reports.

    Not an AppArmor assertion -- this host has no enforcing kernel.  It
    asserts the file shape the walk depends on exists and parses: the tids
    enumerate, the labels read, and the NUL procfs terminates them with is
    stripped (an unstripped one makes every later comparison fail against a
    name that looks identical when printed).
    """
    if not os.path.isdir(DEFAULT_TASK_ATTR_DIR):
        pytest.skip("no /proc/self/task on this platform")
    scan = scan_thread_profiles("a-profile-nothing-is-in")
    assert scan.route == "task_dir"
    assert scan.scanned >= 1
    for _tid, label in scan.divergent:
        assert "\x00" not in label
        assert label == label.strip()


# --------------------------------------------------------------------------
# Pool recycling, by thread identity.
# --------------------------------------------------------------------------

def _saturate(pool, width: int) -> set:
    """Run *width* concurrent tasks so every worker thread is created."""
    barrier = threading.Barrier(width)
    threads = set()
    lock = threading.Lock()

    def _task():
        barrier.wait(timeout=10)
        with lock:
            threads.add(threading.current_thread())

    futures = [pool.submit(_task) for _ in range(width)]
    for fut in futures:
        fut.result(timeout=10)
    return threads


@pytest.fixture()
def runner_rpc():
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )
    rpc = RunnerRPC(
        runner_sock, lambda name, args: (True, {}),
        max_workers=3, control_workers=2,
    )
    try:
        yield rpc
    finally:
        rpc._pool.shutdown(wait=False)
        rpc._control_pool.shutdown(wait=False)
        daemon_sock.close()
        runner_sock.close()


def test_no_worker_predating_the_recycle_serves_later_work(runner_rpc):
    """The property that matters, and it is kernel-independent.

    A worker created before the transition holds the cred it was created
    with and cannot be repaired -- the kernel refuses an ``attr/current``
    write from any task but its own -- so the only remedy is to retire it.
    Asserting by thread IDENTITY (the Thread objects, held for the duration
    so no ident can be recycled) states exactly that: nothing that ran
    before the recycle runs after it.
    """
    before_work = _saturate(runner_rpc._pool, 3)
    before_ctl = _saturate(runner_rpc._control_pool, 2)
    assert len(before_work) == 3 and len(before_ctl) == 2

    stats = runner_rpc.recycle_worker_pools("confined to jaato-ws-sess1")
    assert stats["work_threads_retired"] == 3
    assert stats["ctl_threads_retired"] == 2

    after_work = _saturate(runner_rpc._pool, 3)
    after_ctl = _saturate(runner_rpc._control_pool, 2)

    assert not (before_work & after_work), (
        "a work-lane thread created before the AppArmor transition served "
        "work after it"
    )
    assert not (before_ctl & after_ctl), (
        "a control-lane thread created before the AppArmor transition "
        "served work after it"
    )


def test_recycling_does_not_cancel_work_already_submitted(runner_rpc):
    """Nothing in flight is dropped, because a dropped call never answers.

    ``session.end`` from the previous session of a cascade, and the
    dispatch watchdog's health probe, can legitimately be executing when
    the bootstrap frame is read.  Cancelling them would leave the daemon
    waiting for a response that is never written, so the old executors are
    shut down with neither ``wait`` nor ``cancel_futures``.
    """
    release = threading.Event()
    started = threading.Event()

    def _slow():
        started.set()
        release.wait(timeout=10)
        return "finished anyway"

    fut = runner_rpc._pool.submit(_slow)
    assert started.wait(timeout=10)

    runner_rpc.recycle_worker_pools("confined to jaato-ws-sess1")
    release.set()
    assert fut.result(timeout=10) == "finished anyway"


def test_recycling_leaves_usable_lanes(runner_rpc):
    """A recycled runner must still be able to serve RPCs."""
    runner_rpc.recycle_worker_pools("confined to jaato-ws-sess1")
    assert runner_rpc._pool.submit(lambda: 1 + 1).result(timeout=10) == 2
    assert runner_rpc._control_pool.submit(lambda: "ok").result(timeout=10) == "ok"


# --------------------------------------------------------------------------
# The bootstrap seam: recycle, then verify, then refuse.
# --------------------------------------------------------------------------

def test_retire_and_verify_recycles_before_it_verifies(tmp_path, monkeypatch):
    """Order is load-bearing.

    Recycling first removes the population the framework created and CAN
    remove, so anything the verification then finds is a thread reached by
    neither lane -- the unknown that must not be certified.
    """
    order: List[str] = []

    def _recycle(reason: str):
        order.append(f"recycle:{reason}")
        return {}

    def _verify(profile, **kwargs):
        order.append(f"verify:{profile}")
        return scan_thread_profiles(
            profile, task_dir=_fake_task_dir(tmp_path, {1: "p (enforce)\n"}),
        )

    monkeypatch.setattr(
        "server.runner.bootstrap.verify_thread_confinement", _verify,
    )
    _retire_and_verify_threads("p", _recycle)

    assert order == ["recycle:confined to p", "verify:p"]


def test_divergence_fails_the_bootstrap(tmp_path, monkeypatch):
    """A log line is not a boundary.

    Proceeding with an ERROR would reproduce the incident exactly: a
    session whose record asserts ``sandbox_mode: apparmor`` while
    in-process tools run outside the kernel boundary, and which #1013's
    ``require`` mode would certify.
    """
    def _verify(profile, **kwargs):
        raise ThreadConfinementDivergence(
            profile, [(99, "unconfined")], scanned=4, route="task_dir",
        )

    monkeypatch.setattr(
        "server.runner.bootstrap.verify_thread_confinement", _verify,
    )
    with pytest.raises(BootstrapError) as excinfo:
        _retire_and_verify_threads("jaato-ws-sess1", None)
    assert excinfo.value.stage == "confine"
    assert "99" in excinfo.value.message


def test_an_unverifiable_proc_does_not_fail_the_bootstrap(tmp_path, monkeypatch):
    """Failing closed on a /proc we merely could not read takes hosts down."""
    def _verify(profile, **kwargs):
        task_dir = _fake_task_dir(tmp_path, {7: "p (enforce)\n"})
        attr = Path(task_dir) / "8" / "attr"
        attr.mkdir(parents=True)
        (attr / "current").mkdir()
        return scan_thread_profiles(profile, task_dir=task_dir)

    monkeypatch.setattr(
        "server.runner.bootstrap.verify_thread_confinement", _verify,
    )
    _retire_and_verify_threads("p", None)   # must not raise


def test_a_failed_recycle_does_not_skip_the_verification(monkeypatch):
    """The recycle is a remedy; the verification is the verdict."""
    verified: List[str] = []

    def _verify(profile, **kwargs):
        verified.append(profile)
        return scan_thread_profiles(profile, task_dir="/nonexistent-1023")

    def _boom(_reason):
        raise RuntimeError("executor construction failed")

    monkeypatch.setattr(
        "server.runner.bootstrap.verify_thread_confinement", _verify,
    )
    _retire_and_verify_threads("p", _boom)
    assert verified == ["p"]


# --------------------------------------------------------------------------
# Structural guards: the wiring cannot be dropped silently.
# --------------------------------------------------------------------------

def _source_of(func) -> str:
    return inspect.getsource(func)


def test_the_transition_path_retires_and_verifies():
    """A fresh ``aa_change_profile`` is followed by the #1023 work."""
    from server.runner.session import _maybe_self_confine

    tree = ast.parse(inspect.getsource(_maybe_self_confine))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_retire_and_verify_threads"
    ]
    assert len(calls) == 2, (
        "_maybe_self_confine must retire-and-verify on BOTH paths that end "
        "with the process confined: the fresh transition and the "
        "already-confined idempotent skip"
    )


def test_the_idempotent_path_retires_and_verifies_too():
    """The reused-slot path is the commonest #1023 case, not an exemption.

    A slot serving session 2 of a cascade under the SAME profile takes the
    idempotent skip -- and a worker created before that slot's FIRST
    bootstrap is unconfined on it.
    """
    from server.runner.session import _maybe_self_confine

    src = _source_of(_maybe_self_confine)
    skip_marker = "skipping redundant self-confine"
    assert skip_marker in src
    tail = src.split(skip_marker, 1)[1]
    # The call must appear before that branch's ``return``.
    head, _, _ = tail.partition("        return")
    assert "_retire_and_verify_threads(" in head


def test_the_rpc_bootstrap_hands_the_session_its_recycle_hook():
    """Without the hook the session can verify but cannot remedy."""
    src = _source_of(RunnerRPC._handle_session_bootstrap)
    assert "recycle_pools=self.recycle_worker_pools" in src


def test_nothing_claims_aa_change_profile_is_process_level():
    """The premise this issue disproved must not be restated in the tree.

    ``bootstrap.py`` used to carry a comment calling the transition
    "process-level", which is the belief that made every downstream check
    look sufficient.
    """
    src = Path(inspect.getfile(verify_thread_confinement)).read_text()
    lowered = src.lower()
    assert "is a process-level transition" not in lowered
    assert "per-task" in lowered
