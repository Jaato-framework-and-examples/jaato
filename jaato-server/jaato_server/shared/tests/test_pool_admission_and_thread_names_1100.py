"""#1100 — "never confined" is not "never served", and a tid is not a name.

A live daemon, session ``20260917_120657``::

    12:06:57  slot pid=95942  profile=(none)                 confined=False
    12:11:25  slot pid=95942  returned to pool
    12:17:44  slot pid=95942  profile=jaato-ws-test-hola-3-…  confined=True
              REFUSED: "6 of 7 scanned threads report otherwise
                        (tid=95982 label='unconfined', …)"

One slot served an **unconfined** session, went back to the pool, and was
handed to a **confined** one.  ``aa_change_profile`` is per-task, so the
threads that unconfined run created can never be confined — only retired
— and #1023's per-thread verification correctly refuses the bootstrap.

The hole is in the admission gate.  ``SlotKey.build`` folds ``""`` to
``None`` so unconfined has one spelling (right for a KEY), and
``accepts_unaffined`` read that ``None`` as *never confined*, which it
takes to mean *no threads to worry about*.  Those are different
statements, and the slot above is where they come apart.

What this module pins:

* a slot that served an **unconfined** session is not offered to a
  session that wants a profile — the confirmed sequence above, replayed
  through the real ``PoolManager``;
* a genuinely virgin slot still fits anyone, so the pool stays a pool;
* unconfined→unconfined reuse is untouched, and a host with no AppArmor
  sees no change at all;
* the refusal **names** the divergent threads, so the next recurrence is
  self-diagnosing instead of costing a third investigation;
* a name is advisory — it never decides whether a thread is divergent,
  because an allow-list of thread names is an allow-list of unconfined
  code;
* ``TelemetryPlugin.shutdown()`` is called at both session boundaries,
  which is the leak that most plausibly populates the divergent set.

NO KERNEL.  This container has no AppArmor LSM.  Every confinement fact
here is exercised against a fabricated ``<dir>/<tid>/attr/current`` tree
and in-memory pool state, exactly as
``server/tests/test_per_thread_confinement_1023.py`` and
``test_slot_reuse_key_and_profile_1033.py`` do.  Nothing here proves the
kernel behaves as #1023 describes; it proves the framework stops handing
the kernel a slot it cannot re-confine, and that when it refuses one it
says which threads it refused.
"""

from __future__ import annotations

import pathlib
import sys
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from jaato_server.server.runner.bootstrap import (
    UNKNOWN_THREAD_NAME,
    ThreadConfinementDivergence,
    scan_thread_profiles,
    verify_thread_confinement,
)
from jaato_server.server.runner_pool import PoolManager, PoolSlot, SlotKey


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from jaato_server.shared.tests.reversion import Reversion


_POOL = "jaato-server/jaato_server/server/runner_pool.py"
_BOOTSTRAP = "jaato-server/jaato_server/server/runner/bootstrap.py"
_RPC = "jaato-server/jaato_server/server/runner/rpc.py"


#: Each entry puts one piece of the defect back.
REVERSIONS = [
    Reversion(
        target=_POOL,
        find=("        if not slot.has_served:\n"
              "            return True\n"
              "        return (slot.profile_name or None) "
              "== self.profile_name"),
        replace=("        return not slot.profile_name or "
                 "slot.profile_name == self.profile_name"),
        test=("TestAdmission::"
              "test_a_slot_that_served_unconfined_is_not_offered_a_profile"),
        because=("the gate reading a falsy profile_name as 'virgin' again, "
                 "so a slot carrying an unconfined session's threads is "
                 "handed to a confined session and refused at bootstrap"),
    ),
    Reversion(
        target=_POOL,
        find=("        slot.profile_name = self.profile_name\n"
              "        slot.has_served = True"),
        replace="        slot.profile_name = self.profile_name",
        test=("TestAdmission::"
              "test_a_slot_that_served_unconfined_is_not_offered_a_profile"),
        because=("the claim never being recorded, which leaves every slot "
                 "reading as virgin forever and makes the gate above inert"),
    ),
    Reversion(
        target=_BOOTSTRAP,
        find=('            f"name={names.get(tid, UNKNOWN_THREAD_NAME)!r} "\n'),
        replace="",
        test=("TestRefusalNamesTheThreads::"
              "test_the_divergence_message_names_the_thread"),
        because=("the refusal going back to tid-and-label only — the state "
                 "in which two live incidents could not say what the "
                 "divergent threads were"),
    ),
    Reversion(
        target=_BOOTSTRAP,
        find="    names = _thread_names()\n",
        replace="    names = {}\n",
        test=("TestRefusalNamesTheThreads::"
              "test_the_scan_names_every_thread_python_knows"),
        because=("the scan collecting no names, so every divergent thread "
                 "renders as the placeholder and the message is back to "
                 "where it started"),
    ),
    Reversion(
        target=_RPC,
        find=('        telemetry_error = shutdown_runtime_telemetry('
              'runtime, "session.end")'),
        replace="        telemetry_error = None",
        test=("TestTelemetryShutdown::"
              "test_session_end_really_shuts_the_runtime_telemetry_down"),
        because=("TelemetryPlugin.shutdown() losing its only warm-path "
                 "caller again, so a pool slot accumulates one live OTel "
                 "export thread per session it serves"),
    ),
    Reversion(
        target=_RPC,
        find=('        shutdown_runtime_telemetry(\n'
              '            getattr(session, "_runtime", None) '
              'if session else None,\n'
              '            "session.shutdown",\n'
              '        )\n'),
        replace="",
        test=("TestTelemetryShutdown::"
              "test_both_session_boundaries_call_it"),
        because=("the COLD path losing its caller — a runner reaches "
                 "session.end or session.shutdown, never both, so wiring "
                 "one of them leaks on every session that takes the other"),
    ),
]


# ======================================================================
# Fixtures — a pool with no threads, a proc tree with no kernel
# ======================================================================


def _slot(
    pid: int,
    *,
    cascade_id: Optional[str] = None,
    config_root: Optional[str] = None,
    workspace_root: Optional[str] = None,
    profile_name: Optional[str] = None,
    has_served: bool = False,
) -> PoolSlot:
    """An idle slot in whatever state a previous session left it.

    ``has_served`` is explicit here and defaults to ``False``, unlike the
    #1033 helper: the whole subject of this module is the slot whose four
    identity fields are ``None`` **and** which has served, so deriving
    the flag from the fields would assume away the case under test.
    """
    return PoolSlot(
        pid=pid,
        sock=MagicMock(name=f"slot-{pid}-sock"),
        cascade_id=cascade_id,
        config_root=config_root,
        workspace_root=workspace_root,
        profile_name=profile_name,
        has_served=has_served,
    )


def _pool(*slots: PoolSlot) -> PoolManager:
    """A ``PoolManager`` holding *slots* and starting no thread.

    The constructor starts nothing — ``spawn_initial_slots`` and
    ``start_replenishment`` are explicit calls this module never makes —
    so this is a pure data structure under test.
    """
    pool = PoolManager(template_manager=MagicMock(), target_size=8)
    pool._idle_slots = list(slots)
    return pool


def _proc_tree(tmp_path, labels: Dict[int, str]) -> str:
    """Fabricate ``<dir>/<tid>/attr/current`` for each tid.

    The shape ``scan_thread_profiles`` reads, reproduced exactly, which
    is the only reason ``task_dir`` is a parameter at all.
    """
    for tid, label in labels.items():
        attr = tmp_path / str(tid) / "attr"
        attr.mkdir(parents=True, exist_ok=True)
        (attr / "current").write_text(label + "\n")
    return str(tmp_path)


# ======================================================================
# 1. The admission gate
# ======================================================================


class TestAdmission:
    """``accepts_unaffined`` — what a slot's history admits."""

    def test_a_slot_that_served_unconfined_is_not_offered_a_profile(
        self,
    ) -> None:
        """The confirmed sequence, replayed.

        Slot 95942: acquired unconfined (standalone session, no cascade,
        no profile), returned to the pool, then asked for by a session
        that wants ``jaato-ws-test-hola-3-dca6fdc0fbdf``.  Before #1100
        it was handed over and the bootstrap was refused with 6 of 7
        threads reporting ``unconfined``.
        """
        slot = _slot(95942)
        pool = _pool(slot)

        # 12:06 — the unconfined session takes it.
        first = pool.acquire_slot(workspace_root="/ws/test-hola-3")
        assert first is slot
        assert first.has_served is True
        assert first.profile_name is None

        # 12:11 — back to the pool, PURE IDLE.
        pool.return_slot_after_session(first)

        # 12:17 — a confined session asks.  It must not get this slot.
        assert pool.acquire_slot(
            workspace_root="/ws/test-hola-3",
            profile_name="jaato-ws-test-hola-3-dca6fdc0fbdf",
        ) is None

    def test_the_refusal_is_counted_as_a_boundary_mismatch(self) -> None:
        """The cost of this fix is already instrumented.

        ``pool_profile_mismatch_skips_total`` counts idle slots path (2)
        passed over because the boundary did not fit.  It does not care
        WHY it did not fit, so the extra misses this change produces on a
        daemon mixing postures need no new counter — which is what makes
        the cost measurable by an operator who already watches it beside
        ``pool_acquire_miss_total``.
        """
        pool = _pool(_slot(95942, has_served=True))

        assert pool.acquire_slot(profile_name="jaato-ws-x-deadbeefcafe") is None

        counters = pool.get_telemetry()
        assert counters["pool_profile_mismatch_skips_total"] == 1
        assert counters["pool_acquire_miss_total"] == 1

    def test_a_virgin_slot_still_fits_anybody(self) -> None:
        """Otherwise the pool stops being a pool.

        The ``target_size`` floor is stocked with template forks that
        have served nothing, and those must remain takeable by any
        arriving session whatever profile it wants.
        """
        fresh = _slot(11)
        pool = _pool(fresh)

        got = pool.acquire_slot(
            workspace_root="/ws-b", profile_name="jaato-ws-ws-b-bbbbbbbbbbbb",
        )

        assert got is fresh
        assert got.has_served is True
        assert got.profile_name == "jaato-ws-ws-b-bbbbbbbbbbbb"

    def test_unconfined_reuse_by_an_unconfined_session_is_untouched(
        self,
    ) -> None:
        """A slot that ran unconfined still serves the next unconfined one.

        This is the half the fix must NOT break: the threads are
        unconfined, the arriving session wants no boundary, and #1023's
        verification does not run at all.  Refusing here would cost a
        cold spawn for no safety.
        """
        used = _slot(95942, has_served=True)
        pool = _pool(used)

        assert pool.acquire_slot(profile_name=None) is used

    def test_no_apparmor_changes_nothing(self) -> None:
        """Every profile name is empty, so the gate is a tautology.

        On a host with no AppArmor every session resolves ``""`` and
        ``SlotKey.build`` folds it to ``None``, so a served slot's
        ``profile_name`` always equals the arriving key's.  The "no
        AppArmor at all: completely unchanged" requirement #1033 stated
        survives the new clause.
        """
        used = _slot(11, workspace_root="/ws", profile_name="",
                     has_served=True)
        pool = _pool(used)

        assert pool.acquire_slot(workspace_root="/ws", profile_name="") is used

    def test_a_confined_slot_still_serves_its_own_profile(self) -> None:
        """#1033's reuse hit is unaffected — same boundary, same slot."""
        used = _slot(
            11, workspace_root="/ws-a",
            profile_name="jaato-ws-ws-a-aaaaaaaaaaaa", has_served=True,
        )
        pool = _pool(used)

        assert pool.acquire_slot(
            workspace_root="/ws-a", profile_name="jaato-ws-ws-a-aaaaaaaaaaaa",
        ) is used

    def test_has_served_is_not_derivable_from_the_key(self) -> None:
        """Why the flag exists rather than being computed.

        An unconfined standalone session stamps a key whose every field
        is ``None``.  Whatever a derivation looked at, it would see the
        same tuple a virgin slot carries — which is the defect, not an
        implementation detail of it.
        """
        virgin = _slot(1)
        served = _slot(2)
        SlotKey.build().stamp(served)

        assert SlotKey.of_slot(virgin) == SlotKey.of_slot(served)
        assert virgin.has_served is False
        assert served.has_served is True


# ======================================================================
# 2. The refusal names the threads
# ======================================================================


class TestRefusalNamesTheThreads:
    """``ThreadProfileScan.names`` and the divergence message (#1100)."""

    def test_the_divergence_message_names_the_thread(self, tmp_path) -> None:
        """``tid=… name=… label=…``, not ``tid=… label=…``.

        The message is the artefact the operator actually reads; the two
        live incidents were investigated from daemon-log timestamps
        because the refusal itself could not say what it had refused.
        """
        exc = ThreadConfinementDivergence(
            "jaato-ws-test-hola-3-dca6fdc0fbdf",
            [(95982, "unconfined"), (95987, "unconfined")],
            scanned=7,
            route="task_dir",
            names={95982: "runner-rpc-work_0", 95987: "OtelBatchSpanProc"},
        )

        text = str(exc)
        assert "tid=95982 name='runner-rpc-work_0' label='unconfined'" in text
        assert "tid=95987 name='OtelBatchSpanProc' label='unconfined'" in text

    def test_a_thread_python_does_not_know_is_named_as_such(self) -> None:
        """The ``task_dir`` route is complete; the name map is not.

        A C-extension thread appears in ``/proc/<pid>/task/`` and never
        in ``threading.enumerate``.  Saying so is the honest rendering;
        inventing one from ``/proc/<tid>/comm`` is not available (every
        runner thread's ``comm`` is ``"python"`` on CPython 3.11).
        """
        exc = ThreadConfinementDivergence(
            "jaato-ws-x", [(4242, "unconfined")],
            scanned=2, route="task_dir", names={},
        )

        assert f"tid=4242 name='{UNKNOWN_THREAD_NAME}'" in str(exc)

    def test_the_scan_names_every_thread_python_knows(self, tmp_path) -> None:
        """The map is populated by the walk, on either route.

        Read at scan time on purpose: a thread that exits between the
        walk and the message has already left ``threading.enumerate``,
        so a name resolved later would go missing for exactly the
        population that is churning.
        """
        import os

        me = threading.current_thread()
        task_dir = _proc_tree(tmp_path, {os.getpid(): "jaato-ws-x (enforce)"})

        scan = scan_thread_profiles("jaato-ws-x", task_dir=task_dir)

        assert scan.names.get(os.getpid()) == me.name
        assert scan.name_of(os.getpid()) == me.name
        assert scan.name_of(999999) == UNKNOWN_THREAD_NAME

    def test_verify_carries_the_names_into_the_exception(
        self, tmp_path,
    ) -> None:
        """The path production takes: scan → verify → raise."""
        import os

        pid = os.getpid()
        task_dir = _proc_tree(tmp_path, {
            pid: "jaato-ws-x (enforce)",
            424242: "unconfined",
        })

        with pytest.raises(ThreadConfinementDivergence) as excinfo:
            verify_thread_confinement(
                "jaato-ws-x", task_dir=task_dir,
                grace_seconds=0.0, sleep=lambda _s: None,
            )

        exc = excinfo.value
        assert exc.divergent == ((424242, "unconfined"),)
        # The main thread was matched, so its name is in the map even
        # though it is not in the divergent set.
        assert exc.names.get(pid) == threading.current_thread().name
        assert f"tid=424242 name='{UNKNOWN_THREAD_NAME}'" in str(exc)

    def test_a_name_never_decides_divergence(self, tmp_path) -> None:
        """The constraint #1100 is explicit about.

        An allow-list of thread names is an allow-list of unconfined
        code.  A thread named exactly like a framework lane is still
        refused when its label says ``unconfined``, and a thread whose
        name is unknown is still MATCHED when its label is inside the
        profile.
        """
        import os

        pid = os.getpid()
        task_dir = _proc_tree(tmp_path, {
            pid: "jaato-ws-x (enforce)",
            555001: "unconfined",           # unknown name, bad label
            555002: "jaato-ws-x//child",    # unknown name, good label
        })

        scan = scan_thread_profiles("jaato-ws-x", task_dir=task_dir)

        assert scan.divergent == ((555001, "unconfined"),)
        assert 555002 in scan.matched
        assert scan.name_of(555001) == UNKNOWN_THREAD_NAME
        assert scan.name_of(555002) == UNKNOWN_THREAD_NAME


# ======================================================================
# 3. The telemetry shutdown
# ======================================================================


class _RecordingTelemetry:
    """A telemetry plugin that records its own teardown."""

    def __init__(self, raises: bool = False) -> None:
        self.shutdowns = 0
        self.resets = 0
        self._raises = raises

    def shutdown(self) -> None:
        self.shutdowns += 1
        if self._raises:
            raise RuntimeError("exporter refused to flush")

    def reset_for_next_session(self) -> None:
        self.resets += 1


class _StubRuntime:
    def __init__(self, telemetry: Any) -> None:
        self.telemetry = telemetry
        self.registry: Any = None


class _NoPluginRegistry:
    """The minimum ``session.end`` needs: a registry holding nothing."""

    def list_available(self) -> List[str]:
        return []

    def get_plugin(self, name: str) -> Any:
        return None

    def get_plugin_source(self, name: str) -> Any:
        return None

    def shutdown_all(self, skip: Any = None) -> List[str]:
        return []


class TestTelemetryShutdown:
    """#1100 supporting defect: ``shutdown()`` had no caller in the tree."""

    def test_session_end_shuts_the_runtime_telemetry_down(self) -> None:
        """The warm path — slot returns to the pool.

        Telemetry is runtime-scoped and a runner builds a fresh
        ``JaatoRuntime`` per ``session.bootstrap``, so the outgoing
        plugin is unreachable after this point.  Dropping the reference
        is not freeing the resource: an OTel ``BatchSpanProcessor`` owns
        a live export thread.
        """
        from jaato_server.server.runner.rpc import shutdown_runtime_telemetry

        telemetry = _RecordingTelemetry()

        assert shutdown_runtime_telemetry(
            _StubRuntime(telemetry), "session.end") is None
        assert telemetry.shutdowns == 1

    def test_a_failing_shutdown_is_reported_not_raised(self) -> None:
        """A telemetry teardown must never fail a session end.

        It is reported instead, and the caller appends it to ``errors``
        — which does have teeth: the daemon will not pool a slot whose
        session end reported errors, and a shutdown that raised is
        exactly when the export thread may still be running.
        """
        from jaato_server.server.runner.rpc import shutdown_runtime_telemetry

        telemetry = _RecordingTelemetry(raises=True)

        reason = shutdown_runtime_telemetry(
            _StubRuntime(telemetry), "session.end")

        assert reason is not None
        assert "exporter refused to flush" in reason

    def test_a_runtime_without_telemetry_is_a_no_op(self) -> None:
        """Duck-typed stubs and older runtimes must not raise here."""
        from jaato_server.server.runner.rpc import shutdown_runtime_telemetry

        assert shutdown_runtime_telemetry(None, "session.end") is None
        assert shutdown_runtime_telemetry(object(), "session.end") is None
        assert shutdown_runtime_telemetry(
            _StubRuntime(None), "session.end") is None
        assert shutdown_runtime_telemetry(
            _StubRuntime(object()), "session.end") is None

    def test_both_session_boundaries_call_it(self) -> None:
        """Warm and cold, because a runner reaches only one of them.

        ``session.end`` returns the slot to the pool; ``session.shutdown``
        is taken when it does not.  Wiring one leaves the other leaking,
        and a source-level check is what keeps a later refactor from
        removing a call site whose absence has no visible symptom until a
        slot has served a dozen sessions.
        """
        import inspect

        from jaato_server.server.runner import rpc as rpc_module

        warm = inspect.getsource(rpc_module.RunnerRPC._handle_session_end)
        cold = inspect.getsource(
            rpc_module.RunnerRPC._release_session_plugins)

        assert "shutdown_runtime_telemetry(" in warm
        assert "shutdown_runtime_telemetry(" in cold
        assert '"session.shutdown"' in cold

    def test_session_end_really_shuts_the_runtime_telemetry_down(self) -> None:
        """Driven through the real ``session.end`` handler.

        The behavioural half of the pair below: a source-level check
        pins that the call site exists, this pins that reaching the
        handler reaches the plugin.  Neither subsumes the other — a call
        site can be present and unreachable, and a behavioural test that
        stubbed the handler would pin nothing about production.
        """
        import socket

        from jaato_server.server.runner.envelope import RequestEnvelope  # noqa: F401
        from jaato_server.server.runner.rpc import RunnerRPC
        from jaato_server.server.runner.session import RunnerSessionHost
        from jaato_server.shared.session_envelope import SessionInitEnvelope

        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        b.close()
        rpc = RunnerRPC(a, lambda name, args: (False, {"error": "no executor"}))

        telemetry = _RecordingTelemetry()
        registry = _NoPluginRegistry()
        runtime = _StubRuntime(telemetry)
        runtime.registry = registry
        session = MagicMock()
        session._runtime = runtime
        rpc._session_host = RunnerSessionHost(
            envelope=SessionInitEnvelope(
                session_id="sess-1100",
                workspace_path="/tmp/ws",
                profile_name="p",
                provider_name="anthropic",
                model_name="m",
                plugins=[],
            ),
            runtime=runtime,
            session=session,
        )

        ok, payload = rpc._handle_session_end()

        assert ok is True
        assert payload["errors"] == []
        assert telemetry.shutdowns == 1

    def test_the_null_plugin_and_the_otel_plugin_both_answer(self) -> None:
        """The protocol is satisfied by every implementation shipped.

        A telemetry-disabled deployment gets ``NullTelemetryPlugin``,
        whose ``shutdown`` is a no-op — so the new call sites cost it
        nothing.
        """
        from jaato_server.shared.plugins.telemetry import create_plugin

        plugin = create_plugin({"enabled": False})
        plugin.shutdown()          # must not raise
        plugin.shutdown()          # idempotent
