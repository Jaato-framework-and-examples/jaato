"""The model is told which execution boundary it is under (issue #1012).

The gap this closes: ``establish_containment`` picks one of four tiers, the
choice materially changes what a cell can do, and the model was told none of
it — it discovered the tier by hitting a refusal mid-task and reverse-
engineering the boundary from one error string.  In the reported session that
produced a confident, wrong, three-times-repeated claim that the sandbox
forbids native code in general.

Two surfaces answer, and each covers what the other cannot:

* the plugin's **instruction contribution** — the standing fact, in the system
  prompt, before the first cell;
* the **first ``notebook_execute`` result** from a fresh kernel — the kernel's
  own report, so a respawn under a different posture is visible.

Both render through ``kernel_sandbox.boundary_notice``, which is the single
place a tier's consequences are written down; these tests hold that property
as much as the wording itself, because two texts that can disagree are how the
model ends up believing the wrong one.
"""

import pytest

from jaato_server.shared.plugins.notebook.backends.base import NotebookBackend
from jaato_server.shared.plugins.notebook.backends.subprocess_kernel import (
    SubprocessKernelBackend,
)
from jaato_server.shared.plugins.notebook.kernel_sandbox import (
    BOUNDARY_APPARMOR,
    BOUNDARY_AUDIT,
    BOUNDARY_NONE,
    BOUNDARY_OPT_OUT,
    boundary_notice,
)
from jaato_server.shared.plugins.notebook.plugin import NotebookPlugin
from jaato_server.shared.plugins.notebook.types import (
    ExecutionResult,
    ExecutionStatus,
    NotebookInfo,
)

_APPARMOR = "jaato_server.shared.plugins.notebook.backends.subprocess_kernel.apparmor_enforced_profile"
_WORKSPACE = "jaato_server.shared.plugins.notebook.backends.subprocess_kernel.get_workspace_root"


class TestBoundaryNotice:
    """The one renderer: what a tier means, said in exactly one place."""

    def test_every_tier_has_a_notice(self):
        for kind in (BOUNDARY_APPARMOR, BOUNDARY_AUDIT,
                     BOUNDARY_OPT_OUT, BOUNDARY_NONE):
            assert boundary_notice(kind), f"{kind} says nothing to the model"

    def test_an_unknown_tier_asserts_nothing(self):
        # Inventing a default here would be the second source of truth the
        # whole design avoids: a backend claiming no tier must produce no
        # claim, not a plausible-looking wrong one.
        assert boundary_notice(None) == ()
        assert boundary_notice("some-future-tier") == ()

    def test_the_audit_notice_carries_both_facts_the_failing_session_lacked(self):
        text = " ".join(boundary_notice(BOUNDARY_AUDIT))
        # (1) refused-by-containment is distinguished from not-installed.
        assert "NOT a missing package" in text
        assert "import ctypes" in text
        assert "numpy" in text
        # (2) a subprocess is not audited — the fact that ends the incident.
        assert "subprocess is NOT audited" in text
        assert "`cli`" in text

    def test_the_apparmor_notice_does_not_inherit_the_audit_limitation(self):
        # The AppArmor tier installs NO hook, so telling a model there that
        # ctypes is refused would cost it the scientific stack for nothing.
        text = " ".join(boundary_notice(BOUNDARY_APPARMOR))
        assert "REFUSED" not in text
        assert "import normally" in text


class TestBoundaryKind:
    """The tier is named, by the ladder that already decided it."""

    @pytest.fixture
    def backend(self, tmp_path):
        b = SubprocessKernelBackend()
        b.initialize({"workspace_root": str(tmp_path)})
        return b

    def test_apparmor_wins(self, backend, monkeypatch):
        monkeypatch.setattr(_APPARMOR, lambda: "jaato-ws-x//child")
        assert backend.boundary_kind() == BOUNDARY_APPARMOR

    def test_audit_when_unconfined_with_a_workspace(self, backend, monkeypatch):
        monkeypatch.setattr(_APPARMOR, lambda: None)
        assert backend.boundary_kind() == BOUNDARY_AUDIT

    def test_opt_out_outranks_the_audit_hook(self, backend, monkeypatch):
        monkeypatch.setattr(_APPARMOR, lambda: None)
        backend.initialize({"allow_uncontained_exec": True})
        assert backend.boundary_kind() == BOUNDARY_OPT_OUT

    def test_no_workspace_is_no_boundary(self, monkeypatch):
        monkeypatch.setattr(_APPARMOR, lambda: None)
        monkeypatch.setattr(_WORKSPACE, lambda: None)
        assert SubprocessKernelBackend().boundary_kind() == BOUNDARY_NONE

    def test_execution_boundary_is_worded_from_the_same_ladder(
            self, backend, monkeypatch):
        """Prose and tier must not be derived twice — that is how they drift.

        Before #1012 ``execution_boundary`` wrote the ladder out inline and
        nothing could ask for the answer by name.
        """
        monkeypatch.setattr(_APPARMOR, lambda: None)
        for kind, allowed, fragment in (
            (BOUNDARY_AUDIT, True, "audit-hook workspace containment"),
            (BOUNDARY_OPT_OUT, True, "opt-out"),
        ):
            backend._allow_uncontained = (kind == BOUNDARY_OPT_OUT)
            assert backend.boundary_kind() == kind
            ok, description = backend.execution_boundary()
            assert ok is allowed
            assert fragment in description

    def test_a_backend_that_names_no_tier_says_so(self):
        # The base default. A backend whose containment is not one of the
        # four tiers must return None rather than the nearest-looking one.
        class Unannounced(NotebookBackend):
            capabilities = None

            def initialize(self, config=None): ...
            def create_notebook(self, name, gpu_enabled=False): ...
            def execute(self, notebook_id, code, timeout_seconds=None): ...
            def get_execution_status(self, notebook_id): ...
            def get_variables(self, notebook_id): ...
            def reset_notebook(self, notebook_id): ...
            def delete_notebook(self, notebook_id): ...
            def list_notebooks(self): return []
            def shutdown(self): ...
            def is_available(self): return True

        assert Unannounced().boundary_kind() is None


class TestInstructionContribution:
    """Shape (1): the standing fact, in the system prompt."""

    @pytest.fixture
    def plugin(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_APPARMOR, lambda: None)
        p = NotebookPlugin()
        p.initialize({"workspace_root": str(tmp_path), "backend": "subprocess"})
        yield p
        p.shutdown()

    def test_the_active_tier_reaches_the_system_instructions(self, plugin):
        instructions = plugin.get_system_instructions()
        assert "Notebook execution boundary: audit" in instructions
        assert "subprocess is NOT audited" in instructions

    def test_only_the_active_tier_is_paid_for(self, plugin):
        """This block lands in the prompt-cache prefix on EVERY request.

        Rendering a table of all four tiers would charge every session for
        three boundaries it is not under, so the other tiers' distinctive
        lines must be absent.
        """
        instructions = plugin.get_system_instructions()
        assert "kernel-enforced AppArmor profile" not in instructions
        assert "NO filesystem boundary" not in instructions
        assert "cells are refused outright" not in instructions

    def test_the_block_is_short_enough_to_pay_for(self, plugin):
        block = plugin._boundary_instruction_block()
        assert 0 < len(block) < 1200, (
            "the boundary block is paid on every request; keep it to the "
            "tier and its two practical consequences")

    def test_a_backend_naming_no_tier_contributes_nothing(self, plugin):
        plugin._backends["subprocess"].boundary_kind = lambda: None
        assert plugin._boundary_instruction_block() == ""

    def test_it_renders_what_the_backend_says_not_a_fixed_string(
            self, plugin, monkeypatch):
        monkeypatch.setattr(_APPARMOR, lambda: "jaato-ws-y//child")
        instructions = plugin.get_system_instructions()
        assert "Notebook execution boundary: apparmor" in instructions
        assert "import normally" in instructions


class TestFirstResultAnnouncement:
    """Shape (2): the KERNEL's own report, once per kernel."""

    @pytest.fixture
    def plugin(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_APPARMOR, lambda: None)
        p = NotebookPlugin()
        p.initialize({"workspace_root": str(tmp_path), "backend": "subprocess"})
        yield p
        p.shutdown()

    def _stub_backend(self, plugin, kinds):
        """Replace execute() with one that yields ``kinds`` in order."""
        backend = plugin._backends["subprocess"]
        info = NotebookInfo(notebook_id="nb1", name="n", backend="subprocess")
        remaining = list(kinds)
        backend.list_notebooks = lambda: [info]
        backend.execute = lambda *a, **k: ExecutionResult(
            status=ExecutionStatus.COMPLETED, execution_count=1,
            boundary_kind=remaining.pop(0))
        plugin._current_notebook_id = "nb1"
        return backend

    def test_the_first_result_carries_the_boundary(self, plugin):
        self._stub_backend(plugin, [BOUNDARY_AUDIT])
        result = plugin._execute_code({"code": "1"})
        announced = result["execution_boundary"]
        assert announced["boundary"] == BOUNDARY_AUDIT
        assert announced["notes"] == list(boundary_notice(BOUNDARY_AUDIT))

    def test_later_results_do_not_repeat_it(self, plugin):
        self._stub_backend(plugin, [BOUNDARY_AUDIT, None, None])
        assert "execution_boundary" in plugin._execute_code({"code": "1"})
        assert "execution_boundary" not in plugin._execute_code({"code": "2"})
        assert "execution_boundary" not in plugin._execute_code({"code": "3"})

    def test_a_respawn_under_a_different_posture_is_visible(self, plugin):
        """The case the system prompt cannot cover.

        The prompt states the tier resolved when the session came up. A
        kernel that died and respawned somewhere weaker re-announces, and it
        reports what the KERNEL established rather than what the daemon
        expected — so the contradiction is visible in the result the model
        is reading anyway.
        """
        self._stub_backend(plugin, [BOUNDARY_APPARMOR, None, BOUNDARY_AUDIT])
        first = plugin._execute_code({"code": "1"})
        assert first["execution_boundary"]["boundary"] == BOUNDARY_APPARMOR
        assert "execution_boundary" not in plugin._execute_code({"code": "2"})
        after_respawn = plugin._execute_code({"code": "3"})
        assert after_respawn["execution_boundary"]["boundary"] == BOUNDARY_AUDIT


class TestAnnouncementIsConsumedOncePerKernel:
    """The latch itself, at the backend seam that owns it."""

    def test_armed_by_spawn_and_consumed_once(self):
        class _FakeKernel:
            def __init__(self):
                self.announce_boundary = True
                self.info = NotebookInfo(
                    notebook_id="n", name="n", backend="subprocess",
                    boundary_kind=BOUNDARY_AUDIT)

        kernel = _FakeKernel()
        take = SubprocessKernelBackend._consume_boundary_announcement
        assert take(kernel) == BOUNDARY_AUDIT
        assert take(kernel) is None
        assert take(kernel) is None

    def test_a_kernel_that_reported_no_tier_announces_nothing(self):
        class _FakeKernel:
            def __init__(self):
                self.announce_boundary = True
                self.info = NotebookInfo(
                    notebook_id="n", name="n", backend="subprocess")

        assert SubprocessKernelBackend._consume_boundary_announcement(
            _FakeKernel()) is None


class TestAgainstALiveKernel:
    """End to end: the tier the kernel reports on READY reaches the result.

    The rest of this module stubs the backend so the plugin-side wiring can
    be asserted precisely.  This one spawns a real kernel subprocess, because
    the READY frame is where the tier comes from and #1012's rule is that
    nothing invents a second one.
    """

    def test_a_real_kernel_reports_its_own_tier(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_APPARMOR, lambda: None)
        monkeypatch.delenv("JAATO_NOTEBOOK_ALLOW_UNCONTAINED_EXEC",
                           raising=False)
        plugin = NotebookPlugin()
        plugin.initialize({"workspace_root": str(tmp_path),
                           "backend": "subprocess"})
        try:
            result = plugin._execute_code({"code": "print(6 * 7)"})
            assert result["execution_boundary"]["boundary"] == BOUNDARY_AUDIT
            assert "42" in result["output"]
            # The kernel's own answer, recorded from READY rather than
            # re-derived daemon-side.
            info = plugin._backends["subprocess"].list_notebooks()[0]
            assert info.boundary_kind == BOUNDARY_AUDIT
        finally:
            plugin.shutdown()

    def test_the_refusal_a_cell_hits_names_the_tier_and_a_way_through(
            self, tmp_path, monkeypatch):
        """#1011 ask 3, against the real hook rather than the check alone.

        A refusal naming the tier is also what makes a silent degrade
        visible: this hook is installed on the audit tier and on no other,
        so an operator who believes AppArmor is enforced can read the tier
        off the error their agent just reported.
        """
        monkeypatch.setattr(_APPARMOR, lambda: None)
        monkeypatch.delenv("JAATO_NOTEBOOK_ALLOW_UNCONTAINED_EXEC",
                           raising=False)
        plugin = NotebookPlugin()
        plugin.initialize({"workspace_root": str(tmp_path),
                           "backend": "subprocess"})
        try:
            result = plugin._execute_code({"code": "import ctypes"})
            text = result.get("output", "") + (result.get("error") or "")
            assert "audit tier" in text
            assert "NOT a missing package" in text
            assert "`cli`" in text
        finally:
            plugin.shutdown()
