"""Notebook cells are contained to the workspace (issue #710).

Three layers, tested where each one lives:

1. :class:`ContainmentPolicy` — the verdict, unit-tested in this process.
2. ``NotebookBackend.execution_boundary`` — what each backend declares, and the
   base class's refusal for a backend that declares nothing.
3. The kernel — driven end to end as a real subprocess, because the audit hook
   can only be installed in a process whose whole job is running cells (it
   cannot be removed once added, so it must never be installed here).

**The staging trap** (see ``shared/plugins/CLAUDE.md``): on Linux ``tmp_path``
is itself under ``/tmp``, which the sandbox allows by default, so an escape
test whose "outside" target lives there passes for the wrong reason.  Policy
tests therefore substitute ``SYSTEM_TEMP_PATHS`` with a directory under
``tmp_path``; the subprocess tests use ``/etc/hostname``, which is the path the
issue actually measured.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from ..backends.base import NotebookBackend
from ..backends.local import LocalJupyterBackend, INPROCESS_OPT_IN_ENV
from ..backends.subprocess_kernel import SubprocessKernelBackend
from ..kernel_sandbox import (
    BOUNDARY_APPARMOR,
    BOUNDARY_AUDIT,
    BOUNDARY_NONE,
    BOUNDARY_OPT_OUT,
    UNCONTAINED_OPT_IN_ENV,
    ContainmentPolicy,
    NotebookContainmentError,
    _check_dlopen,
    _check_open,
    _check_spawn,
    _check_system,
    _spawn_path_like,
    _write_intent,
    establish_containment,
)
from ..types import ExecutionStatus, OutputType

# The repository's ``jaato-server`` directory, so a spawned kernel imports the
# ``shared`` package under test rather than whichever one is installed in the
# interpreter (they differ when the tree is checked out as a git worktree).
_SERVER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))))


@pytest.fixture
def isolated_temp(monkeypatch, tmp_path):
    """Point the sandbox's ``/tmp`` allowance at a directory under ``tmp_path``.

    Without this the fixture's own "outside the workspace" paths are inside the
    temp allowance and every escape assertion passes vacuously.
    """
    fake_temp = tmp_path / "systmp"
    fake_temp.mkdir()
    monkeypatch.setattr(
        "shared.plugins.sandbox_utils.SYSTEM_TEMP_PATHS", [str(fake_temp)])
    return fake_temp


@pytest.fixture
def policy(tmp_path, isolated_temp):
    """A policy contained to ``tmp_path/ws``, with nothing else granted."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return ContainmentPolicy(str(workspace), read_roots=[str(tmp_path / "libs")])


class TestPolicyVerdicts:
    """What :meth:`ContainmentPolicy.allows` answers, and why."""

    def test_workspace_read_and_write_allowed(self, policy, tmp_path):
        inside = str(tmp_path / "ws" / "notes.txt")
        assert policy.allows(inside, "read") is True
        assert policy.allows(inside, "write") is True

    def test_outside_refused_both_modes(self, policy, tmp_path):
        outside = str(tmp_path / "elsewhere" / "secret.txt")
        assert policy.allows(outside, "read") is False
        assert policy.allows(outside, "write") is False

    def test_traversal_out_of_workspace_refused(self, policy, tmp_path):
        assert policy.allows(str(tmp_path / "ws" / ".." / "x"), "read") is False

    def test_symlink_out_of_workspace_refused(self, policy, tmp_path):
        target = tmp_path / "elsewhere"
        target.mkdir()
        (target / "secret.txt").write_text("s")
        link = tmp_path / "ws" / "link"
        link.symlink_to(target)
        assert policy.allows(str(link / "secret.txt"), "read") is False

    def test_temp_allowed_like_cli(self, policy, isolated_temp):
        scratch = str(isolated_temp / "scratch.txt")
        assert policy.allows(scratch, "read") is True
        assert policy.allows(scratch, "write") is True

    def test_read_roots_are_read_only(self, tmp_path, isolated_temp):
        libs = tmp_path / "libs"
        libs.mkdir()
        pol = ContainmentPolicy(str(tmp_path / "ws"), read_roots=[str(libs)])
        assert pol.allows(str(libs / "mod.py"), "read") is True
        assert pol.allows(str(libs / "mod.py"), "write") is False

    def test_write_roots_allow_both(self, tmp_path, isolated_temp):
        venv = tmp_path / "venv"
        venv.mkdir()
        pol = ContainmentPolicy(str(tmp_path / "ws"), write_roots=[str(venv)])
        assert pol.allows(str(venv / "lib" / "x"), "write") is True

    def test_no_workspace_refuses_everything(self, tmp_path):
        pol = ContainmentPolicy(None)
        assert pol.allows(str(tmp_path / "anything"), "read") is False


class TestPolicyGrants:
    """``sandbox add`` / ``sandbox deny``, refreshed per cell."""

    def test_granted_read_allowed_write_still_refused(self, policy, tmp_path):
        granted = tmp_path / "corpus"
        granted.mkdir()
        policy.set_extra_allows(read=[str(granted)])
        assert policy.allows(str(granted / "a.txt"), "read") is True
        assert policy.allows(str(granted / "a.txt"), "write") is False

    def test_granted_write_allows_both(self, policy, tmp_path):
        granted = tmp_path / "corpus"
        granted.mkdir()
        policy.set_extra_allows(write=[str(granted)])
        assert policy.allows(str(granted / "a.txt"), "write") is True

    def test_revocation_takes_effect(self, policy, tmp_path):
        granted = tmp_path / "corpus"
        granted.mkdir()
        policy.set_extra_allows(read=[str(granted)])
        assert policy.allows(str(granted / "a.txt"), "read") is True
        policy.set_extra_allows()          # the operator revoked it
        assert policy.allows(str(granted / "a.txt"), "read") is False

    def test_denial_outranks_the_workspace_itself(self, policy, tmp_path):
        secret = tmp_path / "ws" / "secret"
        secret.mkdir()
        policy.set_extra_allows(deny=[str(secret)])
        assert policy.allows(str(secret / "k.pem"), "read") is False
        assert policy.allows(str(tmp_path / "ws" / "ok.txt"), "read") is True


class TestAuditEventChecks:
    """The per-event handlers, called directly (never via an installed hook)."""

    def test_open_write_mode_detected_from_mode_string(self):
        assert _write_intent("w", None) is True
        assert _write_intent("rb", None) is False
        assert _write_intent("r+", None) is True

    def test_open_write_mode_detected_from_os_flags(self):
        assert _write_intent(None, os.O_RDONLY) is False
        assert _write_intent(None, os.O_WRONLY | os.O_CREAT) is True

    def test_open_outside_raises(self, policy, tmp_path):
        with pytest.raises(NotebookContainmentError):
            _check_open(policy, (str(tmp_path / "elsewhere" / "x"), "r", 0))

    def test_open_of_a_descriptor_is_not_a_path(self, policy):
        _check_open(policy, (3, None, os.O_RDONLY))     # must not raise

    def test_spawn_argument_outside_raises(self, policy, tmp_path):
        # The row #710 calls "the point": cli refused `cat /etc/hostname`, and
        # the notebook spawned it.
        with pytest.raises(NotebookContainmentError):
            _check_spawn(policy, ("/bin/cat", ["cat", "/etc/hostname"], None, {}),
                         (0, 1, 2))

    def test_spawn_of_workspace_relative_argument_allowed(self, policy):
        # A bare command name and a bare relative argument both resolve against
        # the kernel's cwd, which IS the workspace — the same reading cli takes.
        _check_spawn(policy, ("ls", ["ls", "notes.txt"], None, {}), (0, 1, 2))

    def test_spawn_of_an_absolute_binary_is_checked_like_cli(self, policy):
        # cli classifies the command word itself when it is path-shaped, so an
        # absolute binary outside the workspace is refused there too.  Matching
        # that is the point: the notebook must not be the softer surface.
        with pytest.raises(NotebookContainmentError):
            _check_spawn(policy, ("/bin/ls", ["/bin/ls"], None, {}), (0, 1, 2))

    def test_spawn_cwd_outside_raises(self, policy, tmp_path):
        with pytest.raises(NotebookContainmentError):
            _check_spawn(policy, ("/bin/ls", ["ls"], str(tmp_path / "out"), {}),
                         (0, 1, 2))

    def test_os_system_argument_outside_raises(self, policy):
        with pytest.raises(NotebookContainmentError):
            _check_system(policy, ("cat /etc/hostname",))

    def test_dlopen_of_a_bare_soname_refused(self, policy):
        # A loaded library's own open(2) calls raise no audit event, so this is
        # the one way past every other check in the module.
        with pytest.raises(NotebookContainmentError):
            _check_dlopen(policy, ("libc.so.6",))

    def test_dlopen_of_none_refused(self, policy):
        with pytest.raises(NotebookContainmentError):
            _check_dlopen(policy, (None,))

    def test_path_like_mirrors_cli(self):
        assert _spawn_path_like("/etc/passwd") is True
        assert _spawn_path_like("../secrets") is True
        assert _spawn_path_like("~/id_rsa") is True
        assert _spawn_path_like("-la") is False
        assert _spawn_path_like("https://example.invalid/x") is False
        assert _spawn_path_like("notes.txt") is False


class TestEstablishContainment:
    """Which boundary is chosen, strongest first."""

    def test_apparmor_wins_and_installs_no_hook(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "shared.plugins.notebook.kernel_sandbox.apparmor_enforced_profile",
            lambda: "jaato-ws-x//child")
        installed = []
        monkeypatch.setattr(
            "shared.plugins.notebook.kernel_sandbox.install", installed.append)
        kind, _ = establish_containment(str(tmp_path))
        assert kind == BOUNDARY_APPARMOR
        assert installed == []       # the kernel already bounds the syscalls

    def test_audit_hook_when_unconfined(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "shared.plugins.notebook.kernel_sandbox.apparmor_enforced_profile",
            lambda: None)
        installed = []
        monkeypatch.setattr(
            "shared.plugins.notebook.kernel_sandbox.install", installed.append)
        kind, description = establish_containment(str(tmp_path))
        assert kind == BOUNDARY_AUDIT
        assert str(tmp_path) in description
        assert len(installed) == 1

    def test_opt_out_is_a_boundary_kind_of_its_own(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "shared.plugins.notebook.kernel_sandbox.apparmor_enforced_profile",
            lambda: None)
        installed = []
        monkeypatch.setattr(
            "shared.plugins.notebook.kernel_sandbox.install", installed.append)
        kind, _ = establish_containment(str(tmp_path), opt_out=True)
        assert kind == BOUNDARY_OPT_OUT
        assert installed == []

    def test_no_workspace_yields_no_boundary(self, monkeypatch):
        monkeypatch.setattr(
            "shared.plugins.notebook.kernel_sandbox.apparmor_enforced_profile",
            lambda: None)
        kind, _ = establish_containment(None)
        assert kind == BOUNDARY_NONE


class TestBackendBoundaryDeclarations:
    """Every backend states a boundary, and the default is a refusal."""

    def test_base_class_refuses_by_default(self):
        class Unannounced(NotebookBackend):
            capabilities = None
            def initialize(self, config=None): ...
            def shutdown(self): ...
            def is_available(self): return True
            def create_notebook(self, name, gpu_enabled=False): ...
            def execute(self, notebook_id, code, timeout_seconds=None): ...
            def get_execution_status(self, notebook_id, execution_id=None): ...
            def get_variables(self, notebook_id): return {}
            def reset_notebook(self, notebook_id): return notebook_id
            def delete_notebook(self, notebook_id): ...
            def list_notebooks(self): return []

        allowed, reason = Unannounced().execution_boundary()
        assert allowed is False
        assert "Unannounced" in reason

    def test_local_backend_defers_to_the_inprocess_gate(self, monkeypatch):
        monkeypatch.delenv(INPROCESS_OPT_IN_ENV, raising=False)
        monkeypatch.setattr(
            "shared.plugins.notebook.backends.local._apparmor_enforced_profile",
            lambda: None)
        backend = LocalJupyterBackend()
        backend.initialize()
        allowed, reason = backend.execution_boundary()
        assert allowed is False
        assert INPROCESS_OPT_IN_ENV in reason

    def test_subprocess_backend_states_containment(self, monkeypatch, tmp_path):
        monkeypatch.delenv(UNCONTAINED_OPT_IN_ENV, raising=False)
        monkeypatch.setattr(
            "shared.plugins.notebook.backends.subprocess_kernel"
            ".apparmor_enforced_profile", lambda: None)
        backend = SubprocessKernelBackend()
        backend.initialize({"workspace_root": str(tmp_path)})
        allowed, reason = backend.execution_boundary()
        assert allowed is True
        assert str(tmp_path) in reason

    def test_subprocess_backend_refuses_without_a_workspace(self, monkeypatch):
        monkeypatch.delenv(UNCONTAINED_OPT_IN_ENV, raising=False)
        monkeypatch.setattr(
            "shared.plugins.notebook.backends.subprocess_kernel"
            ".apparmor_enforced_profile", lambda: None)
        monkeypatch.setattr(
            "shared.plugins.notebook.backends.subprocess_kernel"
            ".get_workspace_root", lambda: None)
        backend = SubprocessKernelBackend()
        backend.initialize({})
        allowed, reason = backend.execution_boundary()
        assert allowed is False
        assert UNCONTAINED_OPT_IN_ENV in reason

    def test_env_opt_out_is_honoured(self, monkeypatch):
        monkeypatch.setenv(UNCONTAINED_OPT_IN_ENV, "1")
        monkeypatch.setattr(
            "shared.plugins.notebook.backends.subprocess_kernel"
            ".apparmor_enforced_profile", lambda: None)
        backend = SubprocessKernelBackend()
        backend.initialize({})
        allowed, reason = backend.execution_boundary()
        assert allowed is True
        assert "opt-out" in reason


class _BoundarylessBackend(NotebookBackend):
    """A backend that states no boundary and records whether it was asked to run.

    Stands in for the shape #710 warns about: a backend added later that never
    decided how cell code is contained.  ``ran`` is the assertion target — the
    plugin must refuse *before* dispatch, not after.
    """

    def __init__(self):
        self.ran = False
        self._info = None

    @property
    def capabilities(self):
        from ..types import BackendCapabilities
        return BackendCapabilities(name="boundaryless")

    def initialize(self, config=None): ...

    def shutdown(self): ...

    def is_available(self): return True

    def create_notebook(self, name, gpu_enabled=False):
        from ..types import NotebookInfo
        self._info = NotebookInfo(notebook_id="nb1", name=name,
                                  backend="boundaryless")
        return self._info

    def execute(self, notebook_id, code, timeout_seconds=None):
        from ..types import ExecutionResult
        self.ran = True
        return ExecutionResult(status=ExecutionStatus.COMPLETED)

    def get_execution_status(self, notebook_id, execution_id=None):
        from ..types import ExecutionResult
        return ExecutionResult(status=ExecutionStatus.COMPLETED)

    def get_variables(self, notebook_id): return {}

    def reset_notebook(self, notebook_id): return notebook_id

    def delete_notebook(self, notebook_id): ...

    def list_notebooks(self):
        return [self._info] if self._info else []


class TestPluginGate:
    """``NotebookPlugin`` refuses a cell no backend will contain — on BOTH paths.

    The streaming path is the live one (``supports_streaming`` is True for
    ``notebook_execute``), and before #710 it ran neither the static analyzer
    nor any boundary check, so a gate placed only on ``_execute_code`` bound
    nothing in a daemon.
    """

    @staticmethod
    def _plugin_with_boundaryless_backend(tmp_path):
        from ..plugin import NotebookPlugin
        plugin = NotebookPlugin()
        plugin.initialize({"workspace_root": str(tmp_path),
                           "sandbox_mode": "disabled"})
        backend = _BoundarylessBackend()
        plugin._backends["boundaryless"] = backend
        plugin._active_backend_name = "boundaryless"
        plugin._current_notebook_id = backend.create_notebook("t").notebook_id
        return plugin, backend

    def test_non_streaming_path_refuses(self, tmp_path):
        plugin, backend = self._plugin_with_boundaryless_backend(tmp_path)
        result = plugin._execute_code({"code": "1 + 1"})
        assert result.get("error") == "Notebook execution refused"
        assert "_BoundarylessBackend" in result["reason"]
        assert backend.ran is False

    def test_streaming_path_refuses(self, tmp_path):
        import asyncio

        plugin, backend = self._plugin_with_boundaryless_backend(tmp_path)

        async def _collect():
            return [c async for c in plugin.execute_streaming(
                "notebook_execute", {"code": "1 + 1"})]

        chunks = asyncio.run(_collect())
        assert [c.chunk_type for c in chunks] == ["error"]
        assert "Notebook execution refused" in chunks[0].content
        assert backend.ran is False

    def test_streaming_path_applies_the_static_analyzer(self, tmp_path):
        # `sandbox_mode: strict` blocked nothing on this path before #710.
        import asyncio

        from ..plugin import NotebookPlugin
        plugin = NotebookPlugin()
        plugin.initialize({"workspace_root": str(tmp_path),
                           "sandbox_mode": "strict"})
        backend = _BoundarylessBackend()
        plugin._backends["boundaryless"] = backend
        plugin._active_backend_name = "boundaryless"
        plugin._current_notebook_id = backend.create_notebook("t").notebook_id

        async def _collect():
            return [c async for c in plugin.execute_streaming(
                "notebook_execute",
                {"code": "import subprocess\n"
                         "subprocess.run(['cat', '/etc/shadow'])"})]

        chunks = asyncio.run(_collect())
        assert [c.chunk_type for c in chunks] == ["error"]
        assert "blocked by sandbox" in chunks[0].content.lower()
        assert backend.ran is False

    def test_a_backend_with_a_boundary_still_runs(self, tmp_path):
        plugin, backend = self._plugin_with_boundaryless_backend(tmp_path)
        backend.execution_boundary = lambda: (True, "test boundary")
        result = plugin._execute_code({"code": "1 + 1"})
        assert "error" not in result
        assert backend.ran is True


def _kernel_env():
    """Environment for a spawned kernel, pinned to the tree under test."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_SERVER_ROOT, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.pop(UNCONTAINED_OPT_IN_ENV, None)
    return env


@pytest.fixture
def kernel(tmp_path, monkeypatch):
    """A real kernel subprocess rooted at ``tmp_path/ws``, torn down after."""
    monkeypatch.setenv(
        "PYTHONPATH", _kernel_env()["PYTHONPATH"])
    monkeypatch.delenv(UNCONTAINED_OPT_IN_ENV, raising=False)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    backend = SubprocessKernelBackend()
    backend.initialize({"workspace_root": str(workspace)})
    try:
        yield backend, backend.create_notebook("t"), workspace
    finally:
        backend.shutdown()


def _text(result):
    return "".join(o.content for o in result.outputs
                   if o.output_type in (OutputType.STDOUT, OutputType.RESULT))


@pytest.mark.skipif(sys.platform == "win32",
                    reason="the measured escapes are POSIX paths")
class TestKernelEndToEnd:
    """The three rows of #710's table, against a live kernel."""

    def test_kernel_reports_its_boundary(self, kernel):
        _backend, info, workspace = kernel
        assert info.boundary is not None
        assert str(workspace) in info.boundary

    def test_read_outside_the_workspace_is_refused(self, kernel):
        backend, info, _ = kernel
        result = backend.execute(info.notebook_id, "open('/etc/hostname').read()")
        assert result.status == ExecutionStatus.FAILED
        assert result.error_name == "NotebookContainmentError"

    def test_spawning_the_command_cli_refuses_is_refused(self, kernel):
        backend, info, _ = kernel
        result = backend.execute(
            info.notebook_id,
            "import subprocess\n"
            "subprocess.run(['cat', '/etc/hostname'], capture_output=True)")
        assert result.status == ExecutionStatus.FAILED
        assert result.error_name == "NotebookContainmentError"

    def test_os_system_is_refused(self, kernel):
        backend, info, _ = kernel
        result = backend.execute(
            info.notebook_id, "import os\nos.system('cat /etc/hostname')")
        assert result.status == ExecutionStatus.FAILED
        assert result.error_name == "NotebookContainmentError"

    def test_ctypes_cannot_load_libc(self, kernel):
        backend, info, _ = kernel
        result = backend.execute(
            info.notebook_id, "import ctypes\nctypes.CDLL('libc.so.6')")
        assert result.status == ExecutionStatus.FAILED
        assert result.error_name == "NotebookContainmentError"

    def test_workspace_work_still_runs(self, kernel):
        backend, info, workspace = kernel
        result = backend.execute(
            info.notebook_id,
            "open('notes.txt', 'w').write('hi')\nopen('notes.txt').read()")
        assert result.status == ExecutionStatus.COMPLETED
        assert "hi" in _text(result)
        assert (workspace / "notes.txt").read_text() == "hi"

    def test_stdlib_imports_still_work(self, kernel):
        # The hook audits every `open`, and the interpreter keeps importing
        # after it is installed; denying its own installation would not contain
        # a model, it would break Python.
        backend, info, _ = kernel
        result = backend.execute(
            info.notebook_id,
            "import json, sqlite3, ssl, zoneinfo\nprint('ok')")
        assert result.status == ExecutionStatus.COMPLETED
        assert "ok" in _text(result)

    def test_in_workspace_subprocess_still_runs(self, kernel):
        backend, info, _ = kernel
        result = backend.execute(
            info.notebook_id,
            "import subprocess\n"
            "print(subprocess.run(['echo', 'hello'], capture_output=True,"
            " text=True).stdout)")
        assert result.status == ExecutionStatus.COMPLETED
        assert "hello" in _text(result)

    def test_operator_opt_out_restores_the_old_behaviour(self, tmp_path,
                                                         monkeypatch):
        monkeypatch.setenv("PYTHONPATH", _kernel_env()["PYTHONPATH"])
        workspace = tmp_path / "ws"
        workspace.mkdir()
        backend = SubprocessKernelBackend()
        backend.initialize({"workspace_root": str(workspace),
                            "allow_uncontained_exec": True})
        try:
            info = backend.create_notebook("t")
            assert "opt-out" in (info.boundary or "")
            result = backend.execute(info.notebook_id, "open('/etc/hostname').read()")
            assert result.status == ExecutionStatus.COMPLETED
        finally:
            backend.shutdown()

    def test_sandbox_grant_reaches_a_running_kernel(self, kernel, tmp_path):
        # `sandbox add` is the documented escape hatch and a kernel outlives
        # many cells, so the grant rides every execute frame rather than argv.
        backend, info, _ = kernel
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text("granted")
        read = backend.execute(
            info.notebook_id, f"open({str(corpus / 'a.txt')!r}).read()")
        # Staged under tmp_path, which IS inside the /tmp allowance the kernel
        # shares with cli — so this reads, and the assertion that matters is
        # the denial below, which must outrank that allowance.
        assert read.status == ExecutionStatus.COMPLETED
        backend.set_sandbox_paths_fn(
            lambda: {"read": [], "write": [], "deny": [str(corpus)]})
        denied = backend.execute(
            info.notebook_id, f"open({str(corpus / 'a.txt')!r}).read()")
        assert denied.status == ExecutionStatus.FAILED
        assert denied.error_name == "NotebookContainmentError"


class TestKernelRefusalFrames:
    """The kernel's own refusal path, without spawning one."""

    def test_refusal_frame_names_the_opt_out(self):
        from .. import kernel_main, kernel_protocol as proto

        class _Sink:
            def __init__(self):
                self.data = bytearray()

            def write(self, chunk):
                self.data.extend(chunk)
                return len(chunk)

            def flush(self):
                pass

        sink = _Sink()
        kernel_main._refuse_uncontained(sink, "c1")
        payload = sink.data[4:].decode("utf-8")
        assert proto.ERROR in payload
        assert "NotebookContainmentUnavailable" in payload
        assert UNCONTAINED_OPT_IN_ENV in payload

    def test_absent_allow_block_does_not_clear_the_policy(self, tmp_path,
                                                          monkeypatch):
        # A runner that does not speak the `allow` field must not be read as
        # "the operator revoked everything".
        from .. import kernel_main

        pol = ContainmentPolicy(str(tmp_path))
        pol.set_extra_allows(read=[str(tmp_path / "corpus")])
        monkeypatch.setattr(
            "shared.plugins.notebook.kernel_sandbox.current_policy", lambda: pol)
        kernel_main._apply_allow_frame({"type": "execute", "code": ""})
        assert pol._extra_read != []
        kernel_main._apply_allow_frame({"allow": {"read": [], "write": [],
                                                  "deny": []}})
        assert pol._extra_read == []
