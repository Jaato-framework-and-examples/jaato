"""PR 1: the subprocess kernel runs with cwd=workspace (relative paths
in-workspace, no process-global chdir), persists state, and reports errors."""
import os
import pytest
from shared.plugins.notebook.backends.subprocess_kernel import SubprocessKernelBackend
from shared.plugins.notebook.types import ExecutionStatus, OutputType


def _text(result):
    return "".join(o.content for o in result.outputs
                   if o.output_type in (OutputType.STDOUT, OutputType.RESULT))


def test_cwd_is_workspace_and_relative_writes_stay_in_workspace(tmp_path):
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    try:
        nb = be.create_notebook("t")
        # os.getcwd() IS the workspace (the whole point of 1c)
        r = be.execute(nb.notebook_id, "import os\nprint(os.getcwd())")
        assert r.status == ExecutionStatus.COMPLETED
        assert _text(r).strip() == str(tmp_path)
        # a RELATIVE write lands inside the workspace, not the launch dir
        be.execute(nb.notebook_id,
                   "os.makedirs('drafts', exist_ok=True)\n"
                   "open('drafts/x.txt','w').write('hi')")
        assert (tmp_path / "drafts" / "x.txt").read_text() == "hi"
    finally:
        be.shutdown()


def test_state_persists_across_cells(tmp_path):
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    try:
        nb = be.create_notebook("t")
        be.execute(nb.notebook_id, "x = 41")
        r = be.execute(nb.notebook_id, "x + 1")          # last-expression value
        assert r.status == ExecutionStatus.COMPLETED
        assert "42" in _text(r)
        assert "x" in be.get_variables(nb.notebook_id)
        be.reset_notebook(nb.notebook_id)
        r2 = be.execute(nb.notebook_id, "print('x' in dir())")
        assert "False" in _text(r2)                      # reset cleared it
    finally:
        be.shutdown()


def test_cell_error_surfaces(tmp_path):
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    try:
        nb = be.create_notebook("t")
        r = be.execute(nb.notebook_id, "1/0")
        assert r.status == ExecutionStatus.FAILED
        assert r.error_name == "ZeroDivisionError"
    finally:
        be.shutdown()


def test_tools_bridge_executes_through_runner(tmp_path):
    # PR 2: notebook tools.X() round-trips to the runner executor cross-process.
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    calls = []
    be.set_tool_executor(
        lambda name, args: (calls.append((name, args)) or True,
                            {"echoed": args.get("q")})[1] and (True, {"echoed": args.get("q")}))
    try:
        nb = be.create_notebook("t")
        r = be.execute(nb.notebook_id,
                       "r = tools.echo(q='hi')\nprint(r['echoed'])")
        assert r.status == ExecutionStatus.COMPLETED, r.error_message
        assert "hi" in _text(r)
        assert calls == [("echo", {"q": "hi"})]
    finally:
        be.shutdown()


def test_tools_bridge_runs_in_trusted_permission_scope(tmp_path):
    from shared.ai_tool_runner import in_trusted_bridge_context
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    seen = []
    be.set_tool_executor(
        lambda name, args: (seen.append(in_trusted_bridge_context()), (True, {}))[1])
    try:
        nb = be.create_notebook("t")
        be.execute(nb.notebook_id, "tools.x()")
        assert seen == [True]   # kernel-originated tool inherits notebook approval
    finally:
        be.shutdown()


def test_tools_bridge_error_raises_in_cell(tmp_path):
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    be.set_tool_executor(lambda name, args: (False, "boom"))
    try:
        nb = be.create_notebook("t")
        r = be.execute(nb.notebook_id, "tools.x()")
        assert r.status == ExecutionStatus.FAILED
        assert r.error_name == "ToolExecutionError"
    finally:
        be.shutdown()


def test_tools_bridge_no_executor_errors(tmp_path):
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    try:
        nb = be.create_notebook("t")
        r = be.execute(nb.notebook_id, "tools.x()")  # no executor wired
        assert r.status == ExecutionStatus.FAILED
    finally:
        be.shutdown()


def test_plugin_defaults_to_subprocess_and_local_opts_out(tmp_path):
    # PR 3 cutover: the plugin defaults to the subprocess kernel; "local" opts
    # back to the in-process backend (the CWD-escape fallback).
    from shared.plugins.notebook.plugin import NotebookPlugin
    p = NotebookPlugin()
    p.initialize({"workspace_root": str(tmp_path)})
    try:
        assert p._active_backend_name == "subprocess"
    finally:
        p.shutdown()
    p2 = NotebookPlugin()
    p2.initialize({"workspace_root": str(tmp_path), "backend": "local"})
    try:
        assert p2._active_backend_name == "local"
    finally:
        p2.shutdown()


def test_session_contextvar_workspace_wins_over_plugin_config(tmp_path):
    # Fix A: _spawn reads the session_context ContextVar (get_workspace_root) —
    # the SAME deterministic per-session source file_edit uses — in preference to
    # the plugin-config value.  This is what makes the kernel chdir to the SESSION
    # workspace regardless of where the daemon was launched (the peer's A/B bug).
    from shared.session_context import set_workspace_root, reset_workspace_root
    ctx_ws = tmp_path / "session_ws"; ctx_ws.mkdir()
    cfg_ws = tmp_path / "config_ws"; cfg_ws.mkdir()
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(cfg_ws)})       # the fallback
    token = set_workspace_root(str(ctx_ws))               # session ContextVar wins
    try:
        nb = be.create_notebook("t")
        r = be.execute(nb.notebook_id, "import os\nprint(os.getcwd())")
        assert _text(r).strip() == str(ctx_ws)
    finally:
        reset_workspace_root(token)
        be.shutdown()


def test_spawn_fails_loud_without_any_workspace(monkeypatch):
    # No os.getcwd() fallback: if neither the ContextVar nor plugin config is set,
    # spawning raises rather than silently chdir'ing to the daemon launch dir.
    monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
    be = SubprocessKernelBackend()                        # no workspace anywhere
    with pytest.raises(RuntimeError, match="no workspace_root"):
        be.create_notebook("t")


def test_streaming_path_propagates_workspace_contextvar(tmp_path, monkeypatch):
    # The STREAMING path (notebook_execute) runs backend.execute in a raw thread.
    # ContextVars don't propagate to a raw thread, so without copy_context() the
    # kernel's get_workspace_root() misses the session workspace and chdir's to
    # the launch dir (config) instead.  This drives execute_streaming with the
    # session ContextVar set to a DIFFERENT dir than the plugin config and
    # asserts the kernel landed in the ContextVar (session) dir.
    import asyncio
    from shared.plugins.notebook.plugin import NotebookPlugin
    from shared.session_context import set_workspace_root, reset_workspace_root
    monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
    cfg_dir = tmp_path / "launch"; cfg_dir.mkdir()       # plugin config (wrong)
    ses_dir = tmp_path / "session"; ses_dir.mkdir()      # session ContextVar (right)
    p = NotebookPlugin()
    p.initialize({"workspace_root": str(cfg_dir)})
    token = set_workspace_root(str(ses_dir))

    async def _run():
        out = []
        async for ch in p.execute_streaming(
                "notebook_execute", {"code": "import os\nprint(os.getcwd())"}):
            out.append(ch.content)
        return "".join(out)

    try:
        out = asyncio.run(_run())
        assert str(ses_dir) in out                       # ContextVar reached the thread
        assert str(cfg_dir) not in out
    finally:
        reset_workspace_root(token)
        p.shutdown()
