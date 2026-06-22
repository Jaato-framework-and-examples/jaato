"""PR 1: the subprocess kernel runs with cwd=workspace (relative paths
in-workspace, no process-global chdir), persists state, and reports errors."""
import os
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
