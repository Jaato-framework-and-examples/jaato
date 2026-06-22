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


def test_tools_bridge_stubbed_until_pr2(tmp_path):
    be = SubprocessKernelBackend()
    be.initialize({"workspace_root": str(tmp_path)})
    try:
        nb = be.create_notebook("t")
        r = be.execute(nb.notebook_id, "tools.web_search(query='x')")
        assert r.status == ExecutionStatus.FAILED  # stub raises (PR 2 wires it)
    finally:
        be.shutdown()
