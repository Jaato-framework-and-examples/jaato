"""Model-called tool bodies run in ``tool_hat``; the runner's own work in base (#1422).

The template rendered a ``tool_hat`` sub-profile that nothing entered, so
every in-process tool (``file_edit``, ``readFile``, memory, ``web_fetch``,
``proposeReference``, out-of-tree plugins) ran in the runner's BASE
profile.  Base therefore carried the tool-scope denies, and those also
blocked the runner's own bookkeeping: ``references bundle add|create|
reconcile|merge|unpack`` got EACCES under ``.jaato/references/**``.

After this change:

- ``tool_hat`` is a HAT, entered per thread with ``change_hat`` and a
  64-bit token around every model-called tool body
  (``shared/apparmor_hat.py``) and left with the same token in
  ``finally``;
- user commands, the completion tools and ``TRAIT_FRAMEWORK_LEVEL``
  tools run in base;
- base no longer write-denies ``.jaato/references/**``; the hat,
  ``//child`` and the isolated sub-runner do;
- #1023's per-thread verification treats a worker left in the hat by a
  failed return as divergence, and a worker mid-tool as inside.

No kernel here: the ``attr/current`` writes and reads are injected, and
the template is checked as rendered text (compiled with
``apparmor_parser`` where it is installed).  What the kernel does with
``changehat`` is for the enforcing-host checklist in the PR.
"""

from __future__ import annotations

import ast
import contextlib
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Dict, List

import pytest

from jaato_server.server.apparmor import AppArmorManager
from jaato_server.server.runner import bootstrap as B
from jaato_server.shared import apparmor_hat as H
from jaato_server.shared.ai_tool_runner import ToolExecutor
from jaato_server.shared.tests.reversion import Reversion

_HAT = "jaato-server/jaato_server/shared/apparmor_hat.py"
_RUNNER = "jaato-server/jaato_server/shared/ai_tool_runner.py"
_AA = "jaato-server/jaato_server/server/apparmor.py"
_BOOT = "jaato-server/jaato_server/server/runner/bootstrap.py"
_SESSION = "jaato-server/jaato_server/server/runner/session.py"
_JSESSION = "jaato-server/jaato_server/shared/jaato_session.py"

REVERSIONS = [
    Reversion(
        target=_RUNNER,
        find=("            with self._tool_body_confinement(name):\n"
              "                # The tool body may record who approved it"),
        replace=("            with nullcontext():\n"
                 "                # The tool body may record who approved it"),
        test="test_a_tool_body_runs_inside_the_hat",
        because="a model-called tool body runs in the base profile",
    ),
    Reversion(
        target=_HAT,
        find='        write(path, f"changehat {token:016x}^")\n        return True',
        replace='        return True',
        test="test_the_hat_is_left_with_the_token_it_was_entered_with",
        because="a thread is never returned from the hat",
    ),
    Reversion(
        target=_BOOT,
        find="    return tid in stuck_hat_tids",
        replace="    return False",
        test="test_a_worker_stuck_in_the_hat_is_divergence",
        because="#1023 verification passes a worker left in the hat",
    ),
    Reversion(
        target=_AA,
        find='        return f"""  hat tool_hat{subprofile_flag_clause} {{',
        replace='        return f"""  profile tool_hat{subprofile_flag_clause} {{',
        test="test_tool_hat_is_a_hat_without_the_unconfine_rule",
        because="change_hat into a child PROFILE is refused by the kernel",
    ),
    Reversion(
        target=_SESSION,
        find="    _arm_tool_hat(envelope, runtime)\n",
        replace="",
        test="test_bootstrap_arms_and_requires_the_hat",
        because="no runner session ever installs the hat",
    ),
    Reversion(
        target=_JSESSION,
        find=("        with self._executor.base_profile_calls():\n"
              "            _ok, result = self._executor.execute(command_name, args)"),
        replace=("        if True:\n"
                 "            _ok, result = self._executor.execute(command_name, args)"),
        test="test_user_commands_and_completion_tools_run_in_base",
        because="`references bundle add` runs in the hat and cannot write the catalog",
    ),
    Reversion(
        target=_RUNNER,
        find="        executor_fn = self._in_tool_confinement(name, executor_fn)\n",
        replace="",
        test="test_an_auto_background_body_enters_the_hat_on_its_own_thread",
        because="an auto-backgrounded tool body runs in base on the pool thread",
    ),
]

BASE = "jaato-ws-test-0123456789ab"
HAT = f"{BASE}//tool_hat"


class _FakeAttr:
    """One thread's ``attr/current``, driven by the writes the kernel honours."""

    def __init__(self, label: str = BASE, fail_write: str = "") -> None:
        self.label = label
        self.writes: List[str] = []
        self.fail_write = fail_write

    def write(self, _path: str, payload: str) -> None:
        self.writes.append(payload)
        if self.fail_write and payload.endswith(self.fail_write):
            raise PermissionError("EACCES")
        verb, _, rest = payload.partition(" ")
        assert verb == "changehat"
        token, _, hat = rest.partition("^")
        assert len(token) == 16 and int(token, 16) != 0
        self.label = HAT if hat else BASE

    def read(self, _path: str):
        from jaato_server.shared.apparmor_label import parse_label
        return parse_label(f"{self.label} (enforce)")


def _hat(attr: _FakeAttr):
    return H.tool_hat(BASE, attr_path="/fake", write=attr.write, read=attr.read)


# ---------------------------------------------------------------------------
# The hat itself
# ---------------------------------------------------------------------------

def test_the_hat_is_left_with_the_token_it_was_entered_with():
    attr = _FakeAttr()
    seen = []
    with _hat(attr):
        seen.append(attr.label)
    assert seen == [HAT]
    assert attr.label == BASE
    enter, leave = attr.writes
    token = enter.split(" ")[1].split("^")[0]
    assert enter == f"changehat {token}^tool_hat"
    assert leave == f"changehat {token}^"


def test_the_hat_is_left_when_the_body_raises():
    attr = _FakeAttr()
    with pytest.raises(ValueError):
        with _hat(attr):
            raise ValueError("tool failed")
    assert attr.label == BASE and len(attr.writes) == 2


def test_a_nested_call_does_not_change_the_label_again():
    attr = _FakeAttr()
    with _hat(attr):
        with _hat(attr):
            assert attr.label == HAT
    assert len(attr.writes) == 2 and attr.label == BASE


def test_a_thread_born_in_the_hat_runs_where_it_is():
    """A fresh token from inside a hat is a wrong token, which the kernel
    answers by killing the task: nothing is written."""
    attr = _FakeAttr(label=HAT)
    with _hat(attr):
        pass
    assert attr.writes == []


def test_a_thread_outside_the_session_profile_runs_nothing():
    attr = _FakeAttr(label="unconfined")
    ran = []
    with pytest.raises(H.ToolHatError):
        with _hat(attr):
            ran.append(1)
    assert ran == [] and attr.writes == []


def test_a_refused_entry_runs_nothing():
    attr = _FakeAttr(fail_write="^tool_hat")
    ran = []
    with pytest.raises(H.ToolHatError):
        with _hat(attr):
            ran.append(1)
    assert ran == []


def test_a_failed_return_marks_the_thread_stuck():
    attr = _FakeAttr(fail_write="^")
    tid = threading.get_native_id()
    try:
        with _hat(attr):
            pass
        assert tid in H.stuck_hat_tids()
    finally:
        H._stuck.discard(tid)


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------

_in_ctx = threading.local()


def _factory(log: List[str]):
    @contextlib.contextmanager
    def _ctx():
        log.append("enter")
        _in_ctx.on = True
        try:
            yield
        finally:
            _in_ctx.on = False
            log.append("exit")
    return _ctx


def _body(seen: List[bool]):
    def _fn(_args):
        seen.append(bool(getattr(_in_ctx, "on", False)))
        return {"ok": True}
    return _fn


def test_a_tool_body_runs_inside_the_hat():
    log: List[str] = []
    seen: List[bool] = []
    e = ToolExecutor()
    e.register("readFile", _body(seen))
    e.set_apparmor_context(_factory(log))
    assert e.execute("readFile", {})[0] is True
    assert seen == [True] and log == ["enter", "exit"]


def test_a_hat_that_cannot_be_entered_refuses_the_call():
    seen: List[bool] = []
    e = ToolExecutor()
    e.register("readFile", _body(seen))

    def _refuse():
        raise H.ToolHatError("no hat")
    e.set_apparmor_context(_refuse)
    ok, result = e.execute("readFile", {})
    assert ok is False and "no hat" in result["error"] and seen == []


def test_user_commands_and_completion_tools_run_in_base():
    log: List[str] = []
    seen: List[bool] = []
    e = ToolExecutor()
    e.register("references", _body(seen))
    e.register("signal_completion", _body(seen))
    e.set_apparmor_context(_factory(log))
    e.mark_base_profile_tools({"signal_completion"})
    with e.base_profile_calls():
        e.execute("references", {})
    e.execute("signal_completion", {})
    assert seen == [False, False] and log == []
    # And the two production call sites use them.
    src = Path(__file__).resolve().parents[2] / "shared" / "jaato_session.py"
    tree = ast.parse(src.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "execute_user_command")
    assert "base_profile_calls" in ast.unparse(fn)
    assert "mark_base_profile_tools" in src.read_text()


def test_an_auto_background_body_enters_the_hat_on_its_own_thread():
    log: List[str] = []
    seen: List[bool] = []
    threads: List[int] = []

    class _Plugin:
        name = "bg"

        def get_executors(self):
            return {"slow": _body(seen)}

        def start_background(self, name, args, executor_fn, output_callback):
            def _run():
                threads.append(threading.get_native_id())
                executor_fn(args)
            t = threading.Thread(target=_run)
            t.start()
            t.join()
            raise RuntimeError("no handle")  # take the sync fallback too

    e = ToolExecutor()
    e.register("slow", _body(seen))
    e.set_apparmor_context(_factory(log))
    e._execute_with_auto_background("slow", {}, _Plugin(), 1.0)
    assert threads and threads[0] != threading.get_native_id()
    assert seen == [True, True], seen


def test_no_path_writes_changeprofile_unconfined():
    """Before #1422 a framework-level tool wrote ``changeprofile
    unconfined``; in a confined runner that succeeds and leaves the worker
    unconfined for good."""
    src = (Path(__file__).resolve().parents[1] / "ai_tool_runner.py").read_text()
    literals = [n.value for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not any(v.strip() == "changeprofile unconfined" for v in literals)


# ---------------------------------------------------------------------------
# Runner bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_arms_and_requires_the_hat():
    from types import SimpleNamespace
    from jaato_server.server.runner import session as S

    rt = SimpleNamespace()
    S._arm_tool_hat(SimpleNamespace(profile_name=BASE), rt)
    assert callable(rt.tool_hat_factory)
    for unconfined in ("", "jaato-ws-a//sub"):
        rt2 = SimpleNamespace()
        S._arm_tool_hat(SimpleNamespace(profile_name=unconfined), rt2)
        assert rt2.tool_hat_factory is None

    bare = SimpleNamespace(_executor=ToolExecutor())
    with pytest.raises(S.BootstrapError):
        S._require_tool_hat(SimpleNamespace(profile_name=BASE), bare)
    bare._executor.set_apparmor_context(rt.tool_hat_factory)
    S._require_tool_hat(SimpleNamespace(profile_name=BASE), bare)

    src = (Path(S.__file__)).read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "bootstrap_session")
    calls = {c.func.id for c in ast.walk(fn)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert {"_arm_tool_hat", "_require_tool_hat"} <= calls


# ---------------------------------------------------------------------------
# #1023 verification
# ---------------------------------------------------------------------------

def _task_dir(tmp_path: Path, labels: Dict[int, str]) -> str:
    for tid, label in labels.items():
        d = tmp_path / str(tid) / "attr"
        d.mkdir(parents=True)
        (d / "current").write_text(label + "\n")
    return str(tmp_path)


def test_a_worker_mid_tool_is_inside_the_boundary(tmp_path):
    task_dir = _task_dir(tmp_path, {1: f"{BASE} (enforce)", 2: f"{HAT} (enforce)"})
    scan = B.scan_thread_profiles(BASE, task_dir=task_dir)
    assert not scan.divergent


def test_a_worker_stuck_in_the_hat_is_divergence(tmp_path):
    task_dir = _task_dir(tmp_path, {1: f"{BASE} (enforce)", 2: f"{HAT} (enforce)"})
    scan = B.scan_thread_profiles(BASE, task_dir=task_dir, stuck_hat_tids={2})
    assert [tid for tid, _ in scan.divergent] == [2]
    with pytest.raises(B.ThreadConfinementDivergence):
        B.verify_thread_confinement(
            BASE, task_dir=task_dir, grace_seconds=0,
            stuck_hat_tids=lambda: {2},
        )


# ---------------------------------------------------------------------------
# The template
# ---------------------------------------------------------------------------

def _bodies(root: Path) -> Dict[str, str]:
    workspace = root / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    manager = AppArmorManager(workspace_root=str(root))
    text = manager._render_profile("sid", str(workspace))
    hat = text.index("  hat tool_hat {")
    child = text.index("profile child {")
    return {"base": text[:hat], "tool_hat": text[hat:child],
            "child": text[child:], "_all": text, "ws": str(workspace),
            "isolated": manager._render_sub_profile("sid", "sub1", str(workspace))}


def _denies_catalog(body: str, ws: str) -> bool:
    return any(line.split() == ["audit", "deny", f'"{ws}/.jaato/references/**"', "wlk,"]
               for line in body.splitlines())


def test_base_lets_the_runner_write_the_catalog_and_tool_scopes_do_not(tmp_path):
    b = _bodies(tmp_path)
    assert not _denies_catalog(b["base"], b["ws"])
    for scope in ("tool_hat", "child", "isolated"):
        assert _denies_catalog(b[scope], b["ws"]), scope
    assert AppArmorManager._TEMPLATE_VERSION >= 44


def test_tool_hat_is_a_hat_without_the_unconfine_rule(tmp_path):
    b = _bodies(tmp_path)
    assert "  hat tool_hat {" in b["_all"]
    assert "profile tool_hat" not in b["_all"]
    hat = [" ".join(line.split()) for line in b["tool_hat"].splitlines()]
    assert "change_profile -> unconfined," not in hat
    assert "change_profile -> jaato-ws-sid//child," in hat
    assert "owner /proc/*/task/*/attr/current rw," in hat


@pytest.mark.skipif(shutil.which("apparmor_parser") is None,
                    reason="apparmor_parser not installed")
def test_the_render_compiles(tmp_path: Path):
    profile = tmp_path / "profile"
    profile.write_text(_bodies(tmp_path / "root")["_all"])
    result = subprocess.run(["apparmor_parser", "-Q", "-K", str(profile)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
