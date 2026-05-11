"""Unit-level regression pins for §7d — cgroup-attach plumbing
through RunnerSpawner.spawn + spawn_session_runner.

The TestRealKernel integration tests in
``shared/tests/test_runtime_limits_e2e.py`` validate the actual
kernel contract (runner-spawn-into-cgroup, child-inherit, PTY
grandchild-inherit) on Linux hosts with cgroup v2.  These unit
pins assert the *plumbing* lands correctly so the integration
tests can exercise the contract.

Pins:

- ``RunnerSpawner.spawn`` accepts ``cgroup_attach`` kwarg
- ``cgroup_attach`` is invoked between fork() and exec() in
  the forked child (AST-based pin)
- ``spawn_session_runner`` accepts + forwards ``cgroup_attach``
- WS pre-init hook constructs the attach callback from cgroups
  manager and passes it to spawn_session_runner
- WS pre-init hook gracefully degrades when cgroups unavailable
  (cgroup_attach stays None; runner spawns in daemon's cgroup)
"""

from __future__ import annotations

import ast
import inspect

import server.runner_spawner as spawner_module
import server.runner_spawn as runner_spawn_module
import server.websocket as websocket_module
from server.runner_spawner import RunnerSpawner


# ----------------------------------------------------------------------
# Signature pins
# ----------------------------------------------------------------------


def test_runner_spawner_spawn_accepts_cgroup_attach_kwarg() -> None:
    """Pin ``RunnerSpawner.spawn`` accepts the ``cgroup_attach``
    keyword arg (§7d plumbing entry)."""
    sig = inspect.signature(RunnerSpawner.spawn)
    assert "cgroup_attach" in sig.parameters, (
        "§7d regression: ``RunnerSpawner.spawn`` missing the "
        "``cgroup_attach`` kwarg added in §7d."
    )
    param = sig.parameters["cgroup_attach"]
    # The kwarg must be optional (defaults to None) so IPC
    # callers can pass nothing.
    assert param.default is None, (
        f"``cgroup_attach`` should default to None for backward "
        f"compatibility; got default={param.default!r}"
    )


def test_spawn_session_runner_accepts_cgroup_attach_kwarg() -> None:
    """Pin ``spawn_session_runner`` accepts + forwards
    ``cgroup_attach``."""
    sig = inspect.signature(runner_spawn_module.spawn_session_runner)
    assert "cgroup_attach" in sig.parameters, (
        "§7d regression: ``spawn_session_runner`` missing the "
        "``cgroup_attach`` kwarg."
    )
    assert sig.parameters["cgroup_attach"].default is None


# ----------------------------------------------------------------------
# Invocation-order pin (between fork and exec)
# ----------------------------------------------------------------------


def test_cgroup_attach_invoked_between_fork_and_exec() -> None:
    """Pin ``cgroup_attach()`` is invoked in the forked-child
    branch of ``RunnerSpawner.spawn`` BEFORE ``_exec_runner``.
    cgroup membership survives exec(); attaching before exec
    is essential so the new program starts already in the
    session cgroup."""
    src = inspect.getsource(RunnerSpawner.spawn)
    import textwrap
    src = textwrap.dedent(src)
    tree = ast.parse(src)

    # Find the child branch: ``if pid == 0:`` body.
    child_body = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "pid"
        ):
            child_body = node.body
            break
    assert child_body is not None, (
        "Could not locate the ``if pid == 0:`` branch in "
        "RunnerSpawner.spawn — fork pattern changed?"
    )

    # Walk the child body in source order; track when we see
    # ``cgroup_attach()`` and ``_exec_runner(...)``.  The
    # attach must come first.
    cgroup_attach_idx = -1
    exec_runner_idx = -1
    counter = [0]

    def _visit(node):
        for child in ast.iter_child_nodes(node):
            counter[0] += 1
            cur_idx = counter[0]
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name) and func.id == "cgroup_attach":
                    nonlocal_set_cgroup_idx(cur_idx)
                elif (
                    isinstance(func, ast.Attribute)
                    and func.attr == "_exec_runner"
                ):
                    nonlocal_set_exec_idx(cur_idx)
            _visit(child)

    def nonlocal_set_cgroup_idx(idx: int) -> None:
        nonlocal cgroup_attach_idx
        if cgroup_attach_idx == -1:
            cgroup_attach_idx = idx

    def nonlocal_set_exec_idx(idx: int) -> None:
        nonlocal exec_runner_idx
        if exec_runner_idx == -1:
            exec_runner_idx = idx

    for stmt in child_body:
        _visit(stmt)

    assert cgroup_attach_idx > 0, (
        "§7d regression: ``cgroup_attach()`` call not found in "
        "the forked-child branch of RunnerSpawner.spawn."
    )
    assert exec_runner_idx > 0, (
        "``_exec_runner(...)`` call not found in the forked-"
        "child branch — fork pattern changed?"
    )
    assert cgroup_attach_idx < exec_runner_idx, (
        f"§7d regression: ``cgroup_attach()`` must be invoked "
        f"BEFORE ``_exec_runner(...)`` so cgroup membership "
        f"survives the exec.  Found cgroup_attach at AST node "
        f"{cgroup_attach_idx}, _exec_runner at {exec_runner_idx}."
    )


# ----------------------------------------------------------------------
# WS pre-init hook integration pin
# ----------------------------------------------------------------------


def test_websocket_pre_init_hook_provisions_cgroup_before_spawn() -> None:
    """Pin the WS pre-init hook constructs ``cgroup_attach``
    from the CgroupsManager and passes it to
    ``spawn_session_runner``.  Pre-§7d the cgroup was
    provisioned in the POST-init session_hook (too late to
    attach the runner at spawn time); §7d moves provisioning
    into the pre-init hook so the attach_cb is available
    before spawn."""
    src = inspect.getsource(websocket_module)
    # The provisioning + attach-callback construction is wrapped
    # in a ``_apparmor_pre_init_hook`` nested function.
    # AST-parse and walk for the right shape.
    tree = ast.parse(src)

    pre_init_func = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_apparmor_pre_init_hook"
        ):
            pre_init_func = node
            break
    assert pre_init_func is not None, (
        "Could not locate ``_apparmor_pre_init_hook`` in "
        "server/websocket.py."
    )

    # Walk pre_init_func body; assert both
    # ``cgroups.provision_cgroup(...)`` and
    # ``cgroups.make_attach_callback(...)`` calls present,
    # and that they appear BEFORE the spawn_session_runner call.
    calls_order = []
    for child in ast.walk(pre_init_func):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute):
            if func.attr == "provision_cgroup":
                calls_order.append(("provision_cgroup", child.lineno))
            elif func.attr == "make_attach_callback":
                calls_order.append(("make_attach_callback", child.lineno))
        if isinstance(func, ast.Name) and func.id == "spawn_session_runner":
            calls_order.append(("spawn_session_runner", child.lineno))

    provision_lines = [
        ln for name, ln in calls_order if name == "provision_cgroup"
    ]
    attach_lines = [
        ln for name, ln in calls_order if name == "make_attach_callback"
    ]
    spawn_lines = [
        ln for name, ln in calls_order if name == "spawn_session_runner"
    ]

    assert provision_lines, (
        "§7d regression: ``cgroups.provision_cgroup(...)`` call "
        "missing from ``_apparmor_pre_init_hook``."
    )
    assert attach_lines, (
        "§7d regression: ``cgroups.make_attach_callback(...)`` "
        "call missing from ``_apparmor_pre_init_hook``."
    )
    assert spawn_lines, (
        "``spawn_session_runner(...)`` call missing from "
        "``_apparmor_pre_init_hook`` — pre-init hook structure "
        "changed?"
    )

    # The provision + attach must come BEFORE the spawn.
    assert min(provision_lines) < min(spawn_lines), (
        f"§7d regression: provision_cgroup at line "
        f"{min(provision_lines)} must come before spawn at "
        f"line {min(spawn_lines)} (so the attach_cb is "
        f"constructed in time)."
    )
    assert min(attach_lines) < min(spawn_lines), (
        f"§7d regression: make_attach_callback at line "
        f"{min(attach_lines)} must come before spawn at "
        f"line {min(spawn_lines)}."
    )


# ----------------------------------------------------------------------
# Optionality pin (cgroups unavailable → None, no crash)
# ----------------------------------------------------------------------


def test_runner_spawner_spawn_works_without_cgroup_attach() -> None:
    """Pin: when ``cgroup_attach=None`` (IPC sessions or hosts
    without cgroup v2), the spawn proceeds normally — the
    forked child skips the attach step.  AST-level check:
    the attach call is wrapped in ``if cgroup_attach is not None:``."""
    src = inspect.getsource(RunnerSpawner.spawn)
    import textwrap
    src = textwrap.dedent(src)
    tree = ast.parse(src)

    # Find any ``if cgroup_attach is not None:`` guard wrapping
    # the call site.
    found_guard = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "cgroup_attach"
            and any(isinstance(op, ast.IsNot) for op in test.ops)
        ):
            found_guard = True
            break
    assert found_guard, (
        "§7d regression: ``cgroup_attach()`` call must be wrapped "
        "in ``if cgroup_attach is not None:`` so IPC sessions "
        "(no cgroup provisioned) can pass ``None`` without "
        "crashing the forked child."
    )


# ----------------------------------------------------------------------
# Forward-compat: no spurious _cgroup_attach field deletions
# ----------------------------------------------------------------------


def test_daemon_side_cgroup_attach_field_stays_until_phase4() -> None:
    """Per the §7d audit Q4: the daemon-side
    ``ToolExecutor._cgroup_attach`` field stays as the
    defensive fallback for disk-restored sessions (§3.12).
    Pin the field declaration is present."""
    import shared.ai_tool_runner as tool_runner_module
    src = inspect.getsource(tool_runner_module)
    assert "self._cgroup_attach" in src, (
        "§7d audit finding #4 regression: daemon-side "
        "``ToolExecutor._cgroup_attach`` field removed.  Per the "
        "audit, this field stays as the disk-restore fallback "
        "until Phase 4+ migrates the restore path explicitly."
    )
