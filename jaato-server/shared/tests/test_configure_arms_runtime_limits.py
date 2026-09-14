"""``configure()`` must ARM the plugins that enforce the caps (#735).

This is the load-bearing half of the #735 guard, and it is deliberately
not a wire test.  Everything on the wire side — the envelope field, the
producer, the round trip — would have gone green on the pre-fix tree
too, because the pre-fix defect was not that the values failed to
travel.  They travelled: the runner's own startup line printed
``tool_timeout_seconds=2.0`` on a cold-spawn run.  They landed on
``server/runner/tool_executor.ToolExecutor``, the Phase-2 cli-only
``execute_fn`` that ``RunnerRPC`` bypasses whenever a session host
exists.  The session's tools run through
``shared.ai_tool_runner.ToolExecutor``, whose ``set_runtime_limits`` had
**no non-test caller at all**, so ``CliPlugin._runtime_limits`` was
``None`` on every path — and a ``sleep 60`` under a declared
``tool_timeout_seconds: 2`` ran for 60.02 s, measured on both spawn
paths against ``main`` @ ``2bd2456``.

So the claim these tests make is about the OBJECT, not the wire: after
``configure()``, the plugin that will call ``subprocess.run(timeout=)``
is holding the number.

``configure()`` is the one application point on purpose.  Every route a
session can be built by converges on it — the runner's
``session.bootstrap`` (pool-served and cold-spawned alike), the isolated
sub-runner, and an in-process ``runtime.create_session`` — so the paths
cannot arm different things, which is what an env vehicle and an
envelope vehicle living side by side would allow.
"""

from __future__ import annotations

import pytest

from shared.jaato_runtime import JaatoRuntime
from shared.jaato_session import JaatoSession
from shared.plugins.registry import PluginRegistry
from shared.runtime_limits import RuntimeLimits


CAPS = RuntimeLimits(tool_timeout_seconds=120.0, max_output_bytes=8192)


@pytest.fixture(scope="module")
def _discovered():
    """One registry discovery for the module — it is the slow part."""
    registry = PluginRegistry()
    registry.discover()
    return registry


@pytest.fixture
def wired(_discovered):
    """A runtime + registry exposing the two subprocess plugins.

    ``cli`` and ``interactive_shell`` are exactly the plugins that
    implement the ``set_runtime_limits`` receiver, so they are what the
    executor's forwarding loop is supposed to find.
    """
    registry = _discovered
    for name in ("cli", "interactive_shell"):
        try:
            registry.expose_tool(name)
        except Exception:  # noqa: BLE001 — optional extra (pexpect)
            pass
    runtime = JaatoRuntime()
    runtime.configure_plugins(registry)
    return runtime, registry


def _configure(runtime, *, limits=None, width=None, plugins=("cli",)):
    """Configure a session the way a runner bootstrap would."""
    session = JaatoSession(runtime, model="m")
    session.configure(
        plugins=list(plugins),
        skip_provider=True,          # no network; the caps are the subject
        runtime_limits=limits,
        max_parallel_tools=width,
    )
    return session


# ----------------------------------------------------------------------
# The application point
# ----------------------------------------------------------------------

def test_configure_arms_the_cli_plugin(wired):
    """THE regression.  Pre-fix, ``configure()`` took no such argument and
    ``CliPlugin._runtime_limits`` stayed ``None`` however loudly the
    profile declared a cap."""
    runtime, registry = wired
    _configure(runtime, limits=CAPS)

    armed = registry.get_plugin("cli")._runtime_limits
    assert armed is not None, (
        "cli._runtime_limits is None after configure() — the cap reached "
        "the session and stopped there; every tool call runs unbounded"
    )
    assert armed.tool_timeout_seconds == 120.0
    assert armed.max_output_bytes == 8192


def test_configure_arms_the_interactive_shell_plugin(wired):
    """The sibling subprocess surface.

    ``interactive_shell`` spawns PTYs rather than ``subprocess.run``, but
    it implements the same receiver and was inert for the same reason.
    """
    runtime, registry = wired
    shell = registry.get_plugin("interactive_shell")
    if shell is None or not hasattr(shell, "set_runtime_limits"):
        pytest.skip("interactive_shell unavailable (optional pexpect extra)")
    _configure(runtime, limits=CAPS, plugins=("cli", "interactive_shell"))
    assert shell._runtime_limits == CAPS


def test_the_executor_itself_holds_the_block(wired):
    """One rung below the plugins: the executor that forwards to them.

    Asserting here as well as on the plugin distinguishes "the session
    never called the setter" from "the setter ran but forwarded to
    nothing", which are different bugs with the same symptom.
    """
    runtime, _registry = wired
    session = _configure(runtime, limits=CAPS)
    assert session._executor.get_runtime_limits() == CAPS


def test_no_limits_declared_leaves_the_plugin_alone(wired):
    """``None`` is a no-op, NOT a clear.

    Sessions on one runtime share the plugin registry, so a limitless
    in-process subagent calling ``set_runtime_limits(None, None)`` would
    strip the cap off its parent's tools.  "Nobody declared limits" must
    not be able to disarm somebody who did.
    """
    runtime, registry = wired
    _configure(runtime, limits=CAPS)
    _configure(runtime, limits=None)          # a limitless sibling
    assert registry.get_plugin("cli")._runtime_limits == CAPS


def test_a_later_session_can_narrow_the_cap(wired):
    """A declared block still replaces a declared block — only absence is
    inert."""
    runtime, registry = wired
    _configure(runtime, limits=CAPS)
    tighter = RuntimeLimits(tool_timeout_seconds=5.0)
    _configure(runtime, limits=tighter)
    assert registry.get_plugin("cli")._runtime_limits == tighter


def test_arming_happens_after_the_registry_is_installed(wired):
    """Ordering is the whole trick, so pin it.

    ``ToolExecutor.set_runtime_limits`` forwards by walking
    ``registry.list_exposed()``.  Called before ``set_registry`` it
    forwards to nothing and fails silently — the executor would hold the
    block and no plugin would.  Asserting the plugin (not the executor)
    is what makes this test sensitive to the order.
    """
    runtime, registry = wired
    session = _configure(runtime, limits=CAPS)
    assert session._executor._registry is registry
    assert registry.get_plugin("cli")._runtime_limits is not None


def test_no_kernel_attach_callback_is_installed(wired):
    """The session passes ``attach_callback=None`` deliberately.

    Kernel limits are applied to the runner PROCESS at fork time and its
    children inherit the cgroup, so a per-plugin ``preexec_fn`` would be
    redundant — and the session has no cgroup handle to offer anyway.
    """
    runtime, registry = wired
    _configure(runtime, limits=RuntimeLimits(memory_max_mb=512,
                                             tool_timeout_seconds=5.0))
    assert registry.get_plugin("cli")._cgroup_attach is None


def test_the_receiver_list_is_what_gets_logged(wired):
    """The effective-caps log line states a fact, not an intention.

    #735's point 4: "a limit that silently does not apply is worse than
    no limit".  The line names the plugins that took the block, and an
    empty list beside a declared cap is the visible form of "this
    profile enables no subprocess plugin, so the cap bounds nothing".
    """
    runtime, _registry = wired
    session = _configure(runtime, limits=CAPS)
    assert "cli" in session._runtime_limit_receivers()


def test_arming_logs_the_effective_caps(wired, caplog):
    """The operator-facing half of point 4."""
    import logging
    runtime, _registry = wired
    with caplog.at_level(logging.INFO, logger="shared.jaato_session"):
        _configure(runtime, limits=CAPS)
    armed = [r.getMessage() for r in caplog.records
             if "runtime_limits armed" in r.getMessage()]
    assert armed, "no effective-caps line was logged"
    assert "120.0" in armed[-1] and "8192" in armed[-1]


# ----------------------------------------------------------------------
# The width: two vehicles, one answer
# ----------------------------------------------------------------------

class TestParallelWidthResolution:
    """``max_parallel_tools`` can arrive two ways and must not disagree.

    v6 carries it as its own envelope field; v7 also carries it inside
    the block.  The standalone kwarg wins so a v6 daemon talking to a v7
    runner keeps working, and so an in-process caller can narrow one
    session without synthesising a whole ``RuntimeLimits``.
    """

    @staticmethod
    def _resolve(*args):
        # Imported per call rather than at module scope so the rest of
        # this module still COLLECTS on a tree without the resolver —
        # a collection error hides which claim actually broke.
        from shared.jaato_session import _resolve_parallel_width
        return _resolve_parallel_width(*args)

    def test_explicit_kwarg_wins(self):
        assert self._resolve(2, RuntimeLimits(max_parallel_tools=9)) == 2

    def test_block_supplies_it_when_the_kwarg_is_absent(self):
        assert self._resolve(None, RuntimeLimits(max_parallel_tools=3)) == 3

    def test_neither_declares_one(self):
        assert self._resolve(None, None) is None
        assert self._resolve(None, RuntimeLimits()) is None

    def test_configure_installs_the_width_from_the_block(self, wired):
        runtime, _registry = wired
        session = _configure(
            runtime, limits=RuntimeLimits(max_parallel_tools=2))
        assert session._max_parallel_tools == 2
        assert session._parallel_worker_cap(100) == 2

    def test_configure_still_honours_the_standalone_kwarg(self, wired):
        """Envelope v6 compatibility, pinned."""
        runtime, _registry = wired
        session = _configure(runtime, limits=None, width=3)
        assert session._max_parallel_tools == 3
