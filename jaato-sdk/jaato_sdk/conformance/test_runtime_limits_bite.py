"""A declared ``runtime_limits`` cap must bound a real tool call (#735).

Every other guard for #735 asserts a link in the chain: the envelope
carries the block, the runner re-parses it, ``configure()`` hands it to
the cli plugin.  This one asserts the OUTCOME, against a real daemon, a
real runner subprocess and a real ``sleep``: a profile declaring
``tool_timeout_seconds: 2`` must not run a 60-second command to
completion.

It is the only test in the set that would have caught the original
defect with no knowledge of the mechanism.  The pre-fix tree forwarded
the caps correctly all the way into the runner process — the runner's
own startup line printed ``tool_timeout_seconds=2.0`` — and the tool
still ran 60.02 s, because the values configured the Phase-2 cli-only
executor rather than the session's.  A test that asserts a value
somewhere in the plumbing cannot tell those apart; a stopwatch can.

Two traps this file has to step around, both measured while reproducing
the issue.  Neither is optional:

* **``cli`` auto-backgrounds at 10 s by default.**  An uncapped
  ``sleep 60`` therefore returns a background handle at ~10 s, which
  looks vaguely like something bounded the tool and makes a 2-vs-60
  measurement unreadable.  ``auto_background_threshold`` is pushed out
  of the way so the ONLY thing that can end the command early is the
  cap.
* **``defaultPolicy: "allow"`` does not suppress the permission
  prompt.**  Every run emits ``PermissionRequestedEvent`` for the cli
  tool regardless, so the client answers it — which is what a real
  interactive client does anyway.  The answer lands before
  ``ToolCallStartEvent``, so it cannot contaminate the tool's measured
  duration.  (That the policy does not suppress it is a separate
  finding, deliberately not chased here.)

Opt-in: ``pytest -m conformance``.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from jaato_sdk import IPCClient
from jaato_sdk.conformance.daemon import ConformanceDaemon, echo_workspace
from jaato_sdk.events import (
    ClientType, ErrorEvent, PermissionRequestedEvent, ToolCallEndEvent,
    ToolCallStartEvent,
)

pytestmark = pytest.mark.conformance


#: Wide on purpose.  60 s against a 2 s cap is 30x, so "enforced" and
#: "ignored" cannot be confused by scheduler noise, a slow container, or
#: the cost of starting a subprocess.
SLEEP_SECONDS = 60
TOOL_TIMEOUT_SECONDS = 2.0

#: Anything comfortably between the cap and the sleep.  A run at or above
#: this means the cap did not bite.
VERDICT_SECONDS = 20.0

#: Far past ``SLEEP_SECONDS`` so the plugin's own auto-background deadline
#: cannot end the command and be mistaken for the cap.
NO_AUTO_BACKGROUND = 3600.0


@pytest.fixture(scope="module")
def capped_daemon():
    """A daemon serving one profile: echo drives ``cli``, under a 2 s cap."""
    root = Path(tempfile.mkdtemp(prefix="jaato-735-ws-"))
    echo_workspace(
        root,
        tool_call={"name": "cli_based_tool",
                   "args": {"command": f"sleep {SLEEP_SECONDS}"}},
        response="done",
        plugins=["cli"],
        runtime_limits={"tool_timeout_seconds": TOOL_TIMEOUT_SECONDS},
        plugin_configs={"cli": {
            "auto_background_threshold": NO_AUTO_BACKGROUND,
        }},
        name="capped",
    )
    d = ConformanceDaemon(root)
    try:
        yield d.start()
    finally:
        d.stop()


async def _run_the_capped_tool(daemon, wait_seconds: float = 150.0):
    """Drive one turn and return the ``ToolCallEndEvent``s it produced."""
    c = IPCClient(socket_path=daemon.socket_path,
                  client_type=ClientType.API,
                  workspace_path=str(daemon.workspace),
                  auto_start=False)
    assert await c.connect(timeout=60), "could not connect to the test daemon"
    ends: list = []
    started = asyncio.Event()
    finished = asyncio.Event()
    try:
        await c.create_session(profile="capped")

        async def collect():
            async for ev in c.events():
                if isinstance(ev, PermissionRequestedEvent):
                    # See the module docstring: the prompt arrives even
                    # under an allow policy, and it arrives BEFORE the
                    # tool starts, so answering costs the measurement
                    # nothing.
                    await c.respond_to_permission(ev.request_id, "a")
                elif isinstance(ev, ToolCallStartEvent):
                    started.set()
                elif isinstance(ev, ToolCallEndEvent):
                    ends.append(ev)
                    finished.set()
                elif isinstance(ev, ErrorEvent):
                    finished.set()

        task = asyncio.create_task(collect())
        await asyncio.sleep(0.3)
        await c.send_message("go")
        try:
            await asyncio.wait_for(finished.wait(), timeout=wait_seconds)
        except asyncio.TimeoutError:
            pass
        task.cancel()
        return ends
    finally:
        await c.disconnect()


def test_a_declared_tool_timeout_actually_bounds_the_tool(capped_daemon):
    """THE end-to-end claim, and the one with a stopwatch in it.

    ``duration_seconds`` is the daemon's own measurement of the tool
    call, which is the number the reproduction reported as 60.02 s
    against this exact 2 s cap — on the pool-served path AND on
    cold-spawn.
    """
    ends = asyncio.run(_run_the_capped_tool(capped_daemon))

    assert ends, (
        "no ToolCallEndEvent arrived within the wait — the tool neither "
        "completed nor was stopped, which is a worse version of the "
        "failure this test exists to catch"
    )
    end = ends[0]
    assert not end.backgrounded, (
        "the tool was auto-backgrounded, so its duration says nothing "
        "about the cap; auto_background_threshold is not being honoured"
    )
    assert end.duration_seconds < VERDICT_SECONDS, (
        f"the tool ran {end.duration_seconds:.2f}s under a declared "
        f"tool_timeout_seconds={TOOL_TIMEOUT_SECONDS}s — the cap is not "
        f"in effect (issue #735: measured at 60.02s pre-fix)"
    )
