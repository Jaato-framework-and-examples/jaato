"""Every tool-id map the daemon sends carries the runner-tier names (#1681).

``_build_tool_id_mappings`` may read the runner's tool schemas only OFF the
event loop (the read is a ``_threadsafe`` RPC that self-deadlocks on the
loop; see ``test_tool_id_mappings_offloop_gate.py``).  So the runner-tier
names (``cli``, ``prompt.*``, client tools) came only from the off-loop
re-emits, while every ON-loop build (``session.info``'s
``tool_id_mappings``, ``emit_current_state`` on re-attach,
``initialize``) sent a daemon-tier-only map.  Clients replace their map
with each one, so a later on-loop map dropped those names, and a client
that re-attached to a running session (no new bootstrap, so no re-emit)
never got them.  Seen as ``list_tools: {"category_id": "c_cf07194e"}``
(the ``prompt`` category) and ``t_c162b0f2`` (``prompt.jaato-sdk``).

The fix caches what the runner answered on its last off-loop read and
folds it into every build.  These tests drive the real method on a
server built without ``__init__`` (as the off-loop gate test does), with
a stand-in runner RPC and no daemon registry.
"""
import asyncio
from types import SimpleNamespace

from jaato_server.server.core import JaatoServer
from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.tool_id_map import name_to_id


REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/server/core.py",
        find="""        mappings.update(getattr(self, "_runner_tool_id_mappings", {}))
""",
        replace="",
        test="test_an_on_loop_build_carries_the_runner_tier_names",
        because="an on-loop map (session.info, a re-attach) sent without "
                "the runner-tier names, which the client then replaces "
                "its full map with",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/core.py",
        find="""            if runner_mappings:
                self._runner_tool_id_mappings = runner_mappings
""",
        replace="""            self._runner_tool_id_mappings = runner_mappings
""",
        test="test_a_failed_runner_read_keeps_the_names_it_had",
        because="a runner read that failed or answered nothing erasing "
                "the names the previous read found",
    ),
]


CLI = SimpleNamespace(name="cli_based_tool", category="system")
PROMPT = SimpleNamespace(name="prompt.jaato-sdk", category="prompt")


class _RPC:
    """The runner side: answers the schemas it is given, or raises."""

    def __init__(self, schemas):
        self.schemas = schemas

    def session_get_tool_schemas_threadsafe(self):
        if isinstance(self.schemas, Exception):
            raise self.schemas
        return list(self.schemas)


def _server(schemas):
    srv = JaatoServer.__new__(JaatoServer)   # skip the heavy __init__
    srv._runner_rpc = _RPC(schemas)
    srv._runner_tool_id_mappings = {}
    srv.registry = None
    return srv


def _on_loop(srv):
    async def _build():
        return srv._build_tool_id_mappings()
    return asyncio.run(_build())


def _runner_names():
    return {
        name_to_id("cli_based_tool"): "cli_based_tool",
        name_to_id("system", prefix="c"): "system",
        name_to_id("prompt.jaato-sdk"): "prompt.jaato-sdk",
        name_to_id("prompt", prefix="c"): "prompt",
    }


def test_the_issue_ids_are_the_names_it_reports():
    """The ids in the report are these names, so the fixture is the case."""
    assert name_to_id("prompt", prefix="c") == "c_cf07194e"
    assert name_to_id("prompt.jaato-sdk") == "t_c162b0f2"


def test_an_on_loop_build_carries_the_runner_tier_names():
    srv = _server([CLI, PROMPT])
    off_loop = srv._build_tool_id_mappings()   # the runner-ready re-emit
    assert off_loop == _runner_names()
    # Then session.info / a re-attach build ON the loop: same names.
    assert _on_loop(srv) == _runner_names()


def test_an_on_loop_build_before_any_runner_read_is_daemon_tier_only():
    """The control: nothing read yet, nothing invented."""
    srv = _server([CLI, PROMPT])
    assert _on_loop(srv) == {}


def test_a_failed_runner_read_keeps_the_names_it_had():
    srv = _server([CLI, PROMPT])
    srv._build_tool_id_mappings()
    srv._runner_rpc.schemas = RuntimeError("runner busy")
    assert srv._build_tool_id_mappings() == _runner_names()
    srv._runner_rpc.schemas = []
    assert srv._build_tool_id_mappings() == _runner_names()


def test_a_later_runner_read_refreshes_the_names():
    srv = _server([CLI])
    srv._build_tool_id_mappings()
    srv._runner_rpc.schemas = [CLI, PROMPT]   # e.g. client tools registered
    srv._build_tool_id_mappings()
    assert _on_loop(srv) == _runner_names()
