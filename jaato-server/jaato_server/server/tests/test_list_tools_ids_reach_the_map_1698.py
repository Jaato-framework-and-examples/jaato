"""Every id ``list_tools`` hands the model is in the daemon's tool-id map (#1698).

The map's runner-tier names came from ``session.get_tool_schemas``, which
answers the session's ACTIVATED schemas only.  A category whose tools are
all runner-tier and still deferred (``prompt``, from ``prompt_library``)
was in no map the daemon sent, so a client tracking ``ToolIdNames`` showed
``list_tools: {"category_id": "c_cf07194e"}`` unresolved: the very call the
model makes to activate those tools.

The runner now answers a ``catalog`` beside the activated ``schemas``:
every tool name and category its registry exposes to the session, filtered
the way ``list_tools`` filters.  These tests drive a real ``PluginRegistry``
with the real introspection plugin, a real ``RunnerRPC`` /
``RunnerRPCClient`` pair over a socketpair, and the real
``JaatoServer._build_tool_id_mappings``; only the session (nothing
activated) and the deferred plugin are stand-ins.
"""
import asyncio
import socket
import threading
from types import SimpleNamespace
from typing import Any, Dict, Set

import pytest

from jaato_sdk.plugins.model_provider.types import ToolSchema
from jaato_server.server.core import JaatoServer
from jaato_server.server.runner.rpc import RunnerRPC
from jaato_server.server.runner.session import RunnerSessionHost
from jaato_server.server.runner_rpc_client import RunnerRPCClient
from jaato_server.shared.plugins.introspection import plugin as _introspection_module
from jaato_server.shared.plugins.introspection.plugin import IntrospectionPlugin
from jaato_server.shared.plugins.registry import PluginRegistry
from jaato_server.shared.session_envelope import SessionInitEnvelope
from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.tool_id_map import name_to_id


REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/server/runner/rpc.py",
        find="""            "catalog": self._tool_id_catalog(session),
""",
        replace="",
        test="test_every_id_list_tools_hands_out_is_in_the_map",
        because="the runner answering only the activated schemas, so a "
                "deferred runner-tier category never reaches the map",
    ),
    Reversion(
        target="jaato-server/jaato_server/server/runner_rpc_client.py",
        find="""        for key in ("schemas", "catalog"):
""",
        replace="""        for key in ("schemas",):
""",
        test="test_every_id_list_tools_hands_out_is_in_the_map",
        because="the daemon reading the catalog the runner sent and "
                "throwing it away",
    ),
]


class _PromptLibrary:
    """A runner-tier plugin whose one tool is deferred (not activated)."""

    name = "prompt_library"

    def initialize(self, config=None):
        pass

    def shutdown(self):
        pass

    def get_tool_schemas(self):
        return [ToolSchema(
            name="prompt.jaato-sdk",
            description="The jaato-sdk skill",
            parameters={"type": "object", "properties": {}},
            category="prompt",
            discoverability="discoverable",
        )]

    def get_executors(self):
        return {}

    def get_user_commands(self):
        return []

    def get_auto_approved_tools(self):
        return []


def _registry(monkeypatch) -> "tuple[PluginRegistry, IntrospectionPlugin]":
    # Introspection keeps its session in a module-level thread-local.
    monkeypatch.setattr(
        _introspection_module._thread_local, "session", None, raising=False,
    )
    registry = PluginRegistry()
    registry.register_plugin(_PromptLibrary(), expose=True)
    intro = IntrospectionPlugin()
    registry.register_plugin(intro, expose=True)
    intro.set_plugin_registry(registry)
    return registry, intro


def _ids_list_tools_hands_out(intro: IntrospectionPlugin) -> Dict[str, str]:
    """Every ``{id: name}`` a model reads from ``list_tools``."""
    ids: Dict[str, str] = {}
    summary = intro._execute_list_tools({})
    for cat in summary["categories"]:
        ids[cat["id"]] = cat["name"]
        detail = intro._execute_list_tools({"category_id": cat["id"]})
        for tool in detail.get("tools", []):
            ids[tool["id"]] = tool["name"]
    return ids


class _Session:
    """A runner-side session that has activated nothing yet."""

    def __init__(self, registry):
        self._runtime = SimpleNamespace(registry=registry)

    def get_tool_schemas(self):
        return []


def _envelope() -> SessionInitEnvelope:
    return SessionInitEnvelope(
        session_id="sess-1698",
        workspace_path="/tmp/ws",
        profile_name="p",
        provider_name="anthropic",
        model_name="m",
        plugins=[],
    )


def _map_the_daemon_builds(registry) -> Dict[str, str]:
    """Build the map through a real runner RPC pair, off the loop."""
    daemon_sock, runner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_STREAM,
    )
    runner = RunnerRPC(runner_sock, lambda n, a: (False, {"error": "none"}))
    runner._session_host = RunnerSessionHost(
        envelope=_envelope(), runtime=None, session=_Session(registry),
    )
    runner_thread = threading.Thread(target=runner.serve, daemon=True)
    runner_thread.start()

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    async def _start():
        c = RunnerRPCClient(daemon_sock, runner_pid=0)   # bound to ``loop``
        await c.start()
        return c

    client = asyncio.run_coroutine_threadsafe(_start(), loop).result(5)
    try:
        srv = JaatoServer.__new__(JaatoServer)   # skip the heavy __init__
        srv._runner_rpc = client
        srv._runner_tool_id_mappings = {}
        srv.registry = None   # the daemon walk excludes runner tier anyway
        return srv._build_tool_id_mappings()   # this thread: off the loop
    finally:
        runner.shutdown()
        runner_thread.join(timeout=2)
        client._closed = True
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2)


def test_the_issue_id_is_the_prompt_category():
    assert name_to_id("prompt", prefix="c") == "c_cf07194e"


def test_list_tools_hands_out_the_deferred_category(monkeypatch):
    """The precondition: the id the report saw is one list_tools offers."""
    _, intro = _registry(monkeypatch)
    assert _ids_list_tools_hands_out(intro)["c_cf07194e"] == "prompt"


def test_every_id_list_tools_hands_out_is_in_the_map(monkeypatch):
    registry, intro = _registry(monkeypatch)
    handed_out = _ids_list_tools_hands_out(intro)
    mappings = _map_the_daemon_builds(registry)
    missing: Set[str] = {
        i for i, name in handed_out.items() if mappings.get(i) != name
    }
    assert not missing, f"ids list_tools hands out with no name: {missing}"


def test_a_session_without_a_registry_still_answers_its_schemas():
    """The catalog is best effort: no registry, no catalog, no error."""
    assert RunnerRPC._tool_id_catalog(SimpleNamespace()) == []
