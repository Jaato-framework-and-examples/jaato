"""A pool slot that loaded an embedding model is retired, not reused (#1565).

The first fix released the provider at ``session.end``.  Measured live, the
slot still went back to the pool with ~1.07 GB of anonymous heap: dropping
the references frees the objects, but glibc keeps the pages and torch's
import-time heap is returned only by process exit.  So the references
plugin now says, through ``slot_retire_reason()``, that a model was loaded
in this process; the runner's ``session.end`` reports it as
``retire_slot``; and the daemon closes the slot instead of returning it,
counting ``pool_slot_retired_total``.

No model and no runner process: the provider is the counting fake of the
#1562 guard, the runner's sweep runs over a real ``PluginRegistry`` stand-in,
and the daemon's decision is driven through ``JaatoServer._settle_pool_slot``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from jaato_server.server.core import JaatoServer
from jaato_server.server.runner.rpc import slot_retire_reasons
from jaato_server.shared.tests.reversion import Reversion
from jaato_server.shared.tests.test_references_embedder_memory_1562 import (  # noqa: F401 — fixture
    DEFAULT_MODEL,
    OTHER_MODEL,
    _plugin,
    _workspace,
    factory,
)

_PLUGIN = "jaato-server/jaato_server/shared/plugins/references/plugin.py"
_RPC = "jaato-server/jaato_server/server/runner/rpc.py"
_CORE = "jaato-server/jaato_server/server/core.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find="        self._model_load_attempted = True\n",
        replace="",
        because="the plugin would never say a model was loaded, so the slot keeps ~1 GB (#1565)",
        test="TestPluginSaysSo::test_a_loaded_model_gives_a_reason",
    ),
    Reversion(
        target=_RPC,
        find="            reasons.append(reason)\n",
        replace="            pass\n",
        because="the runner would drop every plugin's reason (#1565)",
        test="TestRunnerReports::test_reasons_are_collected",
    ),
    Reversion(
        target=_CORE,
        find="        if errors == [] and retire:\n",
        replace="        if False:\n",
        because="the daemon would pool a slot the runner asked to retire (#1565)",
        test="TestDaemonRetires::test_a_slot_with_a_reason_is_not_pooled",
    ),
]


class TestPluginSaysSo:
    def test_a_loaded_model_gives_a_reason(self, tmp_path, factory):
        plugin = _plugin(_workspace(tmp_path, DEFAULT_MODEL))  # eager load
        plugin.reset_for_next_session()
        assert plugin.slot_retire_reason()

    def test_no_load_gives_none(self, tmp_path, factory):
        # The index was built with another model, so nothing loads (#1562).
        plugin = _plugin(_workspace(tmp_path, OTHER_MODEL))
        plugin.reset_for_next_session()
        assert plugin.slot_retire_reason() is None


class _Registry:
    def __init__(self, plugins: Dict[str, Any]) -> None:
        self._plugins = plugins

    def list_available(self) -> List[str]:
        return list(self._plugins)

    def get_plugin(self, name: str) -> Any:
        return self._plugins[name]


class TestRunnerReports:
    def test_reasons_are_collected(self):
        def boom():
            raise RuntimeError("probe")
        registry = _Registry({
            "plain": SimpleNamespace(),
            "quiet": SimpleNamespace(slot_retire_reason=lambda: None),
            "raises": SimpleNamespace(slot_retire_reason=boom),
            "model": SimpleNamespace(slot_retire_reason=lambda: "model loaded"),
        })
        assert slot_retire_reasons(registry) == ["model loaded"]

    def test_no_reasons_when_nobody_answers(self):
        assert slot_retire_reasons(_Registry({"a": SimpleNamespace()})) == []


class _Pool:
    def __init__(self) -> None:
        self.returned: List[Any] = []
        self.retired: List[Any] = []

    def return_slot_after_session(self, slot: Any) -> bool:
        self.returned.append(slot)
        return True

    def note_slot_retired(self, slot: Any) -> None:
        self.retired.append(slot)


def _settle(result: Dict[str, Any]):
    server = JaatoServer.__new__(JaatoServer)
    server._session_id = "s1"
    slot = SimpleNamespace(pid=4242, cascade_id=None, last_session_id=None, rpc=None)
    pool = _Pool()
    rpc = SimpleNamespace(reset_for_slot_reuse=lambda: None)
    kept = server._settle_pool_slot(rpc, slot, pool, result)
    return kept, pool, slot


class TestDaemonRetires:
    def test_a_slot_with_a_reason_is_not_pooled(self):
        kept, pool, slot = _settle(
            {"plugins_reset": 3, "errors": [], "retire_slot": ["model loaded"]})
        assert kept is False
        assert pool.returned == [] and pool.retired == [slot]

    def test_a_slot_without_a_reason_is_pooled(self):
        kept, pool, slot = _settle({"plugins_reset": 3, "errors": []})
        assert kept is True and pool.returned == [slot] and pool.retired == []

    def test_reset_errors_still_close_the_slot(self):
        kept, pool, _ = _settle({"plugins_reset": 3, "errors": ["x: boom"]})
        assert kept is False and pool.returned == [] and pool.retired == []
