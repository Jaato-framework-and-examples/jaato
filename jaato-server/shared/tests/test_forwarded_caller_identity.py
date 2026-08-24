"""A daemon-forwarded executor must be able to name the calling session.

``daemon.plugin_execute`` ships ``plugin_name``, ``tool_name`` and ``args``
— and no caller identity.  So a daemon-side plugin instance answering a
forwarded call has no idea who asked.  For ``subagent.list_siblings``,
whose entire job is "describe the cascade around ME", that is fatal: it
failed EVERY forwarded call.

The identity was never missing from the daemon, only from the CALL.  The
``PluginRegistry`` is built per ``JaatoServer`` and a ``JaatoServer`` is
per session, so the registry already knows — and the plugin already holds
the registry via ``set_plugin_registry``.

The previous code read ``self._daemon_session_id`` with a fallback to
``self._session._session_id``.  NOTHING sets ``_daemon_session_id`` on a
plugin (it is a ``JaatoSession`` attribute), and the daemon-side instance
has no ``_session`` — a fallback chain in which neither link could ever be
reached, which is why the guard could never pass.
"""

from types import SimpleNamespace

from shared.plugins.registry import PluginRegistry
from shared.plugins.subagent.plugin import SubagentPlugin
from shared.tool_result_builder import split_executor_result


def test_registry_exposes_the_session_id_it_was_given():
    reg = PluginRegistry()
    assert reg.session_id is None, "absent until stamped"
    reg.set_session_id("sess-A")
    assert reg.session_id == "sess-A"


def test_registry_session_id_is_per_session():
    """The invariant the identity fix rests on.

    Two sessions mean two registries mean two ids.  If a shared registry
    is ever introduced, this fails HERE — loudly, next to the assumption
    — rather than silently handing one session's roster to another.
    """
    a, b = PluginRegistry(), PluginRegistry()
    a.set_session_id("sess-A")
    b.set_session_id("sess-B")
    assert a.session_id == "sess-A"
    assert b.session_id == "sess-B"
    assert a is not b


def _plugin_with(registry, manager):
    p = SubagentPlugin()
    p.set_plugin_registry(registry)
    if manager is not None:
        p.set_session_manager(manager)
    return p


def test_list_siblings_asks_the_registry_who_is_calling():
    """The forwarded call resolves its caller and builds that roster."""
    asked = {}

    class Manager:
        def build_sibling_roster(self, sid):
            asked["sid"] = sid
            return {"siblings": [{"sibling_name": "sibling-b"}]}

    reg = PluginRegistry()
    reg.set_session_id("sess-A")
    plugin = _plugin_with(reg, Manager())

    ok, data = split_executor_result(plugin._execute_list_siblings({}))

    assert ok is True
    assert asked["sid"] == "sess-A", "the roster must be built for the CALLER"
    assert data["status"] == "ok"
    assert data["siblings"] == [{"sibling_name": "sibling-b"}]


def test_two_sessions_get_their_own_rosters():
    """Identity is read per call, not captured once at wiring time."""
    seen = []

    class Manager:
        def build_sibling_roster(self, sid):
            seen.append(sid)
            return {"siblings": []}

    manager = Manager()
    for sid in ("sess-A", "sess-B"):
        reg = PluginRegistry()
        reg.set_session_id(sid)
        _plugin_with(reg, manager)._execute_list_siblings({})

    assert seen == ["sess-A", "sess-B"]


def test_unidentifiable_caller_fails_visibly():
    """No identity is an ERROR, not an empty roster.

    An unanswerable question must not return a confident empty answer —
    "you have no siblings" and "I could not tell who you are" are
    different facts, and only one of them is safe to act on.
    """
    reg = PluginRegistry()          # never stamped

    class Manager:
        def build_sibling_roster(self, sid):
            raise AssertionError("must not be reached without an identity")

    ok, data = split_executor_result(
        _plugin_with(reg, Manager())._execute_list_siblings({})
    )

    assert ok is False, "must signal failure through the executor contract"
    assert "error" in data, "must carry the key the body check reads"
    assert "session" in data["error"].lower()
