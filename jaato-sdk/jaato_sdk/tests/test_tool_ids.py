"""Hashed tool/category id resolution (#1650).

The rules the TUI and the web coder agreed on, now in the SDK.  Mirror of
``jaato-sdk-ts/src/toolIds.test.ts``.
"""

from jaato_sdk import ToolIdNames, resolve_tool_ids, tool_id_mappings
from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.events import ClientType, EventType, SessionInfoEvent, ToolIdRegistryEvent

NAMES = {"c_bbc5e661": "filesystem", "t_c162b0f2": "readFile"}


def test_resolves_an_exact_id_through_lists_and_dicts():
    args = {"category_id": "c_bbc5e661", "tool_ids": ["t_c162b0f2", "t_ffffffff"], "n": 3}
    assert resolve_tool_ids(args, NAMES) == {
        "category_id": "filesystem",
        "tool_ids": ["readFile", "t_ffffffff"],
        "n": 3,
    }
    assert args["tool_ids"][0] == "t_c162b0f2", "input is not mutated"


def test_unknown_ids_substrings_and_keys_are_left_alone():
    assert resolve_tool_ids("t_00000000", NAMES) == "t_00000000"
    assert resolve_tool_ids("see t_c162b0f2", NAMES) == "see t_c162b0f2"
    assert resolve_tool_ids({"t_c162b0f2": 1}, NAMES) == {"t_c162b0f2": 1}


def test_tool_id_mappings_keeps_str_to_str_only():
    assert tool_id_mappings({"c_1": "system", "bad": 3}) == {"c_1": "system"}
    assert tool_id_mappings(None) is None
    assert tool_id_mappings([]) is None


def test_registry_event_merges_and_never_drops_a_name_1681():
    """A daemon before #1681 sent maps missing its runner-tier names from
    the event loop; replacing with one lost ``cli`` / ``prompt.*``."""
    n = ToolIdNames({"c_cf07194e": "cli"})
    assert n.observe(ToolIdRegistryEvent(mappings=NAMES)) is True
    assert n.mappings == {**NAMES, "c_cf07194e": "cli"}
    assert n.observe(SessionInfoEvent(tool_id_mappings={"t_x": "x"})) is True
    assert n.name("c_cf07194e") == "cli"
    assert n.observe(ToolIdRegistryEvent(mappings={})) is False
    assert n.name("c_cf07194e") == "cli"


def test_replace_is_still_wholesale():
    n = ToolIdNames({"t_old": "old"})
    n.replace(NAMES)
    assert n.mappings == NAMES


def test_session_info_seeds_only_when_non_empty():
    n = ToolIdNames()
    assert n.observe(SessionInfoEvent(tool_id_mappings=NAMES)) is True
    assert n.observe(SessionInfoEvent(tool_id_mappings={})) is False
    assert n.name("t_c162b0f2") == "readFile"


def test_raw_wire_dicts_are_accepted():
    n = ToolIdNames()
    assert n.observe({"type": "tools.id_registry", "mappings": NAMES}) is True
    assert n.observe({"type": "agent.output", "mappings": {"x": "y"}}) is False
    assert n.resolve(["c_bbc5e661"]) == ["filesystem"]


def test_late_registry_still_applies_at_read_time():
    n = ToolIdNames()
    args = {"tool_ids": ["t_c162b0f2"]}
    assert n.resolve(args) == args
    n.observe(ToolIdRegistryEvent(mappings=NAMES))
    assert n.resolve(args) == {"tool_ids": ["readFile"]}


def test_track_follows_a_real_client_dispatch_and_unsubscribes():
    client = IPCClient("/nonexistent.sock", client_type=ClientType.API)
    n = ToolIdNames()
    off = n.track(client)
    client._dispatch(SessionInfoEvent(tool_id_mappings={"t_a": "a"}))
    assert n.name("t_a") == "a"
    client._dispatch(ToolIdRegistryEvent(mappings=NAMES))
    assert n.mappings == {**NAMES, "t_a": "a"}
    off()
    client._dispatch(ToolIdRegistryEvent(mappings={"t_z": "z"}))
    assert n.name("t_z") == "t_z"
    assert EventType.TOOL_ID_REGISTRY.value == "tools.id_registry"
