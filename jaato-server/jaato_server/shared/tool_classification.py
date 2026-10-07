"""One coarse class per tool, so a client can weigh a row and a policy can
say "auto-allow bookkeeping" without either growing its own name list
(the final rollout item from jaato/#1304, "Permissions + diffs").

THE VOCABULARY IS CLOSED, and it is the issue's own: ``housekeeping``
(createPlan, startPlan, setStepStatus, completePlan, list_tools,
get_tool_schemas, listReferences, list_subagent_profiles,
subscribeToEvents) / ``write`` / ``exec`` / ``read`` / ``agent`` / ``other``.
``other`` is not a sixth category to fill in later — it is what an
unrecognised name gets, including every MCP tool (``mcp__server__tool``)
and any in-tree tool this table has no opinion about.  A table is a source
of classes, never of invented ones.

WHY A NAME TABLE, NOT A ``TRAIT_*``.  The tree's own convention (see
CLAUDE.md's "Tool Traits") is that a ``ToolSchema.traits`` declaration is
right when a tool's OWN plugin should assert something about its OWN
contract — ``TRAIT_FILE_WRITER`` binds a result shape (``path`` /
``files_modified`` / ``changes[].file``) a tool's executor must honour, and
``TRAIT_UNTRUSTED_CONTENT`` binds a security boundary the provider
converter enforces.  ``tool_class`` binds neither: it is a presentation /
policy grouping laid ON TOP of ~40 tools spread across a dozen-plus
plugins (``todo``, ``introspection``, ``references``, ``subagent``,
``event_bus_tools``, ``file_edit``, ``cli``, ``interactive_shell``,
``notebook``, ...), several of which declare no traits at all today and
none of which need to know their calls get folded into one transcript
line or auto-approved by a profile flag.  Spreading nine ``TRAIT_*``
constants across those plugins to express one nine-tool list is the
inversion CLAUDE.md's own traits intro warns against — hardcoding tool
names into cross-cutting behaviour is exactly what traits exist to
avoid, and a distributed declaration a UI concern owns is still a
hardcoded name, just filed under more headings.  Two of the five
non-``other`` classes (``exec``, ``agent``) also have no existing trait to
reuse: there is no ``TRAIT_EXEC`` or ``TRAIT_SPAWNS_AGENT`` today, so a
trait-based design would still need new vocabulary invented here, in
the same file, for the same nine-and-forty names — a table gets there
in one module instead of thirteen.

So: ONE table, here, mirroring
``jaato-web-coder-ui/src/protocol/toolClass.ts`` by construction (same
five sets, same names, same fallback rule) so a client that already
carries its own copy — for a daemon predating protocol support, or for
a name this table does not yet cover — reads the identical answer
whichever side computed it.  The two are not enforced to stay identical
by a cross-language guard (out of scope here); a name added to one and
not the other degrades to ``"other"`` on the side that missed it, which
is the safe direction — no tool call is ever assigned a WRONG class,
only possibly the generic one.

WHERE THIS IS READ.  ``ToolCallStartEvent.tool_class`` and
``PermissionRequestedEvent.tool_class`` (both protocol-additive, no
version bump — see the fields' own docstrings) are populated from this
one function at the two points those events are built:
``jaato_server/server/core.py`` (``on_tool_call_start`` and the history-replay
path) and ``jaato_server/server/runner_rpc_handlers/prompt_operator.py`` (the
runner-served ASK relay).  ``plugin_configs.permission.auto_allow_housekeeping``
(see ``shared/plugins/permission/plugin.py``) reuses the exact same
``HOUSEKEEPING_TOOLS`` set rather than restating the list a third time.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

#: The closed vocabulary a tool call may be classified as.  Kept as plain
#: strings (not an enum) because this rides the wire in two Pydantic events
#: and a plugin-config knob's whitelist, all of which read a bare string.
TOOL_CLASSES: FrozenSet[str] = frozenset(
    {"housekeeping", "write", "exec", "read", "agent", "other"}
)

#: The issue's own housekeeping list, verbatim: plan bookkeeping plus the
#: discovery/subscription calls a model makes before it does anything a
#: person cares about.  Also the set ``auto_allow_housekeeping`` whitelists
#: (#3 of the phase-3 server changes) — one definition, two consumers.
HOUSEKEEPING_TOOLS: FrozenSet[str] = frozenset(
    {
        "createPlan",
        "startPlan",
        "setStepStatus",
        "completePlan",
        "list_tools",
        "get_tool_schemas",
        "listReferences",
        "list_subagent_profiles",
        "subscribeToEvents",
    }
)

#: ``file_edit`` -- anything that changes bytes on disk.
_WRITE_TOOLS: FrozenSet[str] = frozenset(
    {
        "writeNewFile",
        "updateFile",
        "removeFile",
        "moveFile",
        "renameFile",
        "multiFileEdit",
        "findAndReplace",
        "restoreFile",
        "undoFileChange",
    }
)

#: A subprocess, a shell session, or a notebook cell -- code that RUNS.
_EXEC_TOOLS: FrozenSet[str] = frozenset(
    {
        "cli_based_tool",
        "shell_spawn",
        "shell_input",
        "shell_read",
        "shell_control",
        "shell_close",
        "notebook_execute",
    }
)

#: Looks without changing anything.
_READ_TOOLS: FrozenSet[str] = frozenset(
    {
        "readFile",
        "glob_files",
        "grep_content",
        "listBackups",
        "shell_list",
        "retrieve_memories",
        "selectReferences",
    }
)

#: Spawns, messages or lists other sessions/subagents.
_AGENT_TOOLS: FrozenSet[str] = frozenset(
    {
        "spawn_subagent",
        "close_subagent",
        "cancel_subagent",
        "send_to_subagent",
        "list_subagents",
        "send_to_sibling",
        "list_siblings",
        "send_to_session",
        "list_group_sessions",
    }
)

_TABLE: Dict[str, str] = {
    **{name: "housekeeping" for name in HOUSEKEEPING_TOOLS},
    **{name: "write" for name in _WRITE_TOOLS},
    **{name: "exec" for name in _EXEC_TOOLS},
    **{name: "read" for name in _READ_TOOLS},
    **{name: "agent" for name in _AGENT_TOOLS},
}


def classify_tool(tool_name: str) -> str:
    """A tool's coarse class by its DISPLAY name.

    ``tool_name`` is the name a client would show a person -- the
    un-hashed name, never the ``t_xxxxxxxx`` wire id (#873's hashed ids
    carry no information about what the tool does, and the classification
    call sites here all still hold the display name at the point they
    build the event).

    An unrecognised name -- an MCP tool, a plugin this table has no
    opinion about (``store_memory``, ``web_search``, ``call_service``,
    ...) -- is ``"other"`` rather than a guess.
    """
    return _TABLE.get(tool_name, "other")
