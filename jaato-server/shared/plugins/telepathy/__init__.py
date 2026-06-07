"""Telepathy plugin — subagent→parent context-sharing tool.

Provides the ``share_context`` tool which lets a subagent push
structured context (file contents, findings, notes) up to its
parent agent without the parent re-reading or re-executing.

**Non-core, opt-in.**  The tool is only meaningful when the session
has a parent (i.e. when the session is a subagent of another
session — ``_parent_session`` set via
``JaatoSession.set_parent_session``).  Operators include this
plugin in a profile's ``plugins:`` list ONLY for profiles that
will be spawned as subagents:

.. code-block:: yaml

    # subagent profile
    plugins:
      - telepathy   # so this subagent can share back to its parent
      - cli

Root / main sessions and one-shot orchestrators don't list
``telepathy`` in their ``plugins:`` — share_context wouldn't make
sense for them anyway (no parent to share to), and the framework's
per-turn visibility filter (PR #241) would hide the tool even if
the plugin were loaded.  Belt-and-braces: the plugin only loads
when explicitly listed, AND the visibility predicate gates it on
``_parent_session is not None`` per turn.

History (2026-06-07): the tool used to live as a session built-in
in ``shared/jaato_session.py`` (``_register_telepathy_tool``,
``_execute_share_context``, ``_format_shared_context``) — registered
unconditionally at ``configure()`` time, so root sessions saw it
on their tool surface even though ``_execute_share_context`` would
reject every call with "No parent session available."  Extracted
to this plugin per maintainer guidance: tools belong in plugins,
not session built-ins.  See the vLLM smoke wire-dump investigation
for the surfacing.
"""

from .plugin import TelepathyPlugin, create_plugin

# Plugin kind identifier for registry discovery
PLUGIN_KIND = "tool"

# Runner tier — the executor calls ``parent_session.inject_prompt``
# which crosses the runner-side session boundary; lives runner-side
# alongside the session it's attached to.
PLUGIN_TIER = "runner"

__all__ = [
    'TelepathyPlugin',
    'create_plugin',
    'PLUGIN_KIND',
    'PLUGIN_TIER',
]
