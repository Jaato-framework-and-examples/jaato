"""TelepathyPlugin — share_context tool implementation.

Extracted from ``shared/jaato_session.py`` (2026-06-07) per the
maintainer's guidance that tools belong in plugins, not session
built-ins.  See the package ``__init__.py`` for the full history
and rationale.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from jaato_sdk.plugins.model_provider.types import (
    ToolSchema,
    DISCOVERABILITY_EAGER,
)
from shared.session_context import get_current_session

if TYPE_CHECKING:
    from shared.jaato_session import JaatoSession


class TelepathyPlugin:
    """Provides the ``share_context`` tool for subagent→parent context sharing.

    Lifecycle:

    1. ``__init__()`` — empty plugin instance.
    2. ``initialize(config)`` — no-op (no config knobs).
    3. ``_get_session()`` — resolve the host session from the
       per-execution ContextVar (``shared.session_context``), NEVER
       from ``self``.  Plugin instances are shared across sibling
       subagents, so storing the session on ``self`` (the old
       ``set_session`` approach) leaked one subagent's session into
       another.  The session wiring sets the ContextVar in
       ``configure()`` and around every tool execution.
    4. ``is_tool_visible(name)`` — per-turn visibility predicate
       consulted by ``JaatoSession._get_tools_for_provider`` (PR #241).
       Returns ``False`` for ``share_context`` when the host session
       has no parent (calling it would just error "No parent session
       available" at the executor); returns ``True`` for any other
       tool name (other plugins' tools are not this plugin's
       concern).
    5. ``get_executors()['share_context']`` runs the actual push to
       ``parent_session.inject_prompt``.

    The visibility predicate is belt-and-braces: this plugin is
    non-core (only loaded when listed in ``profile.plugins``), so
    most root / orchestrator profiles never load it.  Subagent
    profiles that DO list it get the per-turn gate to handle the
    edge case where a session lists the plugin but never has its
    parent wired (e.g. cascade entry-points mis-labeled as
    subagents).
    """

    def __init__(self) -> None:
        self._initialized: bool = False

    @property
    def name(self) -> str:
        return "telepathy"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """No-op initializer.

        Telepathy has no configurable knobs.  The ``config`` arg is
        accepted to satisfy the ``ToolPlugin`` protocol uniformly.
        """
        self._initialized = True

    def shutdown(self) -> None:
        self._initialized = False

    def reset_for_next_session(self) -> None:
        """Cascade-sharing reset — NO-OP for this plugin.

        Added to satisfy the ``ToolPlugin`` protocol's runtime
        ``isinstance`` check (its absence silently dropped the plugin
        at discovery — see the registry protocol-skip warning).  Per
        Daniel's litmus test (``docs/design/runner-cascade-sharing.md``
        §4.3), telepathy holds NO per-session state at all: the host
        session is resolved per-execution from the ContextVar (see
        :meth:`_get_session`), never stored on ``self``.  Nothing to
        clear.
        """
        # Intentionally empty — no per-session state on self.

    # NOTE: deliberately no ``set_session()``.  Plugin instances are
    # shared across sibling subagents within a session, so storing the
    # session on ``self`` would leak one subagent's session into another
    # (and trip tests/test_plugin_session_safety.py).  Resolve it
    # per-execution from the ContextVar instead.
    def _get_session(self) -> Optional['JaatoSession']:
        """Return the current host session, or ``None`` when unset.

        Reads ``shared.session_context.get_current_session()`` — the
        session wiring sets that ContextVar in ``configure()`` (so the
        visibility + instruction phases on the session's main thread see
        it) and around every tool execution (so the executor does too).
        ``LookupError`` means no session in this context (catalog
        discovery before configure, isolated unit tests) → ``None``.
        """
        try:
            return get_current_session()
        except LookupError:
            return None

    def is_tool_visible(self, tool_name: str) -> bool:
        """Per-turn visibility predicate (PR #241 hook).

        ``share_context`` is only meaningful when the host session
        has a parent.  Hide it from the tool surface otherwise so
        the model never sees a tool whose executor would reject
        every call.

        Returns ``True`` for tool names other than ``share_context``
        — this predicate has no opinion on tools owned by other
        plugins.
        """
        if tool_name != "share_context":
            return True
        session = self._get_session()
        if session is None:
            return False
        return getattr(session, '_parent_session', None) is not None

    def get_tool_schemas(self) -> List[ToolSchema]:
        return [
            ToolSchema(
                name='share_context',
                description=(
                    'Share context from your memory with your parent agent. '
                    'Use this to transfer knowledge without the parent needing to '
                    're-read files or re-execute tools.\n\n'
                    'CRITICAL: Share the COMPLETE file content, not summaries or excerpts. '
                    'The parent needs the full content to work with it. Never omit content '
                    '"for brevity" - that defeats the purpose of this tool.\n\n'
                    'IMPORTANT: Do NOT re-read files before sharing. Use your memory of files '
                    'you have already read. Copy the full content from your context.\n\n'
                    'Use this to:\n'
                    '- Share complete file contents you have already read\n'
                    '- Share your analysis or findings\n'
                    '- Share relevant facts you have discovered'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "object",
                            "description": (
                                "Files to share from your memory. Keys are file paths, "
                                "values are the COMPLETE file content from your context. "
                                "Do NOT summarize or omit content - share the full text."
                            ),
                            "additionalProperties": {"type": "string"}
                        },
                        "findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Key findings, facts, or conclusions to share. "
                                "These should be insights from your analysis."
                            )
                        },
                        "notes": {
                            "type": "string",
                            "description": (
                                "Free-form context, analysis, guidance, or explanation "
                                "to help the parent agent understand the shared context."
                            )
                        }
                    },
                    "required": []
                },
                discoverability=DISCOVERABILITY_EAGER,
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        return {"share_context": self._execute_share_context}

    def get_user_commands(self) -> List[Any]:
        return []

    def get_auto_approved_tools(self) -> List[str]:
        # Subagents calling share_context need to push without an
        # interactive permission prompt — there's no human in the
        # subagent's loop.  The parent has its own permission
        # handling on whatever it does with the injected content.
        return ["share_context"]

    def get_system_instructions(self) -> Optional[str]:
        """Describe ``share_context`` to the model — only when usable.

        Required by the ``ToolPlugin`` protocol (its absence silently
        dropped the plugin at discovery).  Mirrors :meth:`is_tool_visible`'s
        parent-gating: the instructions are emitted ONLY when the host
        session has a parent, so the model is never told about a tool the
        per-turn visibility filter is hiding.  Returns ``None`` for root /
        parentless sessions.
        """
        session = self._get_session()
        if session is None:
            return None
        if getattr(session, '_parent_session', None) is None:
            return None
        return (
            "You have access to `share_context`, which pushes context up "
            "to your parent agent without it re-reading files or "
            "re-executing tools. Share COMPLETE file contents (never "
            "summaries or excerpts) from your memory — do not re-read "
            "files first. Use it to hand the parent full file contents you "
            "have already read, plus your findings and explanatory notes."
        )

    def _execute_share_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Push structured context to the host session's parent.

        Body preserved verbatim from the pre-extraction
        ``JaatoSession._execute_share_context`` (2026-06-07 move).
        """
        # Local import to avoid a hard runtime dependency on the
        # session module at plugin-discovery time.
        from shared.message_queue import SourceType

        session = self._get_session()
        if session is None:
            return {
                'success': False,
                'error': (
                    'TelepathyPlugin has no session in context.  The '
                    'session ContextVar must be set (configure() / tool '
                    'execution) before the executor runs.'
                ),
            }

        files = args.get('files', {})
        findings = args.get('findings', [])
        notes = args.get('notes', '')

        # Check if there's anything to share
        if not files and not findings and not notes:
            return {
                'success': False,
                'error': (
                    'No context to share. Provide at least one of: '
                    'files, findings, notes.'
                )
            }

        parent_session = getattr(session, '_parent_session', None)
        if not parent_session:
            # Defensive: the visibility filter should have hidden
            # the tool when no parent — this error path only fires
            # if the model called the tool from cached schema /
            # message history mid-turn after the parent was unset.
            return {
                'success': False,
                'error': (
                    'No parent session available. This agent may '
                    'be the main agent.'
                )
            }

        formatted_context = self._format_shared_context(files, findings, notes)
        agent_id = getattr(session, '_agent_id', 'unknown')

        try:
            # Use same pattern as subagent communication: inject if
            # busy, send if idle.  CHILD source type — will be
            # processed when parent is idle.
            if parent_session.is_running:
                parent_session.inject_prompt(
                    formatted_context,
                    source_id=agent_id,
                    source_type=SourceType.CHILD,
                )
                return {
                    'success': True,
                    'status': 'queued',
                    'message': (
                        'Context queued for parent. Will be '
                        'processed when parent is idle.'
                    ),
                    'shared': {
                        'files': list(files.keys()) if files else [],
                        'findings_count': len(findings) if findings else 0,
                        'has_notes': bool(notes),
                    },
                }

            # Parent is idle — shouldn't normally happen
            # (subagent runs while parent waits) but handle
            # gracefully by injecting anyway.
            parent_session.inject_prompt(
                formatted_context,
                source_id=agent_id,
                source_type=SourceType.CHILD,
            )
            return {
                'success': True,
                'status': 'sent',
                'message': 'Context sent to parent.',
                'shared': {
                    'files': list(files.keys()) if files else [],
                    'findings_count': len(findings) if findings else 0,
                    'has_notes': bool(notes),
                },
            }
        except Exception as e:  # noqa: BLE001
            return {
                'success': False,
                'error': f'Failed to share context: {e}',
            }

    @staticmethod
    def _format_shared_context(
        files: Dict[str, str],
        findings: List[str],
        notes: str,
    ) -> str:
        """Format shared context for injection into parent's conversation.

        Body preserved verbatim from the pre-extraction
        ``JaatoSession._format_shared_context`` (2026-06-07 move).
        """
        parts: List[str] = []

        # Add instruction prefix so the receiving agent knows to use
        # this content directly instead of re-reading files.
        if files:
            parts.append(
                "CONTEXT FROM SUBAGENT: The following files and findings are "
                "shared from the subagent's memory. DO NOT re-read these "
                "files - use the content provided below directly."
            )
            parts.append("")

        parts.append('<shared_context from_agent="subagent">')

        if files:
            parts.append('<files>')
            for path, content in files.items():
                parts.append(f'<file path="{path}">')
                parts.append(content)
                parts.append('</file>')
            parts.append('</files>')

        if findings:
            parts.append('<findings>')
            for finding in findings:
                parts.append(f'  - {finding}')
            parts.append('</findings>')

        if notes:
            parts.append('<notes>')
            parts.append(notes)
            parts.append('</notes>')

        parts.append('</shared_context>')
        return '\n'.join(parts)


def create_plugin() -> TelepathyPlugin:
    return TelepathyPlugin()
