"""Agent lifecycle tools — registered as core tools in JaatoSession.

Provides ``signal_completion``, which lets the main agent declare its
work is done.  This emits ``AgentCompletedEvent`` through the
session's UI hooks — the same mechanism subagents use — enabling
downstream reactors (e.g. memory-advisor) to trigger.

Subagents get completion signaling for free from the subagent plugin
which controls their lifecycle.  The main agent has no host, so it
needs this tool to signal explicitly.

Registered as a core tool (not a plugin) so it is available regardless
of the profile's plugin list.

**Typed completion payloads.**  When the active session's profile
declared a ``completion_payload_schema`` field, the tool's parameters
are dynamically rebuilt: the legacy ``summary: str`` is replaced with
``payload: <schema>``.  The schema is embedded directly in the tool
parameters JSON Schema so providers that constrain tool calls at
sampling (Anthropic, OpenAI, Google, Ollama, LM Studio) enforce the
shape automatically; ``jsonschema.validate`` runs server-side as
defense-in-depth and on validation failure returns a structured error
to the model so it can self-correct on its next turn.  The validated
payload is forwarded to ``hooks.on_agent_completed(payload=...)`` for
reactor consumers to read as typed fields.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from jaato_sdk.plugins.model_provider.types import ToolSchema

from .completion_schema_loader import resolve_completion_schema

if TYPE_CHECKING:
    from shared.jaato_session import JaatoSession

logger = logging.getLogger(__name__)


class LifecycleTools:
    """Agent lifecycle signaling tools.

    Instantiated per-session in ``JaatoSession.configure()`` and
    registered via ``registry.register_core_tool()``.

    On construction, resolves the session's
    ``_completion_payload_schema`` (inline dict or path under
    ``.jaato/completion_schemas/``) so that ``get_tool_schemas()`` can
    decide between the legacy ``summary: str`` shape and the typed
    ``payload: <schema>`` shape without re-resolving on every call.

    Args:
        session: The owning JaatoSession.
    """

    def __init__(self, session: 'JaatoSession') -> None:
        self._session = session
        self._payload_schema: Optional[Dict[str, Any]] = resolve_completion_schema(
            getattr(session, '_completion_payload_schema', None),
            workspace_path=getattr(session, 'workspace_path', None),
        )

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return the ``signal_completion`` schema for the active profile.

        When no ``completion_payload_schema`` is declared, returns the
        legacy shape (``summary: str``).  When one is declared, embeds
        the resolved JSON Schema as the ``payload`` parameter so
        providers enforce it at sampling time.
        """
        if self._payload_schema is None:
            parameters = {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "Brief summary of what was accomplished."
                        ),
                    },
                },
                "required": ["summary"],
            }
        else:
            parameters = {
                "type": "object",
                "properties": {
                    "payload": self._payload_schema,
                },
                "required": ["payload"],
            }

        schemas: List[ToolSchema] = [
            ToolSchema(
                name="signal_completion",
                description=(
                    "Signal that you have finished all your work and have "
                    "nothing left to do.  This triggers downstream agents "
                    "(e.g. memory curator) and allows the session to be "
                    "cleaned up.  Call this as your very last action."
                ),
                parameters=parameters,
                discoverability="core",
            ),
        ]

        # Per-turn model-tier switching.  Only registered when the
        # session has tier mode active — single-model sessions don't
        # see this tool at all (no protocol noise, full backwards
        # compat).  See ``shared/model_tiers.py`` for the resolved
        # config and ``project_backlog_per_turn_model.md`` for the design.
        if getattr(self._session, '_tier_config', None) is not None:
            schemas.append(self._enter_tier_schema())

        return schemas

    def _enter_tier_schema(self) -> ToolSchema:
        """Build the ``enter_tier`` tool schema.

        Three named tiers (``planner`` / ``dispatcher`` / ``executor``)
        constrain the parameter via ``oneOf`` so providers that enforce
        tool params at sampling time reject invalid names before they
        ever reach the executor.  The description block enumerates each
        tier's role explicitly — that's the model's main protocol
        reference once the system-prompt augmentation reminds it of
        which tier it currently occupies.
        """
        from .model_tiers import TIER_PLANNER, TIER_DISPATCHER, TIER_EXECUTOR
        return ToolSchema(
            name="enter_tier",
            description=(
                "Switch the session's active model tier.  Three tiers "
                "are available; pick the one that matches what you're "
                "about to do:\n\n"
                "* `planner` — deep thought, multi-step reasoning, "
                "complex problem decomposition.  Most expensive; use "
                "when you genuinely need the strongest model.\n"
                "* `dispatcher` — coordination, light reasoning, "
                "deciding which tools to call.  Default starting tier.\n"
                "* `executor` — mechanical tool calls and result "
                "interpretation when the plan is clear.  Cheapest; use "
                "when the work doesn't need reasoning.\n\n"
                "Switching is cheap (no network round-trip; just "
                "re-points the active provider).  After your work at "
                "the new tier is done, switch back via another "
                "`enter_tier` call.  Calling with the tier you're "
                "already in is a no-op."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [TIER_PLANNER, TIER_DISPATCHER, TIER_EXECUTOR],
                        "description": (
                            "Target tier name.  Must be one of "
                            f"{TIER_PLANNER}/{TIER_DISPATCHER}/{TIER_EXECUTOR}."
                        ),
                    },
                },
                "required": ["name"],
            },
            discoverability="core",
        )

    def get_executors(self) -> Dict[str, Any]:
        executors: Dict[str, Any] = {
            "signal_completion": self._execute_signal_completion,
        }
        if getattr(self._session, '_tier_config', None) is not None:
            executors["enter_tier"] = self._execute_enter_tier
        return executors

    def get_auto_approved_tools(self) -> List[str]:
        approved = ["signal_completion"]
        if getattr(self._session, '_tier_config', None) is not None:
            approved.append("enter_tier")
        return approved

    def _execute_enter_tier(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Switch the session's active tier per the model's request.

        Validates the ``name`` argument against the three valid tier
        identifiers (the schema's ``enum`` already constrains compliant
        providers, but defence-in-depth — providers without enum
        enforcement could leak through), then delegates to
        ``JaatoSession.switch_tier`` for the actual provider mutation.
        Tool errors are returned as ``error`` fields the model can
        read and self-correct from.
        """
        from .model_tiers import VALID_TIER_NAMES

        requested = args.get("name")
        if not isinstance(requested, str) or not requested.strip():
            return {
                "error": "invalid_argument",
                "message": "enter_tier requires 'name' to be a non-empty string.",
            }
        requested = requested.strip()
        if requested not in VALID_TIER_NAMES:
            return {
                "error": "invalid_tier",
                "message": (
                    f"unknown tier {requested!r}; "
                    f"must be one of {sorted(VALID_TIER_NAMES)}."
                ),
            }
        try:
            return self._session.switch_tier(requested)
        except RuntimeError as exc:
            return {"error": "tier_mode_inactive", "message": str(exc)}
        except Exception as exc:
            logger.warning("enter_tier failed for tier %r: %s", requested, exc)
            return {
                "error": "switch_failed",
                "message": (
                    f"Could not switch to tier {requested!r}: {exc}. "
                    f"The session is still at its previous tier."
                ),
            }

    def _execute_signal_completion(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Emit AgentCompletedEvent for the calling agent.

        With no ``completion_payload_schema``: reads the legacy
        ``summary`` string and emits the event with ``payload=None``.

        With a schema declared: validates ``payload`` against the
        resolved schema using ``jsonschema``.  On validation failure,
        returns a structured error to the model (no event emission)
        so the model can self-correct on its next turn.  On success,
        forwards the validated payload to
        ``hooks.on_agent_completed(payload=...)`` and a derived
        ``summary`` (from the payload's ``summary`` field if present,
        otherwise empty) for reactor consumers that still read the
        legacy field.
        """
        payload: Optional[Dict[str, Any]]
        summary: str

        if self._payload_schema is None:
            summary = args.get("summary", "")
            payload = None
        else:
            payload = args.get("payload")
            try:
                import jsonschema
                jsonschema.validate(instance=payload, schema=self._payload_schema)
            except jsonschema.ValidationError as exc:
                # Return structured error to model, do not emit completion event
                logger.info(
                    "signal_completion payload validation failed: %s",
                    exc.message,
                )
                return {
                    "error": "validation_failed",
                    "message": (
                        "The 'payload' argument did not match the profile's "
                        "completion_payload_schema. Fix the payload and call "
                        "signal_completion again."
                    ),
                    "validation_error": exc.message,
                    "schema_path": list(exc.absolute_path),
                }
            # Derive legacy summary field for backwards-compatible consumers
            summary = (
                payload.get("summary", "")
                if isinstance(payload, dict)
                else ""
            )

        hooks = getattr(self._session, '_ui_hooks', None)
        if not hooks or not hasattr(hooks, 'on_agent_completed'):
            return {"error": "No UI hooks available"}

        agent_id = getattr(self._session, '_agent_id', 'main')
        usage = (
            self._session.get_context_usage()
            if hasattr(self._session, 'get_context_usage')
            else {}
        )

        hooks.on_agent_completed(
            agent_id=agent_id,
            completed_at=datetime.now(),
            success=True,
            token_usage=usage.get('total_tokens'),
            turns_used=usage.get('turns'),
            payload=payload,
        )

        logger.info(
            "Agent %s signaled completion: %s",
            agent_id,
            (summary[:80] if summary else ("(typed payload)" if payload else "(no summary)")),
        )

        result: Dict[str, Any] = {
            "status": "completed",
            "agent_id": agent_id,
            "summary": summary,
        }
        if payload is not None:
            result["payload"] = payload
        return result
