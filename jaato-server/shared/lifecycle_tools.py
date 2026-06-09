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

**Typed completion payloads (Option G, server 0.6.115+).**  When the
active session's profile declared a ``completion_payload_schema``
field, the tool's parameters become the schema directly — top-level
properties of the schema are exposed as flat tool args, NOT wrapped
in a single ``payload`` parameter.  The flat shape mitigates the
Anthropic/Bedrock stringification pathology (the model emits one huge
nested arg as a JSON-string instead of an object).

The schema is sent on the wire in the tool's parameter definition;
``jsonschema.validate`` runs server-side after each ``signal_completion``
call as defense-in-depth, and on validation failure returns a
structured error to the model so it can self-correct on its next
turn.  The validated args dict is forwarded to
``hooks.on_agent_completed(payload=...)`` for reactor consumers —
the ``payload`` shape passed to downstream is identical to the
pre-G shape (a flat dict with the schema's properties), so no
consumer-side changes are needed.

.. warning::
   **The schema is ADVISORY at the model layer unless the model
   supports grammar-constrained sampling on its provider path.**

   - Via **OpenRouter**: the model must be on OpenRouter's
     structured-outputs supported list
     (https://openrouter.ai/docs/guides/features/structured-outputs).
     As of 2026-05-16: OpenAI GPT-4o+, Google Gemini, Anthropic
     Sonnet 4.5 / Opus 4.1+, most OSS, Fireworks.
     ``claude-haiku-4.5`` is **NOT** on the list.
   - Via **direct Anthropic API**: tool definitions need ``strict:
     true`` to force schema adherence — see
     https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use.
     Anthropic's default behavior is "advisory" — the model may
     return incompatible types (e.g. ``"2"`` instead of ``2``).
     jaato's provider plugins do not currently set ``strict: true``.

   **When the active model lacks strict-mode support**, the schema
   only constrains the post-hoc ``jsonschema.validate`` check — not
   the model's emission.  Empirically (v109-v112,
   ``feedback_cascade_completion_schemas_require_strict_model_support``
   memory): claude-haiku-4.5 violated a ``{"type": "string",
   "const": "1.0"}`` constraint for 7+ retries despite the schema
   being visible from turn 1.  Adherence in that regime depends on
   weak model priors + persona prose reinforcement.

   **Implications for cascade authors (per Daniel's 2026-05-16 rule):**
   when a cascade depends on schema constraints for determinism
   (``const``, ``enum``, ``format``, ``additionalProperties: false``),
   the order of operations is:

   1. **Check** the model's documented strict-mode / structured-
      outputs support list (Anthropic strict-tool-use docs,
      OpenRouter structured-outputs list, etc.).
   2. **If listed:** enable strict mode via the provider's knobs.
   3. **If not listed:** switch to a model that does support it.

   The framework deliberately does **not** simulate grammar-
   constrained sampling with framework-side prose injection (e.g.
   auto-injecting "field X is the string Y, not a number" into the
   persona).  That would paper over a model limitation with a
   half-measure.  A previous proposal to do so (Option B,
   ``{{!framework:completion_schema}}`` directive) was rejected on
   exactly these grounds.

   Persona prose authored by the kb (in ``.jaato/agents/*.md``) is
   a separate, kb-author-controlled lever — tactically effective
   for unblocking a specific cascade run on an unsupported model
   (v112 evidence), but a workaround for a wrong model choice,
   not a recommended design pattern.

**Schema authoring convention (server 0.6.27+).**  When you author a
``completion_payload_schema``, declare two optional string-array
escape hatches alongside your data fields::

    {
      "type": "object",
      "additionalProperties": false,
      "required": [...],
      "properties": {
        ...your data fields...,
        "warnings": {
          "type": "array",
          "items": { "type": "string" },
          "description": (
            "Advisory non-fatal notes the agent surfaced (skip "
            "decisions, defaulted values, ambiguities)."
          )
        },
        "errors": {
          "type": "array",
          "items": { "type": "string" },
          "description": (
            "Hard failures the agent recovered from "
            "(degraded-mode signals)."
          )
        }
      }
    }

Without these arrays + ``additionalProperties: false``, an agent
whose persona instructs surfacing skip decisions / fallback choices /
ambiguities has nowhere to put that prose.  The structured-output
constraint forces a schema-violation retry, and the retry path is
non-deterministic — different runs produce different stripped
payloads.  Strip ``warnings`` and ``errors`` from the canonical
hash in determinism tests; they are advisory by design and not
load-bearing.

For a worked example see ``feedback_completion_schema_warnings_field``
in the project memory and the
``docs/design/payload-schema-conventions.md`` design document
(unified spawn-side + completion-side guide).

**Completion processors (server 0.6.125+).**  Some bugs are
structurally well-formed but semantically false — the agent's
payload validates against the schema yet its claims contradict the
session's actual tool-call history.  Concrete example
(kb-enablement-2.0 v117): the codegen agent emitted ``files[]``
with 20 entries, 6 of which were fabricated — their
``renderTemplateToFile`` calls had returned errors, but the schema
doesn't know about tool-call history and the agent rationalized
success.

Profiles declare a list of kb-authored Python processors that
run AFTER ``jsonschema.validate`` passes.  This is one unified
surface (replacing the prior ``completion_artifacts`` +
``completion_validators`` split) — each processor module can
PRODUCE output and/or VALIDATE the payload via probe-by-symbol::

    {
      "name": "codegen",
      "completion_payload_schema": "completion_schemas/step_result.json",
      "completion_processors": [
        {"script": "scripts/processors/codegen_files_exist.py",
         "on_error": "fail_completion"},
        {"script": "scripts/processors/render_audit.py",
         "output": "audit/{case_id}.json"}
      ]
    }

Each module exposes one or both top-level callables::

    def render(payload: dict, context: RenderContext) -> str | bytes:
        # Produces output content; written to disk when the entry
        # declares ``output:`` — validator-as-renderer when ``output``
        # is omitted.

    def validate(payload: dict, context: RenderContext) -> list[str]:
        # Returns error strings; empty = pass, non-empty = block
        # completion per ``on_error`` policy.

``context.tool_calls`` carries the pre-computed ledger of every
function_call + function_response in the session (paired by
call_id) — use it to cross-check payload claims against actual
tool outcomes.  See
:class:`shared.dynamic_instructions.RenderContext` for the full
context shape.

Paths resolve via the same loader as prefetch scripts and reactor
handlers — see :func:`shared.script_loader.resolve_script_path`:

1. absolute path → as-is
2. ``<config_root>/<path>`` (workspace-tier kb)
3. ``~/.jaato/<path>`` (user-tier fallback)

Errors from any processor are aggregated (not short-circuited —
the agent sees the full picture on retry) and returned as the
same ``validation_failed`` shape used by schema failures.  See
``shared/completion_processors.py`` for the implementation and
``shared/tests/test_completion_processors.py`` for end-to-end
examples.
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
            # JaatoSession adopts the runtime's config_root via
            # ``runtime._config_root``; honor it here so a profile that
            # references ``"completion_payload_schema": "completion_schemas/<name>.json"``
            # resolves under the override path instead of the
            # workspace's ``.jaato/completion_schemas/``.
            config_root=getattr(
                getattr(session, 'runtime', None), '_config_root', None,
            ),
        )
        # Lazy-loaded completion processors (kb-authored Python).  None
        # until the first ``signal_completion`` call resolves the
        # configured entries via
        # ``shared.completion_processors.load_processors``; subsequent
        # calls in the same session reuse the cached list.  Each
        # element is a ``LoadedProcessor`` carrying the original
        # ``CompletionProcessor`` entry plus probed ``render`` /
        # ``validate`` callables; ``load_error`` surfaces typos /
        # missing symbols to the agent at signal time.
        self._processors_loaded: Optional[List[Any]] = None

        # Accumulated payload state (server 0.6.198+, 2026-06-09) for the
        # ``prepare_completion`` / ``query_completion`` / arg-less
        # ``signal_completion`` triple.  Closes the composition-burden
        # failure on small models (qwen3-14b @ temp=0): the model
        # collapses to ``args={}`` when forced to compose the entire
        # completion payload in one tool emission, even though it can
        # produce each piece individually (interrogator-probe proven).
        # The triple lets the agent transcribe partials across many
        # turns ("readFile-style" — one observation per call), inspect
        # accumulated state via ``query_completion``, and finally call
        # ``signal_completion()`` arg-less to synthesize from
        # accumulated.  Legacy ``signal_completion(args=full)`` single-
        # shot path remains for capable models.  See
        # ``feedback_small_model_narration_skipping_is_structural`` for
        # the empirical chain that motivated this design.
        self._accumulated_payload: Dict[str, Any] = {}

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return the ``signal_completion`` schema for the active profile.

        When no ``completion_payload_schema`` is declared, returns the
        legacy shape (``summary: str``).  When one is declared, embeds
        the resolved JSON Schema as the ``payload`` parameter so
        providers enforce it at sampling time.

        **Interactive-root filter (server 0.6.61+).**  Sessions that are
        BOTH a root session (no parent_session) AND connected via an
        interactive client (``client_type ∈ {terminal, web, chat}``)
        DO NOT see ``signal_completion`` in their tool surface.  The
        rationale: interactive clients expect the session to remain
        available for further turns until the user disconnects;
        ``signal_completion`` is a terminal-tool that ends the session,
        which is the wrong contract for that workload.  Subagents (any
        client) and headless API clients (``client_type=api``) are
        unaffected — they continue to see the tool because cascade /
        completion-payload contracts depend on it.  See
        :meth:`_should_hide_signal_completion` for the precise gate.
        """
        schemas: List[ToolSchema] = []

        if not self._should_hide_signal_completion():
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
                # Option G (server 0.6.115+, 2026-05-16): the profile's
                # completion_payload_schema IS the tool's parameter schema.
                # Top-level properties become flat tool args, no ``payload``
                # wrapper.  Closes the Anthropic/Bedrock stringification
                # pathology that v109 hit (the model emits one huge nested
                # arg as a JSON-string instead of an object).  Five
                # medium-sized top-level args don't trigger the same
                # stringification heuristic that one huge nested arg does.
                #
                # Provider contract: tool ``parameters`` MUST be a
                # ``type: object`` JSON Schema.  Profiles that declare a
                # non-object completion_payload_schema would produce a
                # malformed tool schema — this is treated as a kb
                # authoring error and surfaced at provider call time.
                parameters = self._payload_schema

            schemas.append(
                ToolSchema(
                    name="signal_completion",
                    description=(
                        "Signal that you have finished all your work and have "
                        "nothing left to do.  This triggers downstream agents "
                        "(e.g. memory curator) and allows the session to be "
                        "cleaned up.  Call this as your very last action.\n\n"
                        "Two paths:\n"
                        "  - Single-shot (capable models): call with the full "
                        "payload matching completion_payload_schema as args.\n"
                        "  - Accumulator: call with NO args after using "
                        "``prepare_completion`` to populate fields across "
                        "multiple turns; the framework synthesizes the payload "
                        "from accumulated state.  See ``query_completion`` to "
                        "inspect accumulated state."
                    ),
                    parameters=parameters,
                    discoverability="core",
                )
            )

            # prepare_completion / query_completion tools — register
            # only when a completion_payload_schema is declared
            # (otherwise there is no schema to accumulate against).
            # Designed for small-model composition-burden mitigation:
            # the agent submits PARTIAL segments of the payload across
            # multiple turns instead of composing the entire structured
            # emission in one tool call.  See
            # ``_execute_prepare_completion`` /
            # ``_execute_query_completion`` for the executor docstrings.
            if self._payload_schema is not None:
                schemas.append(
                    ToolSchema(
                        name="prepare_completion",
                        description=(
                            "Add fields to the completion payload one segment "
                            "at a time.  Pass a partial dict matching part of "
                            "completion_payload_schema's shape (any subset of "
                            "top-level fields, or nested values for fields you "
                            "already started).  The framework merges into "
                            "session-tier accumulated state and returns: "
                            "``accepted`` (what merged), ``rejected`` "
                            "(per-field type errors), "
                            "``pending_required_fields_with_descriptions`` "
                            "(what remains), and ``is_complete`` (True when "
                            "every required field is satisfied).  Call this "
                            "repeatedly as you discover values from your "
                            "tool calls; when ``is_complete`` is True, call "
                            "``signal_completion()`` with no args.  Use "
                            "``query_completion`` to inspect accumulated "
                            "state without contributing."
                        ),
                        parameters={
                            "type": "object",
                            "additionalProperties": True,
                            "description": (
                                "Partial completion payload — any subset of "
                                "fields from completion_payload_schema.  No "
                                "wire-level constraints; framework validates "
                                "each field server-side against the declared "
                                "schema and surfaces rejections in the "
                                "response."
                            ),
                        },
                        discoverability="core",
                    )
                )
                schemas.append(
                    ToolSchema(
                        name="query_completion",
                        description=(
                            "Inspect the current accumulated completion "
                            "payload (read-only — does not mutate state).  "
                            "Returns ``accumulated`` (what you've set so "
                            "far), ``pending_required_fields_with_descriptions`` "
                            "(what's still missing), and ``is_complete`` "
                            "(whether ``signal_completion()`` arg-less is "
                            "ready to fire).  Use this when you lose track of "
                            "what you've contributed, want to verify a value "
                            "you set earlier, or before calling "
                            "``signal_completion`` to confirm completeness."
                        ),
                        parameters={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                        discoverability="core",
                    )
                )

        # Per-turn model-tier switching.  Only registered when the
        # session has tier mode active — single-model sessions don't
        # see this tool at all (no protocol noise, full backwards
        # compat).  See ``shared/model_tiers.py`` for the resolved
        # config and ``project_backlog_per_turn_model.md`` for the design.
        if getattr(self._session, '_tier_config', None) is not None:
            schemas.append(self._enter_tier_schema())

        return schemas

    def _should_hide_signal_completion(self) -> bool:
        """Whether to hide ``signal_completion`` from this session's tool surface.

        Two independent gates, ANY of which hides the tool:

        1. **No declared completion_payload_schema** (2026-06-07+).
           ``signal_completion`` only makes sense when the profile
           declares a typed payload contract via
           ``completion_payload_schema``.  Without one, the legacy
           "unconstrained ``{summary: string}``" path was accepting
           anything from the model with no validation — a profile
           that wanted to reference signal_completion in its persona
           had no rigorous "done" semantic.  Now the tool is
           opt-in via schema declaration: declare a schema → get
           signal_completion; don't declare one → the tool is
           hidden and the session terminates naturally when the
           model emits text without tool calls.

           This gate applies uniformly to ROOT and SUBAGENT
           sessions.  Subagents that need to bubble structured
           payloads up to their parent declare a schema; subagents
           that just emit text and stop don't need it.

        2. **Interactive root filter** (pre-existing).  Root
           sessions (no ``_parent_session`` set) connected via an
           interactive client (``client_type ∈ {TERMINAL, WEB,
           CHAT}``) hide signal_completion because interactive
           clients expect the session to stay available for further
           turns; signal_completion's termination contract is the
           wrong shape for that workload.

           Headless API clients
           (``client_type=API``) keep the tool because cascade
           entry-points and one-shot orchestrators rely on it.

        Returns True iff EITHER gate fires.  Returns False — i.e.
        ``signal_completion`` IS exposed — only when a schema is
        declared AND the interactive-root filter doesn't apply.
        """
        # Gate 1 (2026-06-07+): no schema → hide.  Applies to root
        # AND subagent.  If you want signal_completion, declare a
        # completion_payload_schema in your profile.
        if self._payload_schema is None:
            return True

        # Gate 2 (pre-existing): subagents always see it (when schema
        # declared); interactive root sessions don't.
        if getattr(self._session, '_parent_session', None) is not None:
            return False
        pctx = getattr(self._session, '_presentation_context', None)
        if pctx is None:
            return False
        # Local import to avoid a circular dependency with jaato_sdk
        # at module load time (this module is imported during
        # JaatoSession.configure which itself runs before the SDK
        # event types are guaranteed loaded in some test paths).
        from jaato_sdk.events import ClientType
        return pctx.client_type in (
            ClientType.TERMINAL, ClientType.WEB, ClientType.CHAT,
        )

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
        if self._payload_schema is not None:
            executors["prepare_completion"] = self._execute_prepare_completion
            executors["query_completion"] = self._execute_query_completion
        if getattr(self._session, '_tier_config', None) is not None:
            executors["enter_tier"] = self._execute_enter_tier
        return executors

    def get_auto_approved_tools(self) -> List[str]:
        approved = ["signal_completion"]
        if self._payload_schema is not None:
            approved.append("prepare_completion")
            approved.append("query_completion")
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

        With a schema declared (Option G, server 0.6.115+): the
        incoming ``args`` dict **IS** the payload (no ``payload``
        wrapper).  ``args`` is validated directly against the schema
        using ``jsonschema``.  On validation failure, returns a
        structured error to the model (no event emission) so the model
        can self-correct on its next turn.  On success, forwards the
        validated payload to ``hooks.on_agent_completed(payload=...)``
        and a derived ``summary`` (from the payload's ``summary``
        field if present, otherwise empty) for reactor consumers that
        still read the legacy field.

        **Accumulator path (server 0.6.198+):** when ``args`` is empty
        AND a ``completion_payload_schema`` is declared, the framework
        synthesizes the payload from
        :attr:`_accumulated_payload` (populated by
        ``prepare_completion`` calls across the session).  If the
        accumulated state satisfies the schema (``is_complete=True``),
        proceeds to validation + processors.  Otherwise rejects with
        the same ``validation_failed`` shape used by malformed
        partial-args, including the
        ``pending_required_fields_with_descriptions`` map so the agent
        knows what's left to set via ``prepare_completion``.
        """
        payload: Optional[Dict[str, Any]]
        summary: str

        if self._payload_schema is None:
            summary = args.get("summary", "")
            payload = None
        else:
            # Accumulator path: empty args + non-empty accumulated state
            # → synthesize the payload from accumulated.  See
            # ``_execute_prepare_completion`` for how state gets there.
            if not args and self._accumulated_payload:
                pending = self._compute_pending_required_fields(
                    self._accumulated_payload, self._payload_schema,
                )
                if pending:
                    logger.info(
                        "signal_completion: arg-less call but accumulated "
                        "state has %d required field(s) pending — rejecting",
                        len(pending),
                    )
                    return {
                        "error": "validation_failed",
                        "message": (
                            "signal_completion was called with no args but "
                            "the accumulated payload is not yet complete. "
                            "Use prepare_completion to fill the pending "
                            "fields below, then call signal_completion() "
                            "with no args again.  Or call signal_completion "
                            "with a full payload as args to bypass the "
                            "accumulator."
                        ),
                        "pending_required_fields_with_descriptions": pending,
                    }
                # is_complete=True — synthesize.
                payload = dict(self._accumulated_payload)
                logger.info(
                    "signal_completion: synthesizing from accumulated "
                    "state (%d top-level keys)",
                    len(payload),
                )
            else:
                # Option G: args dict IS the payload — the tool's
                # parameters mirror completion_payload_schema's
                # top-level properties directly.  No "payload" wrapper.
                payload = args
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
                        "The arguments did not match the profile's "
                        "completion_payload_schema. Fix the arguments and "
                        "call signal_completion again."
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

        # Unified completion processors (server 0.6.125+) — kb-authored
        # Python that both PRODUCES output (render symbol) and VALIDATES
        # the payload (validate symbol).  Replaces the prior split
        # between completion_artifacts and completion_validators.
        # Runs AFTER ``jsonschema.validate`` passes; any
        # ``on_error: fail_completion`` failure returns the same
        # ``validation_failed`` shape so the agent retries within
        # ``max_turns``.  See ``shared/completion_processors.py`` for
        # the loader, ledger builder, and per-processor invocation.
        configured_processors = getattr(
            self._session, "_completion_processors", []
        ) or []
        processor_outcome = None
        if configured_processors and payload is not None:
            from .completion_processors import (
                build_tool_call_ledger,
                collect_failure_messages,
                invoke_processors,
                load_processors,
            )
            from .dynamic_instructions import build_render_context
            if self._processors_loaded is None:
                workspace_path_str = getattr(self._session, "workspace_path", None)
                config_root_str = getattr(
                    getattr(self._session, "runtime", None), "_config_root", None,
                )
                self._processors_loaded = load_processors(
                    configured_processors,
                    workspace_path=workspace_path_str,
                    config_root=config_root_str,
                )
            try:
                history = self._session.get_history()
            except Exception:
                history = []
            ledger = build_tool_call_ledger(history)
            ctx = build_render_context(
                self._session,
                agent_params=getattr(self._session, "_agent_params", {}),
                tool_calls=ledger,
            )
            processor_outcome = invoke_processors(
                self._processors_loaded,
                payload=payload,
                context=ctx,
            )
            if processor_outcome.has_fatal:
                failure_messages = collect_failure_messages(processor_outcome)
                logger.info(
                    "signal_completion: %d completion-processor error(s); "
                    "returning self-correction prompt",
                    len(failure_messages),
                )
                return {
                    "error": "validation_failed",
                    "message": (
                        "Your payload structure was valid, but one or "
                        "more completion processors reported errors. "
                        "Read each error below carefully, fix the "
                        "underlying issue (e.g. surface failed tool "
                        "calls in errors[], remove fabricated entries, "
                        "retry failed tool calls, or correct payload "
                        "fields a renderer needs), then call "
                        "signal_completion again."
                    ),
                    "processor_errors": failure_messages,
                }
            # Soft failures — log but proceed.
            for proc, msg in processor_outcome.warned:
                logger.warning(
                    "completion-processor warned (script=%s output=%s): %s",
                    getattr(proc, "script", None),
                    getattr(proc, "output", None),
                    msg,
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

        # AgentCompletedEvent.token_usage is typed Dict[str, int] (pydantic
        # validates this post-Phase-0 migration; pre-migration the dataclass
        # silently accepted whatever was passed).  get_context_usage() returns
        # a dict with prompt/output/total token counts; forward only the int-
        # valued keys so the payload matches the wire schema.
        token_usage = {
            k: v for k, v in usage.items()
            if k in ('total_tokens', 'prompt_tokens', 'output_tokens')
            and isinstance(v, int)
        }
        hooks.on_agent_completed(
            agent_id=agent_id,
            completed_at=datetime.now(),
            success=True,
            token_usage=token_usage if token_usage else None,
            turns_used=usage.get('turns'),
            payload=payload,
        )

        # Flip the per-session completion flag so the loop-exit nudge
        # guard knows this agent did its part — no nudge needed when
        # the loop terminates from here.  Set BEFORE any subsequent
        # work to keep the predicate consistent if anything below
        # raises.
        self._session._signal_completion_called = True

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
        if processor_outcome and processor_outcome.written:
            result["artifacts_written"] = list(processor_outcome.written)
        return result

    # ==================== prepare/query completion (2026-06-09) ====================

    def _execute_prepare_completion(
        self, args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge a partial payload into accumulated state, validate
        per-field, return progress signal.

        Designed for small-model composition-burden mitigation
        ([[feedback_small_model_narration_skipping_is_structural]]).
        Each call is a discrete TRANSCRIPTION operation — the model
        contributes one or more fields at a time, the framework
        accumulates, the response signals what's still pending.  The
        model stays in the same cognitive mode that works for tool
        result transcription instead of composing the entire payload
        as one structured emission.

        Args:
            args: Partial completion payload — any subset of fields
                from ``completion_payload_schema``.  Top-level field
                rejection is per-field: a partial like
                ``{service: "billing", endpoints: <malformed>}``
                accepts ``service``, rejects ``endpoints``, surfaces
                both in the response.  Nested objects merge
                last-write-wins per Q7.

        Returns:
            Dict with:
            - ``accepted``: subset of args that passed per-field
              validation and was merged into accumulated state.
            - ``rejected``: ``{field_path: rejection_reason}`` for
              fields that failed validation; accumulated state was NOT
              mutated for these.
            - ``pending_required_fields_with_descriptions``: list of
              ``{path, description, type, ...}`` dicts for required
              fields still missing from accumulated state.
            - ``is_complete``: True iff
              ``jsonschema.validate(accumulated, full_schema)`` passes.
        """
        if self._payload_schema is None:
            return {
                "error": "no_completion_schema",
                "message": (
                    "prepare_completion is only available when the "
                    "profile declares a completion_payload_schema. This "
                    "session has no schema, so signal_completion(summary) "
                    "is the only completion path."
                ),
            }

        if not isinstance(args, dict):
            return {
                "error": "invalid_argument",
                "message": (
                    "prepare_completion requires a dict argument matching "
                    "part of completion_payload_schema's shape."
                ),
            }

        # Per-field validation: try each top-level key against the
        # schema's properties.  Accept what validates, reject what
        # doesn't — preserves partial submissions where some fields are
        # well-formed and others aren't.
        accepted: Dict[str, Any] = {}
        rejected: Dict[str, str] = {}
        for key, value in args.items():
            err = self._validate_field_against_schema(
                key, value, self._payload_schema,
            )
            if err is None:
                accepted[key] = value
            else:
                rejected[key] = err

        # Merge accepted fields into accumulated state (last-write-wins
        # per-key — Q7).  Deep merge for dict values, replace for
        # everything else.
        for key, value in accepted.items():
            if (
                isinstance(value, dict)
                and isinstance(self._accumulated_payload.get(key), dict)
            ):
                self._deep_merge(self._accumulated_payload[key], value)
            else:
                self._accumulated_payload[key] = value

        pending = self._compute_pending_required_fields(
            self._accumulated_payload, self._payload_schema,
        )
        is_complete = not pending

        logger.info(
            "prepare_completion: accepted=%d rejected=%d pending=%d is_complete=%s",
            len(accepted), len(rejected), len(pending), is_complete,
        )

        return {
            "accepted": accepted,
            "rejected": rejected,
            "pending_required_fields_with_descriptions": pending,
            "is_complete": is_complete,
        }

    def _execute_query_completion(
        self, args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inspect accumulated completion state (read-only).

        Returns the full accumulated payload + pending required fields
        + is_complete flag.  Does NOT mutate state — safe to call
        repeatedly.  Useful when the model loses track of what it has
        contributed, wants to verify a value set earlier, or before
        calling ``signal_completion`` to confirm completeness.
        """
        if self._payload_schema is None:
            return {
                "error": "no_completion_schema",
                "message": (
                    "query_completion is only available when the profile "
                    "declares a completion_payload_schema."
                ),
            }

        pending = self._compute_pending_required_fields(
            self._accumulated_payload, self._payload_schema,
        )
        return {
            "accumulated": dict(self._accumulated_payload),
            "pending_required_fields_with_descriptions": pending,
            "is_complete": not pending,
        }

    def _validate_field_against_schema(
        self,
        key: str,
        value: Any,
        schema: Dict[str, Any],
    ) -> Optional[str]:
        """Validate one top-level field against the schema's property
        spec for that key.  Returns rejection-reason string on
        failure, or None on success.

        Validation uses a RELAXED variant of the property spec where
        all ``required[]`` arrays are stripped recursively.  This
        allows partial submissions — the agent can submit
        ``{stack_config: {language: "java"}}`` even though the schema
        marks ``framework`` as required inside stack_config.  Type
        checking + enum constraints + format constraints remain
        enforced.  Required-field tracking is handled separately by
        the pending walker which uses the ORIGINAL (unrelaxed) schema.
        """
        properties = schema.get("properties", {})
        if key not in properties:
            return (
                f"Field {key!r} is not declared in "
                f"completion_payload_schema.properties. Valid top-level "
                f"keys: {sorted(properties.keys())}."
            )

        # Strip required[] recursively so partial submissions don't
        # fail validation just for being incomplete.
        relaxed = self._strip_required_recursive(properties[key])

        try:
            import jsonschema
            jsonschema.validate(instance=value, schema=relaxed)
        except jsonschema.ValidationError as exc:
            return f"{exc.message} (at {'.'.join(str(p) for p in exc.absolute_path)})"
        return None

    def _strip_required_recursive(self, schema: Any) -> Any:
        """Return a deep-copied schema with all ``required[]`` arrays
        removed.  Recurses into ``properties`` and ``items`` so nested
        objects-in-arrays inherit the relaxation.  Used by
        ``_validate_field_against_schema`` to allow partial
        submissions during accumulation while preserving type / enum /
        format checks.
        """
        if not isinstance(schema, dict):
            return schema
        out = {k: v for k, v in schema.items() if k != "required"}
        if "properties" in out and isinstance(out["properties"], dict):
            out["properties"] = {
                k: self._strip_required_recursive(v)
                for k, v in out["properties"].items()
            }
        if "items" in out:
            out["items"] = self._strip_required_recursive(out["items"])
        return out

    def _compute_pending_required_fields(
        self,
        accumulated: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return list of required fields still missing from
        accumulated state.  Each entry includes the JSON-pointer-style
        path + the schema's description / type / enum (if present) for
        that field so the model gets just-in-time guidance per
        pending field without needing to re-read persona prose.

        Walks the schema recursively for nested-object required
        fields.  Arrays of objects: if the accumulated value is a
        non-empty array, each item's required fields are checked
        against item-schema.  If accumulated array is empty AND the
        array itself is required, surfaces ``<path>`` as pending.
        """
        pending: List[Dict[str, Any]] = []
        self._walk_required(
            accumulated, schema, path_prefix="", pending=pending,
        )
        return pending

    def _walk_required(
        self,
        accumulated: Any,
        schema: Dict[str, Any],
        path_prefix: str,
        pending: List[Dict[str, Any]],
    ) -> None:
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            for req_key in required:
                child_path = (
                    f"{path_prefix}.{req_key}" if path_prefix else req_key
                )
                if not isinstance(accumulated, dict) or req_key not in accumulated:
                    pending.append(
                        self._describe_pending_field(
                            child_path, properties.get(req_key, {}),
                        )
                    )
                else:
                    # Recurse into the present value to surface deeply
                    # nested required fields (e.g. each endpoint's
                    # required ``operation``).
                    self._walk_required(
                        accumulated[req_key],
                        properties.get(req_key, {}),
                        child_path,
                        pending,
                    )
        elif schema.get("type") == "array":
            items_schema = schema.get("items", {})
            if isinstance(accumulated, list):
                for idx, item in enumerate(accumulated):
                    item_path = f"{path_prefix}[{idx}]"
                    self._walk_required(item, items_schema, item_path, pending)

    def _describe_pending_field(
        self,
        path: str,
        field_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the pending-field descriptor surfaced in
        prepare/query responses.  Keeps the schema knowledge JIT-
        accessible so small models don't need to retain all 30K of
        persona prose in working context.
        """
        descriptor: Dict[str, Any] = {"path": path}
        for key in ("description", "type", "enum", "format", "examples"):
            if key in field_schema:
                descriptor[key] = field_schema[key]
        # For arrays-of-objects, surface the items.type hint so the
        # model knows it needs to fill an array (not a scalar).
        if field_schema.get("type") == "array" and "items" in field_schema:
            items = field_schema["items"]
            descriptor["items_type"] = items.get("type", "any")
            if "required" in items:
                descriptor["items_required"] = list(items["required"])
        return descriptor

    def _deep_merge(
        self,
        target: Dict[str, Any],
        source: Dict[str, Any],
    ) -> None:
        """Recursive in-place merge.  Last-write-wins per leaf (Q7).
        Dict values merge; everything else (including arrays)
        replaces.  Symmetric with the accumulated-state mutation
        semantics in ``_execute_prepare_completion``.
        """
        for key, value in source.items():
            if (
                isinstance(value, dict)
                and isinstance(target.get(key), dict)
            ):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
