"""Learning advisor plugin — institutional memory with quality filtering.

A hidden (non-tool-exposing) plugin that silently observes sessions,
distills "lessons learned" from successful approaches, and injects that
wisdom into future sessions with similar goals.

Design principles:
- **Guidance, not prescription** — injected lessons are framed as "here's
  what worked before", not "you must do this".
- **Quality gating** — only lessons from sessions with good outcomes
  propagate.  A ranked pool per goal cluster keeps the top N lessons
  (default 3) and evicts the worst when a better one arrives.
- **Invisible by default** — no tools are exposed to the model.  The
  plugin works entirely through prompt enrichment and session lifecycle
  hooks.  User commands (``advisor``) provide observability.

Lifecycle:
    1. Session starts → ``enrich_prompt()`` fires → matcher finds relevant
       lessons → injects guidance into the prompt.
    2. Session runs → plugin is passive (no tool exposure).
    3. Session ends (``close_session()`` or ``advisor reflect`` command) →
       reflector produces a Lesson → scorer rates quality → storage
       persists → pool evicts worst if capacity exceeded.

Integration points:
    - ``subscribes_to_prompt_enrichment()`` → True
    - ``enrich_prompt()`` → inject lesson guidance
    - ``set_session()`` → capture session reference for reflection
    - ``set_workspace_path()`` → resolve storage path
    - ``get_user_commands()`` → ``advisor`` command for observability
"""

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from jaato_sdk.plugins.base import (
    CommandCompletion,
    HelpLines,
    PromptEnrichmentResult,
    UserCommand,
)
from jaato_sdk.plugins.model_provider.types import ToolSchema
from shared.trace import trace as _trace_write

from .matcher import GoalMatcher
from .models import (
    Lesson,
    LessonMetadata,
    MAX_LESSONS_PER_CLUSTER,
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
    STATUS_DISMISSED,
)
from .reflector import SessionReflector
from .scorer import OutcomeScorer
from .storage import LessonStorage


# Thread-local session storage — ensures subagents don't clobber the
# main agent's session reference.
_thread_local = threading.local()


class LearningAdvisorPlugin:
    """Plugin that learns from past sessions and advises future ones.

    The advisor is "hidden": it exposes no tools for the model to call.
    Instead, it:
    - Observes session starts via prompt enrichment.
    - Injects lessons learned from similar past sessions.
    - Reflects on completed sessions (on demand or at session close).
    - Persists lessons in a quality-ranked pool per goal cluster.

    User commands (``advisor show``, ``advisor rate``, ``advisor reflect``,
    ``advisor clear``) provide transparency and control.

    Attributes:
        _storage: JSONL-backed lesson storage.
        _matcher: In-memory keyword-based goal matcher.
        _reflector: LLM-based session → lesson distiller.
        _scorer: Composite outcome quality scorer.
        _was_reset: Tracks whether the current session was reset
            (lowers the completion signal in scoring).
    """

    def __init__(self) -> None:
        """Initialize the learning advisor plugin.

        Storage and matcher are created lazily during ``initialize()``.
        The session reference is received via ``set_session()`` (auto-wired).
        """
        self._name = "learning_advisor"
        self._storage: Optional[LessonStorage] = None
        self._matcher: Optional[GoalMatcher] = None
        self._reflector = SessionReflector()
        self._scorer = OutcomeScorer()
        self._storage_path_template: str = ".jaato/advisor_lessons.jsonl"
        self._was_reset: bool = False
        self._first_prompt: Optional[str] = None
        self._enrichment_limit: int = 2  # Max lessons to inject per prompt

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        _trace_write("ADVISOR", msg)

    # ------------------------------------------------------------------
    # Plugin protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return plugin name."""
        return self._name

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize storage backend and matcher.

        Args:
            config: Optional configuration dict with keys:
                - storage_path: Path to JSONL file
                  (default: ``.jaato/advisor_lessons.jsonl``).
                - enrichment_limit: Max lessons to inject per prompt
                  (default: 2).
        """
        config = config or {}
        self._storage_path_template = config.get(
            "storage_path", ".jaato/advisor_lessons.jsonl"
        )
        self._enrichment_limit = config.get("enrichment_limit", 2)

        self._storage = LessonStorage(self._storage_path_template)
        self._matcher = GoalMatcher()

        # Build index from existing lessons
        existing = self._storage.load_all()
        self._matcher.build_index(existing)
        self._trace(
            f"initialize: storage_path={self._storage_path_template}, "
            f"lessons={len(existing)}"
        )

    def shutdown(self) -> None:
        """Shutdown the plugin and clean up resources."""
        self._trace("shutdown")
        if self._matcher:
            self._matcher.clear()
        self._storage = None
        self._matcher = None

    def set_workspace_path(self, path: str) -> None:
        """Re-initialize storage under the correct workspace directory.

        Called by PluginRegistry broadcast after plugin initialization.
        Resolves the relative storage path template against the workspace
        root.

        Args:
            path: Absolute path to the workspace root.
        """
        resolved = str(Path(path) / self._storage_path_template)
        self._trace(f"set_workspace_path: {path} -> {resolved}")
        self._storage = LessonStorage(resolved)
        self._matcher = GoalMatcher()
        existing = self._storage.load_all()
        self._matcher.build_index(existing)

    def set_session(self, session: Any) -> None:
        """Receive the session reference (auto-wired by JaatoSession).

        Stored in thread-local storage so subagents don't clobber the
        main agent's reference.

        Args:
            session: The owning JaatoSession instance.
        """
        _thread_local.session = session

    @property
    def _session(self) -> Optional[Any]:
        """Get the session for the current thread context."""
        return getattr(_thread_local, "session", None)

    # ------------------------------------------------------------------
    # Tool protocol (empty — advisor is hidden from the model)
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return empty list — advisor exposes no tools to the model."""
        return []

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return executors for user commands only.

        The advisor has no model-invocable tools. The only executor
        is for the ``advisor`` user command.
        """
        return {
            "advisor": self._execute_user_command,
        }

    def get_system_instructions(self) -> Optional[str]:
        """No system instructions — the advisor injects via enrichment."""
        return None

    def get_auto_approved_tools(self) -> List[str]:
        """User command is auto-approved (invoked directly by user)."""
        return ["advisor"]

    # ------------------------------------------------------------------
    # User commands
    # ------------------------------------------------------------------

    def get_user_commands(self) -> List[UserCommand]:
        """Return the ``advisor`` user command for observability.

        Subcommands:
        - ``advisor show`` — display stored lessons.
        - ``advisor rate good|bad`` — rate current session quality.
        - ``advisor reflect`` — manually trigger reflection.
        - ``advisor clear`` — delete all stored lessons.
        - ``advisor help`` — show usage information.
        """
        return [
            UserCommand(
                name="advisor",
                description="Learning advisor: show, rate, reflect, clear, help",
                share_with_model=False,
            )
        ]

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        """Provide autocompletion for advisor command arguments."""
        if command != "advisor":
            return []

        subcommands = [
            CommandCompletion("show", "Display stored lessons"),
            CommandCompletion("rate", "Rate current session (good/bad)"),
            CommandCompletion("reflect", "Trigger manual reflection"),
            CommandCompletion("clear", "Delete all lessons"),
            CommandCompletion("help", "Show detailed help"),
        ]

        if not args:
            return subcommands

        if len(args) == 1:
            partial = args[0].lower()
            return [c for c in subcommands if c.value.startswith(partial)]

        if len(args) == 2 and args[0].lower() == "rate":
            ratings = [
                CommandCompletion("good", "Mark session as high-quality"),
                CommandCompletion("bad", "Mark session as low-quality"),
            ]
            partial = args[1].lower()
            return [r for r in ratings if r.value.startswith(partial)]

        return []

    def _execute_user_command(self, args: Dict[str, Any]) -> Any:
        """Execute the ``advisor`` user command.

        Dispatches to subcommand handlers based on the first argument.

        Args:
            args: Command arguments. The raw command text is in the
                ``"action"`` key or as the first positional argument.

        Returns:
            Formatted response string or ``HelpLines`` for pager display.
        """
        # Parse subcommand from raw args
        raw = args.get("action", "").strip()
        if not raw:
            # Try positional args format
            raw = args.get("args", "").strip()

        parts = raw.split(None, 1)
        subcommand = parts[0].lower() if parts else "help"
        sub_args = parts[1] if len(parts) > 1 else ""

        if subcommand == "show":
            return self._cmd_show()
        elif subcommand == "rate":
            return self._cmd_rate(sub_args)
        elif subcommand == "reflect":
            return self._cmd_reflect()
        elif subcommand == "clear":
            return self._cmd_clear()
        elif subcommand == "help":
            return self._cmd_help()
        else:
            return f"Unknown advisor subcommand: {subcommand!r}. Try: advisor help"

    def _cmd_show(self) -> str:
        """Display all active lessons."""
        if not self._storage:
            return "Advisor not initialized."

        lessons = self._storage.load_active()
        if not lessons:
            return "No lessons stored yet. Lessons are created when sessions complete."

        lines = [f"**Learning Advisor** — {len(lessons)} active lesson(s)\n"]
        for i, lesson in enumerate(lessons, 1):
            tags = ", ".join(lesson.context_tags[:5])
            q = f"{lesson.quality.composite:.2f}"
            lines.append(
                f"{i}. [{q}] {lesson.goal_summary}\n"
                f"   Tags: {tags}\n"
                f"   Insights: {len(lesson.key_insights)} | "
                f"Anti-patterns: {len(lesson.anti_patterns)} | "
                f"Used: {lesson.used_count}x"
            )

        return "\n".join(lines)

    def _cmd_rate(self, rating_str: str) -> str:
        """Apply an explicit quality rating to the current session.

        Args:
            rating_str: ``"good"`` (maps to 0.9) or ``"bad"`` (maps to 0.1).
        """
        rating_str = rating_str.strip().lower()
        rating_map = {"good": 0.9, "bad": 0.1}
        rating = rating_map.get(rating_str)

        if rating is None:
            return "Usage: advisor rate good|bad"

        session = self._session
        if not session:
            return "No active session to rate."

        # Trigger reflection with the user rating
        turn_accounting = session.get_turn_accounting()
        quality = self._scorer.score_session(
            turn_accounting,
            was_reset=self._was_reset,
            user_rating=rating,
        )

        lesson = self._reflector.reflect(
            session, quality, session_id=getattr(session, "_agent_id", None)
        )

        if lesson:
            self._store_and_rank(lesson)
            return (
                f"Session rated as **{rating_str}** (score: {quality.composite:.2f}). "
                f"Lesson stored: {lesson.goal_summary}"
            )

        return f"Session rated as **{rating_str}** but no lesson could be extracted (too few turns?)."

    def _cmd_reflect(self) -> str:
        """Manually trigger reflection on the current session."""
        session = self._session
        if not session:
            return "No active session to reflect on."

        turn_accounting = session.get_turn_accounting()
        if not turn_accounting:
            return "Session has no turns yet — nothing to reflect on."

        quality = self._scorer.score_session(
            turn_accounting, was_reset=self._was_reset
        )

        lesson = self._reflector.reflect(
            session, quality, session_id=getattr(session, "_agent_id", None)
        )

        if lesson:
            self._store_and_rank(lesson)
            return (
                f"Lesson extracted (score: {quality.composite:.2f}):\n"
                f"  Goal: {lesson.goal_summary}\n"
                f"  Insights: {', '.join(lesson.key_insights[:3]) or '(none)'}\n"
                f"  Tags: {', '.join(lesson.context_tags[:5])}"
            )

        return "Reflection produced no lesson (LLM call may have failed)."

    def _cmd_clear(self) -> str:
        """Delete all stored lessons."""
        if not self._storage:
            return "Advisor not initialized."

        count = self._storage.count()
        self._storage.clear()
        if self._matcher:
            self._matcher.clear()

        return f"Cleared {count} lesson(s)."

    def _cmd_help(self) -> HelpLines:
        """Show detailed help for the advisor command."""
        return HelpLines(lines=[
            ("Learning Advisor", "bold"),
            ("", ""),
            ("A hidden plugin that learns from past sessions and advises future ones.", "dim"),
            ("Lessons are automatically injected via prompt enrichment when a similar", "dim"),
            ("goal is detected. Only high-quality outcomes propagate.", "dim"),
            ("", ""),
            ("USAGE", "bold"),
            ("    advisor <action>", ""),
            ("", ""),
            ("ACTIONS", "bold"),
            ("    show        Display all active lessons with quality scores", ""),
            ("    rate good   Mark current session as high-quality (score: 0.9)", ""),
            ("    rate bad    Mark current session as low-quality (score: 0.1)", ""),
            ("    reflect     Manually trigger lesson extraction for current session", ""),
            ("    clear       Delete all stored lessons", ""),
            ("    help        Show this help", ""),
            ("", ""),
            ("HOW IT WORKS", "bold"),
            ("    1. When a session starts, the advisor checks for lessons matching", "dim"),
            ("       the goal and injects guidance via prompt enrichment.", "dim"),
            ("    2. When a session ends, the advisor scores the outcome and", "dim"),
            ("       distills a lesson via an LLM reflection call.", "dim"),
            ("    3. Lessons are ranked per goal cluster. Better outcomes", "dim"),
            ("       replace worse ones (top 3 kept per cluster).", "dim"),
            ("    4. Use 'advisor rate good/bad' to explicitly signal quality.", "dim"),
            ("", ""),
            ("STORAGE", "bold"),
            ("    Lessons are stored in .jaato/advisor_lessons.jsonl", "dim"),
        ])

    # ------------------------------------------------------------------
    # Prompt enrichment protocol
    # ------------------------------------------------------------------

    def get_enrichment_priority(self) -> int:
        """Return enrichment priority.

        Runs at priority 70 — after environment/context plugins but
        before memory (80), so the model sees lesson guidance before
        memory hints.
        """
        return 70

    def subscribes_to_prompt_enrichment(self) -> bool:
        """Subscribe to enrich prompts with lessons learned."""
        return True

    def enrich_prompt(self, prompt: str) -> PromptEnrichmentResult:
        """Inject relevant lessons into the prompt.

        Extracts keywords from the prompt, matches against stored lessons,
        and appends non-prescriptive guidance about what worked in similar
        past sessions.

        Args:
            prompt: The user's prompt text.

        Returns:
            PromptEnrichmentResult with enriched prompt and metadata.
        """
        if not self._matcher or not self._storage:
            return PromptEnrichmentResult(
                prompt=prompt,
                metadata={"error": "Advisor not initialized"},
            )

        # Track the first prompt as the session goal
        if self._first_prompt is None:
            self._first_prompt = prompt

        # Extract keywords and find matching lessons
        keywords = GoalMatcher.extract_keywords(prompt)
        if not keywords:
            return PromptEnrichmentResult(
                prompt=prompt, metadata={"advisor_matches": 0}
            )

        matches = self._matcher.find_matching_lessons(
            keywords, limit=self._enrichment_limit
        )

        if not matches:
            return PromptEnrichmentResult(
                prompt=prompt, metadata={"advisor_matches": 0}
            )

        # Load full lessons for the matched metadata
        full_lessons = []
        for meta in matches:
            lesson = self._storage.get_by_id(meta.id)
            if lesson:
                full_lessons.append(lesson)

        if not full_lessons:
            return PromptEnrichmentResult(
                prompt=prompt, metadata={"advisor_matches": 0}
            )

        # Build guidance block
        guidance = self._format_guidance(full_lessons)
        enriched_prompt = prompt + "\n\n" + guidance

        # Update usage counts
        now = datetime.now(timezone.utc).isoformat()
        for lesson in full_lessons:
            lesson.used_count += 1
            lesson.last_used_at = now
            self._storage.update(lesson)

        # Build notification
        tag_summary = ", ".join(
            f'"{t}"' for t in full_lessons[0].context_tags[:3]
        )

        self._trace(
            f"enrich_prompt: injected {len(full_lessons)} lesson(s), "
            f"tags={tag_summary}"
        )

        return PromptEnrichmentResult(
            prompt=enriched_prompt,
            metadata={
                "advisor_matches": len(full_lessons),
                "matched_ids": [l.id for l in full_lessons],
                "notification": {
                    "message": (
                        f"advisor: injected {len(full_lessons)} lesson(s) "
                        f"from similar past sessions (tags: {tag_summary})"
                    )
                },
            },
        )

    def _format_guidance(self, lessons: List[Lesson]) -> str:
        """Format lessons into a non-prescriptive guidance block.

        The guidance is framed as "prior experience" to encourage the
        model to treat it as a helpful hint, not a mandate.

        Args:
            lessons: Full lesson objects to format.

        Returns:
            Formatted guidance string for prompt injection.
        """
        lines = [
            "---",
            "**Prior Experience** (for reference, not prescription)",
            "",
        ]

        for lesson in lessons:
            q = lesson.quality.composite
            confidence = "high" if q >= 0.7 else "moderate" if q >= 0.4 else "low"

            lines.append(
                f"For similar tasks involving **{lesson.goal_summary}** "
                f"(confidence: {confidence}, score: {q:.2f}):"
            )

            if lesson.key_insights:
                lines.append("  What worked:")
                for insight in lesson.key_insights[:3]:
                    lines.append(f"  - {insight}")

            if lesson.anti_patterns:
                lines.append("  What to avoid:")
                for anti in lesson.anti_patterns[:2]:
                    lines.append(f"  - {anti}")

            if lesson.approach_summary:
                lines.append(f"  Approach: {lesson.approach_summary}")

            lines.append("")

        lines.append("---")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Session lifecycle observation
    # ------------------------------------------------------------------

    def on_session_reset(self) -> None:
        """Called when the session is reset.

        Marks the session as having been reset, which lowers the
        completion signal in quality scoring.
        """
        self._was_reset = True

    # ------------------------------------------------------------------
    # Storage & ranking
    # ------------------------------------------------------------------

    def _store_and_rank(self, new_lesson: Lesson) -> None:
        """Store a lesson and enforce per-cluster capacity limits.

        If lessons with overlapping goals already exist, the new lesson
        competes on quality:
        - If the cluster is under capacity, the lesson is simply added.
        - If the cluster is at capacity, the worst lesson is archived
          and the new one takes its place (if it scores higher).

        Args:
            new_lesson: The newly created lesson.
        """
        if not self._storage or not self._matcher:
            return

        # Find existing lessons with similar goals
        keywords = GoalMatcher.extract_keywords(new_lesson.goal_summary)
        keywords.extend(new_lesson.context_tags)

        existing_matches = self._matcher.find_matching_lessons(
            keywords,
            limit=MAX_LESSONS_PER_CLUSTER + 1,
            min_overlap=2,  # Require stronger overlap for cluster membership
        )

        # Count active lessons in the cluster
        active_in_cluster = [
            m for m in existing_matches if m.status == STATUS_ACTIVE
        ]

        if len(active_in_cluster) >= MAX_LESSONS_PER_CLUSTER:
            # Find the worst in the cluster
            worst = min(active_in_cluster, key=lambda m: m.quality_composite)

            if new_lesson.quality.composite > worst.quality_composite:
                # Archive the worst and replace
                old_lesson = self._storage.get_by_id(worst.id)
                if old_lesson:
                    old_lesson.status = STATUS_ARCHIVED
                    self._storage.update(old_lesson)
                    self._matcher.index_lesson(old_lesson)
                    self._trace(
                        f"archived lesson {worst.id} "
                        f"(score: {worst.quality_composite:.2f}) "
                        f"in favour of new lesson "
                        f"(score: {new_lesson.quality.composite:.2f})"
                    )
            else:
                # New lesson isn't better — still store it but archived
                new_lesson.status = STATUS_ARCHIVED
                self._trace(
                    f"new lesson archived — cluster at capacity and "
                    f"score {new_lesson.quality.composite:.2f} didn't beat "
                    f"worst {worst.quality_composite:.2f}"
                )

        # Persist and index
        self._storage.save(new_lesson)
        self._matcher.index_lesson(new_lesson)
        self._trace(
            f"stored lesson {new_lesson.id}: "
            f"goal={new_lesson.goal_summary!r}, "
            f"score={new_lesson.quality.composite:.2f}, "
            f"status={new_lesson.status}"
        )


def create_plugin() -> LearningAdvisorPlugin:
    """Factory function to create the learning advisor plugin instance.

    Returns:
        LearningAdvisorPlugin instance.
    """
    return LearningAdvisorPlugin()
