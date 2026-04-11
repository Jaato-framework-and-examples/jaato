"""Tests for the learning advisor plugin integration."""

import pytest

from shared.plugins.learning_advisor.models import (
    Lesson,
    QualityScore,
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
)
from shared.plugins.learning_advisor.plugin import LearningAdvisorPlugin


def _make_lesson(
    id: str = "test-1",
    goal: str = "Fix authentication bug",
    tags: list = None,
    score: float = 0.7,
    status: str = STATUS_ACTIVE,
) -> Lesson:
    """Helper to create a Lesson with defaults."""
    qs = QualityScore(
        tool_success_rate=score,
        token_efficiency=score,
        completion_signal=score,
    )
    qs.compute()
    return Lesson(
        id=id,
        goal_summary=goal,
        approach_summary="Systematic debugging approach",
        key_insights=["Check token expiry first"],
        anti_patterns=["Don't restart blindly"],
        context_tags=tags or ["authentication", "debugging", "token"],
        quality=qs,
        status=status,
        created_at="2025-01-01T00:00:00+00:00",
    )


class TestPluginProtocol:
    """Tests for basic ToolPlugin protocol compliance."""

    def test_name(self):
        """Plugin has correct name."""
        plugin = LearningAdvisorPlugin()
        assert plugin.name == "learning_advisor"

    def test_no_tool_schemas(self):
        """Plugin exposes no tools to the model."""
        plugin = LearningAdvisorPlugin()
        assert plugin.get_tool_schemas() == []

    def test_auto_approved_tools(self):
        """Only the user command is auto-approved."""
        plugin = LearningAdvisorPlugin()
        assert plugin.get_auto_approved_tools() == ["advisor"]

    def test_no_system_instructions(self):
        """Plugin has no system instructions (works via enrichment)."""
        plugin = LearningAdvisorPlugin()
        assert plugin.get_system_instructions() is None

    def test_user_commands(self):
        """Plugin exposes the 'advisor' user command."""
        plugin = LearningAdvisorPlugin()
        commands = plugin.get_user_commands()
        assert len(commands) == 1
        assert commands[0].name == "advisor"
        assert commands[0].share_with_model is False

    def test_subscribes_to_enrichment(self):
        """Plugin subscribes to prompt enrichment."""
        plugin = LearningAdvisorPlugin()
        assert plugin.subscribes_to_prompt_enrichment() is True

    def test_enrichment_priority(self):
        """Plugin runs at priority 70 (before memory at 80)."""
        plugin = LearningAdvisorPlugin()
        assert plugin.get_enrichment_priority() == 70


class TestPluginLifecycle:
    """Tests for plugin initialization and workspace management."""

    def test_initialize(self, tmp_path):
        """Initialize creates storage and matcher."""
        plugin = LearningAdvisorPlugin()
        plugin.initialize()
        plugin.set_workspace_path(str(tmp_path))

        # Should work without errors
        assert plugin._storage is not None
        assert plugin._matcher is not None

    def test_set_workspace_path(self, tmp_path):
        """set_workspace_path resolves storage under workspace root."""
        plugin = LearningAdvisorPlugin()
        plugin.initialize()
        plugin.set_workspace_path(str(tmp_path))

        # Storage path should be under tmp_path
        assert str(tmp_path) in str(plugin._storage.path)

    def test_shutdown(self, tmp_path):
        """Shutdown cleans up resources."""
        plugin = LearningAdvisorPlugin()
        plugin.initialize()
        plugin.set_workspace_path(str(tmp_path))

        plugin.shutdown()
        assert plugin._storage is None
        assert plugin._matcher is None


class TestPromptEnrichment:
    """Tests for prompt enrichment with lesson injection."""

    def test_no_enrichment_without_init(self):
        """Enrichment without initialization returns prompt unchanged."""
        plugin = LearningAdvisorPlugin()
        result = plugin.enrich_prompt("Fix the auth bug")
        assert result.prompt == "Fix the auth bug"
        assert result.metadata.get("error")

    def test_no_match_returns_prompt_unchanged(self, tmp_path):
        """When no lessons match, prompt is returned unchanged."""
        plugin = LearningAdvisorPlugin()
        plugin.initialize()
        plugin.set_workspace_path(str(tmp_path))

        result = plugin.enrich_prompt("Fix the authentication bug")
        assert result.prompt == "Fix the authentication bug"
        assert result.metadata.get("advisor_matches") == 0

    def test_matching_lesson_injected(self, tmp_path):
        """When a matching lesson exists, it's injected into the prompt."""
        plugin = LearningAdvisorPlugin()
        plugin.initialize()
        plugin.set_workspace_path(str(tmp_path))

        # Store a lesson manually
        lesson = _make_lesson(tags=["authentication", "debugging", "token"])
        plugin._storage.save(lesson)
        plugin._matcher.index_lesson(lesson)

        result = plugin.enrich_prompt("Fix the authentication token bug")
        assert "Prior Experience" in result.prompt
        assert "Check token expiry first" in result.prompt
        assert result.metadata.get("advisor_matches") == 1

    def test_enrichment_includes_anti_patterns(self, tmp_path):
        """Injected guidance includes anti-patterns."""
        plugin = LearningAdvisorPlugin()
        plugin.initialize()
        plugin.set_workspace_path(str(tmp_path))

        lesson = _make_lesson()
        plugin._storage.save(lesson)
        plugin._matcher.index_lesson(lesson)

        result = plugin.enrich_prompt("Fix the authentication issue")
        assert "Don't restart blindly" in result.prompt

    def test_enrichment_limit(self, tmp_path):
        """Only enrichment_limit lessons are injected."""
        plugin = LearningAdvisorPlugin()
        plugin.initialize({"enrichment_limit": 1})
        plugin.set_workspace_path(str(tmp_path))

        for i in range(5):
            lesson = _make_lesson(
                id=f"l{i}",
                tags=["shared_tag", "common"],
            )
            plugin._storage.save(lesson)
            plugin._matcher.index_lesson(lesson)

        result = plugin.enrich_prompt("Work on the shared_tag common task")
        assert result.metadata.get("advisor_matches") == 1

    def test_usage_count_incremented(self, tmp_path):
        """Injected lessons have their usage count incremented."""
        plugin = LearningAdvisorPlugin()
        plugin.initialize()
        plugin.set_workspace_path(str(tmp_path))

        lesson = _make_lesson(id="track-use", tags=["authentication"])
        plugin._storage.save(lesson)
        plugin._matcher.index_lesson(lesson)

        plugin.enrich_prompt("Fix the authentication problem")

        updated = plugin._storage.get_by_id("track-use")
        assert updated is not None
        assert updated.used_count == 1
        assert updated.last_used_at is not None


class TestStoreAndRank:
    """Tests for the quality-ranked lesson pool."""

    def test_store_within_capacity(self, tmp_path):
        """Lessons are stored when cluster is under capacity."""
        plugin = LearningAdvisorPlugin()
        plugin.initialize()
        plugin.set_workspace_path(str(tmp_path))

        lesson = _make_lesson(id="new-1", score=0.7)
        plugin._store_and_rank(lesson)

        assert plugin._storage.count() == 1

    def test_better_lesson_replaces_worst(self, tmp_path):
        """When cluster is at capacity, better lesson archives worst."""
        plugin = LearningAdvisorPlugin()
        plugin.initialize()
        plugin.set_workspace_path(str(tmp_path))

        # Fill cluster with 3 lessons (MAX_LESSONS_PER_CLUSTER = 3)
        for i in range(3):
            lesson = _make_lesson(
                id=f"fill-{i}",
                tags=["database", "migration"],
                score=0.3 + (i * 0.1),  # 0.3, 0.4, 0.5
            )
            plugin._storage.save(lesson)
            plugin._matcher.index_lesson(lesson)

        # Add a better lesson
        better = _make_lesson(
            id="better",
            tags=["database", "migration"],
            score=0.9,
        )
        plugin._store_and_rank(better)

        # The worst should be archived
        worst = plugin._storage.get_by_id("fill-0")
        assert worst is not None
        assert worst.status == STATUS_ARCHIVED

    def test_worse_lesson_gets_archived(self, tmp_path):
        """When new lesson is worse than cluster's worst, it's archived."""
        plugin = LearningAdvisorPlugin()
        plugin.initialize()
        plugin.set_workspace_path(str(tmp_path))

        # Fill cluster with high-quality lessons
        for i in range(3):
            lesson = _make_lesson(
                id=f"high-{i}",
                tags=["database", "migration"],
                score=0.9,
            )
            plugin._storage.save(lesson)
            plugin._matcher.index_lesson(lesson)

        # Add a worse lesson
        worse = _make_lesson(
            id="worse",
            tags=["database", "migration"],
            score=0.1,
        )
        plugin._store_and_rank(worse)

        # The new lesson should be archived
        stored = plugin._storage.get_by_id("worse")
        assert stored is not None
        assert stored.status == STATUS_ARCHIVED


class TestCommandCompletions:
    """Tests for user command autocompletion."""

    def test_root_completions(self):
        """Root level shows all subcommands."""
        plugin = LearningAdvisorPlugin()
        completions = plugin.get_command_completions("advisor", [])
        values = [c.value for c in completions]
        assert "show" in values
        assert "rate" in values
        assert "reflect" in values
        assert "clear" in values
        assert "help" in values

    def test_partial_completion(self):
        """Partial subcommand filters matches."""
        plugin = LearningAdvisorPlugin()
        completions = plugin.get_command_completions("advisor", ["sh"])
        assert len(completions) == 1
        assert completions[0].value == "show"

    def test_rate_sub_completions(self):
        """Rate subcommand shows good/bad options."""
        plugin = LearningAdvisorPlugin()
        completions = plugin.get_command_completions("advisor", ["rate", ""])
        values = [c.value for c in completions]
        assert "good" in values
        assert "bad" in values

    def test_wrong_command(self):
        """Non-advisor command returns empty."""
        plugin = LearningAdvisorPlugin()
        completions = plugin.get_command_completions("other", [])
        assert completions == []
