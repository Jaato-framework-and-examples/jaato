"""Tests for learning advisor goal similarity matcher."""

import pytest

from shared.plugins.learning_advisor.matcher import GoalMatcher
from shared.plugins.learning_advisor.models import (
    Lesson,
    QualityScore,
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
)


def _make_lesson(
    id: str,
    goal: str = "Test goal",
    tags: list = None,
    score: float = 0.7,
    status: str = STATUS_ACTIVE,
    created_at: str = "2025-01-01T00:00:00+00:00",
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
        approach_summary="Test approach",
        key_insights=[],
        anti_patterns=[],
        context_tags=tags or ["testing"],
        quality=qs,
        status=status,
        created_at=created_at,
    )


class TestKeywordExtraction:
    """Tests for GoalMatcher.extract_keywords()."""

    def test_basic_extraction(self):
        """Extracts meaningful keywords from a sentence."""
        keywords = GoalMatcher.extract_keywords(
            "Fix the authentication token validation bug"
        )
        assert "authentication" in keywords
        assert "token" in keywords
        assert "validation" in keywords

    def test_stopwords_filtered(self):
        """Common stopwords are removed."""
        keywords = GoalMatcher.extract_keywords(
            "the and but from with"
        )
        assert keywords == []

    def test_short_words_filtered(self):
        """Words shorter than 4 characters are removed."""
        keywords = GoalMatcher.extract_keywords("fix the bug in api")
        # "fix" (3), "the" (stopword), "bug" (3), "api" (3) → all filtered
        assert keywords == []

    def test_deduplication(self):
        """Duplicate keywords are removed."""
        keywords = GoalMatcher.extract_keywords(
            "database database migration database"
        )
        assert keywords.count("database") == 1

    def test_case_insensitive(self):
        """Keywords are lowercased."""
        keywords = GoalMatcher.extract_keywords("PostgreSQL Migration")
        assert "postgresql" in keywords
        assert "migration" in keywords

    def test_underscore_words(self):
        """Words with underscores are preserved."""
        keywords = GoalMatcher.extract_keywords(
            "The oauth_pkce_flow function"
        )
        assert "oauth_pkce_flow" in keywords


class TestGoalMatcher:
    """Tests for GoalMatcher indexing and matching."""

    def test_build_index(self):
        """Build index from lessons."""
        matcher = GoalMatcher()
        lessons = [
            _make_lesson("l1", tags=["authentication", "oauth"]),
            _make_lesson("l2", tags=["database", "migration"]),
        ]
        matcher.build_index(lessons)
        assert matcher.get_lesson_count() == 2

    def test_exact_match(self):
        """Exact tag match returns the lesson."""
        matcher = GoalMatcher()
        matcher.build_index([
            _make_lesson("l1", tags=["authentication", "oauth"]),
        ])

        results = matcher.find_matching_lessons(["authentication"])
        assert len(results) == 1
        assert results[0].id == "l1"

    def test_partial_match(self):
        """Partial substring match works."""
        matcher = GoalMatcher()
        matcher.build_index([
            _make_lesson("l1", tags=["postgresql_indexing"]),
        ])

        # "postgresql" is a substring of "postgresql_indexing"
        results = matcher.find_matching_lessons(["postgresql"])
        assert len(results) == 1
        assert results[0].id == "l1"

    def test_no_match(self):
        """No matching tags returns empty list."""
        matcher = GoalMatcher()
        matcher.build_index([
            _make_lesson("l1", tags=["authentication"]),
        ])

        results = matcher.find_matching_lessons(["completely_unrelated"])
        assert len(results) == 0

    def test_active_only_filter(self):
        """Archived lessons are excluded by default."""
        matcher = GoalMatcher()
        matcher.build_index([
            _make_lesson("l1", tags=["testing"], status=STATUS_ACTIVE),
            _make_lesson("l2", tags=["testing"], status=STATUS_ARCHIVED),
        ])

        results = matcher.find_matching_lessons(["testing"])
        assert len(results) == 1
        assert results[0].id == "l1"

    def test_include_archived(self):
        """active_only=False includes archived lessons."""
        matcher = GoalMatcher()
        matcher.build_index([
            _make_lesson("l1", tags=["testing"], status=STATUS_ACTIVE),
            _make_lesson("l2", tags=["testing"], status=STATUS_ARCHIVED),
        ])

        results = matcher.find_matching_lessons(
            ["testing"], active_only=False
        )
        assert len(results) == 2

    def test_ranking_by_quality(self):
        """Higher quality lessons rank higher."""
        matcher = GoalMatcher()
        matcher.build_index([
            _make_lesson("low", tags=["database"], score=0.3),
            _make_lesson("high", tags=["database"], score=0.9),
        ])

        results = matcher.find_matching_lessons(["database"])
        assert len(results) == 2
        assert results[0].id == "high"
        assert results[1].id == "low"

    def test_limit(self):
        """Limit restricts number of results."""
        matcher = GoalMatcher()
        matcher.build_index([
            _make_lesson("l1", tags=["shared_tag"]),
            _make_lesson("l2", tags=["shared_tag"]),
            _make_lesson("l3", tags=["shared_tag"]),
        ])

        results = matcher.find_matching_lessons(["shared_tag"], limit=2)
        assert len(results) == 2

    def test_min_overlap(self):
        """min_overlap filters lessons with too few matching tags.

        Tags are chosen so they have no substring overlap with each other
        (e.g. "database" is not a substring of "migration").  Otherwise
        ``find_matching_lessons`` would count both as matches via its
        substring fallback and the min_overlap filter would not kick in.
        """
        matcher = GoalMatcher()
        matcher.build_index([
            _make_lesson("l1", tags=["database", "migration", "postgresql"]),
            _make_lesson("l2", tags=["database"]),  # Only 1 overlap
        ])

        results = matcher.find_matching_lessons(
            ["database", "migration"], min_overlap=2
        )
        assert len(results) == 1
        assert results[0].id == "l1"

    def test_remove_lesson(self):
        """Removing a lesson cleans it from all indexes."""
        matcher = GoalMatcher()
        matcher.build_index([
            _make_lesson("l1", tags=["testing"]),
            _make_lesson("l2", tags=["testing"]),
        ])

        matcher.remove_lesson("l1")
        assert matcher.get_lesson_count() == 1

        results = matcher.find_matching_lessons(["testing"])
        assert len(results) == 1
        assert results[0].id == "l2"

    def test_clear(self):
        """Clear empties all indexes."""
        matcher = GoalMatcher()
        matcher.build_index([
            _make_lesson("l1", tags=["testing"]),
        ])

        matcher.clear()
        assert matcher.get_lesson_count() == 0
        assert matcher.find_matching_lessons(["testing"]) == []

    def test_index_single_lesson(self):
        """index_lesson adds a single lesson to existing index."""
        matcher = GoalMatcher()
        matcher.build_index([
            _make_lesson("l1", tags=["existing"]),
        ])

        matcher.index_lesson(
            _make_lesson("l2", tags=["new_tag"])
        )
        assert matcher.get_lesson_count() == 2
