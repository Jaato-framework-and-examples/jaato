"""Tests for learning advisor data models."""

import pytest

from shared.plugins.learning_advisor.models import (
    Lesson,
    LessonMetadata,
    QualityScore,
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
    STATUS_DISMISSED,
    MAX_LESSONS_PER_CLUSTER,
)


class TestQualityScore:
    """Tests for QualityScore composite computation."""

    def test_compute_weighted_average(self):
        """Composite is a weighted average of components."""
        qs = QualityScore(
            tool_success_rate=1.0,
            token_efficiency=1.0,
            completion_signal=1.0,
        )
        result = qs.compute()
        assert result == pytest.approx(1.0)
        assert qs.composite == pytest.approx(1.0)

    def test_compute_partial_scores(self):
        """Partial scores produce expected weighted average."""
        qs = QualityScore(
            tool_success_rate=0.5,  # 40% weight → 0.2
            token_efficiency=0.0,   # 30% weight → 0.0
            completion_signal=1.0,  # 30% weight → 0.3
        )
        result = qs.compute()
        assert result == pytest.approx(0.5)

    def test_compute_zero_scores(self):
        """All zero components yield zero composite."""
        qs = QualityScore(
            tool_success_rate=0.0,
            token_efficiency=0.0,
            completion_signal=0.0,
        )
        result = qs.compute()
        assert result == pytest.approx(0.0)

    def test_user_rating_overrides_composite(self):
        """Explicit user rating replaces computed composite."""
        qs = QualityScore(
            tool_success_rate=0.0,
            token_efficiency=0.0,
            completion_signal=0.0,
            user_rating=0.9,
        )
        result = qs.compute()
        assert result == pytest.approx(0.9)
        assert qs.composite == pytest.approx(0.9)

    def test_user_rating_clamped(self):
        """User rating is clamped to [0.0, 1.0]."""
        qs = QualityScore(user_rating=1.5)
        assert qs.compute() == pytest.approx(1.0)

        qs2 = QualityScore(user_rating=-0.5)
        assert qs2.compute() == pytest.approx(0.0)


class TestLesson:
    """Tests for Lesson dataclass."""

    def test_default_values(self):
        """Lesson has sensible defaults."""
        lesson = Lesson(
            id="test-1",
            goal_summary="Test goal",
            approach_summary="Test approach",
            key_insights=["insight1"],
            anti_patterns=[],
            context_tags=["testing"],
            quality=QualityScore(),
        )
        assert lesson.status == STATUS_ACTIVE
        assert lesson.used_count == 0
        assert lesson.last_used_at is None
        assert lesson.scope == "project"

    def test_all_statuses_valid(self):
        """Verify status constants are distinct."""
        assert STATUS_ACTIVE != STATUS_ARCHIVED
        assert STATUS_ARCHIVED != STATUS_DISMISSED
        assert STATUS_ACTIVE != STATUS_DISMISSED


class TestLessonMetadata:
    """Tests for LessonMetadata lightweight representation."""

    def test_metadata_from_lesson(self):
        """LessonMetadata captures key fields from a Lesson."""
        qs = QualityScore(tool_success_rate=0.8, completion_signal=1.0)
        qs.compute()

        meta = LessonMetadata(
            id="test-1",
            goal_summary="Test goal",
            context_tags=["testing", "models"],
            quality_composite=qs.composite,
        )
        assert meta.id == "test-1"
        assert meta.status == STATUS_ACTIVE
        assert meta.quality_composite == qs.composite


class TestConstants:
    """Tests for module-level constants."""

    def test_max_lessons_per_cluster(self):
        """Cluster capacity is a positive integer."""
        assert MAX_LESSONS_PER_CLUSTER > 0
        assert isinstance(MAX_LESSONS_PER_CLUSTER, int)
