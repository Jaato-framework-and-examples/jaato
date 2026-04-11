"""Tests for learning advisor JSONL storage."""

import json
import tempfile
from pathlib import Path

import pytest

from shared.plugins.learning_advisor.models import (
    Lesson,
    QualityScore,
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
)
from shared.plugins.learning_advisor.storage import LessonStorage


def _make_lesson(
    id: str = "test-1",
    goal: str = "Fix authentication bug",
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
        approach_summary="Used systematic debugging",
        key_insights=["Check token expiry first"],
        anti_patterns=["Don't restart the server blindly"],
        context_tags=["authentication", "debugging"],
        quality=qs,
        status=status,
        created_at="2025-01-01T00:00:00+00:00",
    )


class TestLessonStorage:
    """Tests for LessonStorage JSONL persistence."""

    def test_save_and_load(self, tmp_path):
        """Round-trip: save a lesson, load it back."""
        path = str(tmp_path / "lessons.jsonl")
        storage = LessonStorage(path)

        lesson = _make_lesson()
        storage.save(lesson)

        loaded = storage.load_all()
        assert len(loaded) == 1
        assert loaded[0].id == "test-1"
        assert loaded[0].goal_summary == "Fix authentication bug"
        assert loaded[0].quality.composite == pytest.approx(lesson.quality.composite)

    def test_save_multiple(self, tmp_path):
        """Multiple saves append to the file."""
        path = str(tmp_path / "lessons.jsonl")
        storage = LessonStorage(path)

        storage.save(_make_lesson(id="l1"))
        storage.save(_make_lesson(id="l2"))
        storage.save(_make_lesson(id="l3"))

        loaded = storage.load_all()
        assert len(loaded) == 3
        assert {l.id for l in loaded} == {"l1", "l2", "l3"}

    def test_load_empty_file(self, tmp_path):
        """Load from non-existent file returns empty list."""
        storage = LessonStorage(str(tmp_path / "missing.jsonl"))
        assert storage.load_all() == []

    def test_load_corrupt_lines(self, tmp_path):
        """Corrupt JSONL lines are skipped, valid ones loaded."""
        path = tmp_path / "lessons.jsonl"
        valid = _make_lesson(id="valid")
        from dataclasses import asdict
        with open(path, "w") as f:
            f.write(json.dumps(asdict(valid)) + "\n")
            f.write("not valid json\n")
            f.write("{}\n")  # Missing required fields → TypeError

        storage = LessonStorage(str(path))
        loaded = storage.load_all()
        assert len(loaded) == 1
        assert loaded[0].id == "valid"

    def test_update_existing(self, tmp_path):
        """Update modifies an existing lesson in place."""
        path = str(tmp_path / "lessons.jsonl")
        storage = LessonStorage(path)

        original = _make_lesson(id="upd1", goal="Original goal")
        storage.save(original)

        modified = _make_lesson(id="upd1", goal="Modified goal")
        storage.update(modified)

        loaded = storage.load_all()
        assert len(loaded) == 1
        assert loaded[0].goal_summary == "Modified goal"

    def test_update_missing_appends(self, tmp_path):
        """Update of a non-existent ID appends the lesson."""
        path = str(tmp_path / "lessons.jsonl")
        storage = LessonStorage(path)

        storage.save(_make_lesson(id="existing"))
        storage.update(_make_lesson(id="new_one"))

        loaded = storage.load_all()
        assert len(loaded) == 2

    def test_delete(self, tmp_path):
        """Delete removes a lesson by ID."""
        path = str(tmp_path / "lessons.jsonl")
        storage = LessonStorage(path)

        storage.save(_make_lesson(id="keep"))
        storage.save(_make_lesson(id="remove"))

        assert storage.delete("remove") is True
        loaded = storage.load_all()
        assert len(loaded) == 1
        assert loaded[0].id == "keep"

    def test_delete_missing(self, tmp_path):
        """Delete of non-existent ID returns False."""
        path = str(tmp_path / "lessons.jsonl")
        storage = LessonStorage(path)

        storage.save(_make_lesson(id="only"))
        assert storage.delete("nonexistent") is False
        assert len(storage.load_all()) == 1

    def test_load_active(self, tmp_path):
        """load_active returns only active lessons, sorted by quality."""
        path = str(tmp_path / "lessons.jsonl")
        storage = LessonStorage(path)

        storage.save(_make_lesson(id="low", score=0.3))
        storage.save(_make_lesson(id="high", score=0.9))
        storage.save(_make_lesson(id="archived", score=1.0, status=STATUS_ARCHIVED))

        active = storage.load_active()
        assert len(active) == 2
        # Higher quality first
        assert active[0].id == "high"
        assert active[1].id == "low"

    def test_count_by_status(self, tmp_path):
        """count_by_status returns correct counts per status."""
        path = str(tmp_path / "lessons.jsonl")
        storage = LessonStorage(path)

        storage.save(_make_lesson(id="a1", status=STATUS_ACTIVE))
        storage.save(_make_lesson(id="a2", status=STATUS_ACTIVE))
        storage.save(_make_lesson(id="ar1", status=STATUS_ARCHIVED))

        counts = storage.count_by_status()
        assert counts[STATUS_ACTIVE] == 2
        assert counts[STATUS_ARCHIVED] == 1

    def test_clear(self, tmp_path):
        """clear removes all lessons."""
        path = str(tmp_path / "lessons.jsonl")
        storage = LessonStorage(path)

        storage.save(_make_lesson(id="l1"))
        storage.save(_make_lesson(id="l2"))
        assert storage.count() == 2

        storage.clear()
        assert storage.count() == 0

    def test_get_by_id(self, tmp_path):
        """get_by_id retrieves a specific lesson."""
        path = str(tmp_path / "lessons.jsonl")
        storage = LessonStorage(path)

        storage.save(_make_lesson(id="target", goal="Target lesson"))
        storage.save(_make_lesson(id="other"))

        found = storage.get_by_id("target")
        assert found is not None
        assert found.goal_summary == "Target lesson"

        assert storage.get_by_id("missing") is None

    def test_lazy_directory_creation(self, tmp_path):
        """Directory is not created until first write."""
        nested = tmp_path / "deep" / "nested" / "lessons.jsonl"
        storage = LessonStorage(str(nested))

        # Directory doesn't exist yet
        assert not nested.parent.exists()

        # Reading doesn't create directory
        assert storage.load_all() == []
        assert not nested.parent.exists()

        # Writing creates directory
        storage.save(_make_lesson())
        assert nested.parent.exists()

    def test_backward_compatible_loading(self, tmp_path):
        """Old records with missing fields load with defaults."""
        path = tmp_path / "lessons.jsonl"
        # Write a minimal record (missing many optional fields)
        minimal = {
            "id": "old-1",
            "goal_summary": "Old goal",
            "approach_summary": "Old approach",
            "key_insights": [],
            "anti_patterns": [],
            "context_tags": ["legacy"],
            "quality": {"tool_success_rate": 0.5},
        }
        with open(path, "w") as f:
            f.write(json.dumps(minimal) + "\n")

        storage = LessonStorage(str(path))
        loaded = storage.load_all()
        assert len(loaded) == 1
        assert loaded[0].status == STATUS_ACTIVE  # Default
        assert loaded[0].used_count == 0  # Default

    def test_unknown_keys_dropped(self, tmp_path):
        """Unknown keys in JSONL lines are silently dropped."""
        path = tmp_path / "lessons.jsonl"
        record = {
            "id": "extra-1",
            "goal_summary": "Goal",
            "approach_summary": "Approach",
            "key_insights": [],
            "anti_patterns": [],
            "context_tags": [],
            "quality": {},
            "unknown_field": "should be dropped",
            "another_unknown": 42,
        }
        with open(path, "w") as f:
            f.write(json.dumps(record) + "\n")

        storage = LessonStorage(str(path))
        loaded = storage.load_all()
        assert len(loaded) == 1
        assert not hasattr(loaded[0], "unknown_field")
