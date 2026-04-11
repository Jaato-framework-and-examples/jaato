"""JSONL-based storage backend for the learning advisor plugin.

Follows the same patterns as the memory plugin's storage:
- Append-only writes for normal operation.
- Full rewrite for updates/deletes (acceptable for small stores).
- Backward-compatible loading: unknown keys are dropped, missing keys
  get dataclass defaults.
- Lazy directory creation (only on first write, not on init).
"""

import json
from dataclasses import asdict, fields as dc_fields
from pathlib import Path
from typing import Dict, List, Optional, Set

from .models import (
    Lesson,
    QualityScore,
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
)


# Fields on the Lesson dataclass, for filtering unknown keys.
_LESSON_FIELD_NAMES: Set[str] = {f.name for f in dc_fields(Lesson)}


def _lesson_from_dict(data: dict) -> Lesson:
    """Construct a ``Lesson`` from a raw JSON dict.

    Provides backward compatibility:
    - Drops unknown keys from hand-edited files.
    - Fills missing fields via dataclass defaults.
    - Reconstructs the nested ``QualityScore`` from its dict representation.
    """
    filtered = {k: v for k, v in data.items() if k in _LESSON_FIELD_NAMES}

    # Reconstruct nested QualityScore
    quality_raw = filtered.get("quality")
    if isinstance(quality_raw, dict):
        filtered["quality"] = QualityScore(**quality_raw)
    elif quality_raw is None:
        filtered["quality"] = QualityScore()

    return Lesson(**filtered)


class LessonStorage:
    """JSONL-based storage for lessons learned.

    Each lesson is stored as a single JSON line.  The format supports
    easy appending and sequential reading.

    The storage is workspace-scoped: a relative path template (e.g.
    ``.jaato/advisor_lessons.jsonl``) is resolved against the workspace
    root when ``set_workspace_path()`` is called on the owning plugin.

    Attributes:
        path: Resolved ``Path`` to the JSONL file.
    """

    def __init__(self, path: str):
        """Initialize storage with a file path.

        Directory creation is deferred to the first write so that
        read-only usage against a not-yet-existing path does not
        create stale directories.

        Args:
            path: Path to the JSONL file for storing lessons.
        """
        self.path = Path(path)

    def _ensure_parent_dir(self) -> None:
        """Create parent directory if it does not exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save(self, lesson: Lesson) -> None:
        """Append a lesson to the JSONL file.

        Args:
            lesson: Lesson object to persist.
        """
        self._ensure_parent_dir()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(lesson)) + "\n")

    def update(self, lesson: Lesson) -> None:
        """Update an existing lesson (full file rewrite).

        If the lesson ID is not found, it is appended instead.

        Args:
            lesson: Lesson with updated fields.
        """
        all_lessons = self.load_all()

        updated = False
        for i, existing in enumerate(all_lessons):
            if existing.id == lesson.id:
                all_lessons[i] = lesson
                updated = True
                break

        if not updated:
            all_lessons.append(lesson)

        self._rewrite(all_lessons)

    def delete(self, lesson_id: str) -> bool:
        """Delete a lesson by ID (full file rewrite).

        Args:
            lesson_id: Unique identifier of the lesson to delete.

        Returns:
            True if the lesson was found and deleted, False otherwise.
        """
        all_lessons = self.load_all()
        before = len(all_lessons)
        all_lessons = [l for l in all_lessons if l.id != lesson_id]

        if len(all_lessons) == before:
            return False

        self._rewrite(all_lessons)
        return True

    def _rewrite(self, lessons: List[Lesson]) -> None:
        """Rewrite the entire JSONL file with the given lessons."""
        self._ensure_parent_dir()
        with open(self.path, "w", encoding="utf-8") as f:
            for lesson in lessons:
                f.write(json.dumps(asdict(lesson)) + "\n")

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def load_all(self) -> List[Lesson]:
        """Load all lessons from the JSONL file.

        Backward-compatible: lines with unknown or missing fields are
        handled gracefully.

        Returns:
            List of Lesson objects, or empty list if file doesn't exist.
        """
        if not self.path.exists():
            return []

        lessons: List[Lesson] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    lessons.append(_lesson_from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    # Skip corrupt lines rather than crash
                    continue
        return lessons

    def load_active(self) -> List[Lesson]:
        """Load only active lessons (not archived or dismissed).

        Returns:
            Active lessons sorted by quality (best first).
        """
        active = [l for l in self.load_all() if l.status == STATUS_ACTIVE]
        active.sort(key=lambda l: l.quality.composite, reverse=True)
        return active

    def count(self) -> int:
        """Return total number of stored lessons."""
        return len(self.load_all())

    def count_by_status(self) -> Dict[str, int]:
        """Return ``{status: count}`` for all stored lessons."""
        counts: Dict[str, int] = {}
        for lesson in self.load_all():
            counts[lesson.status] = counts.get(lesson.status, 0) + 1
        return counts

    def get_by_id(self, lesson_id: str) -> Optional[Lesson]:
        """Retrieve a specific lesson by ID.

        Args:
            lesson_id: Unique identifier.

        Returns:
            The matching Lesson or None.
        """
        for lesson in self.load_all():
            if lesson.id == lesson_id:
                return lesson
        return None

    def clear(self) -> None:
        """Delete all lessons by truncating the file."""
        if self.path.exists():
            self._ensure_parent_dir()
            with open(self.path, "w", encoding="utf-8") as f:
                pass  # Truncate
