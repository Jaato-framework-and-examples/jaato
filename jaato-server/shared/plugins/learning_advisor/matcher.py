"""Goal similarity matcher for the learning advisor plugin.

Implements a hybrid matching strategy:

1. **Tag-based pre-filter** — fast, deterministic keyword overlap using the
   same stopword/keyword extraction approach as the memory plugin's indexer.
2. **Scored ranking** — tag overlap count × quality composite, so that
   higher-quality lessons surface first when multiple match.

The matcher maintains an in-memory index of ``LessonMetadata`` keyed by
context tags, rebuilt on startup from the JSONL store.  Only *active*
lessons are eligible for matching (archived/dismissed are excluded).
"""

import re
from typing import Dict, List, Set

from .models import (
    Lesson,
    LessonMetadata,
    STATUS_ACTIVE,
)


# Common English stopwords — same list as memory plugin indexer.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "could", "may", "might", "can", "about", "how", "what", "when", "where",
    "which", "who", "why", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "them", "their", "my", "your", "our",
    "please", "help", "want", "need", "make", "create", "change", "update",
    "also", "just", "like", "using", "used", "use",
}


class GoalMatcher:
    """Matches incoming goals against stored lessons by keyword similarity.

    Maintains lightweight in-memory indexes for fast lookup:

    - ``_tag_index``:  Maps each normalised tag to the set of lesson IDs
      whose ``context_tags`` contain it.
    - ``_lessons``:  Maps lesson ID to ``LessonMetadata`` (no full content).

    Only *active* lessons participate in matching.  Archived and dismissed
    lessons remain indexed for administrative queries but are filtered out
    at match time.

    Attributes:
        _tag_index: Inverted index from tag → lesson IDs.
        _lessons: Lesson ID → lightweight metadata.
    """

    def __init__(self) -> None:
        """Initialize empty indexes."""
        self._tag_index: Dict[str, List[str]] = {}  # tag → [lesson_id, …]
        self._lessons: Dict[str, LessonMetadata] = {}  # id → metadata

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, lessons: List[Lesson]) -> None:
        """Build the index from a list of full ``Lesson`` objects.

        Called once on plugin startup with the contents of the JSONL store.

        Args:
            lessons: All persisted lessons (any status).
        """
        self._tag_index.clear()
        self._lessons.clear()
        for lesson in lessons:
            self.index_lesson(lesson)

    def index_lesson(self, lesson: Lesson) -> None:
        """Add or update a single lesson in the index.

        Args:
            lesson: Full lesson object.
        """
        meta = LessonMetadata(
            id=lesson.id,
            goal_summary=lesson.goal_summary,
            context_tags=lesson.context_tags,
            quality_composite=lesson.quality.composite,
            status=lesson.status,
            created_at=lesson.created_at,
            scope=lesson.scope,
        )
        self._lessons[lesson.id] = meta

        for tag in lesson.context_tags:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = []
            if lesson.id not in self._tag_index[tag_lower]:
                self._tag_index[tag_lower].append(lesson.id)

    def remove_lesson(self, lesson_id: str) -> None:
        """Remove a lesson from the index.

        Args:
            lesson_id: ID of the lesson to remove.
        """
        self._lessons.pop(lesson_id, None)
        for tag, ids in list(self._tag_index.items()):
            if lesson_id in ids:
                ids.remove(lesson_id)
                if not ids:
                    del self._tag_index[tag]

    # ------------------------------------------------------------------
    # Keyword extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """Extract searchable keywords from a goal description or prompt.

        Strategy (mirrors memory plugin indexer):
        1. Tokenise on word boundaries (including underscores).
        2. Lowercase.
        3. Drop stopwords.
        4. Drop very short tokens (< 4 chars).

        Args:
            text: Free-form goal description or user prompt.

        Returns:
            De-duplicated list of keywords.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        seen: Set[str] = set()
        keywords: List[str] = []
        for w in words:
            if w in STOPWORDS or len(w) < 4 or w in seen:
                continue
            seen.add(w)
            keywords.append(w)
        return keywords

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def find_matching_lessons(
        self,
        keywords: List[str],
        limit: int = 3,
        *,
        active_only: bool = True,
        min_overlap: int = 1,
    ) -> List[LessonMetadata]:
        """Find lessons whose context tags overlap the given keywords.

        Matching strategy:
        1. Exact tag match (keyword == tag).
        2. Partial substring match (keyword in tag or tag in keyword).
        3. Score = overlap_count × quality_composite.
        4. Sort by score descending, then by recency.

        Only active lessons are returned by default.

        Args:
            keywords: Extracted keywords from the incoming goal/prompt.
            limit: Maximum number of lessons to return.
            active_only: If True, exclude archived/dismissed lessons.
            min_overlap: Minimum number of overlapping tags required.

        Returns:
            Ranked list of ``LessonMetadata``, best match first.
        """
        keywords_lower = [kw.lower() for kw in keywords]

        # Collect (lesson_id → overlap count)
        overlap_counts: Dict[str, int] = {}

        for kw in keywords_lower:
            # Exact matches
            if kw in self._tag_index:
                for lid in self._tag_index[kw]:
                    overlap_counts[lid] = overlap_counts.get(lid, 0) + 1

            # Partial substring matches
            for tag, ids in self._tag_index.items():
                if kw != tag and (kw in tag or tag in kw):
                    for lid in ids:
                        overlap_counts[lid] = overlap_counts.get(lid, 0) + 1

        # Build scored results
        scored: List[tuple] = []
        for lid, overlap in overlap_counts.items():
            if overlap < min_overlap:
                continue
            meta = self._lessons.get(lid)
            if meta is None:
                continue
            if active_only and meta.status != STATUS_ACTIVE:
                continue
            # Score = overlap × quality (rewards both relevance and quality)
            score = overlap * meta.quality_composite
            scored.append((score, meta.created_at, meta))

        # Sort: score desc, then recency desc
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        return [meta for _, _, meta in scored[:limit]]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_lesson_count(self, *, active_only: bool = False) -> int:
        """Return the number of indexed lessons.

        Args:
            active_only: If True, count only active lessons.
        """
        if not active_only:
            return len(self._lessons)
        return sum(1 for m in self._lessons.values() if m.status == STATUS_ACTIVE)

    def clear(self) -> None:
        """Clear all index data."""
        self._tag_index.clear()
        self._lessons.clear()
