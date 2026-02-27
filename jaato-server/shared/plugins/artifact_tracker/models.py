"""Data models for the Artifact Tracker plugin.

Defines the core data structures for tracking artifacts (documents, tests,
configs, etc.) that the model creates or modifies during a session.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ArtifactType(Enum):
    """Types of artifacts that can be tracked."""
    DOCUMENT = "document"       # README, docs, markdown files
    TEST = "test"               # Test files, test suites
    CONFIG = "config"           # Configuration files
    CODE = "code"               # Source code files
    SCHEMA = "schema"           # Database schemas, API schemas
    SCRIPT = "script"           # Build scripts, automation
    DATA = "data"               # Data files, fixtures
    OTHER = "other"             # Anything else


class ReviewStatus(Enum):
    """Review status for artifacts."""
    CURRENT = "current"             # No review needed
    NEEDS_REVIEW = "needs_review"   # Changes may be needed
    IN_REVIEW = "in_review"         # Currently being reviewed
    ACKNOWLEDGED = "acknowledged"   # Reviewed but not updated


@dataclass
class ArtifactRecord:
    """A tracked artifact.

    Attributes:
        artifact_id: Unique identifier for this artifact.
        path: File path or identifier for the artifact.
        artifact_type: Category of artifact (document, test, etc.).
        description: Brief description of what this artifact is.
        created_at: When the artifact was first tracked.
        updated_at: When the artifact was last modified.
        review_status: Current review status.
        review_reason: Why review is needed (if applicable).
        tags: Labels for categorization and filtering.
        related_to: Paths/IDs of related artifacts.
        notes: Additional context or reminders.
    """

    artifact_id: str
    path: str
    artifact_type: ArtifactType
    description: str
    created_at: str  # ISO8601
    updated_at: str  # ISO8601
    review_status: ReviewStatus = ReviewStatus.CURRENT
    review_reason: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    related_to: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    @classmethod
    def create(
        cls,
        path: str,
        artifact_type: ArtifactType,
        description: str,
        tags: Optional[List[str]] = None,
        related_to: Optional[List[str]] = None,
        notes: Optional[str] = None
    ) -> 'ArtifactRecord':
        """Create a new artifact record with auto-generated ID."""
        now = datetime.now(timezone.utc).isoformat() + "Z"
        return cls(
            artifact_id=str(uuid.uuid4()),
            path=path,
            artifact_type=artifact_type,
            description=description,
            created_at=now,
            updated_at=now,
            tags=tags or [],
            related_to=related_to or [],
            notes=notes,
        )

    def mark_updated(self) -> None:
        """Mark the artifact as recently updated."""
        self.updated_at = datetime.now(timezone.utc).isoformat() + "Z"
        self.review_status = ReviewStatus.CURRENT
        self.review_reason = None

    def mark_for_review(self, reason: str) -> None:
        """Flag this artifact as needing review."""
        self.review_status = ReviewStatus.NEEDS_REVIEW
        self.review_reason = reason

    def start_review(self) -> None:
        """Mark that review has started."""
        self.review_status = ReviewStatus.IN_REVIEW

    def acknowledge_review(self, notes: Optional[str] = None) -> None:
        """Acknowledge the review without updating the artifact."""
        self.review_status = ReviewStatus.ACKNOWLEDGED
        if notes:
            self.notes = notes

    def add_relation(self, path: str) -> None:
        """Add a related artifact path."""
        if path not in self.related_to:
            self.related_to.append(path)

    def remove_relation(self, path: str) -> None:
        """Remove a related artifact path."""
        if path in self.related_to:
            self.related_to.remove(path)

    def add_tag(self, tag: str) -> None:
        """Add a tag."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag."""
        if tag in self.tags:
            self.tags.remove(tag)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "artifact_type": self.artifact_type.value,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "review_status": self.review_status.value,
            "review_reason": self.review_reason,
            "tags": self.tags,
            "related_to": self.related_to,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArtifactRecord':
        """Create from dictionary."""
        # Parse artifact type
        type_str = data.get("artifact_type", "other")
        try:
            artifact_type = ArtifactType(type_str)
        except ValueError:
            artifact_type = ArtifactType.OTHER

        # Parse review status
        status_str = data.get("review_status", "current")
        try:
            review_status = ReviewStatus(status_str)
        except ValueError:
            review_status = ReviewStatus.CURRENT

        return cls(
            artifact_id=data.get("artifact_id", str(uuid.uuid4())),
            path=data.get("path", ""),
            artifact_type=artifact_type,
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat() + "Z"),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat() + "Z"),
            review_status=review_status,
            review_reason=data.get("review_reason"),
            tags=data.get("tags", []),
            related_to=data.get("related_to", []),
            notes=data.get("notes"),
        )


@dataclass
class ArtifactRegistry:
    """Collection of tracked artifacts with lookup capabilities."""

    artifacts: Dict[str, ArtifactRecord] = field(default_factory=dict)
    # Index by path for quick lookup
    _path_index: Dict[str, str] = field(default_factory=dict)

    def add(self, artifact: ArtifactRecord) -> None:
        """Add an artifact to the registry."""
        self.artifacts[artifact.artifact_id] = artifact
        self._path_index[artifact.path] = artifact.artifact_id

    def remove(self, artifact_id: str) -> Optional[ArtifactRecord]:
        """Remove an artifact by ID."""
        artifact = self.artifacts.pop(artifact_id, None)
        if artifact:
            self._path_index.pop(artifact.path, None)
        return artifact

    def get_by_id(self, artifact_id: str) -> Optional[ArtifactRecord]:
        """Get an artifact by its ID."""
        return self.artifacts.get(artifact_id)

    def get_by_path(self, path: str) -> Optional[ArtifactRecord]:
        """Get an artifact by its path."""
        artifact_id = self._path_index.get(path)
        if artifact_id:
            return self.artifacts.get(artifact_id)
        return None

    def get_all(self) -> List[ArtifactRecord]:
        """Get all artifacts."""
        return list(self.artifacts.values())

    def get_by_type(self, artifact_type: ArtifactType) -> List[ArtifactRecord]:
        """Get all artifacts of a specific type."""
        return [a for a in self.artifacts.values() if a.artifact_type == artifact_type]

    def get_by_tag(self, tag: str) -> List[ArtifactRecord]:
        """Get all artifacts with a specific tag."""
        return [a for a in self.artifacts.values() if tag in a.tags]

    def get_needing_review(self) -> List[ArtifactRecord]:
        """Get all artifacts that need review."""
        return [
            a for a in self.artifacts.values()
            if a.review_status == ReviewStatus.NEEDS_REVIEW
        ]

    def get_related(self, path: str) -> List[ArtifactRecord]:
        """Get all artifacts related to a given path."""
        results = []
        for artifact in self.artifacts.values():
            if path in artifact.related_to or artifact.path == path:
                results.append(artifact)
        # Also find artifacts that this path relates to
        target = self.get_by_path(path)
        if target:
            for related_path in target.related_to:
                related = self.get_by_path(related_path)
                if related and related not in results:
                    results.append(related)
        return results

    def find_affected_by_change(self, changed_path: str) -> List[ArtifactRecord]:
        """Find artifacts that might be affected by a change to the given path.

        This finds:
        1. The artifact itself (if tracked)
        2. All artifacts that list this path in related_to
        3. All artifacts related to this one
        """
        affected: Set[str] = set()
        changed_artifact = self.get_by_path(changed_path)

        # The changed artifact itself
        if changed_artifact:
            affected.add(changed_artifact.artifact_id)
            # Add its relations
            for related_path in changed_artifact.related_to:
                related = self.get_by_path(related_path)
                if related:
                    affected.add(related.artifact_id)

        # Find artifacts that reference the changed path
        for artifact in self.artifacts.values():
            if changed_path in artifact.related_to:
                affected.add(artifact.artifact_id)

        return [self.artifacts[aid] for aid in affected if aid in self.artifacts]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "artifacts": {
                aid: a.to_dict() for aid, a in self.artifacts.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArtifactRegistry':
        """Create from dictionary."""
        registry = cls()
        artifacts_data = data.get("artifacts", {})
        for artifact_data in artifacts_data.values():
            artifact = ArtifactRecord.from_dict(artifact_data)
            registry.add(artifact)
        return registry
