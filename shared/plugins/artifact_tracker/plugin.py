"""Artifact Tracker plugin for tracking created/modified artifacts.

This plugin helps the model keep track of artifacts (documents, tests, configs,
etc.) it creates or modifies during a session. The key feature is the system
instructions that remind the model to review related artifacts when making
changes, ensuring consistency across related files.

Example workflow:
1. Model creates README.md -> tracks it with trackArtifact
2. Model creates tests/test_api.py -> tracks it, relates to src/api.py
3. Model modifies src/api.py -> plugin reminds to check test_api.py
4. Model reviews and updates test_api.py -> marks as reviewed

The plugin persists state to a JSON file so artifacts survive session restarts.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional

from .models import (
    ArtifactRecord,
    ArtifactRegistry,
    ArtifactType,
    ReviewStatus,
)
from ..model_provider.types import ToolSchema
from ..base import UserCommand


# Default storage location
DEFAULT_STORAGE_PATH = ".artifact_tracker.json"


class ArtifactTrackerPlugin:
    """Plugin that tracks artifacts created/modified by the model.

    Key features:
    - Track documents, tests, configs, and other artifacts
    - Define relationships between artifacts (artifact depends on source files)
    - Auto-flag artifacts for review when related source files change
    - System instructions that guide the model through the workflow

    Tools provided:
    - trackArtifact: Register a new artifact with its dependencies
    - updateArtifact: Update artifact metadata
    - listArtifacts: Show all tracked artifacts (with filtering)
    - checkRelated: Find artifacts that depend on a file (BEFORE modifying)
    - notifyChange: Auto-flag dependent artifacts (AFTER modifying a source file)
    - acknowledgeReview: Mark artifact as reviewed
    - flagForReview: Manually flag a single artifact
    - removeArtifact: Stop tracking an artifact
    """

    def __init__(self):
        self._registry: Optional[ArtifactRegistry] = None
        self._storage_path: Optional[str] = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "artifact_tracker"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the artifact tracker plugin.

        Args:
            config: Optional configuration dict:
                - storage_path: Path to JSON file for persistence
                - auto_load: Whether to load existing state (default: True)
        """
        config = config or {}

        # Set storage path
        self._storage_path = config.get("storage_path", DEFAULT_STORAGE_PATH)

        # Initialize registry
        self._registry = ArtifactRegistry()

        # Load existing state if available
        if config.get("auto_load", True):
            self._load_state()

        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the plugin and save state."""
        if self._registry:
            self._save_state()
        self._registry = None
        self._initialized = False

    def _load_state(self) -> None:
        """Load state from storage file."""
        if not self._storage_path or not os.path.exists(self._storage_path):
            return

        try:
            with open(self._storage_path, 'r') as f:
                data = json.load(f)
                self._registry = ArtifactRegistry.from_dict(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load artifact tracker state: {e}")
            self._registry = ArtifactRegistry()

    def _save_state(self) -> None:
        """Save state to storage file."""
        if not self._storage_path or not self._registry:
            return

        try:
            with open(self._storage_path, 'w') as f:
                json.dump(self._registry.to_dict(), f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save artifact tracker state: {e}")

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for artifact tracking tools."""
        return [
            ToolSchema(
                name="trackArtifact",
                description=(
                    "WORKFLOW STEP: Register a new artifact after creating it. "
                    "Use for documents, tests, configs that should stay in sync with code. "
                    "Set `related_to` to list source files this artifact depends on. "
                    "NEXT: When you later modify those source files, call `notifyChange` to flag this artifact for review."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path or identifier for the artifact"
                        },
                        "artifact_type": {
                            "type": "string",
                            "enum": ["document", "test", "config", "code", "schema", "script", "data", "other"],
                            "description": "Category of artifact"
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of what this artifact is/does"
                        },
                        "related_to": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Paths of related artifacts that should trigger review when changed"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Labels for categorization (e.g., 'api', 'auth', 'frontend')"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Additional context or reminders about this artifact"
                        }
                    },
                    "required": ["path", "artifact_type", "description"]
                }
            ),
            ToolSchema(
                name="updateArtifact",
                description=(
                    "Update metadata for a tracked artifact (description, relations, tags). "
                    "Use `mark_updated=true` after modifying the artifact's content to clear review flags. "
                    "Use `add_related` to link to additional source files this artifact depends on."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path of the artifact to update"
                        },
                        "description": {
                            "type": "string",
                            "description": "New description (optional)"
                        },
                        "add_related": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Paths to add as related artifacts"
                        },
                        "remove_related": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Paths to remove from relations"
                        },
                        "add_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to add to existing tags"
                        },
                        "remove_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to remove from existing tags"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Replace ALL tags with this list (use add_tags/remove_tags for incremental changes)"
                        },
                        "notes": {
                            "type": "string",
                            "description": "New notes (replaces existing)"
                        },
                        "mark_updated": {
                            "type": "boolean",
                            "description": "Set to true if you've updated the artifact content"
                        }
                    },
                    "required": ["path"]
                }
            ),
            ToolSchema(
                name="listArtifacts",
                description=(
                    "List all tracked artifacts. Use `needs_review=true` to see only artifacts "
                    "flagged for review. Filter by `artifact_type` or `tag` to narrow results. "
                    "Check this periodically to see what needs attention."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "artifact_type": {
                            "type": "string",
                            "enum": ["document", "test", "config", "code", "schema", "script", "data", "other"],
                            "description": "Filter by artifact type"
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter by tag"
                        },
                        "needs_review": {
                            "type": "boolean",
                            "description": "Set to true to only show artifacts needing review"
                        }
                    },
                    "required": []
                }
            ),
            ToolSchema(
                name="flagForReview",
                description=(
                    "Manually mark a single artifact as needing review. "
                    "PREFER using `notifyChange` instead - it automatically flags ALL dependent artifacts. "
                    "Use this only when you need to flag a specific artifact with a custom reason."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path of the artifact to flag"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why this artifact needs review"
                        }
                    },
                    "required": ["path", "reason"]
                }
            ),
            ToolSchema(
                name="acknowledgeReview",
                description=(
                    "WORKFLOW STEP: Call after reviewing a flagged artifact. "
                    "Set `was_updated=true` if you modified the artifact content. "
                    "Set `notes` to explain what you checked/changed. "
                    "This clears the review flag so it won't appear in reminders."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path of the artifact"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Notes about the review (e.g., 'No changes needed')"
                        },
                        "was_updated": {
                            "type": "boolean",
                            "description": "Set to true if you updated the artifact"
                        }
                    },
                    "required": ["path"]
                }
            ),
            ToolSchema(
                name="checkRelated",
                description=(
                    "WORKFLOW STEP: Call BEFORE modifying a file to preview impact. "
                    "Shows all tracked artifacts that depend on this file (have it in `related_to`). "
                    "If artifacts are found, plan to review them after your changes. "
                    "NEXT: After modifying the file, call `notifyChange` to flag dependent artifacts."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to check for related artifacts"
                        }
                    },
                    "required": ["path"]
                }
            ),
            ToolSchema(
                name="removeArtifact",
                description=(
                    "Stop tracking an artifact. Use when deleting an artifact file "
                    "or when it no longer needs to stay in sync with other files."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path of the artifact to remove"
                        }
                    },
                    "required": ["path"]
                }
            ),
            ToolSchema(
                name="notifyChange",
                description=(
                    "WORKFLOW STEP: Call AFTER modifying a source file to auto-flag dependent artifacts. "
                    "This finds all tracked artifacts that have the changed file in their `related_to` list "
                    "and marks them as needing review. Much easier than manually calling `flagForReview` for each. "
                    "NEXT: Review each flagged artifact and call `acknowledgeReview` when done."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path of the file that was modified"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief description of what changed (e.g., 'Added new endpoint', 'Renamed function')"
                        }
                    },
                    "required": ["path", "reason"]
                }
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executors for artifact tracking tools."""
        return {
            "trackArtifact": self._execute_track_artifact,
            "updateArtifact": self._execute_update_artifact,
            "listArtifacts": self._execute_list_artifacts,
            "flagForReview": self._execute_flag_for_review,
            "acknowledgeReview": self._execute_acknowledge_review,
            "checkRelated": self._execute_check_related,
            "removeArtifact": self._execute_remove_artifact,
            "notifyChange": self._execute_notify_change,
            # User command aliases
            "artifacts": self._execute_list_artifacts,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the artifact tracker plugin.

        This is the key feature - reminding the model to check related artifacts.
        """
        # Build dynamic reminder based on current state
        reminders = []

        if self._registry:
            # Get artifacts needing review
            needs_review = self._registry.get_needing_review()
            if needs_review:
                reminders.append(
                    f"**ATTENTION**: {len(needs_review)} artifact(s) need review:\n" +
                    "\n".join(f"  - {a.path}: {a.review_reason}" for a in needs_review)
                )

            # Count tracked artifacts
            total = len(self._registry.get_all())
            if total > 0:
                reminders.append(f"Currently tracking {total} artifact(s).")

        reminder_text = "\n\n".join(reminders) if reminders else ""

        return f"""You have access to ARTIFACT TRACKING tools to keep related files in sync.

**PURPOSE**: Track documents, tests, and configs so you remember to update them when related code changes.

**COMPLETE WORKFLOW** (follow these steps):

┌─────────────────────────────────────────────────────────────────────┐
│  WHEN CREATING A NEW ARTIFACT (doc, test, config):                  │
│                                                                     │
│  1. Create the file                                                 │
│  2. Call `trackArtifact` with:                                      │
│     - path: the file you created                                    │
│     - related_to: source files it depends on                        │
│     - description: what this artifact is                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  WHEN MODIFYING A SOURCE FILE:                                      │
│                                                                     │
│  1. BEFORE editing: `checkRelated(path)` → see what depends on it   │
│  2. Make your changes to the file                                   │
│  3. AFTER editing: `notifyChange(path, reason)` → auto-flags deps   │
│  4. Review each flagged artifact                                    │
│  5. For each: `acknowledgeReview(path, was_updated, notes)`         │
└─────────────────────────────────────────────────────────────────────┘

**TOOL QUICK REFERENCE**:
- `trackArtifact` → register new artifact with dependencies
- `checkRelated` → preview what artifacts depend on a file (BEFORE edit)
- `notifyChange` → auto-flag all dependent artifacts (AFTER edit)
- `acknowledgeReview` → clear review flag after checking artifact
- `listArtifacts` → see all tracked artifacts and their status
- `updateArtifact` → modify artifact metadata
- `flagForReview` → manually flag single artifact (prefer notifyChange)
- `removeArtifact` → stop tracking an artifact

**RELATIONSHIP PATTERN**:
The artifact's `related_to` lists what SOURCE FILES it depends on.
When those source files change, the artifact needs review.

Example: `tests/test_api.py` has `related_to: ["src/api.py"]`
→ When you modify `src/api.py`, call `notifyChange("src/api.py", "reason")`
→ This automatically flags `tests/test_api.py` for review

{reminder_text}"""

    def get_auto_approved_tools(self) -> List[str]:
        """Return artifact tracking tools as auto-approved (no security implications)."""
        return [
            "trackArtifact",
            "updateArtifact",
            "listArtifacts",
            "flagForReview",
            "acknowledgeReview",
            "checkRelated",
            "removeArtifact",
            "notifyChange",
            "artifacts",
        ]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user-facing commands for direct invocation."""
        return [
            UserCommand(
                "artifacts",
                "Show all tracked artifacts and their status",
                share_with_model=True  # Model should see this to know what's tracked
            ),
        ]

    # ==================== Tool Executors ====================

    def _execute_track_artifact(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the trackArtifact tool."""
        path = args.get("path", "")
        type_str = args.get("artifact_type", "other")
        description = args.get("description", "")
        related_to = args.get("related_to", [])
        tags = args.get("tags", [])
        notes = args.get("notes")

        if not path:
            return {"error": "path is required"}
        if not description:
            return {"error": "description is required"}

        # Check if already tracked
        if self._registry and self._registry.get_by_path(path):
            return {"error": f"Artifact already tracked: {path}. Use updateArtifact to modify."}

        # Parse artifact type
        try:
            artifact_type = ArtifactType(type_str)
        except ValueError:
            artifact_type = ArtifactType.OTHER

        # Create artifact
        artifact = ArtifactRecord.create(
            path=path,
            artifact_type=artifact_type,
            description=description,
            tags=tags,
            related_to=related_to,
            notes=notes,
        )

        # Add to registry
        if self._registry:
            self._registry.add(artifact)
            self._save_state()

        return {
            "success": True,
            "artifact_id": artifact.artifact_id,
            "path": artifact.path,
            "artifact_type": artifact.artifact_type.value,
            "description": artifact.description,
            "related_to": artifact.related_to,
            "tags": artifact.tags,
            "message": f"Now tracking: {path}"
        }

    def _execute_update_artifact(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the updateArtifact tool."""
        path = args.get("path", "")

        if not path:
            return {"error": "path is required"}

        if not self._registry:
            return {"error": "Plugin not initialized"}

        artifact = self._registry.get_by_path(path)
        if not artifact:
            return {"error": f"Artifact not found: {path}"}

        # Apply updates
        if "description" in args and args["description"]:
            artifact.description = args["description"]

        for rel_path in args.get("add_related", []):
            artifact.add_relation(rel_path)

        for rel_path in args.get("remove_related", []):
            artifact.remove_relation(rel_path)

        # Handle tags - "tags" replaces all, add_tags/remove_tags are incremental
        if "tags" in args:
            # Replace all tags
            artifact.tags = list(args["tags"]) if args["tags"] else []
        else:
            # Incremental tag changes
            for tag in args.get("add_tags", []):
                artifact.add_tag(tag)

            for tag in args.get("remove_tags", []):
                artifact.remove_tag(tag)

        if "notes" in args:
            artifact.notes = args["notes"]

        if args.get("mark_updated", False):
            artifact.mark_updated()

        self._save_state()

        return {
            "success": True,
            "artifact_id": artifact.artifact_id,
            "path": artifact.path,
            "description": artifact.description,
            "related_to": artifact.related_to,
            "tags": artifact.tags,
            "review_status": artifact.review_status.value,
        }

    def _execute_list_artifacts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the listArtifacts tool."""
        if not self._registry:
            return {"error": "Plugin not initialized"}

        # Get all artifacts
        artifacts = self._registry.get_all()

        # Apply filters
        type_filter = args.get("artifact_type")
        if type_filter:
            try:
                artifact_type = ArtifactType(type_filter)
                artifacts = [a for a in artifacts if a.artifact_type == artifact_type]
            except ValueError:
                pass

        tag_filter = args.get("tag")
        if tag_filter:
            artifacts = [a for a in artifacts if tag_filter in a.tags]

        if args.get("needs_review", False):
            artifacts = [a for a in artifacts if a.review_status == ReviewStatus.NEEDS_REVIEW]

        # Format results
        results = []
        for artifact in artifacts:
            results.append({
                "path": artifact.path,
                "type": artifact.artifact_type.value,
                "description": artifact.description,
                "review_status": artifact.review_status.value,
                "review_reason": artifact.review_reason,
                "related_to": artifact.related_to,
                "tags": artifact.tags,
                "updated_at": artifact.updated_at,
            })

        return {
            "total": len(results),
            "artifacts": results,
        }

    def _execute_flag_for_review(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the flagForReview tool."""
        path = args.get("path", "")
        reason = args.get("reason", "")

        if not path:
            return {"error": "path is required"}
        if not reason:
            return {"error": "reason is required"}

        if not self._registry:
            return {"error": "Plugin not initialized"}

        artifact = self._registry.get_by_path(path)
        if not artifact:
            return {"error": f"Artifact not found: {path}"}

        artifact.mark_for_review(reason)
        self._save_state()

        return {
            "success": True,
            "path": artifact.path,
            "review_status": artifact.review_status.value,
            "review_reason": artifact.review_reason,
            "message": f"Flagged for review: {path}",
        }

    def _execute_acknowledge_review(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the acknowledgeReview tool."""
        path = args.get("path", "")
        notes = args.get("notes")
        was_updated = args.get("was_updated", False)

        if not path:
            return {"error": "path is required"}

        if not self._registry:
            return {"error": "Plugin not initialized"}

        artifact = self._registry.get_by_path(path)
        if not artifact:
            return {"error": f"Artifact not found: {path}"}

        if was_updated:
            artifact.mark_updated()
        else:
            artifact.acknowledge_review(notes)

        self._save_state()

        return {
            "success": True,
            "path": artifact.path,
            "review_status": artifact.review_status.value,
            "notes": artifact.notes,
            "message": f"Review acknowledged: {path}",
        }

    def _execute_check_related(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the checkRelated tool."""
        path = args.get("path", "")

        if not path:
            return {"error": "path is required"}

        if not self._registry:
            return {"error": "Plugin not initialized"}

        # Find affected artifacts
        affected = self._registry.find_affected_by_change(path)

        if not affected:
            return {
                "path": path,
                "related_count": 0,
                "related": [],
                "message": f"No tracked artifacts are related to: {path}",
            }

        results = []
        for artifact in affected:
            results.append({
                "path": artifact.path,
                "type": artifact.artifact_type.value,
                "description": artifact.description,
                "review_status": artifact.review_status.value,
                "is_source": artifact.path == path,
            })

        return {
            "path": path,
            "related_count": len(results),
            "related": results,
            "message": f"Found {len(results)} artifact(s) related to: {path}",
            "recommendation": "Consider reviewing/updating these artifacts if you modify this file.",
        }

    def _execute_remove_artifact(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the removeArtifact tool."""
        path = args.get("path", "")

        if not path:
            return {"error": "path is required"}

        if not self._registry:
            return {"error": "Plugin not initialized"}

        artifact = self._registry.get_by_path(path)
        if not artifact:
            return {"error": f"Artifact not found: {path}"}

        self._registry.remove(artifact.artifact_id)
        self._save_state()

        return {
            "success": True,
            "path": path,
            "message": f"Stopped tracking: {path}",
        }

    def _execute_notify_change(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the notifyChange tool.

        Automatically flags all artifacts that depend on the changed file.
        """
        path = args.get("path", "")
        reason = args.get("reason", "")

        if not path:
            return {"error": "path is required"}
        if not reason:
            return {"error": "reason is required"}

        if not self._registry:
            return {"error": "Plugin not initialized"}

        # Find all artifacts that have this path in their related_to list
        flagged = []
        for artifact in self._registry.get_all():
            if path in artifact.related_to:
                artifact.mark_for_review(f"Source changed: {reason}")
                flagged.append({
                    "path": artifact.path,
                    "type": artifact.artifact_type.value,
                    "description": artifact.description,
                })

        if flagged:
            self._save_state()

        return {
            "success": True,
            "changed_path": path,
            "reason": reason,
            "flagged_count": len(flagged),
            "flagged_artifacts": flagged,
            "message": (
                f"Flagged {len(flagged)} artifact(s) for review due to changes in: {path}"
                if flagged else f"No tracked artifacts depend on: {path}"
            ),
            "next_step": (
                "Review each flagged artifact and call `acknowledgeReview` when done."
                if flagged else None
            ),
        }


def create_plugin() -> ArtifactTrackerPlugin:
    """Factory function to create the Artifact Tracker plugin instance."""
    return ArtifactTrackerPlugin()
