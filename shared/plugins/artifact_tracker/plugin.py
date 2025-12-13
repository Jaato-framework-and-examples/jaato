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
    - Define relationships between artifacts
    - Flag artifacts for review when related files change
    - System instructions that remind model to check related artifacts

    Tools provided:
    - trackArtifact: Register a new artifact to track
    - updateArtifact: Update artifact metadata
    - listArtifacts: Show all tracked artifacts
    - flagForReview: Mark artifact as needing review
    - acknowledgeReview: Mark artifact as reviewed
    - removeArtifact: Stop tracking an artifact
    - checkRelated: Find artifacts related to a path
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
                    "Register a new artifact (document, test, config, etc.) to track. "
                    "Use this whenever you create or significantly modify a file that "
                    "should be kept in sync with other parts of the codebase. "
                    "Tracking artifacts helps ensure they stay up-to-date when related code changes."
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
                    "Update metadata for a tracked artifact. Use when you've made "
                    "changes to an artifact and want to update its tracking info, "
                    "add relations, or update the description."
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
                            "description": "Tags to add"
                        },
                        "remove_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to remove"
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
                    "List all tracked artifacts, optionally filtered by type, tag, "
                    "or review status. Use this to see what artifacts are being tracked "
                    "and which ones need attention."
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
                    "Mark an artifact as needing review. Use this when you've made "
                    "changes that might affect a related artifact, to remind yourself "
                    "to check and update it."
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
                    "Mark an artifact as reviewed. Use after you've checked an artifact "
                    "and either updated it or confirmed no changes are needed."
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
                    "Find all artifacts related to a given path. Use this BEFORE making "
                    "changes to a file to see what other artifacts might need updating. "
                    "This helps maintain consistency across related files."
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
                    "Stop tracking an artifact. Use when an artifact is deleted or "
                    "no longer needs to be tracked."
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

**PURPOSE**:
These tools help you remember what documents, tests, and configs you've created
or modified, and remind you to update them when related code changes.

**WHEN TO TRACK ARTIFACTS**:
- README files and documentation you create/update
- Test files (especially when related to specific source files)
- Configuration files that depend on code structure
- Schema files (database, API) that affect multiple files
- Any file that should stay in sync with other parts of the codebase

**CRITICAL WORKFLOW**:
1. **BEFORE modifying a file**: Use `checkRelated` to see if any tracked
   artifacts depend on it. If so, plan to review/update them too.

2. **AFTER creating/modifying important files**: Use `trackArtifact` to
   register them with appropriate `related_to` links.

3. **When you see "ATTENTION" above**: Review and address flagged artifacts
   before continuing with other work.

**RELATIONSHIP EXAMPLES**:
- Test file `tests/test_api.py` → related_to: `["src/api.py"]`
- README `docs/auth.md` → related_to: `["src/auth/", "src/middleware/auth.py"]`
- Config `config/routes.json` → related_to: `["src/routes/"]`

**BEST PRACTICES**:
- Track tests and link them to the code they test
- Track documentation and link to the code it documents
- Use descriptive `reason` when flagging for review
- Don't forget to `acknowledgeReview` after checking artifacts

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


def create_plugin() -> ArtifactTrackerPlugin:
    """Factory function to create the Artifact Tracker plugin instance."""
    return ArtifactTrackerPlugin()
