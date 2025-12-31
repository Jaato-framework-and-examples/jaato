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
import tempfile
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .models import (
    ArtifactRecord,
    ArtifactRegistry,
    ArtifactType,
    ReviewStatus,
)
from ..model_provider.types import ToolSchema
from ..base import UserCommand, ToolResultEnrichmentResult


# Default storage location (inside .jaato directory)
DEFAULT_STORAGE_PATH = ".jaato/.artifact_tracker.json"

# Tools that write/modify files - same as LSP plugin for consistency
FILE_WRITING_TOOLS = {
    'updateFile',
    'writeNewFile',
    'lsp_rename_symbol',
    'lsp_apply_code_action',
}


def _normalize_path(path: str) -> str:
    """Normalize a path to prevent duplicates.

    Handles:
    - Removes leading ./
    - Normalizes path separators
    - Removes trailing slashes (except for root)
    - Collapses redundant separators and .. references

    Args:
        path: The path to normalize

    Returns:
        Normalized path string
    """
    if not path:
        return path

    # Use os.path.normpath to handle .., redundant separators, etc.
    normalized = os.path.normpath(path)

    # normpath keeps leading ./ as just the path, but we want consistency
    # Also handle the case where path starts with ./
    if normalized == '.':
        return '.'

    return normalized


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
        self._agent_name: Optional[str] = None
        self._plugin_registry = None  # Set via set_plugin_registry() for cross-plugin access

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    agent_prefix = f"@{self._agent_name}" if self._agent_name else ""
                    f.write(f"[{ts}] [ARTIFACT_TRACKER{agent_prefix}] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass

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
        self._agent_name = config.get("agent_name")

        # Set storage path
        self._storage_path = config.get("storage_path", DEFAULT_STORAGE_PATH)

        # Initialize registry
        self._registry = ArtifactRegistry()

        # Load existing state if available
        if config.get("auto_load", True):
            self._load_state()

        self._initialized = True
        self._trace(f"initialize: storage_path={self._storage_path}")

    def shutdown(self) -> None:
        """Shutdown the plugin and save state."""
        self._trace("shutdown")
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
            # Ensure parent directory exists
            storage_path = os.path.abspath(self._storage_path)
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)

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
                    "WORKFLOW STEP: Call after reviewing flagged artifact(s). "
                    "Supports both single path and array of paths for batch acknowledgment. "
                    "Set `was_updated=true` if you modified the artifact content. "
                    "Set `notes` to explain what you checked/changed. "
                    "This clears the review flag so it won't appear in reminders."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path of a single artifact to acknowledge"
                        },
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of artifact paths to acknowledge (batch operation)"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Notes about the review (e.g., 'No changes needed') - applied to all"
                        },
                        "was_updated": {
                            "type": "boolean",
                            "description": "Set to true if you updated the artifact(s)"
                        }
                    },
                    "required": []
                }
            ),
            ToolSchema(
                name="checkRelated",
                description=(
                    "WORKFLOW STEP: Call BEFORE modifying a file to preview impact. "
                    "Shows all tracked artifacts that depend on this file (have it in `related_to`). "
                    "If artifacts are found, plan to review them after your changes. "
                    "NEXT: After modifying the file, call `notifyChange` to flag dependent artifacts. "
                    "Use `verbose=false` for concise output without workflow reminders."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to check for related artifacts"
                        },
                        "verbose": {
                            "type": "boolean",
                            "description": "Include workflow reminders in output (default: true). Set to false for concise output."
                        }
                    },
                    "required": ["path"]
                }
            ),
            ToolSchema(
                name="removeArtifact",
                description=(
                    "Stop tracking artifact(s). Use when deleting artifact files "
                    "or when they no longer need to stay in sync with other files. "
                    "Supports both single path and array of paths."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path of a single artifact to remove"
                        },
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of artifact paths to remove (batch operation)"
                        }
                    },
                    "required": []
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
                            "description": (
                                "Brief description of what changed (e.g., 'Added new endpoint', 'Renamed function'). "
                                "This becomes the REVIEW COMMENT shown to reviewers when they check flagged artifacts."
                            )
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
        self._trace(f"trackArtifact: path={path}, type={type_str}")

        if not path:
            return {"error": "path is required"}
        if not description:
            return {"error": "description is required"}

        # Normalize paths to prevent duplicates (e.g., ./doc.md vs doc.md)
        path = _normalize_path(path)
        related_to = [_normalize_path(p) for p in related_to]

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
        self._trace(f"updateArtifact: path={path}")

        if not path:
            return {"error": "path is required"}

        # Normalize path for lookup
        path = _normalize_path(path)

        if not self._registry:
            return {"error": "Plugin not initialized"}

        artifact = self._registry.get_by_path(path)
        if not artifact:
            return {"error": f"Artifact not found: {path}"}

        # Apply updates
        if "description" in args and args["description"]:
            artifact.description = args["description"]

        # Normalize relation paths
        for rel_path in args.get("add_related", []):
            artifact.add_relation(_normalize_path(rel_path))

        for rel_path in args.get("remove_related", []):
            artifact.remove_relation(_normalize_path(rel_path))

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
        type_filter = args.get("artifact_type")
        needs_review = args.get("needs_review", False)
        self._trace(f"listArtifacts: type_filter={type_filter}, needs_review={needs_review}")
        if not self._registry:
            return {"error": "Plugin not initialized"}

        # Get all artifacts
        artifacts = self._registry.get_all()

        # Apply filters
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
        self._trace(f"flagForReview: path={path}")

        if not path:
            return {"error": "path is required"}
        if not reason:
            return {"error": "reason is required"}

        # Normalize path for lookup
        path = _normalize_path(path)

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
        """Execute the acknowledgeReview tool.

        Supports both single path and array of paths.
        """
        single_path = args.get("path", "")
        paths_array = args.get("paths", [])
        notes = args.get("notes")
        was_updated = args.get("was_updated", False)
        self._trace(f"acknowledgeReview: path={single_path}, paths_count={len(paths_array)}")

        # Build list of paths to acknowledge
        paths_to_ack = []
        if single_path:
            paths_to_ack.append(single_path)
        if paths_array:
            paths_to_ack.extend(paths_array)

        if not paths_to_ack:
            return {"error": "Either 'path' or 'paths' is required"}

        # Normalize all paths
        paths_to_ack = [_normalize_path(p) for p in paths_to_ack]

        if not self._registry:
            return {"error": "Plugin not initialized"}

        # Acknowledge each artifact
        acknowledged = []
        not_found = []
        for path in paths_to_ack:
            artifact = self._registry.get_by_path(path)
            if artifact:
                if was_updated:
                    artifact.mark_updated()
                else:
                    artifact.acknowledge_review(notes)
                acknowledged.append({
                    "path": artifact.path,
                    "review_status": artifact.review_status.value,
                })
            else:
                not_found.append(path)

        if acknowledged:
            self._save_state()

        # Build response
        if len(paths_to_ack) == 1:
            # Single path - simple response
            if acknowledged:
                return {
                    "success": True,
                    "path": acknowledged[0]["path"],
                    "review_status": acknowledged[0]["review_status"],
                    "notes": notes,
                    "message": f"Review acknowledged: {acknowledged[0]['path']}",
                }
            else:
                return {"error": f"Artifact not found: {not_found[0]}"}
        else:
            # Multiple paths - detailed response
            return {
                "success": len(not_found) == 0,
                "acknowledged": acknowledged,
                "acknowledged_count": len(acknowledged),
                "not_found": not_found,
                "message": f"Acknowledged {len(acknowledged)} artifact(s)" + (
                    f", {len(not_found)} not found" if not_found else ""
                ),
            }

    def _execute_check_related(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the checkRelated tool."""
        path = args.get("path", "")
        verbose = args.get("verbose", True)  # Default to verbose for discoverability
        self._trace(f"checkRelated: path={path}")

        if not path:
            return {"error": "path is required"}

        # Normalize path for lookup
        path = _normalize_path(path)

        if not self._registry:
            return {"error": "Plugin not initialized"}

        # Find affected artifacts
        affected = self._registry.find_affected_by_change(path)

        if not affected:
            result = {
                "path": path,
                "related_count": 0,
                "related": [],
                "message": f"No tracked artifacts depend on: {path}",
            }
            if verbose:
                result["next_step"] = "You can proceed with your changes. No artifacts will need review."
            return result

        results = []
        for artifact in affected:
            results.append({
                "path": artifact.path,
                "type": artifact.artifact_type.value,
                "description": artifact.description,
                "review_status": artifact.review_status.value,
                "is_source": artifact.path == path,
            })

        result = {
            "path": path,
            "related_count": len(results),
            "related": results,
            "message": f"⚠️  {len(results)} artifact(s) depend on this file and will need review if you modify it.",
        }

        # Add workflow guidance in verbose mode
        if verbose:
            result["workflow_reminder"] = (
                "IMPORTANT: After you finish editing this file, you MUST call:\n"
                f"  notifyChange(path=\"{path}\", reason=\"<describe your changes>\")\n"
                "This will automatically flag the dependent artifacts for review."
            )
            result["next_steps"] = [
                f"1. Make your changes to {path}",
                f"2. Call notifyChange(\"{path}\", \"<reason>\") to flag dependents",
                "3. Review each flagged artifact",
                "4. Call acknowledgeReview() for each after reviewing",
            ]

        return result

    def _execute_remove_artifact(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the removeArtifact tool.

        Supports both single path and array of paths.
        """
        single_path = args.get("path", "")
        paths_array = args.get("paths", [])
        self._trace(f"removeArtifact: path={single_path}, paths_count={len(paths_array)}")

        # Build list of paths to remove
        paths_to_remove = []
        if single_path:
            paths_to_remove.append(single_path)
        if paths_array:
            paths_to_remove.extend(paths_array)

        if not paths_to_remove:
            return {"error": "Either 'path' or 'paths' is required"}

        # Normalize all paths
        paths_to_remove = [_normalize_path(p) for p in paths_to_remove]

        if not self._registry:
            return {"error": "Plugin not initialized"}

        # Remove each artifact
        removed = []
        not_found = []
        for path in paths_to_remove:
            artifact = self._registry.get_by_path(path)
            if artifact:
                self._registry.remove(artifact.artifact_id)
                removed.append(path)
            else:
                not_found.append(path)

        if removed:
            self._save_state()

        # Build response
        if len(paths_to_remove) == 1:
            # Single path - simple response
            if removed:
                return {
                    "success": True,
                    "path": removed[0],
                    "message": f"Stopped tracking: {removed[0]}",
                }
            else:
                return {"error": f"Artifact not found: {not_found[0]}"}
        else:
            # Multiple paths - detailed response
            return {
                "success": len(not_found) == 0,
                "removed": removed,
                "removed_count": len(removed),
                "not_found": not_found,
                "message": f"Removed {len(removed)} artifact(s)" + (
                    f", {len(not_found)} not found" if not_found else ""
                ),
            }

    def _execute_notify_change(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the notifyChange tool.

        Automatically flags all artifacts that depend on the changed file.
        The `reason` parameter becomes the review comment shown to reviewers.
        """
        path = args.get("path", "")
        reason = args.get("reason", "")
        self._trace(f"notifyChange: path={path}")

        if not path:
            return {"error": "path is required"}
        if not reason:
            return {"error": "reason is required"}

        # Normalize path for matching
        path = _normalize_path(path)

        if not self._registry:
            return {"error": "Plugin not initialized"}

        # Find all artifacts that have this path in their related_to list
        flagged = []
        for artifact in self._registry.get_all():
            # Check if normalized path matches any of the artifact's relations
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

    # ==================== Plugin Registry Integration ====================

    def set_plugin_registry(self, registry) -> None:
        """Set the plugin registry for cross-plugin access.

        This enables the artifact tracker to discover file dependencies
        by calling the LSP plugin's get_file_dependents() method.

        Args:
            registry: The PluginRegistry instance.
        """
        self._plugin_registry = registry
        self._trace(f"set_plugin_registry: registry set")

    # ==================== Tool Result Enrichment ====================

    def subscribes_to_tool_result_enrichment(self) -> bool:
        """Subscribe to tool result enrichment for automatic dependency discovery.

        When a file is modified, this plugin will use the LSP plugin to discover
        which other files depend on the modified file, and automatically flag
        them for review.
        """
        return True

    def get_tool_result_enrichment_priority(self) -> int:
        """Run AFTER LSP plugin (priority 30) to use its results.

        Priority 50 ensures LSP diagnostics are added first, then we add
        dependency tracking.
        """
        return 50

    def enrich_tool_result(
        self,
        tool_name: str,
        result: str
    ) -> ToolResultEnrichmentResult:
        """Auto-discover dependents when files are modified.

        When a file-writing tool completes, this method:
        1. Extracts the modified file path from the result
        2. Uses LSP to find files that depend on the modified file
        3. Flags those dependent files for review
        4. Appends a summary to the result for user visibility

        Args:
            tool_name: Name of the tool that produced the result.
            result: The tool's output as a string (JSON-serialized dict).

        Returns:
            ToolResultEnrichmentResult with dependency info appended if applicable.
        """
        # Only process file-writing tools
        if tool_name not in FILE_WRITING_TOOLS:
            return ToolResultEnrichmentResult(result=result)

        self._trace(f"enrich_tool_result: processing {tool_name}")

        # Need both registries to work
        if not self._plugin_registry:
            self._trace("enrich_tool_result: skipped - _plugin_registry not set (call set_plugin_registry())")
            return ToolResultEnrichmentResult(result=result)

        if not self._registry:
            self._trace("enrich_tool_result: skipped - _registry not set (plugin not initialized?)")
            return ToolResultEnrichmentResult(result=result)

        # Get LSP plugin for dependency discovery
        lsp_plugin = self._plugin_registry.get_plugin("lsp")
        if not lsp_plugin:
            self._trace("enrich_tool_result: skipped - LSP plugin not found in registry")
            return ToolResultEnrichmentResult(result=result)

        if not hasattr(lsp_plugin, 'get_file_dependents'):
            self._trace("enrich_tool_result: skipped - LSP plugin missing get_file_dependents method")
            return ToolResultEnrichmentResult(result=result)

        # Extract file paths from result
        file_paths = self._extract_file_paths_from_result(tool_name, result)
        if not file_paths:
            self._trace("enrich_tool_result: no file paths found")
            return ToolResultEnrichmentResult(result=result)

        self._trace(f"enrich_tool_result: analyzing dependencies for {file_paths}")

        # Collect all dependents across all modified files
        all_dependents: List[str] = []

        for file_path in file_paths:
            dependents = lsp_plugin.get_file_dependents(file_path)
            self._trace(f"enrich_tool_result: {file_path} has {len(dependents)} dependents")

            for dep_path in dependents:
                if dep_path not in all_dependents:
                    all_dependents.append(dep_path)

                    # Flag for review - the source file changed
                    self._flag_dependent_for_review(dep_path, file_path)

        if not all_dependents:
            self._trace("enrich_tool_result: no dependents found")
            return ToolResultEnrichmentResult(result=result)

        # Save state after flagging
        self._save_state()

        # Build enriched result with dependency summary
        enriched_result = self._append_dependency_summary(result, all_dependents)

        # Build notification for user visibility
        notification_message = self._format_dependents_message(all_dependents)

        metadata = {
            "dependents_flagged": all_dependents,
            "notification": {
                "message": notification_message
            }
        }

        self._trace(f"enrich_tool_result: flagged {len(all_dependents)} dependents")
        return ToolResultEnrichmentResult(result=enriched_result, metadata=metadata)

    def _extract_file_paths_from_result(
        self,
        tool_name: str,
        result: str
    ) -> List[str]:
        """Extract file paths from a tool result.

        Args:
            tool_name: The tool that produced the result.
            result: The tool's output as a JSON string.

        Returns:
            List of file paths that were modified.
        """
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return []

        file_paths = []

        if isinstance(data, dict):
            # Handle updateFile/writeNewFile: {"path": "...", "success": true}
            if 'path' in data:
                file_paths.append(data['path'])

            # Handle lsp_rename_symbol/lsp_apply_code_action: {"files_modified": [...]}
            if 'files_modified' in data:
                file_paths.extend(data['files_modified'])

            # Also check changes array
            changes = data.get("changes", [])
            for change in changes:
                if isinstance(change, dict) and change.get("file"):
                    file_paths.append(change["file"])

        return [_normalize_path(p) for p in file_paths if p]

    def _flag_dependent_for_review(self, dep_path: str, source_path: str) -> None:
        """Flag a dependent file for review.

        If the file is already tracked as an artifact, flag it for review.
        If not tracked, auto-register it as an artifact.

        Args:
            dep_path: Path to the dependent file.
            source_path: Path to the source file that changed.
        """
        dep_path = _normalize_path(dep_path)
        source_path = _normalize_path(source_path)

        # Check if already tracked
        artifact = self._registry.get_by_path(dep_path)

        if artifact:
            # Already tracked - add relationship if not present and flag for review
            if source_path not in artifact.related_to:
                artifact.related_to.append(source_path)
            artifact.mark_for_review(f"Dependency {os.path.basename(source_path)} was modified")
        else:
            # Auto-track as a new artifact
            artifact = ArtifactRecord(
                path=dep_path,
                artifact_type=ArtifactType.CODE,
                description=f"Auto-tracked: depends on {os.path.basename(source_path)}",
                related_to=[source_path],
                review_status=ReviewStatus.NEEDS_REVIEW,
            )
            artifact.mark_for_review(f"Dependency {os.path.basename(source_path)} was modified")
            self._registry.add(artifact)

    def _append_dependency_summary(self, result: str, dependents: List[str]) -> str:
        """Append a dependency summary to the tool result.

        Args:
            result: The original tool result (JSON string).
            dependents: List of dependent file paths.

        Returns:
            Result with dependency summary appended.
        """
        # Get basenames for readability
        names = [os.path.basename(p) for p in dependents]

        if len(names) <= 5:
            files_str = ", ".join(names)
        else:
            files_str = ", ".join(names[:5]) + f" +{len(names) - 5} more"

        summary = f"\n\n[Artifact Tracker: Flagged {len(dependents)} dependent file(s) for review: {files_str}]"

        return result + summary

    def _format_dependents_message(self, dependents: List[str]) -> str:
        """Format dependents list for notification display.

        Args:
            dependents: List of dependent file paths.

        Returns:
            Human-readable message for the notification.
        """
        if not dependents:
            return "no dependents found"

        # Use basenames for brevity
        names = [os.path.basename(p) for p in dependents]

        if len(names) <= 4:
            return f"flagged for review: {', '.join(names)}"
        else:
            shown = ', '.join(names[:4])
            return f"flagged for review: {shown} +{len(names) - 4} more"


def create_plugin() -> ArtifactTrackerPlugin:
    """Factory function to create the Artifact Tracker plugin instance."""
    return ArtifactTrackerPlugin()
