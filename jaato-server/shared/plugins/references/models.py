"""Data models for the references plugin.

Defines core data structures for reference sources and their metadata.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# Valid keys for the ``contents`` mapping on a ReferenceSource.
# Each key names a type of subfolder that a reference directory may contain.
VALID_CONTENTS_KEYS: Set[str] = {"templates", "validation", "policies", "scripts"}


@dataclass
class ReferenceContents:
    """Declares which typed subfolders exist within a reference directory.

    Each field is either a relative subfolder path (string) when the
    reference contains that type of content, or ``None`` when it does not.

    Fields:
        templates: Subfolder with authoritative template files (.tpl/.tmpl)
            that the model must use via ``renderTemplateToFile`` instead of
            extracting embedded templates from documentation.
        validation: Subfolder with mandatory post-implementation validation
            shell scripts that the model must run after completing an
            implementation that used this reference.
        policies: Subfolder with markdown documents defining implementation
            constraints the model must follow.
        scripts: Subfolder with deterministic helper scripts the model can
            invoke during implementation to avoid re-inventing common
            operations.
    """
    templates: Optional[str] = None
    validation: Optional[str] = None
    policies: Optional[str] = None
    scripts: Optional[str] = None

    def has_any(self) -> bool:
        """Return True if any content type is declared."""
        return any([self.templates, self.validation, self.policies, self.scripts])

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Serialize to a dict (always includes all keys, null for absent)."""
        return {
            "templates": self.templates,
            "validation": self.validation,
            "policies": self.policies,
            "scripts": self.scripts,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'ReferenceContents':
        """Create from a dict, tolerating missing or extra keys.

        Args:
            data: Raw dict from JSON, or None (returns all-None instance).
        """
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            templates=data.get("templates"),
            validation=data.get("validation"),
            policies=data.get("policies"),
            scripts=data.get("scripts"),
        )


class SourceType(Enum):
    """How the reference content can be accessed by the model."""
    LOCAL = "local"      # Local file - model uses CLI tool to read
    URL = "url"          # HTTP URL - model fetches directly
    MCP = "mcp"          # MCP tool - model calls the specified tool
    INLINE = "inline"    # Content embedded in config - no fetch needed


class InjectionMode(Enum):
    """When the reference should be offered to the model."""
    AUTO = "auto"            # Include in system instructions at startup
    SELECTABLE = "selectable"  # User must explicitly select via channel


@dataclass
class ReferenceSource:
    """Represents a reference source in the catalog.

    The plugin maintains metadata about available references. The model
    is responsible for fetching content using the appropriate access method.
    """

    id: str
    name: str
    description: str
    type: SourceType
    mode: InjectionMode

    # Type-specific access info (model uses these to fetch)
    path: Optional[str] = None           # For LOCAL type (original path from config)
    resolved_path: Optional[str] = None  # For LOCAL type (absolute path resolved at load time)
    url: Optional[str] = None            # For URL type
    server: Optional[str] = None         # For MCP type
    tool: Optional[str] = None           # For MCP type
    args: Optional[Dict[str, Any]] = None  # For MCP type
    content: Optional[str] = None        # For INLINE type

    # Optional hint for the model on how to access
    fetch_hint: Optional[str] = None

    # Tags for topic-based discovery
    tags: List[str] = field(default_factory=list)

    # Typed subfolders present in this reference directory.
    # Non-None values are relative paths to the subfolder (e.g., "templates/").
    # Only meaningful for LOCAL directory references.
    contents: ReferenceContents = field(default_factory=ReferenceContents)

    def to_instruction(self) -> str:
        """Generate instruction text for the model describing how to access this reference."""
        if self.type == SourceType.INLINE:
            return f"### {self.name}\n\n{self.content}"

        parts = [f"### {self.name}"]
        parts.append(f"*{self.description}*")
        parts.append("")

        if self.tags:
            parts.append(f"**Tags**: {', '.join(self.tags)}")

        if self.type == SourceType.LOCAL:
            # Use resolved path if available, otherwise original path
            effective_path = self.resolved_path if self.resolved_path else self.path
            path_obj = Path(effective_path) if effective_path else None

            # Check if path is a directory
            if path_obj and path_obj.is_dir():
                # List files in directory recursively
                files = self._list_directory_files(path_obj)
                if files:
                    parts.append(f"**Location**: Directory `{effective_path}` containing {len(files)} file(s):")
                    if self.resolved_path and self.resolved_path != self.path:
                        parts.append(f"*(configured as: `{self.path}`)*")
                    parts.append("")
                    for f in files:
                        parts.append(f"  - `{f}`")
                    parts.append("")
                    parts.append("**Access**: Read each file listed above using the CLI tool")
                else:
                    parts.append(f"**Location**: Directory `{effective_path}` (empty)")
                    parts.append("**Access**: Directory contains no readable files")
            else:
                # Regular file
                if self.resolved_path and self.resolved_path != self.path:
                    parts.append(f"**Location**: `{self.resolved_path}`")
                    parts.append(f"*(configured as: `{self.path}`)*")
                else:
                    parts.append(f"**Location**: `{self.path}`")
                parts.append("**Access**: Read this file using the CLI tool")
        elif self.type == SourceType.URL:
            parts.append(f"**URL**: {self.url}")
            parts.append("**Access**: Fetch this URL to incorporate the content")
        elif self.type == SourceType.MCP:
            parts.append(f"**Server**: {self.server}")
            parts.append(f"**Tool**: `{self.tool}`")
            if self.args:
                parts.append(f"**Args**: `{self.args}`")
            parts.append("**Access**: Call the MCP tool to retrieve this content")

        if self.fetch_hint:
            parts.append(f"**Hint**: {self.fetch_hint}")

        return "\n".join(parts)

    def _list_directory_files(self, directory: Path, max_files: int = 50) -> List[str]:
        """List files in a directory recursively.

        Args:
            directory: Path to the directory to list.
            max_files: Maximum number of files to return (to avoid overwhelming output).

        Returns:
            List of file paths relative to the directory, sorted alphabetically.
        """
        files: List[str] = []
        try:
            for item in sorted(directory.rglob("*")):
                if item.is_file():
                    # Get path relative to the directory
                    rel_path = item.relative_to(directory)
                    # Use the full path from resolved_path base for the model
                    full_rel_path = str(Path(self.resolved_path or self.path) / rel_path)
                    files.append(full_rel_path)
                    if len(files) >= max_files:
                        break
        except (PermissionError, OSError):
            pass  # Skip directories we can't read
        return files

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "mode": self.mode.value,
            "tags": self.tags,
        }

        if self.path is not None:
            result["path"] = self.path
        if self.resolved_path is not None:
            result["resolved_path"] = self.resolved_path
        if self.url is not None:
            result["url"] = self.url
        if self.server is not None:
            result["server"] = self.server
        if self.tool is not None:
            result["tool"] = self.tool
        if self.args is not None:
            result["args"] = self.args
        if self.content is not None:
            result["content"] = self.content
        if self.fetch_hint is not None:
            result["fetchHint"] = self.fetch_hint

        if self.contents.has_any():
            result["contents"] = self.contents.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReferenceSource':
        """Create from dictionary."""
        type_str = data.get("type", "local")
        try:
            source_type = SourceType(type_str)
        except ValueError:
            source_type = SourceType.LOCAL

        mode_str = data.get("mode", "selectable")
        try:
            mode = InjectionMode(mode_str)
        except ValueError:
            mode = InjectionMode.SELECTABLE

        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            type=source_type,
            mode=mode,
            path=data.get("path"),
            resolved_path=data.get("resolved_path"),
            url=data.get("url"),
            server=data.get("server"),
            tool=data.get("tool"),
            args=data.get("args"),
            content=data.get("content"),
            fetch_hint=data.get("fetchHint"),
            tags=data.get("tags", []),
            contents=ReferenceContents.from_dict(data.get("contents")),
        )


@dataclass
class SelectionRequest:
    """Request sent to an channel for reference selection."""

    request_id: str
    timestamp: str
    available_sources: List[ReferenceSource]
    context: Optional[str] = None  # Why the model needs these references

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "context": self.context,
            "sources": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "type": s.type.value,
                    "tags": s.tags,
                }
                for s in self.available_sources
            ],
        }


@dataclass
class SelectionResponse:
    """Response from an channel with selected reference IDs."""

    request_id: str
    selected_ids: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelectionResponse':
        """Create from dictionary."""
        return cls(
            request_id=data.get("request_id", ""),
            selected_ids=data.get("selected_ids", []),
        )
