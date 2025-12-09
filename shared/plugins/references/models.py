"""Data models for the references plugin.

Defines core data structures for reference sources and their metadata.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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
    path: Optional[str] = None           # For LOCAL type
    url: Optional[str] = None            # For URL type
    server: Optional[str] = None         # For MCP type
    tool: Optional[str] = None           # For MCP type
    args: Optional[Dict[str, Any]] = None  # For MCP type
    content: Optional[str] = None        # For INLINE type

    # Optional hint for the model on how to access
    fetch_hint: Optional[str] = None

    # Tags for topic-based discovery
    tags: List[str] = field(default_factory=list)

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
            url=data.get("url"),
            server=data.get("server"),
            tool=data.get("tool"),
            args=data.get("args"),
            content=data.get("content"),
            fetch_hint=data.get("fetchHint"),
            tags=data.get("tags", []),
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
