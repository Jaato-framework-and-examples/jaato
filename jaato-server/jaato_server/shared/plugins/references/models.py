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


#: This reference was copied into a bundle it did not start in, by
#: ``references bundle merge``.  A string rather than a bool, and the same key
#: ``generated_by`` uses (``jaato_sdk.events.ai_generated_by`` mints
#: ``{"kind": "ai", ...}``), so a reader branches on ``kind`` identically
#: wherever provenance appears and the vocabulary can grow without the field
#: changing shape.
ORIGIN_IMPORTED = "imported"

#: This reference was PROPOSED by an agent through ``proposeReference``.  The
#: plugin stamps it from the proposing session -- the model binding, the
#: authenticated creator -- and never from the tool's arguments.
ORIGIN_AGENT = "agent"


@dataclass
class ReferenceOrigin:
    """Where a reference came from, as the framework OBSERVED it arrive.

    Two arrivals are events the framework is present for, and each ``kind``
    carries only the fields its stamper can observe:

    * :data:`ORIGIN_IMPORTED` -- copied in from another bundle by
      ``merge_bundle``: ``bundle``, ``source_id``, ``at``.
    * :data:`ORIGIN_AGENT` -- proposed by an agent through
      ``proposeReference``: ``generated_by`` (the model binding, the
      ``ai_generated_by`` shape a memory carries), ``created_by`` (the
      session's authenticated creator, the ``get_client_user`` chain, never
      an environment value), ``claim_id``, ``at``.

    A reference authored by a human or emitted by ``gen-references`` is not
    an event the framework witnesses, so it carries no origin at all; a
    field naming an author with no stamper would be an inert mechanism,
    which is worse than an absent one.

    Absent means **origin unobserved**, never "authored here".  A reference
    that predates this field, one hand-written into the catalog and one
    installed by ``bundle unpack`` (which copies whole directories rather
    than rewriting each reference) all carry ``None``, and none of the three
    may be read as a claim.

    Why per-REFERENCE rather than per-bundle, which is where the federation
    unit otherwise lives: ``merge_bundle`` copies source references *into the
    target bundle's own directory*, so after a merge the source bundle
    boundary is gone.  Bundle-level provenance would be erased by exactly the
    operation that creates foreign references, and is the one shape that
    cannot survive it.

    Attributes:
        kind: What was observed: :data:`ORIGIN_IMPORTED` or
            :data:`ORIGIN_AGENT`.  An unknown string round-trips untouched.
        bundle: The name of the bundle it was copied FROM.
        source_id: The id it carried in that bundle.  ``bundle merge
            --prefix`` renames on collision, so the local id is not
            necessarily the one the other workspace knows it by, and an
            operator reconciling two catalogs needs the one that is.
        at: ISO-8601 UTC instant of the copy, or of the proposal.
        generated_by: ``{"kind": "ai", provider, model, session_id,
            agent_id}`` of the proposing session (agent kind only).
        created_by: The proposing session's authenticated creator
            (``app:user`` on WS, the OS account on IPC).  ``None`` when the
            transport authenticated nobody -- absence, not a guess.
        claim_id: The claim this reference was proposed as, so a promoted
            entry can be traced back to the claim file a curator read.
        curated_by: Who PROMOTED the claim into the catalog, stamped by the
            daemon from the connection that asked (``{"kind": "human",
            "via": "reference.promote", "user": ...}``, the shape a memory's
            ``curated_by`` has).  Only a promoted agent reference carries it;
            a claim still in the claims directory never does, and one that
            says so was written by something other than the promotion verb.
    """

    kind: str
    bundle: Optional[str] = None
    source_id: Optional[str] = None
    at: Optional[str] = None
    generated_by: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = None
    claim_id: Optional[str] = None
    curated_by: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize, omitting keys whose value was never established.

        Absent rather than ``null``, the rule ``ai_generated_by`` follows:
        a key that is not there is "not measured", and one rendered ``null``
        invites a reader to treat the absence as a measured negative.
        """
        payload: Dict[str, Any] = {"kind": self.kind}
        for key, value in (("bundle", self.bundle),
                           ("source_id", self.source_id),
                           ("at", self.at),
                           ("generated_by", self.generated_by),
                           ("created_by", self.created_by),
                           ("claim_id", self.claim_id),
                           ("curated_by", self.curated_by)):
            if value:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional['ReferenceOrigin']:
        """Create from a dict, or ``None`` when there is nothing to read.

        A payload with no ``kind`` answers ``None`` rather than inventing
        one: this field exists to say what was observed, so a record that
        does not say must not be made to.
        """
        if not isinstance(data, dict):
            return None
        kind = data.get("kind")
        if not isinstance(kind, str) or not kind:
            return None
        return cls(
            kind=kind,
            bundle=data.get("bundle") or None,
            source_id=data.get("source_id") or None,
            at=data.get("at") or None,
            generated_by=_dict_or_none(data.get("generated_by")),
            created_by=data.get("created_by") or None,
            claim_id=data.get("claim_id") or None,
            curated_by=_dict_or_none(data.get("curated_by")),
        )

    def describe(self) -> str:
        """One human/model-readable clause naming the arrival."""
        if self.kind == ORIGIN_IMPORTED:
            return self._describe_imported()
        if self.kind == ORIGIN_AGENT:
            return self._describe_agent()
        return self.kind

    def _describe_imported(self) -> str:
        where = f" from bundle '{self.bundle}'" if self.bundle else ""
        when = f" on {self.at}" if self.at else ""
        alias = (f" (known there as '{self.source_id}')"
                 if self.source_id else "")
        return f"imported{where}{when}{alias}"

    def _describe_agent(self) -> str:
        gen = self.generated_by or {}
        agent = f" agent '{gen['agent_id']}'" if gen.get("agent_id") else " an agent"
        binding = "/".join(v for v in (gen.get("provider"), gen.get("model")) if v)
        model = f" ({binding})" if binding else ""
        session = f" in session {gen['session_id']}" if gen.get("session_id") else ""
        user = f" for {self.created_by}" if self.created_by else ""
        when = f" on {self.at}" if self.at else ""
        curator = (self.curated_by or {}).get("user")
        promoted = (f", promoted by {curator}" if curator
                    else ", promoted" if self.curated_by else "")
        return f"proposed by{agent}{model}{session}{user}{when}{promoted}"


def _dict_or_none(value: Any) -> Optional[Dict[str, Any]]:
    """A non-empty dict, else ``None`` -- a malformed stamp reads as absent."""
    return dict(value) if isinstance(value, dict) and value else None


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
class EmbeddingMetadata:
    """Embedding metadata for a reference source.

    Marks a reference as having an entry in its bundle's sidecar matrix.
    Produced by the ``gen-references`` agent when it calls the
    ``compute_embedding`` tool during indexing.

    The reference's row position in the sidecar matrix is *not* stored here;
    it is derived from the bundle's ``embedding_config.json`` ``rows`` list
    (see ``ReferencesConfig.embedding_rows``). Only the metadata fingerprint
    stays on the reference itself, so a drop-in reference never has to agree
    with the other references on numbering.

    Attributes:
        source_hash: SHA-256 hash of the metadata that was embedded
            (name + description + tags + fetchHint). Used by the reconcile
            pass to detect "this reference's metadata drifted since its
            vector was computed → re-embed it".
    """
    source_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict."""
        return {
            "source_hash": self.source_hash,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional['EmbeddingMetadata']:
        """Create from a dict, or return None if data is absent/invalid.

        Args:
            data: Raw dict from JSON with a ``source_hash`` key, or None.
        """
        if not data or not isinstance(data, dict):
            return None
        source_hash = data.get("source_hash")
        if source_hash is None:
            return None
        return cls(source_hash=str(source_hash))


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

    # Embedding metadata linking this source to the sidecar matrix.
    # None when the source has not been embedded (gen-references not run,
    # or source was added after the last indexing pass).
    embedding: Optional[EmbeddingMetadata] = None

    # Runtime-only: which bundle this source was discovered from.
    # Empty string denotes the root bundle (``.jaato/references/``); a
    # non-empty value is the name of a sub-bundle subdirectory. Not
    # serialized — membership is re-established every load from the
    # directory the JSON file lives in. See ``bundle.Bundle``.
    bundle_name: str = ""

    # Where this reference came from, when the framework observed it arrive
    # (a ``bundle merge`` copy).  ``None`` = origin unobserved, which covers
    # a hand-authored reference and one predating the field alike -- see
    # ``ReferenceOrigin``.
    origin: Optional[ReferenceOrigin] = None

    def _origin_lines(self) -> List[str]:
        """The ``**Origin**`` line, or nothing when arrival was unobserved.

        Named where the MODEL reads the reference, not only in an operator
        listing: content copied in from another workspace is third-party, and
        a model weighing it should know that at the point of use.  Stated,
        never enforced -- this annotates, it does not fence (the wikiLLM
        brainstorm, §8, is where the fence is argued for).

        A method rather than a branch inside :meth:`to_instruction` because
        that function sits on the complexity ratchet.
        """
        if self.origin is None:
            return []
        return [f"**Origin**: {self.origin.describe()}"]

    def to_instruction(self) -> str:
        """Generate instruction text for the model describing how to access this reference."""
        if self.type == SourceType.INLINE:
            return f"### {self.name}\n\n{self.content}"

        parts = [f"### {self.name}"]
        parts.append(f"*{self.description}*")
        parts.append("")

        if self.tags:
            parts.append(f"**Tags**: {', '.join(self.tags)}")

        parts.extend(self._origin_lines())

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

        if self.embedding is not None:
            result["embedding"] = self.embedding.to_dict()

        if self.origin is not None:
            result["origin"] = self.origin.to_dict()

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
            embedding=EmbeddingMetadata.from_dict(data.get("embedding")),
            origin=ReferenceOrigin.from_dict(data.get("origin")),
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
