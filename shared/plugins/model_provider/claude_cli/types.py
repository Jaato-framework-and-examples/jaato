"""Type definitions for Claude CLI stream-json protocol.

These types mirror the Claude Code CLI's NDJSON message format as documented
in the Claude Agent SDK specification.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
import json


def _deserialize_tool_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize tool input values that CLI has stringified.

    The Claude CLI stringifies all non-string values in tool_use blocks:
    - Arrays become '["a", "b"]' instead of ["a", "b"]
    - Objects become '{"key": "value"}' instead of {"key": "value"}
    - Numbers become '50' instead of 50
    - Booleans become 'true' instead of True

    This function parses them back to their proper types.
    """
    if not input_data:
        return input_data

    result = {}
    for key, value in input_data.items():
        if isinstance(value, str):
            # Try to parse JSON arrays and objects
            stripped = value.strip()
            if (stripped.startswith('[') and stripped.endswith(']')) or \
               (stripped.startswith('{') and stripped.endswith('}')):
                try:
                    result[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass  # Keep as string if parsing fails

            # Try to parse booleans
            if stripped.lower() == 'true':
                result[key] = True
                continue
            elif stripped.lower() == 'false':
                result[key] = False
                continue

            # Try to parse numbers (int first, then float)
            try:
                result[key] = int(value)
                continue
            except ValueError:
                pass
            try:
                result[key] = float(value)
                continue
            except ValueError:
                pass

            # Keep as string
            result[key] = value
        else:
            # Non-string values pass through unchanged
            result[key] = value

    return result


class CLIMode(str, Enum):
    """Operating mode for the Claude CLI provider."""

    DELEGATED = "delegated"
    """CLI handles tool execution. jaato sends messages, CLI runs tools."""

    PASSTHROUGH = "passthrough"
    """jaato handles tool execution. CLI returns tool_use blocks for jaato to execute."""


class MessageType(str, Enum):
    """Types of messages in the stream-json protocol."""

    SYSTEM = "system"
    """Session initialization and configuration."""

    ASSISTANT = "assistant"
    """Response from Claude."""

    USER = "user"
    """User input or tool results."""

    RESULT = "result"
    """Final result of a query."""

    STREAM_EVENT = "stream_event"
    """Streaming event (with --include-partial-messages)."""


class ContentBlockType(str, Enum):
    """Types of content blocks within messages."""

    TEXT = "text"
    """Plain text content."""

    TOOL_USE = "tool_use"
    """Tool invocation request."""

    TOOL_RESULT = "tool_result"
    """Result of tool execution."""


# ==================== Content Blocks ====================


@dataclass
class TextBlock:
    """Text content block."""

    text: str

    @property
    def type(self) -> ContentBlockType:
        return ContentBlockType.TEXT

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "text", "text": self.text}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextBlock":
        return cls(text=data.get("text", ""))


@dataclass
class ToolUseBlock:
    """Tool invocation request block."""

    id: str
    name: str
    input: Dict[str, Any]

    @property
    def type(self) -> ContentBlockType:
        return ContentBlockType.TOOL_USE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolUseBlock":
        # Deserialize stringified values from CLI output
        raw_input = data.get("input", {})
        deserialized_input = _deserialize_tool_input(raw_input)
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            input=deserialized_input,
        )


@dataclass
class ToolResultBlock:
    """Tool execution result block."""

    tool_use_id: str
    content: Any
    is_error: bool = False

    @property
    def type(self) -> ContentBlockType:
        return ContentBlockType.TOOL_RESULT

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }
        if self.is_error:
            result["is_error"] = True
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResultBlock":
        return cls(
            tool_use_id=data.get("tool_use_id", ""),
            content=data.get("content", ""),
            is_error=data.get("is_error", False),
        )


ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock]


def parse_content_block(data: Dict[str, Any]) -> ContentBlock:
    """Parse a content block from its dict representation."""
    block_type = data.get("type", "")

    if block_type == "text":
        return TextBlock.from_dict(data)
    elif block_type == "tool_use":
        return ToolUseBlock.from_dict(data)
    elif block_type == "tool_result":
        return ToolResultBlock.from_dict(data)
    else:
        # Unknown block type, treat as text
        return TextBlock(text=str(data))


# ==================== Messages ====================


@dataclass
class SystemMessage:
    """Session initialization and configuration message."""

    subtype: str  # e.g., "init"
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    api_key_source: Optional[str] = None
    cwd: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    mcp_servers: List[Dict[str, str]] = field(default_factory=list)
    model: Optional[str] = None
    permission_mode: Optional[str] = None

    @property
    def type(self) -> MessageType:
        return MessageType.SYSTEM

    @property
    def content(self) -> List[ContentBlock]:
        return []

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": "system",
            "subtype": self.subtype,
            "session_id": self.session_id,
        }
        if self.api_key_source:
            result["apiKeySource"] = self.api_key_source
        if self.cwd:
            result["cwd"] = self.cwd
        if self.tools:
            result["tools"] = self.tools
        if self.mcp_servers:
            result["mcp_servers"] = self.mcp_servers
        if self.model:
            result["model"] = self.model
        if self.permission_mode:
            result["permissionMode"] = self.permission_mode
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemMessage":
        created_at_str = data.get("created_at")
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = datetime.now()
        else:
            created_at = datetime.now()

        return cls(
            subtype=data.get("subtype", ""),
            session_id=data.get("session_id", ""),
            created_at=created_at,
            api_key_source=data.get("apiKeySource"),
            cwd=data.get("cwd"),
            tools=data.get("tools", []),
            mcp_servers=data.get("mcp_servers", []),
            model=data.get("model"),
            permission_mode=data.get("permissionMode"),
        )


@dataclass
class AssistantMessage:
    """Response from Claude."""

    content_blocks: List[ContentBlock]
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    parent_tool_use_id: Optional[str] = None

    @property
    def type(self) -> MessageType:
        return MessageType.ASSISTANT

    @property
    def content(self) -> List[ContentBlock]:
        return self.content_blocks

    @property
    def text(self) -> str:
        """Extract all text content from this message."""
        texts = []
        for block in self.content_blocks:
            if isinstance(block, TextBlock):
                texts.append(block.text)
        return "".join(texts)

    @property
    def tool_uses(self) -> List[ToolUseBlock]:
        """Extract all tool use blocks from this message."""
        return [b for b in self.content_blocks if isinstance(b, ToolUseBlock)]

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": "assistant",
            "content": [b.to_dict() for b in self.content_blocks],
            "session_id": self.session_id,
        }
        if self.parent_tool_use_id:
            result["parent_tool_use_id"] = self.parent_tool_use_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssistantMessage":
        created_at_str = data.get("created_at")
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = datetime.now()
        else:
            created_at = datetime.now()

        # Handle both formats:
        # 1. Direct content: {"type":"assistant","content":[...]}
        # 2. Nested message: {"type":"assistant","message":{"content":[...]}}
        content_data = data.get("content", [])
        if not content_data and "message" in data:
            # Nested format from CLI --verbose output
            message_data = data.get("message", {})
            content_data = message_data.get("content", [])

        content_blocks = [
            parse_content_block(b) for b in content_data
        ]

        return cls(
            content_blocks=content_blocks,
            session_id=data.get("session_id", ""),
            created_at=created_at,
            parent_tool_use_id=data.get("parent_tool_use_id"),
        )


@dataclass
class UserMessage:
    """User input or tool results message."""

    content_blocks: List[ContentBlock]
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    parent_tool_use_id: Optional[str] = None

    @property
    def type(self) -> MessageType:
        return MessageType.USER

    @property
    def content(self) -> List[ContentBlock]:
        return self.content_blocks

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": "user",
            "content": [b.to_dict() for b in self.content_blocks],
            "session_id": self.session_id,
        }
        if self.parent_tool_use_id:
            result["parent_tool_use_id"] = self.parent_tool_use_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserMessage":
        created_at_str = data.get("created_at")
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = datetime.now()
        else:
            created_at = datetime.now()

        # Handle both formats:
        # 1. Direct content: {"type":"user","content":[...]}
        # 2. Nested message: {"type":"user","message":{"content":[...]}}
        content_data = data.get("content", [])
        if not content_data and "message" in data:
            # Nested format from CLI --verbose output
            message_data = data.get("message", {})
            content_data = message_data.get("content", [])

        content_blocks = [
            parse_content_block(b) for b in content_data
        ]

        return cls(
            content_blocks=content_blocks,
            session_id=data.get("session_id", ""),
            created_at=created_at,
            parent_tool_use_id=data.get("parent_tool_use_id"),
        )


@dataclass
class Usage:
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Usage":
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            cache_creation_tokens=data.get("cache_creation_tokens", 0),
        )


@dataclass
class ResultMessage:
    """Final result of a query."""

    subtype: str  # e.g., "success", "error"
    session_id: str
    duration_ms: int = 0
    duration_api_ms: int = 0
    is_error: bool = False
    num_turns: int = 0
    total_cost_usd: Optional[float] = None
    usage: Optional[Usage] = None
    result: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def type(self) -> MessageType:
        return MessageType.RESULT

    @property
    def content(self) -> List[ContentBlock]:
        if self.result:
            return [TextBlock(text=self.result)]
        return []

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": "result",
            "subtype": self.subtype,
            "session_id": self.session_id,
            "duration_ms": self.duration_ms,
            "duration_api_ms": self.duration_api_ms,
            "is_error": self.is_error,
            "num_turns": self.num_turns,
        }
        if self.total_cost_usd is not None:
            result["total_cost_usd"] = self.total_cost_usd
        if self.usage:
            result["usage"] = self.usage.to_dict()
        if self.result:
            result["result"] = self.result
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResultMessage":
        created_at_str = data.get("created_at")
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = datetime.now()
        else:
            created_at = datetime.now()

        usage_data = data.get("usage")
        usage = Usage.from_dict(usage_data) if usage_data else None

        return cls(
            subtype=data.get("subtype", ""),
            session_id=data.get("session_id", ""),
            duration_ms=data.get("duration_ms", 0),
            duration_api_ms=data.get("duration_api_ms", 0),
            is_error=data.get("is_error", False),
            num_turns=data.get("num_turns", 0),
            total_cost_usd=data.get("total_cost_usd"),
            usage=usage,
            result=data.get("result"),
            created_at=created_at,
        )


@dataclass
class StreamEvent:
    """Streaming event for incremental output (with --include-partial-messages)."""

    event_type: str  # message_start, content_block_start, content_block_delta, etc.
    session_id: str
    delta_text: Optional[str] = None  # Text delta from content_block_delta
    raw_event: Dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> MessageType:
        return MessageType.STREAM_EVENT

    @property
    def is_text_delta(self) -> bool:
        """Check if this is a text delta event with content."""
        return self.event_type == "content_block_delta" and self.delta_text is not None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamEvent":
        event = data.get("event", {})
        event_type = event.get("type", "")

        # Extract text delta if present
        delta_text = None
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                delta_text = delta.get("text", "")

        return cls(
            event_type=event_type,
            session_id=data.get("session_id", ""),
            delta_text=delta_text,
            raw_event=event,
        )


# Union type for all messages
CLIMessage = Union[SystemMessage, AssistantMessage, UserMessage, ResultMessage, StreamEvent]


def parse_message(data: Dict[str, Any]) -> CLIMessage:
    """Parse a message from its dict representation."""
    msg_type = data.get("type", "")

    if msg_type == "system":
        return SystemMessage.from_dict(data)
    elif msg_type == "assistant":
        return AssistantMessage.from_dict(data)
    elif msg_type == "user":
        return UserMessage.from_dict(data)
    elif msg_type == "result":
        return ResultMessage.from_dict(data)
    elif msg_type == "stream_event":
        return StreamEvent.from_dict(data)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")


def parse_ndjson_line(line: str) -> CLIMessage:
    """Parse a single NDJSON line into a message."""
    data = json.loads(line)
    return parse_message(data)
