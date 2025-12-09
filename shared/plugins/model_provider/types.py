"""Provider-agnostic types for model interactions.

This module defines internal types that abstract away provider-specific
SDK types (e.g., google.genai.types.Content, google.genai.types.FunctionDeclaration).

These types are used throughout the plugin system and JaatoClient to enable
support for multiple AI providers (Google GenAI, Anthropic, etc.).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class Role(str, Enum):
    """Message role in a conversation."""
    USER = "user"
    MODEL = "model"
    TOOL = "tool"


@dataclass
class ToolSchema:
    """Provider-agnostic tool/function declaration.

    This replaces google.genai.types.FunctionDeclaration with a format
    that can be converted to any provider's tool schema.

    Attributes:
        name: Unique tool name (e.g., 'cli_based_tool').
        description: Human-readable description of what the tool does.
        parameters: JSON Schema object describing the tool's parameters.
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionCall:
    """A function/tool call requested by the model.

    Attributes:
        id: Unique identifier for this call (used for result correlation).
        name: Name of the function to call.
        args: Arguments to pass to the function.
    """
    id: str
    name: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Attachment:
    """Multimodal attachment for tool results.

    Used to include binary data (images, files, etc.) in tool responses.
    The provider converts these to the appropriate SDK-specific format.

    Attributes:
        mime_type: MIME type of the data (e.g., 'image/png', 'application/pdf').
        data: Raw binary data.
        display_name: Optional name for referencing in the response.
    """
    mime_type: str
    data: bytes
    display_name: Optional[str] = None


@dataclass
class ToolResult:
    """Result of executing a tool/function.

    Attributes:
        call_id: ID of the FunctionCall this result corresponds to.
        name: Name of the function that was called.
        result: The result data (must be JSON-serializable).
        is_error: Whether this result represents an error.
        attachments: Optional multimodal attachments (images, files, etc.).
    """
    call_id: str
    name: str
    result: Any
    is_error: bool = False
    attachments: Optional[List['Attachment']] = None


@dataclass
class Part:
    """A part of a message content.

    Messages can contain multiple parts: text, function calls, function results, etc.

    Attributes:
        text: Text content (mutually exclusive with other fields).
        function_call: A function call from the model.
        function_response: A function result being sent back.
        inline_data: Binary data with mime type (for multimodal).
    """
    text: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    function_response: Optional[ToolResult] = None
    inline_data: Optional[Dict[str, Any]] = None  # {"mime_type": str, "data": bytes}

    @classmethod
    def from_text(cls, text: str) -> 'Part':
        """Create a text part."""
        return cls(text=text)

    @classmethod
    def from_function_call(cls, call: FunctionCall) -> 'Part':
        """Create a function call part."""
        return cls(function_call=call)

    @classmethod
    def from_function_response(cls, result: ToolResult) -> 'Part':
        """Create a function response part."""
        return cls(function_response=result)


@dataclass
class Message:
    """A message in a conversation.

    This replaces google.genai.types.Content with a provider-agnostic format.

    Attributes:
        role: The role of the message sender (user, model, or tool).
        parts: List of content parts (text, function calls, etc.).
    """
    role: Role
    parts: List[Part] = field(default_factory=list)

    @classmethod
    def from_text(cls, role: Union[Role, str], text: str) -> 'Message':
        """Create a simple text message."""
        if isinstance(role, str):
            role = Role(role)
        return cls(role=role, parts=[Part.from_text(text)])

    @property
    def text(self) -> Optional[str]:
        """Extract concatenated text from all text parts."""
        texts = [p.text for p in self.parts if p.text]
        return ''.join(texts) if texts else None

    @property
    def function_calls(self) -> List[FunctionCall]:
        """Extract all function calls from this message."""
        return [p.function_call for p in self.parts if p.function_call]


@dataclass
class TokenUsage:
    """Token usage statistics from a model response.

    Attributes:
        prompt_tokens: Tokens used in the prompt/input.
        output_tokens: Tokens generated in the response.
        total_tokens: Total tokens used.
    """
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class FinishReason(str, Enum):
    """Reason why the model stopped generating."""
    STOP = "stop"              # Normal completion
    MAX_TOKENS = "max_tokens"  # Hit token limit
    TOOL_USE = "tool_use"      # Stopped to execute tools
    SAFETY = "safety"          # Safety filter triggered
    ERROR = "error"            # Error occurred
    UNKNOWN = "unknown"        # Unknown reason


@dataclass
class ProviderResponse:
    """Unified response from any AI provider.

    Wraps the provider-specific response with a common interface.

    Attributes:
        text: The text content of the response (if any).
        function_calls: List of function calls requested by the model.
        usage: Token usage statistics.
        finish_reason: Why the model stopped generating.
        raw: The original provider-specific response object.
        structured_output: Parsed JSON when response_schema was requested.
            This is populated when the model returns structured JSON output
            conforming to a requested schema.
    """
    text: Optional[str] = None
    function_calls: List[FunctionCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: FinishReason = FinishReason.UNKNOWN
    raw: Any = None
    structured_output: Optional[Dict[str, Any]] = None

    @property
    def has_function_calls(self) -> bool:
        """Check if the response contains function calls."""
        return len(self.function_calls) > 0

    @property
    def has_structured_output(self) -> bool:
        """Check if the response contains structured output."""
        return self.structured_output is not None
