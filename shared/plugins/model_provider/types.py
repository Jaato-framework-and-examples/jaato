"""Provider-agnostic types for model interactions.

This module defines internal types that abstract away provider-specific
SDK types (e.g., google.genai.types.Content, google.genai.types.FunctionDeclaration).

These types are used throughout the plugin system and JaatoClient to enable
support for multiple AI providers (Google GenAI, Anthropic, etc.).
"""

import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Union


TRAIT_FILE_WRITER = "file_writer"
"""Trait for tools that write or modify files on disk.

Tools declaring this trait participate in the file-enrichment pipeline:
the session passes their full JSON result through all enrichment plugins
(LSP diagnostics, artifact tracking, etc.) instead of treating the result
as plain text.

**Result format contract** — tools with this trait MUST include at least one
of the following keys in their result dict so enrichment plugins can discover
which files were affected:

- ``"path"`` (str): path of the single file written/modified.
- ``"files_modified"`` (list[str]): paths when multiple files are affected.
- ``"changes"`` (list[dict]): detailed per-file change records, each
  containing a ``"file"`` key with the affected path.

Usage::

    from shared.plugins.model_provider.types import ToolSchema, TRAIT_FILE_WRITER

    ToolSchema(
        name="myWriteTool",
        ...,
        traits=frozenset({TRAIT_FILE_WRITER}),
    )
"""


class Role(str, Enum):
    """Message role in a conversation."""
    USER = "user"
    MODEL = "model"
    TOOL = "tool"


# Standard tool categories for consistent classification
TOOL_CATEGORIES = [
    "filesystem",   # File reading, writing, editing, navigation
    "code",         # Code editing, refactoring, analysis
    "search",       # Searching files, content, web
    "memory",       # Persistent memory, notes, context storage
    "coordination", # Task planning, delegation, subagents, parallel execution
    "system",       # System commands, shell execution, environment
    "web",          # Web fetching, API calls, external resources
    "communication",  # User interaction, prompts, questions
    "MCP",          # Tools from external MCP (Model Context Protocol) servers
]


# Standard discoverability modes for tool loading behavior
TOOL_DISCOVERABILITY = [
    "core",          # Always loaded in initial context
    "discoverable",  # Loaded on-demand via introspection tools
]


@dataclass
class EditableContent:
    """Declares which tool parameters are user-editable at permission time.

    Tools that manage "content" (plans, code, configs) can opt-in to being
    user-editable by setting this on their ToolSchema. When permission is
    requested, the user gets an additional "Edit" option that opens the
    content in their $EDITOR.

    Attributes:
        parameters: List of parameter names that are editable (e.g., ["title", "steps"]).
        format: How to present content for editing. Options:
            - "yaml": YAML format (default, most user-friendly for structured data)
            - "json": JSON format
            - "text": Plain text
            - "markdown": Markdown format
        template: Optional header/instructions to show in the editor.
            This text is stripped when parsing the edited content back.
    """
    parameters: List[str]
    format: str = "yaml"
    template: Optional[str] = None


@dataclass
class ToolSchema:
    """Provider-agnostic tool/function declaration.

    This replaces google.genai.types.FunctionDeclaration with a format
    that can be converted to any provider's tool schema.

    Attributes:
        name: Unique tool name (e.g., 'cli_based_tool').
        description: Human-readable description of what the tool does.
        parameters: JSON Schema object describing the tool's parameters.
        category: Optional category for tool organization and filtering.
            Standard categories: filesystem, code, search, memory, planning,
            system, web, communication. Custom categories are also allowed.
        discoverability: Controls when the tool schema is loaded into context.
            - "core": Always present in initial context (default for essential tools)
            - "discoverable": Loaded on-demand when model requests via introspection
            Default is "discoverable" to minimize initial context size.
        editable: Optional EditableContent declaring which parameters are
            user-editable at permission time. When set, the permission prompt
            includes an "Edit" option that opens an external editor.
        traits: Semantic capability tags that drive cross-cutting behavior
            (e.g., enrichment routing).  Use the ``TRAIT_*`` constants defined
            in this module.  See :data:`TRAIT_FILE_WRITER` for an example.
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None
    discoverability: str = "discoverable"
    editable: Optional[EditableContent] = None
    traits: FrozenSet[str] = field(default_factory=frozenset)


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
        thought: Model's internal reasoning/thinking (Gemini 2.0+ thinking mode).
        executable_code: Code generated by the model for execution.
        code_execution_result: Result from code execution.
    """
    text: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    function_response: Optional[ToolResult] = None
    inline_data: Optional[Dict[str, Any]] = None  # {"mime_type": str, "data": bytes}
    thought: Optional[str] = None  # Model's internal reasoning
    executable_code: Optional[str] = None  # Code for execution
    code_execution_result: Optional[str] = None  # Code execution output

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

    @classmethod
    def from_thought(cls, thought: str) -> 'Part':
        """Create a thought/reasoning part."""
        return cls(thought=thought)


def _generate_message_id() -> str:
    """Generate a unique message ID."""
    return str(uuid.uuid4())


@dataclass
class Message:
    """A message in a conversation.

    This replaces google.genai.types.Content with a provider-agnostic format.

    Attributes:
        role: The role of the message sender (user, model, or tool).
        parts: List of content parts (text, function calls, etc.).
        message_id: Unique identifier for this message (for GC history-budget sync).
    """
    role: Role
    parts: List[Part] = field(default_factory=list)
    message_id: str = field(default_factory=_generate_message_id)

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
        cache_read_tokens: Tokens read from cache (reduced cost).
            Supported by: Anthropic, OpenAI, Google Gemini.
        cache_creation_tokens: Tokens written to cache (Anthropic-specific).
            Anthropic charges 1.25x for 5-min cache, 2x for 1-hour cache.
        reasoning_tokens: Tokens used for reasoning/thinking (OpenAI o-series).
            For Anthropic/Gemini, thinking tokens are included in output_tokens.
        thinking_tokens: Tokens used for extended thinking (Anthropic/Gemini).
            Subset of output_tokens spent on thinking content.
            Extracted from API when available, otherwise estimated from text.
    """
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Cache tokens (prompt caching)
    cache_read_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    # Reasoning tokens (OpenAI o-series models)
    reasoning_tokens: Optional[int] = None
    # Thinking tokens (Anthropic/Gemini extended thinking)
    thinking_tokens: Optional[int] = None


class FinishReason(str, Enum):
    """Reason why the model stopped generating."""
    STOP = "stop"              # Normal completion
    MAX_TOKENS = "max_tokens"  # Hit token limit
    TOOL_USE = "tool_use"      # Stopped to execute tools
    SAFETY = "safety"          # Safety filter triggered
    ERROR = "error"            # Error occurred
    CANCELLED = "cancelled"    # Cancelled via CancelToken
    UNKNOWN = "unknown"        # Unknown reason


@dataclass
class ProviderResponse:
    """Unified response from any AI provider.

    Wraps the provider-specific response with a common interface.

    Attributes:
        parts: Ordered list of response parts preserving the interleaving
            of text and function calls as they were produced by the model.
            Use this to process text and tool calls in their original order.
        usage: Token usage statistics.
        finish_reason: Why the model stopped generating.
        raw: The original provider-specific response object.
        structured_output: Parsed JSON when response_schema was requested.
            This is populated when the model returns structured JSON output
            conforming to a requested schema.
        thinking: Extended thinking/reasoning content from the model.
            Populated when models expose their internal reasoning, e.g.
            Anthropic extended thinking or DeepSeek-R1 reasoning_content.
            OpenAI o-series models use reasoning internally but do not
            surface it through this field.
    """
    parts: List[Part] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: FinishReason = FinishReason.UNKNOWN
    raw: Any = None
    structured_output: Optional[Dict[str, Any]] = None
    thinking: Optional[str] = None

    def get_text(self) -> str:
        """Extract concatenated text from all text parts."""
        texts = [p.text for p in self.parts if p.text]
        return ''.join(texts) if texts else ''

    def get_function_calls(self) -> List[FunctionCall]:
        """Extract all function calls from parts."""
        return [p.function_call for p in self.parts if p.function_call]

    def has_function_calls(self) -> bool:
        """Check if the response contains function calls."""
        return any(p.function_call for p in self.parts)

    def has_structured_output(self) -> bool:
        """Check if the response contains structured output."""
        return self.structured_output is not None

    @property
    def has_thinking(self) -> bool:
        """Check if the response contains extended thinking."""
        return self.thinking is not None


class CancelledException(Exception):
    """Raised when an operation is cancelled via CancelToken."""

    def __init__(self, message: str = "Operation was cancelled"):
        self.message = message
        super().__init__(self.message)


class CancelToken:
    """Thread-safe cancellation token for stopping operations.

    Used to signal cancellation requests across threads. Supports:
    - Simple cancellation via cancel()
    - Polling via is_cancelled property
    - Blocking wait via wait()
    - Callback registration for cancellation notifications

    Example:
        token = CancelToken()

        # In worker thread
        def work():
            while not token.is_cancelled:
                do_work_chunk()

        # In main thread
        token.cancel()  # Signals worker to stop

    Thread Safety:
        All methods are thread-safe and can be called from any thread.
    """

    def __init__(self):
        """Initialize a new cancel token."""
        self._cancelled = False
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        """Request cancellation.

        This is idempotent - calling cancel() multiple times has no effect
        after the first call. All registered callbacks are invoked once.
        """
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            callbacks = list(self._callbacks)

        # Set event to wake up any waiters
        self._event.set()

        # Invoke callbacks outside lock to avoid deadlock
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass  # Swallow callback errors

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested.

        Returns:
            True if cancel() has been called, False otherwise.
        """
        return self._cancelled

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for cancellation or timeout.

        Blocks until cancel() is called or timeout expires.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            True if cancelled, False if timeout expired.
        """
        return self._event.wait(timeout=timeout)

    def raise_if_cancelled(self) -> None:
        """Raise CancelledException if cancelled.

        Convenience method for checking cancellation at safe points.

        Raises:
            CancelledException: If cancel() has been called.
        """
        if self._cancelled:
            raise CancelledException()

    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Register a callback to be invoked when cancelled.

        If already cancelled, callback is invoked immediately.

        Args:
            callback: Function to call when cancellation is requested.
        """
        with self._lock:
            if self._cancelled:
                # Already cancelled, invoke immediately
                try:
                    callback()
                except Exception:
                    pass
                return
            self._callbacks.append(callback)

    def reset(self) -> None:
        """Reset the token for reuse.

        Warning: This is not safe if the token is still being used
        by other threads. Only call this when you're certain no
        other code is using this token.
        """
        with self._lock:
            self._cancelled = False
            self._event.clear()
            self._callbacks.clear()


@dataclass
class ThinkingConfig:
    """Configuration for extended thinking/reasoning modes.

    This is a provider-agnostic configuration for thinking capabilities:
    - Anthropic: Extended thinking with budget_tokens
    - Google Gemini: Thinking mode (Gemini 2.0+)
    - GitHub Models: Reasoning content extraction (DeepSeek-R1, etc.)

    Attributes:
        enabled: Whether thinking mode is enabled.
        budget: Token budget for extended thinking.
            Interpretation is provider-specific:
            - Anthropic: Max tokens for thinking (default 10000)
            - Gemini: May be used for thinking budget
    """
    enabled: bool = False
    budget: int = 10000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"enabled": self.enabled, "budget": self.budget}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThinkingConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            budget=data.get("budget", 10000)
        )


@dataclass
class PresentationContext:
    """Display capabilities and constraints of the connected client.

    Assembled by each client at connection time and transmitted to the server
    via ``ClientConfigRequest.presentation``.  The server stores it on the
    ``JaatoSession`` so that:

    1. **System instructions** include a compact display-context block, letting
       the model adapt its output format (e.g. vertical key-value lists instead
       of wide tables on a narrow mobile screen).
    2. **Formatter pipelines** can use the context for any client-side
       reformatting they choose to implement.

    Clients that support expandable/collapsible UI (Telegram inline keyboards,
    HTML ``<details>`` blocks, TUI scrollable panels) should set
    ``supports_expandable_content = True``.  When enabled, the model is *not*
    asked to avoid wide content — instead the client is expected to collapse
    overflow and offer a "show more" affordance.

    Attributes:
        content_width: Available content width in characters.
        content_height: Available content height in lines (None = unlimited scroll).
        supports_markdown: Whether the client renders markdown.
        supports_tables: Whether markdown pipe-tables render acceptably.
        supports_code_blocks: Whether fenced code blocks are rendered.
        supports_images: Whether inline images can be displayed.
        supports_rich_text: Whether bold/italic/links are rendered.
        supports_unicode: Whether wide characters and emoji are safe.
        supports_mermaid: Whether Mermaid diagrams are rendered.
        supports_expandable_content: Whether the client can collapse overflow
            behind an expand/click affordance (e.g. Telegram inline buttons,
            HTML details, TUI panels).
        client_type: Hint identifying the client kind.
    """

    # ── Dimensions ──────────────────────────────────────────────
    content_width: int = 80
    content_height: Optional[int] = None

    # ── Format capabilities ─────────────────────────────────────
    supports_markdown: bool = True
    supports_tables: bool = True
    supports_code_blocks: bool = True
    supports_images: bool = False
    supports_rich_text: bool = True
    supports_unicode: bool = True
    supports_mermaid: bool = False
    supports_expandable_content: bool = False

    # ── Client hint ─────────────────────────────────────────────
    client_type: str = "terminal"

    # ──────────────────────────────────────────────────────────

    def to_system_instruction(self) -> str:
        """Generate a compact display-context block for system instructions.

        The returned string is appended to the model's system prompt so it
        can make intelligent formatting decisions.  The instruction is kept
        deliberately short (30-80 tokens) to minimise overhead.

        Returns:
            A multi-line instruction string describing display constraints.
        """
        lines: List[str] = [
            "## Display Context",
            f"Output width: {self.content_width} characters.",
        ]

        # ── Width-dependent guidance ────────────────────────────
        if self.supports_expandable_content:
            # Client handles overflow via expand/collapse UI — no need
            # to restrict the model's formatting choices.
            lines.append(
                "The client can collapse wide or long content behind an "
                "expandable control. You may use full-width tables and "
                "detailed output freely."
            )
        elif self.content_width < 60:
            lines.append(
                "This is a NARROW display. Avoid markdown tables — "
                "use vertical key: value lists instead. "
                f"Keep lines under {self.content_width} characters."
            )
        elif self.content_width < 100:
            lines.append(
                "Prefer compact tables (3-4 columns max). "
                "For wider data, use vertical key: value format."
            )

        # ── Capability restrictions ─────────────────────────────
        if not self.supports_tables:
            lines.append(
                "Markdown tables are NOT supported. "
                "Use bullet lists or indented key: value pairs."
            )

        if not self.supports_code_blocks:
            lines.append(
                "Fenced code blocks are NOT supported. "
                "Indent code with 4 spaces instead."
            )

        if not self.supports_markdown:
            lines.append("Markdown is NOT supported. Use plain text only.")

        if self.supports_images:
            lines.append("Inline images are supported.")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for event transport."""
        return {
            "content_width": self.content_width,
            "content_height": self.content_height,
            "supports_markdown": self.supports_markdown,
            "supports_tables": self.supports_tables,
            "supports_code_blocks": self.supports_code_blocks,
            "supports_images": self.supports_images,
            "supports_rich_text": self.supports_rich_text,
            "supports_unicode": self.supports_unicode,
            "supports_mermaid": self.supports_mermaid,
            "supports_expandable_content": self.supports_expandable_content,
            "client_type": self.client_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PresentationContext':
        """Create from a dict (e.g. deserialized from ClientConfigRequest)."""
        return cls(
            content_width=data.get("content_width", 80),
            content_height=data.get("content_height"),
            supports_markdown=data.get("supports_markdown", True),
            supports_tables=data.get("supports_tables", True),
            supports_code_blocks=data.get("supports_code_blocks", True),
            supports_images=data.get("supports_images", False),
            supports_rich_text=data.get("supports_rich_text", True),
            supports_unicode=data.get("supports_unicode", True),
            supports_mermaid=data.get("supports_mermaid", False),
            supports_expandable_content=data.get("supports_expandable_content", False),
            client_type=data.get("client_type", "terminal"),
        )
