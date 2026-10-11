"""Provider-agnostic types for model interactions.

Re-exports from types.py for convenient access.
"""

from .types import (
    TRAIT_FILE_WRITER,
    Attachment,
    CancelledException,
    CancelToken,
    EditableContent,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    ThinkingConfig,
    TokenUsage,
    ToolResult,
    ToolSchema,
    TOOL_CATEGORIES,
    TOOL_DISCOVERABILITY,
    DISCOVERABILITY_EAGER,
    DISCOVERABILITY_DEFERRED,
    normalize_inclusive_usage,
    uncached_prompt_tokens,
)
from .decisions import (
    DecisionAnswer,
    DecisionModelOnlyError,
    DecisionQuestion,
    DecisionResult,
)

__all__ = [
    "TRAIT_FILE_WRITER",
    "DISCOVERABILITY_EAGER",
    "DISCOVERABILITY_DEFERRED",
    "Attachment",
    "CancelledException",
    "CancelToken",
    "EditableContent",
    "FinishReason",
    "FunctionCall",
    "Message",
    "Part",
    "ProviderResponse",
    "Role",
    "ThinkingConfig",
    "TokenUsage",
    "ToolResult",
    "ToolSchema",
    "TOOL_CATEGORIES",
    "TOOL_DISCOVERABILITY",
    "DISCOVERABILITY_EAGER",
    "DISCOVERABILITY_DEFERRED",
    # An out-of-tree provider on an OpenAI-shaped wire needs this to
    # meet TokenUsage's prompt-token convention; see its docstring.
    "normalize_inclusive_usage",
    "uncached_prompt_tokens",
    # Decision models (docs/design/decision-models.md); the rest of the
    # contract is in jaato_sdk.plugins.model_provider.decisions.
    "DecisionAnswer",
    "DecisionModelOnlyError",
    "DecisionQuestion",
    "DecisionResult",
]
