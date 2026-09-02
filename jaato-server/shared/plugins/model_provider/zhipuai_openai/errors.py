"""Error types for the Zhipu AI OpenAI-compatible provider.

These exceptions wrap underlying SDK/API errors with actionable guidance
for users to resolve authentication and configuration issues.
"""

from typing import List, Optional


class ZhipuAIOpenAIError(Exception):
    """Base class for Zhipu AI OpenAI-compatible provider errors."""
    pass


class APIKeyNotFoundError(ZhipuAIOpenAIError):
    """No API key could be located.

    Raised when the provider cannot find a Z.AI API key in any of
    the checked locations.
    """

    def __init__(
        self,
        checked_locations: Optional[List[str]] = None,
    ):
        self.checked_locations = checked_locations or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = [
            "No Zhipu AI API key found.",
            "",
        ]

        if self.checked_locations:
            lines.append("Checked locations:")
            for loc in self.checked_locations:
                lines.append(f"  - {loc}")
            lines.append("")

        lines.extend([
            "To authenticate:",
            "  1. Get an API key from https://z.ai/model-api or https://open.bigmodel.cn/",
            "  2. Set ZHIPUAI_API_KEY=<your_key>",
            "  3. Or run: zhipuai-auth key <your_key>",
        ])

        return "\n".join(lines)


class AuthenticationError(ZhipuAIOpenAIError):
    """API key was rejected by the Z.AI endpoint.

    Raised when a key exists but fails authentication.
    """

    def __init__(self, original_error: Optional[str] = None):
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["Zhipu AI API key is invalid or was rejected."]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. Verify your API key at https://z.ai/model-api",
            "  2. Check that the key has not expired",
            "  3. Run 'zhipuai-auth key <new_key>' to update",
        ])
        return "\n".join(lines)


class RateLimitError(ZhipuAIOpenAIError):
    """Rate limit exceeded for Z.AI API.

    Raised when too many requests have been made in a short period.
    """

    def __init__(
        self,
        retry_after: Optional[float] = None,
        original_error: Optional[str] = None,
    ):
        self.retry_after = retry_after
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["Zhipu AI rate limit exceeded."]
        if self.retry_after:
            lines.append(f"Retry after: {self.retry_after} seconds")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "The request will be automatically retried.",
        ])
        return "\n".join(lines)


class ModelNotFoundError(ZhipuAIOpenAIError):
    """Requested model is not available on the Z.AI endpoint.

    Raised when the specified model ID doesn't exist.
    """

    def __init__(
        self,
        model: str,
        known_models: Optional[List[str]] = None,
        original_error: Optional[str] = None,
    ):
        self.model = model
        self.known_models = known_models
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = [f"Model not found: {self.model}"]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        if self.known_models:
            lines.append("")
            lines.append("Available models:")
            for m in self.known_models:
                lines.append(f"  - {m}")
        return "\n".join(lines)


class ContextLimitError(ZhipuAIOpenAIError):
    """Request exceeds the model's context window.

    Raised when the conversation history + system instructions + prompt
    exceeds the model's maximum token limit.
    """

    def __init__(
        self,
        model: str,
        max_tokens: Optional[int] = None,
        original_error: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = [f"Request too large for model: {self.model}"]
        if self.max_tokens:
            lines.append(f"Maximum tokens: {self.max_tokens}")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. Clear conversation history with 'clear' command",
            "  2. Reduce the size of your prompt",
            "  3. Set ZHIPUAI_OPENAI_CONTEXT_LENGTH to the model's actual limit",
        ])
        return "\n".join(lines)


class InfrastructureError(ZhipuAIOpenAIError):
    """Transient infrastructure error from Z.AI API.

    Raised when the API returns a 5xx error or a network error.
    These are typically retriable.
    """

    def __init__(
        self,
        status_code: int = 0,
        original_error: Optional[str] = None,
    ):
        self.status_code = status_code
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.status_code == 0:
            lines = ["Z.AI API network error."]
        else:
            lines = [f"Z.AI API infrastructure error (HTTP {self.status_code})."]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "This is a transient error.",
            "The request will be automatically retried.",
        ])
        return "\n".join(lines)
