"""Antigravity provider exceptions.

Custom exception hierarchy for Antigravity-specific error conditions.
"""

from typing import List, Optional


class AntigravityProviderError(Exception):
    """Base exception for Antigravity provider errors."""

    pass


class AuthenticationError(AntigravityProviderError):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Antigravity authentication failed",
        checked_locations: Optional[List[str]] = None,
    ):
        self.checked_locations = checked_locations or []
        if checked_locations:
            locations = ", ".join(checked_locations)
            message = f"{message}. Checked: {locations}"
        super().__init__(message)


class TokenExpiredError(AntigravityProviderError):
    """Raised when OAuth tokens are expired and refresh fails."""

    def __init__(self, message: str = "OAuth tokens expired and refresh failed"):
        super().__init__(message)


class TokenRefreshError(AntigravityProviderError):
    """Raised when token refresh fails."""

    def __init__(self, message: str = "Failed to refresh OAuth token"):
        super().__init__(message)


class RateLimitError(AntigravityProviderError):
    """Raised when rate limit is exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying (if available).
        email: The account that was rate limited.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        email: Optional[str] = None,
    ):
        self.retry_after = retry_after
        self.email = email
        super().__init__(message)


class QuotaExceededError(AntigravityProviderError):
    """Raised when quota is exhausted."""

    def __init__(
        self,
        message: str = "Antigravity quota exceeded",
        quota_type: str = "antigravity",
    ):
        self.quota_type = quota_type
        super().__init__(message)


class ModelNotFoundError(AntigravityProviderError):
    """Raised when requested model is not available."""

    def __init__(self, model: str):
        self.model = model
        super().__init__(f"Model not found: {model}")


class ProjectNotFoundError(AntigravityProviderError):
    """Raised when project ID cannot be resolved."""

    def __init__(self, message: str = "Could not resolve Antigravity project ID"):
        super().__init__(message)


class EndpointError(AntigravityProviderError):
    """Raised when all API endpoints fail."""

    def __init__(
        self,
        message: str = "All Antigravity endpoints failed",
        tried_endpoints: Optional[List[str]] = None,
    ):
        self.tried_endpoints = tried_endpoints or []
        super().__init__(message)


class ContextLimitError(AntigravityProviderError):
    """Raised when context window limit is exceeded."""

    def __init__(
        self,
        tokens_used: int,
        context_limit: int,
        message: Optional[str] = None,
    ):
        self.tokens_used = tokens_used
        self.context_limit = context_limit
        if message is None:
            message = (
                f"Context limit exceeded: {tokens_used:,} tokens used, "
                f"limit is {context_limit:,} tokens"
            )
        super().__init__(message)


class APIError(AntigravityProviderError):
    """Raised for general API errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class StreamingError(AntigravityProviderError):
    """Raised when streaming response parsing fails."""

    def __init__(self, message: str = "Streaming response parsing failed"):
        super().__init__(message)


class ToolResultMissingError(AntigravityProviderError):
    """Raised when tool results are missing (session recovery needed)."""

    def __init__(self, message: str = "Tool result missing - session may need recovery"):
        super().__init__(message)
