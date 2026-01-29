"""Authentication and configuration error types for GitHub Models provider.

These exceptions wrap underlying SDK/API errors with actionable guidance
for users to resolve authentication and configuration issues.
"""

from typing import List, Optional


class GitHubModelsError(Exception):
    """Base class for GitHub Models provider errors."""
    pass


class TokenNotFoundError(GitHubModelsError):
    """No valid GitHub token could be located.

    Raised when the provider cannot find a valid token through
    the configured authentication method.
    """

    def __init__(
        self,
        auth_method: str = "auto",
        checked_locations: Optional[List[str]] = None,
        suggestion: Optional[str] = None,
    ):
        self.auth_method = auth_method
        self.checked_locations = checked_locations or []
        self.suggestion = suggestion

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = [
            "No GitHub token found.",
            "",
        ]

        if self.checked_locations:
            lines.append("Checked locations:")
            for loc in self.checked_locations:
                lines.append(f"  - {loc}")
            lines.append("")

        if self.suggestion:
            lines.append("To fix:")
            lines.append(f"  {self.suggestion}")
        else:
            lines.extend(self._default_suggestions())

        return "\n".join(lines)

    def _default_suggestions(self) -> List[str]:
        return [
            "To authenticate, choose one of these options:",
            "",
            "Option 1: Device Code Flow (recommended for GitHub Copilot)",
            "  Run 'github-auth login' to authenticate via browser",
            "",
            "Option 2: Personal Access Token (PAT)",
            "  1. Create a PAT at https://github.com/settings/tokens",
            "  2. For fine-grained PAT: select 'models: read' permission",
            "  3. Set GITHUB_TOKEN=your-token",
            "",
            "For GitHub Enterprise with SSO:",
            "  - Fine-grained PATs are auto-authorized during creation",
            "  - Classic PATs require manual SSO authorization per organization",
        ]


class TokenInvalidError(GitHubModelsError):
    """Token was found but is invalid or rejected.

    Raised when a token exists but cannot be used for authentication.
    """

    def __init__(
        self,
        reason: str,
        token_prefix: Optional[str] = None,
        original_error: Optional[str] = None,
    ):
        self.reason = reason
        self.token_prefix = token_prefix
        self.original_error = original_error

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = ["GitHub token is invalid or was rejected."]

        if self.token_prefix:
            lines.append(f"Token type: {self._identify_token_type()}")

        lines.append(f"Reason: {self.reason}")

        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Verify your token is not expired",
            "  2. Ensure the token has 'models: read' permission",
            "  3. For enterprise SSO: authorize the token for your organization",
            "  4. Generate a new token if needed: https://github.com/settings/tokens",
        ])

        return "\n".join(lines)

    def _identify_token_type(self) -> str:
        if not self.token_prefix:
            return "unknown"
        if self.token_prefix.startswith("ghp_"):
            return "classic PAT"
        elif self.token_prefix.startswith("github_pat_"):
            return "fine-grained PAT"
        elif self.token_prefix.startswith("gho_"):
            return "OAuth token"
        elif self.token_prefix.startswith("ghu_"):
            return "GitHub App user token"
        elif self.token_prefix.startswith("ghs_"):
            return "GitHub App installation token"
        return "unknown"


class TokenPermissionError(GitHubModelsError):
    """Token lacks required permissions.

    Raised when authentication succeeds but the token does not
    have sufficient permissions for the requested operation.
    """

    def __init__(
        self,
        organization: Optional[str] = None,
        enterprise: Optional[str] = None,
        missing_permission: Optional[str] = None,
        original_error: Optional[str] = None,
    ):
        self.organization = organization
        self.enterprise = enterprise
        self.missing_permission = missing_permission
        self.original_error = original_error

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = ["GitHub token lacks required permissions for GitHub Models."]

        if self.organization:
            lines.append(f"Organization: {self.organization}")
        if self.enterprise:
            lines.append(f"Enterprise: {self.enterprise}")
        if self.missing_permission:
            lines.append(f"Missing permission: {self.missing_permission}")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "To fix:",
            "  1. Create a fine-grained PAT with 'models: read' permission",
            "  2. For organization access, ensure the token is authorized for the org",
        ])

        if self.organization:
            lines.extend([
                "",
                f"For SSO-enabled organization '{self.organization}':",
                "  - Fine-grained PAT: auto-authorized during creation",
                "  - Classic PAT: go to https://github.com/settings/tokens",
                f"    and click 'Authorize' next to your organization",
            ])

        return "\n".join(lines)


class ModelsDisabledError(GitHubModelsError):
    """GitHub Models is disabled for the enterprise/organization.

    Raised when the API returns a 401 indicating GitHub Models
    is disabled by an administrator.
    """

    def __init__(
        self,
        organization: Optional[str] = None,
        enterprise: Optional[str] = None,
        original_error: Optional[str] = None,
    ):
        self.organization = organization
        self.enterprise = enterprise
        self.original_error = original_error

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = ["GitHub Models is disabled."]

        if self.enterprise:
            lines.append(f"Enterprise: {self.enterprise}")
        if self.organization:
            lines.append(f"Organization: {self.organization}")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "GitHub Models must be enabled by an administrator.",
            "",
            "To fix:",
        ])

        if self.enterprise:
            lines.extend([
                f"  1. An enterprise owner must enable GitHub Models at:",
                f"     https://github.com/enterprises/{self.enterprise}/settings/models",
                "  2. Enterprise policy must allow 'Enabled' or 'No policy'",
            ])
        elif self.organization:
            lines.extend([
                f"  1. An organization admin must enable GitHub Models at:",
                f"     https://github.com/organizations/{self.organization}/settings/models",
                "  2. If enterprise-managed, the enterprise must enable it first",
            ])
        else:
            lines.extend([
                "  Contact your GitHub Enterprise administrator to enable GitHub Models",
            ])

        return "\n".join(lines)


class ModelNotFoundError(GitHubModelsError):
    """Requested model is not available.

    Raised when the specified model ID doesn't exist or isn't
    available in GitHub Models.
    """

    def __init__(
        self,
        model: str,
        available_models: Optional[List[str]] = None,
        original_error: Optional[str] = None,
    ):
        self.model = model
        self.available_models = available_models
        self.original_error = original_error

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = [f"Model not found: {self.model}"]

        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        if self.available_models:
            lines.extend([
                "",
                "Available models:",
            ])
            for model in self.available_models[:10]:  # Limit to first 10
                lines.append(f"  - {model}")
            if len(self.available_models) > 10:
                lines.append(f"  ... and {len(self.available_models) - 10} more")

        lines.extend([
            "",
            "To fix:",
            "  1. Check the model ID format (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet')",
            "  2. Use list_models() to see available models",
            "  3. Verify the model is available in your GitHub subscription tier",
        ])

        return "\n".join(lines)


class RateLimitError(GitHubModelsError):
    """Rate limit exceeded for GitHub Models API.

    Raised when too many requests have been made in a short period.
    """

    def __init__(
        self,
        retry_after: Optional[int] = None,
        limit_type: Optional[str] = None,
        original_error: Optional[str] = None,
    ):
        self.retry_after = retry_after
        self.limit_type = limit_type
        self.original_error = original_error

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        lines = ["GitHub Models rate limit exceeded."]

        if self.limit_type:
            lines.append(f"Limit type: {self.limit_type}")
        if self.retry_after:
            lines.append(f"Retry after: {self.retry_after} seconds")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "Rate limits vary by model and GitHub plan.",
            "",
            "To fix:",
            "  1. Wait for the retry period to elapse",
            "  2. Consider using a different model with higher limits",
            "  3. Upgrade your GitHub Copilot plan for higher rate limits",
            "  4. Enable paid usage for your organization/enterprise",
        ])

        return "\n".join(lines)


class ContextLimitError(GitHubModelsError):
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

        message = self._format_message()
        super().__init__(message)

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
            "  2. Use a model with a larger context window",
            "  3. Reduce the size of your prompt",
        ])

        return "\n".join(lines)


class InfrastructureError(GitHubModelsError):
    """Transient infrastructure error from GitHub API.

    Raised when the API returns a 5xx error indicating a temporary
    server-side issue. These errors are typically retriable.
    """

    def __init__(
        self,
        status_code: int,
        original_error: Optional[str] = None,
    ):
        self.status_code = status_code
        self.original_error = original_error

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        if self.status_code == 0:
            # Network-level error (no HTTP status)
            lines = ["GitHub API network error."]
        else:
            lines = [f"GitHub API infrastructure error (HTTP {self.status_code})."]

        if self.original_error:
            lines.append(f"Error: {self.original_error}")

        lines.extend([
            "",
            "This is a transient error.",
            "The request will be automatically retried.",
        ])

        return "\n".join(lines)
