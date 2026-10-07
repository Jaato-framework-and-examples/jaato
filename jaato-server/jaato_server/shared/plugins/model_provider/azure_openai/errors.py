"""Error types for the Azure OpenAI provider.

The taxonomy mirrors the other OpenAI-compatible providers (so the shared
``_handle_api_error`` mapping is reusable), but the remedies are Azure's:
almost every one of these has a *deployment* or an *api-version* behind
it rather than a model id or an account balance.
"""

from typing import List, Optional


class AzureOpenAIError(Exception):
    """Base class for Azure OpenAI provider errors."""
    pass


class ConfigurationError(AzureOpenAIError):
    """A required piece of Azure configuration is missing.

    Azure needs an endpoint and an api-version before a request can even
    be addressed, and neither has a defensible default.  This is raised
    at ``initialize()`` so the failure names the missing key rather than
    arriving later as a 404 on a URL nobody meant to build.
    """

    def __init__(self, missing: str, detail: str = ""):
        self.missing = missing
        self.detail = detail
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = [f"Azure OpenAI: {self.missing} is not configured."]
        if self.detail:
            lines.append(self.detail)
        return "\n".join(lines)


class APIKeyNotFoundError(AzureOpenAIError):
    """No credential could be located for the Azure OpenAI resource."""

    def __init__(self, checked_locations: Optional[List[str]] = None):
        self.checked_locations = checked_locations or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["No Azure OpenAI credential found.", ""]
        if self.checked_locations:
            lines.append("Checked locations:")
            lines.extend(f"  - {loc}" for loc in self.checked_locations)
            lines.append("")
        lines.extend([
            "To authenticate, either:",
            "  A. Resource key — copy it from the Azure portal (your Azure",
            "     OpenAI resource → Keys and Endpoint) and set",
            "     JAATO_AZURE_OPENAI_API_KEY (AZURE_OPENAI_API_KEY also works),",
            "     or plugin_configs.azure_openai.api_key in a profile (which",
            "     may carry a pass:// URI rather than the literal secret)",
            "  B. Microsoft Entra ID — set",
            "     plugin_configs.azure_openai.auth: aad (or",
            "     JAATO_AZURE_OPENAI_AUTH=aad), install 'azure-identity', and",
            "     make sure the identity holds the 'Cognitive Services OpenAI",
            "     User' role on the resource.  No key is then needed at all.",
        ])
        return "\n".join(lines)


class AuthenticationError(AzureOpenAIError):
    """The credential was rejected by the Azure OpenAI resource."""

    def __init__(self, original_error: Optional[str] = None):
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["Azure OpenAI credential was rejected."]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. Check the key against the resource's 'Keys and Endpoint'",
            "     blade — keys are per RESOURCE, so one from a different",
            "     resource authenticates against nothing here",
            "  2. Check the endpoint matches that same resource",
            "  3. On Entra auth: the identity needs the 'Cognitive Services",
            "     OpenAI User' role assignment on the resource, and role",
            "     assignments can take a few minutes to take effect",
        ])
        return "\n".join(lines)


class RateLimitError(AzureOpenAIError):
    """The deployment's provisioned throughput was exceeded."""

    def __init__(
        self,
        retry_after: Optional[float] = None,
        original_error: Optional[str] = None,
    ):
        self.retry_after = retry_after
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = ["Azure OpenAI rate limit exceeded."]
        if self.retry_after:
            lines.append(f"Retry after: {self.retry_after} seconds")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. Wait for the retry period to elapse (jaato retries these)",
            "  2. Azure rate limits are per DEPLOYMENT, not per account —",
            "     raise the deployment's TPM/RPM quota in the portal, or",
            "     spread load across deployments",
        ])
        return "\n".join(lines)


class ModelNotFoundError(AzureOpenAIError):
    """The named deployment does not exist on this resource."""

    def __init__(self, model: str, original_error: Optional[str] = None):
        self.model = model
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = [f"Azure OpenAI deployment not found: {self.model}"]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. On Azure, `model:` is the DEPLOYMENT NAME your subscription",
            "     chose — not the model id.  'gpt-4o' works only if that is",
            "     what the deployment was called",
            "  2. List deployments in the portal (Azure AI Foundry →",
            "     Deployments) or with the provider's list_models()",
            "  3. Check the endpoint points at the resource that holds it",
        ])
        return "\n".join(lines)


class ContextLimitError(AzureOpenAIError):
    """Request exceeds the deployed model's context window."""

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
        lines = [f"Request too large for deployment: {self.model}"]
        if self.max_tokens:
            lines.append(f"Maximum tokens: {self.max_tokens}")
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "To fix:",
            "  1. Clear conversation history with 'clear' command",
            "  2. Reduce the size of your prompt",
            "  3. Check plugin_configs.azure_openai.context_length matches",
            "     the model VERSION behind this deployment — the deployment",
            "     name says nothing about the window, and an upgrade in the",
            "     portal changes it without changing the name",
        ])
        return "\n".join(lines)


class InfrastructureError(AzureOpenAIError):
    """Transient infrastructure error from Azure OpenAI (5xx / network)."""

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
            lines = ["Azure OpenAI network error."]
        else:
            lines = [
                f"Azure OpenAI infrastructure error (HTTP {self.status_code})."
            ]
        if self.original_error:
            lines.append(f"Error: {self.original_error}")
        lines.extend([
            "",
            "This is a transient error.",
            "The request will be automatically retried.",
        ])
        return "\n".join(lines)
