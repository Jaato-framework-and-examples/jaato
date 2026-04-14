"""Error types for the LM Studio provider."""

from typing import List, Optional


class LMStudioError(Exception):
    """Base class for LM Studio provider errors."""
    pass


class LMStudioConnectionError(LMStudioError):
    """LM Studio server is not reachable at the configured host."""

    def __init__(self, host: str, message: str = ""):
        self.host = host
        detail = f": {message}" if message else ""
        super().__init__(
            f"Cannot connect to LM Studio server at {host}{detail}\n"
            f"Make sure LM Studio's local server is running "
            f"(Developer tab → Start Server)."
        )


class LMStudioModelNotFoundError(LMStudioError):
    """Requested model is not available in LM Studio."""

    def __init__(self, model: str, available: Optional[List[str]] = None):
        self.model = model
        self.available = available or []
        if self.available:
            avail_str = ", ".join(self.available[:5])
            if len(self.available) > 5:
                avail_str += f", ... ({len(self.available)} total)"
            super().__init__(
                f"Model '{model}' not found in LM Studio.\n"
                f"Available models: {avail_str}\n"
                f"Download it via LM Studio's Discover tab or the `lms get` CLI."
            )
        else:
            super().__init__(
                f"Model '{model}' not found in LM Studio.\n"
                f"Download it via LM Studio's Discover tab or the `lms get` CLI."
            )


class LMStudioLoadError(LMStudioError):
    """LM Studio refused to load the model with the supplied configuration.

    Raised when ``POST /api/v1/models/load`` returns a non-2xx response —
    typically indicates the config (context length, GPU offload, etc.)
    exceeds what the model or hardware can support.
    """

    def __init__(
        self,
        model: str,
        status_code: int,
        body: str,
        load_config: Optional[dict] = None,
    ):
        self.model = model
        self.status_code = status_code
        self.body = body
        self.load_config = load_config or {}
        config_summary = ", ".join(f"{k}={v}" for k, v in self.load_config.items())
        super().__init__(
            f"LM Studio failed to load '{model}' (HTTP {status_code}).\n"
            f"Requested config: {config_summary or '(none)'}\n"
            f"Response: {body[:500]}"
        )


class LMStudioAuthenticationError(LMStudioError):
    """LM Studio rejected the bearer token.

    Only raised when ``LMSTUDIO_API_TOKEN`` is set but either missing or
    incorrect for a server configured with ``Require API Token``.
    """

    def __init__(self, original_error: Optional[str] = None):
        self.original_error = original_error
        lines = ["LM Studio rejected the bearer token."]
        if original_error:
            lines.append(f"Error: {original_error}")
        lines.extend([
            "",
            "If the server has 'Require API Token' enabled:",
            "  1. Copy the token from LM Studio's Developer tab",
            "  2. Set LMSTUDIO_API_TOKEN=<token>",
        ])
        super().__init__("\n".join(lines))
