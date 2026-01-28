"""GitHub Copilot API client for OAuth-based authentication.

This client uses the Copilot internal API at api.githubcopilot.com which
accepts OAuth tokens obtained through the device code flow.

The API is OpenAI-compatible, so we implement a simple HTTP client that
mirrors the Azure SDK interface for easy integration.
"""

import json
import requests
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

# Copilot API endpoints
COPILOT_API_BASE = "https://api.githubcopilot.com"
COPILOT_CHAT_ENDPOINT = f"{COPILOT_API_BASE}/chat/completions"
COPILOT_MODELS_ENDPOINT = f"{COPILOT_API_BASE}/models"

# Required headers for Copilot API
COPILOT_HEADERS = {
    "Copilot-Integration-Id": "vscode-chat",
    "Editor-Version": "jaato/1.0",
    "Editor-Plugin-Version": "copilot-chat/0.1.0",
    "Content-Type": "application/json",
    "Accept": "application/json",
}


@dataclass
class CopilotMessage:
    """Message in OpenAI format."""
    role: str
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class CopilotChoice:
    """Choice from completion response."""
    index: int
    message: CopilotMessage
    finish_reason: str


@dataclass
class CopilotUsage:
    """Token usage from response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CopilotResponse:
    """Response from chat completion."""
    id: str
    model: str
    choices: List[CopilotChoice]
    usage: CopilotUsage
    created: int = 0


@dataclass
class CopilotStreamDelta:
    """Delta from streaming response."""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None


@dataclass
class CopilotStreamChoice:
    """Choice from streaming response."""
    index: int
    delta: CopilotStreamDelta
    finish_reason: Optional[str] = None


class CopilotClient:
    """Client for GitHub Copilot API.

    This client provides an interface similar to the Azure SDK but uses
    the Copilot internal API which accepts OAuth tokens.
    """

    def __init__(self, token: str):
        """Initialize the Copilot client.

        Args:
            token: Copilot API token (from /copilot_internal/v2/token exchange)
        """
        self._token = token
        self._timeout = 120  # 2 minute timeout for completions
        self._session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create requests session with appropriate proxy configuration.

        Uses shared.http module for unified proxy/Kerberos configuration.

        Returns:
            Configured requests.Session.
        """
        from shared.http import get_requests_session
        return get_requests_session()

    def _make_request(
        self,
        url: str,
        data: Optional[Dict] = None,
        method: str = "POST",
        stream: bool = False,
    ) -> Any:
        """Make HTTP request to Copilot API.

        Supports multiple proxy configurations:
        1. JAATO_NO_PROXY: Exact host matching to bypass proxy
        2. JAATO_KERBEROS_PROXY: Kerberos/SPNEGO authentication with proxy
        3. Standard proxy env vars (HTTP_PROXY, HTTPS_PROXY, NO_PROXY)

        Args:
            url: Request URL
            data: JSON data for POST requests
            method: HTTP method
            stream: If True, return response object for streaming

        Returns:
            Parsed JSON response or response object for streaming

        Raises:
            RuntimeError: If request fails
        """
        from shared.http import should_bypass_proxy

        headers = {
            **COPILOT_HEADERS,
            "Authorization": f"Bearer {self._token}",
        }

        # Bypass proxy if URL matches JAATO_NO_PROXY
        proxies = {} if should_bypass_proxy(url) else None

        try:
            response = self._session.request(
                method=method,
                url=url,
                json=data,
                headers=headers,
                stream=stream,
                timeout=self._timeout,
                proxies=proxies,
            )
            response.raise_for_status()
            if stream:
                return response
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_body = e.response.text if e.response else str(e)
            try:
                error_data = json.loads(error_body)
                error_msg = error_data.get("error", {}).get("message", error_body)
            except json.JSONDecodeError:
                error_msg = error_body
            raise RuntimeError(f"Copilot API error (HTTP {e.response.status_code if e.response else 'unknown'}): {error_msg}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Copilot API request failed: {e}") from e

    def list_models(self) -> List[str]:
        """List available models from Copilot API.

        Returns:
            List of model IDs
        """
        try:
            response = self._make_request(COPILOT_MODELS_ENDPOINT, method="GET")
            models = []
            for model in response.get("data", []):
                model_id = model.get("id")
                if model_id:
                    models.append(model_id)
            return models
        except Exception:
            # Fallback to common models if API fails
            return [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "claude-3.5-sonnet",
                "o1-preview",
                "o1-mini",
            ]

    def complete(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
    ) -> CopilotResponse:
        """Send chat completion request.

        Args:
            model: Model ID (e.g., 'gpt-4o')
            messages: List of message dicts with role and content
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tools: Optional tool definitions
            stream: If True, return streaming response

        Returns:
            CopilotResponse with completion
        """
        # Build request payload
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True

        response = self._make_request(COPILOT_CHAT_ENDPOINT, payload)

        # Parse response
        choices = []
        for choice_data in response.get("choices", []):
            msg_data = choice_data.get("message", {})
            message = CopilotMessage(
                role=msg_data.get("role", "assistant"),
                content=msg_data.get("content", ""),
                tool_calls=msg_data.get("tool_calls"),
            )
            choices.append(CopilotChoice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason", "stop"),
            ))

        usage_data = response.get("usage", {})
        usage = CopilotUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return CopilotResponse(
            id=response.get("id", ""),
            model=response.get("model", model),
            choices=choices,
            usage=usage,
            created=response.get("created", 0),
        )

    def complete_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict]] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> Iterator[CopilotStreamChoice]:
        """Send streaming chat completion request.

        Args:
            model: Model ID
            messages: List of message dicts
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            tools: Optional tool definitions
            on_chunk: Optional callback for each text chunk

        Yields:
            CopilotStreamChoice for each chunk
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools

        response = self._make_request(COPILOT_CHAT_ENDPOINT, payload, stream=True)

        # Parse SSE stream using iter_lines for proper timeout handling
        # Force UTF-8 encoding - SSE streams may not have charset in Content-Type,
        # causing requests to default to ISO-8859-1 per HTTP/1.1 RFC
        response.encoding = 'utf-8'
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        return

                    try:
                        chunk = json.loads(data)
                        for choice_data in chunk.get("choices", []):
                            delta_data = choice_data.get("delta", {})
                            delta = CopilotStreamDelta(
                                role=delta_data.get("role"),
                                content=delta_data.get("content"),
                                tool_calls=delta_data.get("tool_calls"),
                            )

                            if on_chunk and delta.content:
                                on_chunk(delta.content)

                            yield CopilotStreamChoice(
                                index=choice_data.get("index", 0),
                                delta=delta,
                                finish_reason=choice_data.get("finish_reason"),
                            )
                    except json.JSONDecodeError:
                        continue
        finally:
            response.close()

    def close(self) -> None:
        """Close the client and its session."""
        self._session.close()
