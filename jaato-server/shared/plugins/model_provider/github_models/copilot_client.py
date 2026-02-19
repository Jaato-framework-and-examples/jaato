"""GitHub Copilot API client for OAuth-based authentication.

This client uses the Copilot internal API at api.githubcopilot.com which
accepts OAuth tokens obtained through the device code flow.

The API is OpenAI-compatible, so we implement a simple HTTP client that
mirrors the Azure SDK interface for easy integration.
"""

import json
import httpx
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# Copilot API endpoints
COPILOT_API_BASE = "https://api.githubcopilot.com"
COPILOT_CHAT_ENDPOINT = f"{COPILOT_API_BASE}/chat/completions"
COPILOT_RESPONSES_ENDPOINT = f"{COPILOT_API_BASE}/responses"
COPILOT_MODELS_ENDPOINT = f"{COPILOT_API_BASE}/models"


def is_responses_api_model(model: str) -> bool:
    """Check if a model requires the Responses API instead of Chat Completions.

    Models with "codex" in the name use the Responses API endpoint.
    This includes: gpt-5-codex, gpt-5.1-codex, gpt-5.2-codex, etc.

    Args:
        model: Model name/ID.

    Returns:
        True if the model requires the Responses API.
    """
    model_lower = model.lower()
    return "codex" in model_lower

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


# Responses API specific types
@dataclass
class ResponsesUsage:
    """Token usage from Responses API."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ResponsesFunctionCall:
    """Function call from Responses API output."""
    call_id: str
    name: str
    arguments: str


@dataclass
class ResponsesOutputItem:
    """Output item from Responses API.

    Can be a message or a function_call.
    """
    type: str  # "message" or "function_call"
    # For type="message"
    role: Optional[str] = None
    content: Optional[List[Dict[str, Any]]] = None  # [{"type": "output_text", "text": "..."}]
    # For type="function_call"
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None


@dataclass
class ResponsesAPIResponse:
    """Response from Responses API."""
    id: str
    model: str
    output: List[ResponsesOutputItem]
    usage: ResponsesUsage
    created: int = 0


class CopilotClient:
    """Client for GitHub Copilot API.

    This client provides an interface similar to the Azure SDK but uses
    the Copilot internal API which accepts OAuth tokens.

    Token auto-refresh: Before each API request, the client checks if the
    token needs to be refreshed and obtains a new one if needed.
    """

    def __init__(self, token: str):
        """Initialize the Copilot client.

        Args:
            token: Copilot API token (from /copilot_internal/v2/token exchange)
        """
        self._token = token
        self._timeout = 120  # 2 minute timeout for completions
        self._session = self._create_session()

    def _ensure_valid_token(self) -> None:
        """Ensure the token is valid, refreshing if needed.

        This method checks with the OAuth module if the token needs
        to be refreshed and updates self._token if a new token is obtained.
        """
        try:
            from .oauth import get_stored_access_token, _oauth_trace
            _oauth_trace("_ensure_valid_token: checking if token needs refresh")
            fresh_token = get_stored_access_token()
            if fresh_token and fresh_token != self._token:
                _oauth_trace("_ensure_valid_token: token refreshed")
                self._token = fresh_token
            elif not fresh_token:
                _oauth_trace("_ensure_valid_token: get_stored_access_token returned None")
        except Exception as e:
            # Log the error for debugging but continue with existing token
            # The API call will fail with a clear error if token is invalid
            try:
                from .oauth import _oauth_trace
                _oauth_trace(f"_ensure_valid_token: refresh failed: {e}")
            except Exception:
                pass

    def _force_token_refresh(self) -> None:
        """Force a token refresh by clearing cached token and re-exchanging.

        Used when a 401 is received, indicating the token may be invalid
        even if it appeared valid according to expiry time.
        """
        try:
            from .oauth import clear_copilot_token, get_stored_access_token, _oauth_trace
            _oauth_trace("_force_token_refresh: clearing cached Copilot token")
            # Clear the cached Copilot token to force re-exchange
            clear_copilot_token()
            # Get fresh token (will exchange OAuth token for new Copilot token)
            _oauth_trace("_force_token_refresh: exchanging OAuth for new Copilot token")
            fresh_token = get_stored_access_token()
            if fresh_token:
                _oauth_trace("_force_token_refresh: token refreshed successfully")
                self._token = fresh_token
            else:
                _oauth_trace("_force_token_refresh: get_stored_access_token returned None - OAuth token may be invalid")
        except Exception as e:
            # Log the error for debugging but keep existing token for retry
            try:
                from .oauth import _oauth_trace
                _oauth_trace(f"_force_token_refresh: refresh failed: {e}")
            except Exception:
                pass

    def _create_session(self) -> httpx.Client:
        """Create httpx client with appropriate proxy configuration.

        Uses shared.http module for unified proxy/Kerberos/SSL configuration.

        Returns:
            Configured httpx.Client.
        """
        from shared.http import get_httpx_client
        return get_httpx_client(timeout=float(self._timeout))

    def _make_request(
        self,
        url: str,
        data: Optional[Dict] = None,
        method: str = "POST",
        stream: bool = False,
    ) -> Any:
        """Make HTTP request to Copilot API.

        Proxy, SSL, and Kerberos configuration are handled at the client
        level by ``get_httpx_client()`` (called in ``_create_session``).

        Args:
            url: Request URL
            data: JSON data for POST requests
            method: HTTP method
            stream: If True, return response object for streaming.
                    Caller must close the response when done.

        Returns:
            Parsed JSON response, or an httpx.Response for streaming

        Raises:
            RuntimeError: If request fails
        """
        # Ensure token is valid before making request
        self._ensure_valid_token()

        headers = {
            **COPILOT_HEADERS,
            "Authorization": f"Bearer {self._token}",
        }

        try:
            if stream:
                # Build request and send with stream=True so the response
                # body is not eagerly consumed.  Caller iterates and closes.
                req = self._session.build_request(
                    method, url, json=data, headers=headers,
                )
                response = self._session.send(req, stream=True)
            else:
                response = self._session.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=headers,
                )
            response.raise_for_status()
            if stream:
                return response
            return response.json()
        except httpx.HTTPStatusError as e:
            # On 401, try to refresh token and retry once
            if e.response.status_code == 401:
                old_token = self._token
                self._force_token_refresh()
                if self._token != old_token:
                    # Token was refreshed, retry the request
                    headers["Authorization"] = f"Bearer {self._token}"
                    try:
                        if stream:
                            req = self._session.build_request(
                                method, url, json=data, headers=headers,
                            )
                            response = self._session.send(req, stream=True)
                        else:
                            response = self._session.request(
                                method=method,
                                url=url,
                                json=data,
                                headers=headers,
                            )
                        response.raise_for_status()
                        if stream:
                            return response
                        return response.json()
                    except httpx.HTTPStatusError:
                        pass  # Fall through to original error handling

            status_code = e.response.status_code
            error_body = e.response.text
            try:
                error_data = json.loads(error_body)
                error_msg = error_data.get("error", {}).get("message", error_body)
            except json.JSONDecodeError:
                error_msg = error_body

            # For 400 errors, add diagnostic context about the request
            diagnostic = ""
            if status_code == 400 and data:
                messages = data.get("messages", [])
                msg_summary = []
                for m in messages[-5:]:  # Last 5 messages
                    role = m.get("role", "?")
                    content_len = len(m.get("content") or "") if m.get("content") else 0
                    tool_calls = len(m.get("tool_calls", []))
                    tool_call_id = m.get("tool_call_id", "")[:16] if m.get("tool_call_id") else ""
                    if role == "assistant" and tool_calls:
                        ids = [tc.get("id", "?")[:16] for tc in m.get("tool_calls", [])]
                        msg_summary.append(f"{role}(tools={tool_calls}, ids={ids})")
                    elif role == "tool":
                        msg_summary.append(f"{role}(id={tool_call_id}, content_len={content_len})")
                    else:
                        msg_summary.append(f"{role}(content_len={content_len})")
                diagnostic = f" | Last messages: {msg_summary}"

            raise RuntimeError(f"Copilot API error (HTTP {status_code}): {error_msg}{diagnostic}") from e
        except httpx.HTTPError as e:
            from shared.ssl_helper import is_ssl_cert_failure, log_ssl_guidance
            if is_ssl_cert_failure(e):
                log_ssl_guidance("Copilot API", e)
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

    def list_models_with_info(self) -> Dict[str, Dict[str, Any]]:
        """List available models with detailed info including token limits.

        Returns:
            Dict mapping model IDs to model info dicts with keys:
            - id: Model ID
            - max_input_tokens: Maximum input tokens
            - max_output_tokens: Maximum output tokens
            - context_window: Total context window size
        """
        try:
            response = self._make_request(COPILOT_MODELS_ENDPOINT, method="GET")
            models_info: Dict[str, Dict[str, Any]] = {}

            for model in response.get("data", []):
                model_id = model.get("id")
                if not model_id:
                    continue

                # Extract token limits from various possible field names
                max_input = (
                    model.get("max_input_tokens") or
                    model.get("maxInputTokens") or
                    model.get("input_token_limit") or
                    model.get("context_length") or
                    0
                )
                max_output = (
                    model.get("max_output_tokens") or
                    model.get("maxOutputTokens") or
                    model.get("output_token_limit") or
                    0
                )
                context_window = (
                    model.get("context_window") or
                    model.get("contextWindow") or
                    model.get("context_length") or
                    0
                )

                models_info[model_id] = {
                    "id": model_id,
                    "max_input_tokens": int(max_input) if max_input else 0,
                    "max_output_tokens": int(max_output) if max_output else 0,
                    "context_window": int(context_window) if context_window else 0,
                }

            return models_info
        except Exception:
            # Return empty dict to trigger fallback in provider
            return {}

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

        # Parse SSE stream
        try:
            for line in response.iter_lines():
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

    # ==================== Responses API Methods ====================

    def _convert_messages_to_responses_input(
        self,
        messages: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Convert chat messages to Responses API input format.

        The Responses API uses a different format:
        - System message becomes "instructions" parameter
        - User/assistant messages use same format
        - Tool results become {"type": "function_call_output", "call_id": "...", "output": "..."}
        - Assistant tool calls need to be represented as function_call items

        Args:
            messages: Messages in chat completions format.
            system_instruction: Optional system instruction.

        Returns:
            Tuple of (input items, instructions).
        """
        input_items: List[Dict[str, Any]] = []
        instructions = system_instruction

        for msg in messages:
            role = msg.get("role", "")

            if role == "system":
                # System messages become instructions
                instructions = msg.get("content", "")

            elif role == "user":
                # User messages stay similar
                input_items.append({
                    "role": "user",
                    "content": msg.get("content", ""),
                })

            elif role == "assistant":
                # Assistant messages - check for tool calls
                content = msg.get("content")
                tool_calls = msg.get("tool_calls", [])

                if content:
                    input_items.append({
                        "role": "assistant",
                        "content": content,
                    })

                # Add function calls as separate items
                for tc in tool_calls:
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", "{}"),
                    })

            elif role == "tool":
                # Tool results become function_call_output
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })

        return input_items, instructions

    def _convert_tools_to_responses_format(
        self,
        tools: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert chat completions tools to Responses API format.

        Responses API tools have a flatter structure:
        {"type": "function", "name": "...", "description": "...", "parameters": {...}}

        Args:
            tools: Tools in chat completions format.

        Returns:
            Tools in Responses API format.
        """
        if not tools:
            return None

        responses_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                responses_tools.append({
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })

        return responses_tools if responses_tools else None

    def complete_responses(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict]] = None,
    ) -> ResponsesAPIResponse:
        """Send request to Responses API (for Codex models).

        Args:
            model: Model ID (e.g., 'gpt-5.2-codex')
            messages: List of message dicts in chat completions format
            system_instruction: Optional system instruction
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tools: Optional tool definitions in chat completions format

        Returns:
            ResponsesAPIResponse with completion
        """
        # Convert messages to Responses API format
        input_items, instructions = self._convert_messages_to_responses_input(
            messages, system_instruction
        )

        # Build request payload
        payload: Dict[str, Any] = {
            "model": model,
            "input": input_items,
        }

        if instructions:
            payload["instructions"] = instructions
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        # Convert tools to Responses API format
        responses_tools = self._convert_tools_to_responses_format(tools)
        if responses_tools:
            payload["tools"] = responses_tools

        response = self._make_request(COPILOT_RESPONSES_ENDPOINT, payload)

        # Parse response
        output_items = []
        for item in response.get("output", []):
            item_type = item.get("type", "")
            output_item = ResponsesOutputItem(
                type=item_type,
                role=item.get("role"),
                content=item.get("content"),
                call_id=item.get("call_id"),
                name=item.get("name"),
                arguments=item.get("arguments"),
            )
            output_items.append(output_item)

        usage_data = response.get("usage", {})
        usage = ResponsesUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return ResponsesAPIResponse(
            id=response.get("id", ""),
            model=response.get("model", model),
            output=output_items,
            usage=usage,
            created=response.get("created", 0),
        )

    def complete_responses_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict]] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Send streaming request to Responses API (for Codex models).

        Args:
            model: Model ID (e.g., 'gpt-5.2-codex')
            messages: List of message dicts in chat completions format
            system_instruction: Optional system instruction
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            tools: Optional tool definitions
            on_chunk: Optional callback for each text chunk

        Yields:
            Dictionaries with streaming data:
            - {"type": "text", "text": "..."} for text content
            - {"type": "function_call", "call_id": "...", "name": "...", "arguments": "..."}
            - {"type": "done", "usage": {...}} for completion
        """
        # Convert messages to Responses API format
        input_items, instructions = self._convert_messages_to_responses_input(
            messages, system_instruction
        )

        # Build request payload
        payload: Dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": True,
        }

        if instructions:
            payload["instructions"] = instructions
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        # Convert tools to Responses API format
        responses_tools = self._convert_tools_to_responses_format(tools)
        if responses_tools:
            payload["tools"] = responses_tools

        response = self._make_request(COPILOT_RESPONSES_ENDPOINT, payload, stream=True)

        # Parse SSE stream
        try:
            # Accumulate function call data since it may come in pieces
            current_function_call: Optional[Dict[str, Any]] = None

            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        # Yield completion signal
                        yield {"type": "done"}
                        return

                    try:
                        event = json.loads(data)

                        # Handle different event types from Responses API
                        event_type = event.get("type", "")

                        if event_type == "response.output_item.added":
                            # New output item starting
                            item = event.get("item", {})
                            if item.get("type") == "function_call":
                                current_function_call = {
                                    "call_id": item.get("call_id", ""),
                                    "name": item.get("name", ""),
                                    "arguments": "",
                                }

                        elif event_type == "response.function_call_arguments.delta":
                            # Function call arguments streaming
                            if current_function_call:
                                delta = event.get("delta", "")
                                current_function_call["arguments"] += delta

                        elif event_type == "response.function_call_arguments.done":
                            # Function call complete
                            if current_function_call:
                                yield {
                                    "type": "function_call",
                                    "call_id": current_function_call["call_id"],
                                    "name": current_function_call["name"],
                                    "arguments": current_function_call["arguments"],
                                }
                                current_function_call = None

                        elif event_type == "response.output_text.delta":
                            # Text content streaming
                            text = event.get("delta", "")
                            if text:
                                if on_chunk:
                                    on_chunk(text)
                                yield {"type": "text", "text": text}

                        elif event_type == "response.completed":
                            # Response complete - extract usage
                            resp = event.get("response", {})
                            usage = resp.get("usage", {})
                            yield {
                                "type": "done",
                                "usage": {
                                    "input_tokens": usage.get("input_tokens", 0),
                                    "output_tokens": usage.get("output_tokens", 0),
                                    "total_tokens": usage.get("total_tokens", 0),
                                }
                            }
                            return

                    except json.JSONDecodeError:
                        continue
        finally:
            response.close()

    def close(self) -> None:
        """Close the client and its session."""
        self._session.close()
