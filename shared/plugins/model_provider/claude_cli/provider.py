"""Claude CLI Model Provider implementation.

This provider wraps the Claude Code CLI, communicating via the stream-json
protocol for programmatic access to Claude's agentic capabilities.
"""

import json
import logging
import subprocess
import threading
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional

from ..base import (
    FunctionCallDetectedCallback,
    ProviderConfig,
    StreamingCallback,
    UsageUpdateCallback,
)
from ..types import (
    CancelToken,
    FinishReason,
    FunctionCall,
    Message,
    Part,
    ProviderResponse,
    Role,
    TokenUsage,
    ToolResult,
    ToolSchema,
)
from .env import (
    resolve_cli_mode,
    resolve_cli_path,
    resolve_max_turns,
    resolve_permission_mode,
)
from .types import (
    AssistantMessage,
    CLIMessage,
    CLIMode,
    ContentBlock,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    Usage,
    parse_ndjson_line,
)

logger = logging.getLogger(__name__)


class CLINotFoundError(Exception):
    """Claude CLI executable not found."""

    pass


class CLIConnectionError(Exception):
    """Failed to connect to Claude CLI."""

    pass


class CLIProcessError(Exception):
    """CLI process exited with an error."""

    def __init__(self, message: str, exit_code: int = -1, stderr: str = ""):
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


class ClaudeCLIProvider:
    """Model provider that uses Claude Code CLI as backend.

    This provider spawns the `claude` CLI in print mode with stream-json
    output format, enabling programmatic access to Claude's agentic
    capabilities without direct API calls.

    Attributes:
        name: Provider identifier ("claude_cli").
        is_connected: Whether the provider is ready to accept messages.
        model_name: Currently configured model name.
    """

    def __init__(self) -> None:
        self._cli_path: Optional[str] = None
        self._mode: CLIMode = CLIMode.DELEGATED
        self._model_name: Optional[str] = None
        self._max_turns: Optional[int] = None
        self._permission_mode: Optional[str] = None

        # Session state
        self._system_instruction: Optional[str] = None
        self._tools: List[ToolSchema] = []
        self._history: List[Message] = []
        self._session_id: Optional[str] = None

        # Process state
        self._process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()

        # Last response tracking
        self._last_usage: Optional[TokenUsage] = None
        self._last_result: Optional[ResultMessage] = None

        # Agent context (for tracing)
        self._agent_type: str = "main"
        self._agent_name: Optional[str] = None
        self._agent_id: str = "main"

        # Tool executor callback (for passthrough mode)
        self._tool_executor: Optional[Callable[[str, Dict[str, Any]], ToolResult]] = None

    @property
    def name(self) -> str:
        return "claude_cli"

    @property
    def is_connected(self) -> bool:
        return self._cli_path is not None and self._model_name is not None

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @property
    def mode(self) -> CLIMode:
        """Current operating mode (delegated or passthrough)."""
        return self._mode

    # ==================== Lifecycle ====================

    def initialize(self, config: Optional[ProviderConfig] = None) -> None:
        """Initialize the provider with configuration.

        Args:
            config: Provider configuration. Relevant extra fields:
                - cli_path: Path to claude CLI executable
                - cli_mode: "delegated" or "passthrough"
                - max_turns: Maximum agentic turns
                - permission_mode: CLI permission mode

        Raises:
            CLINotFoundError: If claude CLI is not found.
        """
        extra = config.extra if config else {}

        # Resolve configuration
        try:
            self._cli_path = resolve_cli_path(extra.get("cli_path"))
        except FileNotFoundError as e:
            raise CLINotFoundError(str(e)) from e

        self._mode = resolve_cli_mode(extra.get("cli_mode"))
        self._max_turns = resolve_max_turns(extra.get("max_turns"))
        self._permission_mode = resolve_permission_mode(extra.get("permission_mode"))

        # Store tool executor if provided (for passthrough mode)
        if "tool_executor" in extra:
            self._tool_executor = extra["tool_executor"]

        logger.info(
            f"Initialized Claude CLI provider: path={self._cli_path}, "
            f"mode={self._mode.value}"
        )

    def verify_auth(
        self,
        allow_interactive: bool = False,
        on_message: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Verify that the CLI is authenticated.

        This checks if the claude CLI is installed and can be executed.
        The CLI handles its own authentication state.

        Note: This method can be called before initialize() to check if
        the CLI is available. In that case, it will attempt to find the
        CLI in PATH.

        Args:
            allow_interactive: Not used (CLI handles interactive auth).
            on_message: Optional callback for status messages.

        Returns:
            True if CLI is available and presumably authenticated.
        """
        # Get CLI path - use cached value if available, otherwise try to find it
        cli_path = self._cli_path
        if not cli_path:
            try:
                cli_path = resolve_cli_path(None)
            except FileNotFoundError:
                if on_message:
                    on_message("Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code")
                return False

        try:
            # Check CLI version to verify it's working
            result = subprocess.run(
                [cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                if on_message:
                    version = result.stdout.strip()
                    on_message(f"Claude CLI version: {version}")
                return True
            else:
                logger.warning(f"CLI version check failed: {result.stderr}")
                if on_message:
                    on_message(f"Claude CLI check failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.warning("CLI version check timed out")
            return False
        except Exception as e:
            logger.warning(f"CLI version check error: {e}")
            return False

    def shutdown(self) -> None:
        """Clean up resources."""
        self._terminate_process()
        self._history.clear()
        self._session_id = None

    # ==================== Connection ====================

    def connect(self, model: str) -> None:
        """Set the model to use.

        Args:
            model: Model name/alias. Can be:
                - "sonnet", "opus", "haiku" (aliases)
                - Full model name like "claude-sonnet-4-20250514"
        """
        self._model_name = model
        logger.info(f"Connected to model: {model}")

    def list_models(self, prefix: Optional[str] = None) -> List[str]:
        """List available models.

        Returns commonly available Claude models. The actual availability
        depends on the user's Claude subscription.
        """
        models = [
            "sonnet",
            "opus",
            "haiku",
            "claude-sonnet-4-20250514",
            "claude-opus-4-5-20251101",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ]
        if prefix:
            models = [m for m in models if m.startswith(prefix)]
        return models

    # ==================== Session Management ====================

    def create_session(
        self,
        system_instruction: Optional[str] = None,
        tools: Optional[List[ToolSchema]] = None,
        history: Optional[List[Message]] = None,
    ) -> None:
        """Create or reset the chat session.

        Args:
            system_instruction: System prompt for Claude.
            tools: List of available tools (used in passthrough mode).
            history: Previous conversation history to restore.
        """
        # Terminate any existing process
        self._terminate_process()

        self._system_instruction = system_instruction
        self._tools = tools or []
        self._history = list(history) if history else []
        self._session_id = str(uuid.uuid4())[:8]
        self._last_usage = None
        self._last_result = None

        logger.debug(
            f"Created session {self._session_id} with "
            f"{len(self._tools)} tools, mode={self._mode.value}"
        )

    def get_history(self) -> List[Message]:
        """Get the current conversation history."""
        return list(self._history)

    # ==================== Messaging ====================

    def generate(self, prompt: str) -> ProviderResponse:
        """Simple one-shot generation without session context."""
        # Create temporary session, generate, restore
        old_history = self._history
        old_session = self._session_id

        self._history = []
        self._session_id = str(uuid.uuid4())[:8]

        try:
            return self.send_message(prompt)
        finally:
            self._history = old_history
            self._session_id = old_session

    def send_message(
        self,
        message: str,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        """Send a user message and get a response.

        In delegated mode, the CLI handles tool execution automatically.
        In passthrough mode, tool_use blocks are returned for jaato to execute.

        Args:
            message: The user's message text.
            response_schema: Not supported by CLI provider.

        Returns:
            ProviderResponse with text and/or function calls.
        """
        if response_schema:
            logger.warning("response_schema not supported by Claude CLI provider")

        return self._execute_query(message)

    def send_message_with_parts(
        self,
        parts: List[Part],
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        """Send a message with multiple parts.

        Currently only text parts are supported. Images would need to be
        handled through file references.
        """
        # Extract text from parts
        text_parts = []
        for part in parts:
            if part.text:
                text_parts.append(part.text)
            elif part.inline_data:
                logger.warning(
                    "Inline data (images) not directly supported by CLI provider. "
                    "Consider saving to a file and referencing it."
                )

        message = "\n".join(text_parts) if text_parts else ""
        return self.send_message(message, response_schema)

    def send_tool_results(
        self,
        results: List[ToolResult],
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> ProviderResponse:
        """Send tool execution results back to the model.

        This is used in passthrough mode where jaato executes tools and
        sends results back. In delegated mode, this is a no-op since
        the CLI handles tool execution internally.

        Args:
            results: List of tool execution results.
            response_schema: Not supported by CLI provider.

        Returns:
            ProviderResponse with the model's next response.
        """
        if self._mode == CLIMode.DELEGATED:
            logger.warning(
                "send_tool_results called in delegated mode - CLI handles tools"
            )
            return ProviderResponse(
                text="",
                finish_reason=FinishReason.STOP,
            )

        # In passthrough mode, we need to continue the conversation
        # by feeding tool results back through stdin
        # This requires maintaining a persistent process, which we'll implement
        # in the streaming version
        raise NotImplementedError(
            "send_tool_results in passthrough mode requires streaming. "
            "Use send_tool_results_streaming instead."
        )

    # ==================== Streaming ====================

    def supports_streaming(self) -> bool:
        """Check if streaming is supported."""
        return True

    def supports_stop(self) -> bool:
        """Check if mid-turn cancellation is supported."""
        return True

    def send_message_streaming(
        self,
        message: str,
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
    ) -> ProviderResponse:
        """Send a message with streaming response.

        Args:
            message: The user's message text.
            on_chunk: Callback for each text chunk.
            cancel_token: Optional token for cancellation.
            response_schema: Not supported.
            on_usage_update: Optional callback for token usage updates.
            on_function_call: Optional callback when function call detected.

        Returns:
            ProviderResponse with accumulated text and/or function calls.
        """
        return self._execute_query_streaming(
            message,
            on_chunk=on_chunk,
            cancel_token=cancel_token,
            on_usage_update=on_usage_update,
            on_function_call=on_function_call,
        )

    def send_tool_results_streaming(
        self,
        results: List[ToolResult],
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
    ) -> ProviderResponse:
        """Send tool results with streaming response.

        This is used in passthrough mode. The tool results are formatted
        and sent to continue the conversation.
        """
        if self._mode == CLIMode.DELEGATED:
            logger.warning(
                "send_tool_results_streaming called in delegated mode - "
                "CLI handles tools"
            )
            return ProviderResponse(text="", finish_reason=FinishReason.STOP)

        # Format tool results for the model
        result_text = self._format_tool_results_as_message(results)
        return self._execute_query_streaming(
            result_text,
            on_chunk=on_chunk,
            cancel_token=cancel_token,
            on_usage_update=on_usage_update,
            on_function_call=on_function_call,
        )

    # ==================== Token Management ====================

    def count_tokens(self, content: str) -> int:
        """Estimate token count.

        Uses a simple heuristic since CLI doesn't expose token counting.
        """
        # Rough estimate: ~4 characters per token
        return len(content) // 4

    def get_context_limit(self) -> int:
        """Get the context window size."""
        # Claude models support 200k tokens
        return 200000

    def get_token_usage(self) -> TokenUsage:
        """Get token usage from the last response."""
        return self._last_usage or TokenUsage(prompt_tokens=0, output_tokens=0)

    # ==================== Serialization ====================

    def serialize_history(self, history: List[Message]) -> str:
        """Serialize conversation history to JSON."""
        data = []
        for msg in history:
            msg_data = {
                "role": msg.role.value,
                "parts": [],
            }
            for part in msg.parts:
                if part.text:
                    msg_data["parts"].append({"type": "text", "text": part.text})
                elif part.function_call:
                    msg_data["parts"].append({
                        "type": "function_call",
                        "id": part.function_call.id,
                        "name": part.function_call.name,
                        "args": part.function_call.args,
                    })
                elif part.function_response:
                    msg_data["parts"].append({
                        "type": "function_response",
                        "call_id": part.function_response.call_id,
                        "name": part.function_response.name,
                        "result": part.function_response.result,
                    })
            data.append(msg_data)
        return json.dumps(data)

    def deserialize_history(self, data: str) -> List[Message]:
        """Deserialize conversation history from JSON."""
        messages = []
        for msg_data in json.loads(data):
            role = Role(msg_data["role"])
            parts = []
            for part_data in msg_data.get("parts", []):
                part_type = part_data.get("type")
                if part_type == "text":
                    parts.append(Part.from_text(part_data["text"]))
                elif part_type == "function_call":
                    parts.append(Part.from_function_call(FunctionCall(
                        id=part_data["id"],
                        name=part_data["name"],
                        args=part_data["args"],
                    )))
                elif part_type == "function_response":
                    parts.append(Part.from_function_response(
                        call_id=part_data["call_id"],
                        name=part_data["name"],
                        result=part_data["result"],
                    ))
            messages.append(Message(role=role, parts=parts))
        return messages

    # ==================== Capabilities ====================

    def supports_structured_output(self) -> bool:
        """Check if structured output is supported."""
        return False  # CLI doesn't support response_schema

    # ==================== Agent Context ====================

    def set_agent_context(
        self,
        agent_type: str = "main",
        agent_name: Optional[str] = None,
        agent_id: str = "main",
    ) -> None:
        """Set agent context for tracing."""
        self._agent_type = agent_type
        self._agent_name = agent_name
        self._agent_id = agent_id

    # ==================== Passthrough Mode Support ====================

    def set_tool_executor(
        self,
        executor: Callable[[str, Dict[str, Any]], ToolResult],
    ) -> None:
        """Set the tool executor callback for passthrough mode.

        In passthrough mode, when the CLI returns tool_use blocks,
        this callback is invoked to execute the tools.

        Args:
            executor: Callable that takes (tool_name, args) and returns ToolResult.
        """
        self._tool_executor = executor

    # ==================== Internal Methods ====================

    def _build_cli_args(self, prompt: str) -> List[str]:
        """Build CLI command arguments."""
        args = [
            self._cli_path,
            "--print",  # Non-interactive mode
            "--output-format", "stream-json",  # NDJSON output
        ]

        # Model selection
        if self._model_name:
            args.extend(["--model", self._model_name])

        # Max turns
        if self._max_turns is not None:
            args.extend(["--max-turns", str(self._max_turns)])

        # Permission mode
        if self._permission_mode:
            args.extend(["--permission-mode", self._permission_mode])

        # System prompt
        if self._system_instruction:
            args.extend(["--system-prompt", self._system_instruction])

        # In passthrough mode, disable all built-in tools so we get tool_use blocks
        if self._mode == CLIMode.PASSTHROUGH:
            # Disallow all tools - CLI will return tool_use blocks instead of executing
            args.extend([
                "--disallowedTools", "*",
            ])

        # The prompt itself
        args.append(prompt)

        return args

    def _execute_query(self, prompt: str) -> ProviderResponse:
        """Execute a query synchronously."""
        accumulated_text = ""
        function_calls: List[FunctionCall] = []
        finish_reason = FinishReason.STOP

        for msg in self._stream_cli_messages(prompt):
            if isinstance(msg, AssistantMessage):
                for block in msg.content_blocks:
                    if isinstance(block, TextBlock):
                        accumulated_text += block.text
                    elif isinstance(block, ToolUseBlock):
                        function_calls.append(FunctionCall(
                            id=block.id,
                            name=block.name,
                            args=block.input,
                        ))

            elif isinstance(msg, ResultMessage):
                self._last_result = msg
                if msg.usage:
                    self._last_usage = TokenUsage(
                        prompt_tokens=msg.usage.input_tokens,
                        output_tokens=msg.usage.output_tokens,
                        cache_read_tokens=msg.usage.cache_read_tokens,
                        cache_creation_tokens=msg.usage.cache_creation_tokens,
                    )
                if msg.is_error:
                    finish_reason = FinishReason.ERROR

        # Determine finish reason
        if function_calls:
            finish_reason = FinishReason.TOOL_USE

        # Add to history
        self._add_to_history(prompt, accumulated_text, function_calls)

        return ProviderResponse(
            text=accumulated_text,
            function_calls=function_calls if function_calls else None,
            finish_reason=finish_reason,
        )

    def _execute_query_streaming(
        self,
        prompt: str,
        on_chunk: StreamingCallback,
        cancel_token: Optional[CancelToken] = None,
        on_usage_update: Optional[UsageUpdateCallback] = None,
        on_function_call: Optional[FunctionCallDetectedCallback] = None,
    ) -> ProviderResponse:
        """Execute a query with streaming."""
        accumulated_text = ""
        function_calls: List[FunctionCall] = []
        finish_reason = FinishReason.STOP
        cancelled = False

        try:
            for msg in self._stream_cli_messages(prompt, cancel_token):
                # Check cancellation
                if cancel_token and cancel_token.is_cancelled:
                    cancelled = True
                    break

                if isinstance(msg, AssistantMessage):
                    for block in msg.content_blocks:
                        if isinstance(block, TextBlock):
                            accumulated_text += block.text
                            on_chunk(block.text)
                        elif isinstance(block, ToolUseBlock):
                            fc = FunctionCall(
                                id=block.id,
                                name=block.name,
                                args=block.input,
                            )
                            function_calls.append(fc)
                            if on_function_call:
                                on_function_call(fc)

                elif isinstance(msg, ResultMessage):
                    self._last_result = msg
                    if msg.usage:
                        self._last_usage = TokenUsage(
                            prompt_tokens=msg.usage.input_tokens,
                            output_tokens=msg.usage.output_tokens,
                            cache_read_tokens=msg.usage.cache_read_tokens,
                            cache_creation_tokens=msg.usage.cache_creation_tokens,
                        )
                        if on_usage_update:
                            on_usage_update(self._last_usage)

                    if msg.is_error:
                        finish_reason = FinishReason.ERROR

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            finish_reason = FinishReason.ERROR

        # Determine finish reason
        if cancelled:
            finish_reason = FinishReason.CANCELLED
        elif function_calls:
            finish_reason = FinishReason.TOOL_USE

        # Add to history (unless cancelled)
        if not cancelled:
            self._add_to_history(prompt, accumulated_text, function_calls)

        return ProviderResponse(
            text=accumulated_text,
            function_calls=function_calls if function_calls else None,
            finish_reason=finish_reason,
        )

    def _stream_cli_messages(
        self,
        prompt: str,
        cancel_token: Optional[CancelToken] = None,
    ) -> Iterator[CLIMessage]:
        """Stream messages from the CLI process."""
        args = self._build_cli_args(prompt)

        logger.debug(f"Spawning CLI: {' '.join(args[:5])}...")

        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
        )

        with self._process_lock:
            self._process = process

        try:
            # Read NDJSON lines from stdout
            for line in process.stdout:
                # Check cancellation
                if cancel_token and cancel_token.is_cancelled:
                    logger.debug("Cancellation requested, terminating CLI")
                    process.terminate()
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    msg = parse_ndjson_line(line)
                    yield msg

                    # Stop on result message
                    if isinstance(msg, ResultMessage):
                        break

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse NDJSON line: {line[:100]}...")
                    continue
                except ValueError as e:
                    logger.warning(f"Unknown message type: {e}")
                    continue

            # Wait for process to complete
            process.wait(timeout=5)

            if process.returncode != 0 and not (cancel_token and cancel_token.is_cancelled):
                stderr = process.stderr.read() if process.stderr else ""
                logger.error(f"CLI exited with code {process.returncode}: {stderr}")

        except subprocess.TimeoutExpired:
            logger.warning("CLI process timed out, killing")
            process.kill()
        finally:
            with self._process_lock:
                self._process = None

    def _terminate_process(self) -> None:
        """Terminate any running CLI process."""
        with self._process_lock:
            if self._process:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=2)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
                self._process = None

    def _add_to_history(
        self,
        user_message: str,
        assistant_text: str,
        function_calls: List[FunctionCall],
    ) -> None:
        """Add messages to conversation history."""
        # Add user message
        self._history.append(Message(
            role=Role.USER,
            parts=[Part.from_text(user_message)],
        ))

        # Add assistant response
        parts: List[Part] = []
        if assistant_text:
            parts.append(Part.from_text(assistant_text))
        for fc in function_calls:
            parts.append(Part.from_function_call(fc))

        if parts:
            self._history.append(Message(
                role=Role.MODEL,
                parts=parts,
            ))

    def _format_tool_results_as_message(self, results: List[ToolResult]) -> str:
        """Format tool results as a user message for passthrough mode.

        In passthrough mode, we need to send tool results back to the model.
        This formats them in a way the model can understand.
        """
        result_parts = []
        for result in results:
            if result.is_error:
                result_parts.append(
                    f"Tool '{result.name}' (call_id: {result.call_id}) failed:\n"
                    f"{result.result}"
                )
            else:
                result_parts.append(
                    f"Tool '{result.name}' (call_id: {result.call_id}) returned:\n"
                    f"{result.result}"
                )
        return "\n\n".join(result_parts)
