"""Claude CLI Model Provider implementation.

This provider wraps the Claude Code CLI, communicating via the stream-json
protocol for programmatic access to Claude's agentic capabilities.
"""

import json
import logging
import os
import subprocess
import threading
import uuid
from datetime import datetime
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
    StreamEvent,
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

    # CLI built-in tools that must be blocked in passthrough mode
    # These are disallowed via --disallowed-tools and documented in the system prompt
    CLI_BUILTIN_TOOLS = [
        "Task", "TaskOutput", "Bash", "Glob", "Grep", "ExitPlanMode",
        "Read", "Edit", "Write", "NotebookEdit", "WebFetch", "TodoWrite",
        "WebSearch", "KillShell", "AskUserQuestion", "Skill", "EnterPlanMode",
    ]

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

        # Workspace root for CLI working directory
        self._workspace_root: Optional[str] = None

        # CLI session ID for multi-turn conversations
        # Captured from SystemMessage, used with --resume for subsequent calls
        self._cli_session_id: Optional[str] = None

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        trace_path = os.environ.get('JAATO_PROVIDER_TRACE')
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] [claude_cli] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass

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
                - workspace_root: Working directory for CLI (auto-detected if not provided)

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

        # Detect workspace root for CLI working directory
        self._workspace_root = self._detect_workspace_root(extra.get("workspace_root"))

        logger.info(
            f"Initialized Claude CLI provider: path={self._cli_path}, "
            f"mode={self._mode.value}, workspace={self._workspace_root}"
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
            tools: List of available tools (used in passthrough mode only).
            history: Previous conversation history to restore.
        """
        # Terminate any existing process
        self._terminate_process()

        self._system_instruction = system_instruction
        self._history = list(history) if history else []
        self._session_id = str(uuid.uuid4())[:8]

        # Log what we received
        self._trace(f"create_session: mode={self._mode.value}, tools_received={len(tools) if tools else 0}, "
                    f"sys_instr_len={len(system_instruction) if system_instruction else 0}")

        # Only reset CLI session ID when starting a truly new conversation (no history)
        # Preserve it when updating with existing history (e.g., tool refresh, history update)
        if not history:
            self._trace(f"create_session: new conversation, resetting cli_session_id")
            self._cli_session_id = None
        else:
            self._trace(f"create_session: preserving cli_session_id={self._cli_session_id}, history_len={len(history)}")
        self._last_usage = None
        self._last_result = None

        # In delegated mode, CLI uses its own built-in tools - ignore external tools
        if self._mode == CLIMode.DELEGATED:
            if tools:
                logger.debug(
                    f"Ignoring {len(tools)} external tools in delegated mode - "
                    "CLI handles tool execution with its own tools"
                )
            self._tools = []
        else:
            self._tools = tools or []

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
                parts=[],
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
            return ProviderResponse(parts=[], finish_reason=FinishReason.STOP)

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

    def uses_external_tools(self) -> bool:
        """Check if this provider uses external tool plugins.

        In delegated mode, the CLI manages its own built-in tools (Read, Write,
        Bash, etc.) and any MCP servers it discovers. External tool plugins
        from jaato should NOT be configured.

        In passthrough mode, the CLI is used only for model access and jaato
        handles tool execution, so external tools ARE needed.

        Returns:
            True if external tools should be configured, False otherwise.
        """
        return self._mode == CLIMode.PASSTHROUGH

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

    def _detect_workspace_root(self, config_value: Optional[str] = None) -> Optional[str]:
        """Detect workspace root from config or environment.

        Priority:
        1. Explicit config value
        2. JAATO_WORKSPACE_ROOT environment variable
        3. workspaceRoot environment variable (from .env file)

        Args:
            config_value: Explicit workspace root from config.

        Returns:
            Resolved absolute path to workspace root, or None if not found.
        """
        # Priority 1: Explicit config value
        if config_value:
            resolved = os.path.realpath(os.path.abspath(config_value))
            logger.debug(f"Using workspace_root from config: {resolved}")
            return resolved

        # Priority 2: JAATO_WORKSPACE_ROOT environment variable
        workspace = os.environ.get('JAATO_WORKSPACE_ROOT')
        if workspace:
            resolved = os.path.realpath(os.path.abspath(workspace))
            logger.debug(f"Using JAATO_WORKSPACE_ROOT: {resolved}")
            return resolved

        # Priority 3: workspaceRoot from .env
        workspace = os.environ.get('workspaceRoot')
        if workspace:
            resolved = os.path.realpath(os.path.abspath(workspace))
            logger.debug(f"Using workspaceRoot: {resolved}")
            return resolved

        logger.debug("No workspace root configured, using current directory")
        return None

    def _format_tools_for_prompt(self) -> str:
        """Format jaato's tool schemas as text for the system prompt.

        Used in passthrough mode to expose jaato's tools to the CLI model.
        The model will generate tool_use blocks that jaato then executes.

        Returns:
            Tool descriptions formatted for inclusion in system prompt.
        """
        if not self._tools:
            return ""

        # Use class constant for blocked tools list
        blocked_tools_list = ", ".join(self.CLI_BUILTIN_TOOLS)

        lines = [
            "",
            "# IMPORTANT: Tool Execution Mode",
            "",
            "You are running in **passthrough mode** where tool execution is handled by an external framework (jaato).",
            "",
            "**CRITICAL INSTRUCTIONS:**",
            "- ONLY use the tools listed below in the 'Available Tools' section",
            f"- DO NOT use any of Claude Code CLI's built-in tools: {blocked_tools_list}",
            "- When you need to perform file operations, shell commands, or other actions, use the equivalent tools from the 'Available Tools' list below",
            "",
            "**TOOL EXECUTION BEHAVIOR:**",
            "- When you call a tool, the jaato framework will execute it and provide you with the result",
            "- DO NOT assume or hallucinate that a tool is unavailable or failed before receiving a tool result",
            "- Simply call the tool and wait for the actual result from the framework",
            "- The framework handles all tool execution - you will always receive a proper tool_result response",
            "- When you need to perform multiple independent operations toward DIFFERENT goals, issue all tool calls in a single response",
            "- Never parallelize alternative approaches to the SAME goal - try the preferred tool first and wait for results before falling back",
            "- Do not rush to call alternative tools in parallel when one tool can achieve the goal - this avoids unnecessary permission prompts for the user",
            "",
            "**IGNORING INTERMEDIATE CLI ERRORS:**",
            "- You may receive errors like 'No such tool available: <tool_name>' from the Claude CLI layer",
            "- If the tool IS listed in your 'Available Tools' section below, IGNORE this CLI error",
            "- The CLI does not know about jaato's tools - this is a false negative",
            "- WAIT for the actual jaato framework response (permission request or execution result)",
            "- DO NOT respond to the user until you receive jaato's final resolution",
            "- Only treat a tool as unavailable if it is NOT in your 'Available Tools' section",
            "",
            "# Available Tools",
            "",
            "You have access to the following tools. Use them by generating tool_use blocks:",
            "",
        ]

        for tool in self._tools:
            lines.append(f"## {tool.name}")
            if tool.description:
                lines.append(f"{tool.description}")
            if tool.parameters:
                lines.append("")
                lines.append("Parameters:")
                params = tool.parameters.get("properties", {})
                required = tool.parameters.get("required", [])
                for param_name, param_info in params.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    req_marker = " (required)" if param_name in required else ""
                    lines.append(f"  - {param_name}: {param_type}{req_marker}")
                    if param_desc:
                        lines.append(f"    {param_desc}")
            lines.append("")

        return "\n".join(lines)

    def _build_cli_args(self, prompt: str) -> List[str]:
        """Build CLI command arguments."""
        self._trace(f"_build_cli_args: cli_session_id={self._cli_session_id}")

        args = [
            self._cli_path,
            "--print",  # Non-interactive mode
            "--output-format", "stream-json",  # NDJSON output
            "--verbose",  # Required for stream-json
            "--include-partial-messages",  # Enable token-level streaming
        ]

        # Resume existing CLI session for multi-turn conversation
        if self._cli_session_id:
            self._trace(f"_build_cli_args: using --resume {self._cli_session_id}")
            args.extend(["--resume", self._cli_session_id])
        else:
            # First message only - configure session-level settings
            # Model selection
            if self._model_name:
                args.extend(["--model", self._model_name])

            # Max turns
            if self._max_turns is not None:
                args.extend(["--max-turns", str(self._max_turns)])

            # Permission mode
            if self._permission_mode:
                args.extend(["--permission-mode", self._permission_mode])

        # Tool configuration must be sent on EVERY invocation
        # CLI's --resume does not preserve these settings
        if self._mode == CLIMode.PASSTHROUGH:
            # Disallow CLI built-in tools - jaato provides its own tools
            disallowed = ",".join(self.CLI_BUILTIN_TOOLS)
            args.extend(["--disallowed-tools", disallowed])
            self._trace(f"_build_cli_args: disallowed-tools={disallowed}")

            # Allow jaato's tools so CLI recognizes them
            if self._tools:
                allowed = ",".join(tool.name for tool in self._tools)
                args.extend(["--allowed-tools", allowed])
                self._trace(f"_build_cli_args: allowed-tools count={len(self._tools)}")

            # Build system prompt with tool schemas
            base_prompt_len = len(self._system_instruction) if self._system_instruction else 0
            system_prompt = self._system_instruction or ""
            tool_prompt = self._format_tools_for_prompt()
            self._trace(f"_build_cli_args: passthrough mode, {len(self._tools)} tools, "
                       f"base_prompt_len={base_prompt_len}, tool_prompt_len={len(tool_prompt) if tool_prompt else 0}")
            if tool_prompt:
                system_prompt = f"{system_prompt}\n\n{tool_prompt}" if system_prompt else tool_prompt
            combined_len = len(system_prompt) if system_prompt else 0
            self._trace(f"_build_cli_args: combined_system_prompt_len={combined_len}")
            if system_prompt:
                args.extend(["--system-prompt", system_prompt])

        # Non-passthrough mode: just use the system instruction (also every time)
        elif self._system_instruction:
            args.extend(["--append-system-prompt", self._system_instruction])

        # Use -- to separate options from the prompt
        args.append("--")

        # The prompt itself
        args.append(prompt)

        # Log the CLI flags being used (not the full content)
        flags_used = [a for a in args if a.startswith("--")]
        self._trace(f"_build_cli_args: flags={flags_used}")

        return args

    def _execute_query(self, prompt: str) -> ProviderResponse:
        """Execute a query synchronously."""
        accumulated_text = ""
        function_calls: List[FunctionCall] = []
        finish_reason = FinishReason.STOP

        for msg in self._stream_cli_messages(prompt):
            if isinstance(msg, SystemMessage):
                # Capture CLI session ID for multi-turn conversation
                if msg.session_id and not self._cli_session_id:
                    self._cli_session_id = msg.session_id
                    logger.debug(f"Captured CLI session ID: {self._cli_session_id}")

            elif isinstance(msg, AssistantMessage):
                for block in msg.content_blocks:
                    if isinstance(block, TextBlock):
                        accumulated_text += block.text
                    elif isinstance(block, ToolUseBlock):
                        # Only return function calls in passthrough mode
                        # In delegated mode, CLI handles tool execution internally
                        if self._mode == CLIMode.PASSTHROUGH:
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

        # Build parts list
        parts: List[Part] = []
        if accumulated_text:
            parts.append(Part.from_text(accumulated_text))
        for fc in function_calls:
            parts.append(Part.from_function_call(fc))

        return ProviderResponse(
            parts=parts,
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

        got_stream_events = False  # Track if we got streaming deltas

        try:
            for msg in self._stream_cli_messages(prompt, cancel_token):
                self._trace(f"Got message type: {type(msg).__name__}")

                # Check cancellation
                if cancel_token and cancel_token.is_cancelled:
                    cancelled = True
                    break

                if isinstance(msg, SystemMessage):
                    # Capture CLI session ID for multi-turn conversation
                    self._trace(f"Got SystemMessage: session_id={msg.session_id}, current_cli_session_id={self._cli_session_id}")
                    if msg.session_id and not self._cli_session_id:
                        self._cli_session_id = msg.session_id
                        self._trace(f"Captured CLI session ID: {self._cli_session_id}")

                elif isinstance(msg, StreamEvent):
                    # Handle streaming text deltas
                    if msg.is_text_delta and msg.delta_text:
                        got_stream_events = True
                        accumulated_text += msg.delta_text
                        on_chunk(msg.delta_text)

                elif isinstance(msg, AssistantMessage):
                    # If we got stream events, don't double-count text
                    # Only process tool use blocks from the final message
                    if not got_stream_events:
                        for block in msg.content_blocks:
                            if isinstance(block, TextBlock):
                                accumulated_text += block.text
                                on_chunk(block.text)

                    # Process tool use blocks only in passthrough mode
                    # In delegated mode, CLI handles tool execution internally
                    if self._mode == CLIMode.PASSTHROUGH:
                        for block in msg.content_blocks:
                            if isinstance(block, ToolUseBlock):
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

        # Build parts list
        parts: List[Part] = []
        if accumulated_text:
            parts.append(Part.from_text(accumulated_text))
        for fc in function_calls:
            parts.append(Part.from_function_call(fc))

        return ProviderResponse(
            parts=parts,
            finish_reason=finish_reason,
        )

    def _stream_cli_messages(
        self,
        prompt: str,
        cancel_token: Optional[CancelToken] = None,
    ) -> Iterator[CLIMessage]:
        """Stream messages from the CLI process."""
        args = self._build_cli_args(prompt)

        # Determine working directory for CLI process
        # Use workspace_root if set, otherwise try to detect from env at runtime
        cli_cwd = self._workspace_root
        if not cli_cwd:
            # Try runtime detection in case env was loaded after initialize()
            cli_cwd = self._detect_workspace_root()
        if not cli_cwd:
            # Final fallback: use current directory
            cli_cwd = os.getcwd()

        logger.debug(f"Spawning CLI: {' '.join(args[:5])}... cwd={cli_cwd}")

        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            cwd=cli_cwd,  # Use workspace root as working directory
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
