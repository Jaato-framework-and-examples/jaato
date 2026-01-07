#!/usr/bin/env python3
"""Rich TUI client with sticky plan display.

This client provides a terminal UI experience with:
- Sticky plan panel at the top showing current plan status
- Scrolling output panel below for model responses and tool output
- Full-screen alternate buffer for immersive experience

Supports two modes via Backend abstraction:
- Direct mode: Local JaatoClient (non-IPC)
- IPC mode: Connection to jaato server daemon

Requires an interactive TTY. For non-TTY environments, use simple-client.
"""

import asyncio
import os
import sys
import pathlib
import tempfile
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# Add project root to path for imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Add simple-client to path for reusable components
SIMPLE_CLIENT = ROOT / "simple-client"
if str(SIMPLE_CLIENT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_CLIENT))

from dotenv import load_dotenv

from shared import (
    JaatoClient,
    TokenLedger,
    PluginRegistry,
    PermissionPlugin,
    TodoPlugin,
    active_cert_bundle,
)
from shared.plugins.session import create_plugin as create_session_plugin, load_session_config
from shared.plugins.base import parse_command_args
from shared.plugins.gc import load_gc_from_file
from shared.plugins.code_validation_formatter import create_plugin as create_code_validation_formatter

# Reuse input handling from simple-client
from input_handler import InputHandler

# Rich TUI components
from pt_display import PTDisplay
from plan_reporter import create_live_reporter
from agent_registry import AgentRegistry
from keybindings import load_keybindings, detect_terminal, list_available_profiles

# Backend abstraction for mode-agnostic operation
from backend import Backend, DirectBackend, IPCBackend


class RichClient:
    """Rich TUI client with sticky plan display.

    Uses PTDisplay (prompt_toolkit-based) to manage a full-screen layout with:
    - Sticky plan panel at top (hidden when no plan)
    - Scrolling output below
    - Integrated input prompt at bottom

    The plan panel updates in-place as plan steps progress,
    while model output scrolls naturally below.
    """

    def __init__(
        self,
        env_file: str = ".env",
        verbose: bool = True,
        provider: Optional[str] = None,
        backend: Optional[Backend] = None,
    ):
        self.verbose = verbose
        self.env_file = env_file
        self._provider = provider  # CLI override for provider

        # Backend abstraction - supports both direct and IPC modes
        self._backend: Optional[Backend] = backend
        self._jaato: Optional[JaatoClient] = None  # For direct mode compatibility
        self._is_ipc_mode = backend is not None and isinstance(backend, IPCBackend)

        # Async event loop for backend calls from threads
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None

        self.registry: Optional[PluginRegistry] = None
        self.permission_plugin: Optional[PermissionPlugin] = None
        self.todo_plugin: Optional[TodoPlugin] = None
        self.ledger = TokenLedger()

        # Agent registry for tracking agents and their state
        self._agent_registry = AgentRegistry()

        # Rich TUI display (prompt_toolkit-based)
        self._display: Optional[PTDisplay] = None

        # Input handler (for file expansion, history, completions)
        self._input_handler = InputHandler()

        # Track original inputs for session export
        self._original_inputs: list[dict] = []

        # Flag to signal exit from input loop
        self._should_exit = False

        # Queue for permission/clarification input routing
        import queue
        self._channel_input_queue: queue.Queue[str] = queue.Queue()
        self._waiting_for_channel_input: bool = False
        # Pending response options from permission plugin (single source of truth)
        self._pending_response_options: Optional[list] = None

        # Background model thread tracking
        self._model_thread: Optional[threading.Thread] = None
        self._model_running: bool = False

        # Model info for status bar
        self._model_provider: str = ""
        self._model_name: str = ""

        # GC info for status bar (set during initialization if GC is configured)
        self._gc_threshold: Optional[float] = None
        self._gc_strategy: Optional[str] = None

        # UI hooks reference for signaling agent status changes
        self._ui_hooks: Optional[Any] = None

    def log(self, msg: str) -> None:
        """Log message to output panel."""
        if self.verbose and self._display:
            self._display.add_system_message(msg, style="cyan")

    def _run_async(self, coro):
        """Run an async coroutine from sync context.

        Uses the stored event loop if available, otherwise creates a new one.
        Handles being called from within an already-running event loop (e.g., prompt_toolkit).
        """
        if self._async_loop and self._async_loop.is_running():
            # Submit to running loop from another thread
            future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
            return future.result(timeout=30)
        else:
            # Check if we're inside a running event loop (e.g., prompt_toolkit)
            try:
                asyncio.get_running_loop()
                # We're inside a running loop - run coroutine in a separate thread
                # with its own event loop. We need to create a new loop in that thread.
                import threading
                result = None
                exception = None

                def run_in_thread():
                    nonlocal result, exception
                    try:
                        # Create a new event loop for this thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(coro)
                        finally:
                            loop.close()
                    except Exception as e:
                        exception = e

                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join(timeout=30)

                if exception:
                    raise exception
                return result
            except RuntimeError:
                # No running loop, create one
                return asyncio.run(coro)

    def _create_output_callback(self,
                                  suppress_sources: Optional[set] = None,
                                  force_display: bool = False) -> Callable[[str, str, str], None]:
        """Create callback for real-time output to display.

        Args:
            suppress_sources: Set of source names to suppress (e.g., {"permission"})
            force_display: If True, always display output even when agent_registry is active.
                          Use this for user commands that don't go through agent hooks.
        """
        suppress = suppress_sources or set()

        def callback(source: str, text: str, mode: str) -> None:
            if self._display:
                # Skip suppressed sources (e.g., permission output shown in tool tree)
                if source in suppress:
                    return
                # Note: Spinner is NOT stopped here. It remains active during the
                # entire turn to show "thinking..." when model pauses between chunks.
                # The spinner is stopped when status changes to "done".
                # Skip ALL sources when UI hooks are active - the hooks handle
                # routing all output (model, system, plugin) to the correct buffer
                # via on_agent_output. Without this, output gets duplicated.
                # Exception: force_display=True bypasses this for user commands.
                if self._agent_registry and not force_display:
                    return
                self._display.append_output(source, text, mode)
        return callback

    def _try_execute_plugin_command(self, user_input: str) -> Optional[Any]:
        """Try to execute user input as a plugin-provided command."""
        if not self._backend:
            return None

        user_commands = self._run_async(self._backend.get_user_commands())
        if not user_commands:
            return None

        parts = user_input.strip().split(maxsplit=1)
        input_cmd = parts[0].lower() if parts else ""
        raw_args = parts[1] if len(parts) > 1 else ""

        command = None
        for cmd_name, cmd in user_commands.items():
            if input_cmd == cmd_name.lower():
                command = cmd
                break

        if not command:
            return None

        args = parse_command_args(command, raw_args)

        if command.name.lower() == "save":
            args["user_inputs"] = self._original_inputs.copy()

        # Find the plugin that provides this command and set output callback
        plugin = None
        if self.registry:
            for plugin_name in self.registry.list_exposed():
                p = self.registry.get_plugin(plugin_name)
                if p and hasattr(p, 'get_user_commands'):
                    for cmd in p.get_user_commands():
                        if cmd.name == command.name:
                            plugin = p
                            break
                if plugin:
                    break

        # Set output callback on plugin if it supports it
        # Use force_display=True since user commands don't go through agent hooks
        if plugin and hasattr(plugin, 'set_output_callback') and self._display:
            output_callback = self._create_output_callback(force_display=True)
            plugin.set_output_callback(output_callback)

        try:
            result, shared = self._run_async(
                self._backend.execute_user_command(command.name, args)
            )
            self._display_command_result(command.name, result, shared)

            # Update status bar if model was changed
            if command.name.lower() == "model" and isinstance(result, dict):
                if result.get("success") and result.get("current_model"):
                    self._model_name = result["current_model"]
                    if self._display:
                        self._display.set_model_info(self._model_provider, self._model_name)

            if command.name.lower() == "resume" and isinstance(result, dict):
                user_inputs = result.get("user_inputs", [])
                if user_inputs:
                    self._restore_user_inputs(user_inputs)

            return result

        except Exception as e:
            if self._display:
                self._display.show_lines([(f"Error: {e}", "red")])
            return {"error": str(e)}

        finally:
            # Clear output callback after execution
            if plugin and hasattr(plugin, 'set_output_callback'):
                plugin.set_output_callback(None)

    def _display_command_result(
        self,
        command_name: str,
        result: Any,
        shared: bool
    ) -> None:
        """Display command result in output panel."""
        if not self._display:
            return

        # For plan command, the sticky panel handles display
        if command_name == "plan":
            return

        # Skip pager for empty results (command handles its own output via callback)
        if result is None or result == "":
            return

        lines = [(f"[{command_name}]", "bold")]

        if isinstance(result, dict):
            for key, value in result.items():
                if not key.startswith('_'):
                    lines.append((f"  {key}: {value}", "dim"))
        else:
            # Handle multi-line strings by splitting on newlines
            result_str = str(result)
            if '\n' in result_str:
                for line in result_str.split('\n'):
                    lines.append((f"  {line}", "dim"))
            else:
                lines.append((f"  {result_str}", "dim"))

        if shared:
            lines.append(("  [Result shared with model]", "dim cyan"))

        self._display.show_lines(lines)

    def _restore_user_inputs(self, user_inputs: List[str]) -> None:
        """Restore user inputs to prompt history after session resume."""
        self._original_inputs = list(user_inputs)
        count = self._input_handler.restore_history(
            [entry["text"] if isinstance(entry, dict) else entry for entry in user_inputs]
        )
        if count:
            self.log(f"Restored {count} inputs to prompt history")

    def initialize(self) -> bool:
        """Initialize the client.

        For direct mode: Creates JaatoClient and DirectBackend.
        For IPC mode: Uses provided backend (server handles plugins).
        """
        # IPC mode - backend already provided, minimal local setup
        if self._is_ipc_mode:
            return self._initialize_ipc_mode()

        # Direct mode - full local initialization
        return self._initialize_direct_mode()

    def _initialize_ipc_mode(self) -> bool:
        """Initialize for IPC mode (server connection)."""
        # In IPC mode, the server handles:
        # - Model connection
        # - Plugin registry
        # - Permission handling
        # - Tool configuration
        # - Session management

        # We just need to set up local UI components
        self._model_name = self._backend.model_name
        self._model_provider = self._backend.provider_name

        return True

    def _initialize_direct_mode(self) -> bool:
        """Initialize for direct mode (local JaatoClient)."""
        # Load environment from CWD or explicit --env-file path
        load_dotenv(self.env_file)

        # Check CA bundle
        active_bundle = active_cert_bundle(verbose=False)

        # Check required vars - MODEL_NAME always required
        model_name = os.environ.get("MODEL_NAME")
        if not model_name:
            print("Error: Missing required environment variable: MODEL_NAME")
            return False

        # Check auth method: API key (AI Studio) or Vertex AI
        api_key = os.environ.get("GOOGLE_GENAI_API_KEY")
        project_id = os.environ.get("PROJECT_ID")
        location = os.environ.get("LOCATION")

        if not api_key and (not project_id or not location):
            print("Error: Set GOOGLE_GENAI_API_KEY for AI Studio, or PROJECT_ID and LOCATION for Vertex AI")
            return False

        # Initialize JaatoClient with optional provider override
        try:
            self._jaato = JaatoClient(provider_name=self._provider)
            if api_key:
                # AI Studio mode - just need model
                self._jaato.connect(model=model_name)
            else:
                # Vertex AI mode
                self._jaato.connect(project_id, location, model_name)
        except Exception as e:
            print(f"Error: Failed to connect: {e}")
            return False

        # Create DirectBackend wrapping the JaatoClient
        self._backend = DirectBackend(self._jaato)

        # Store model info for status bar (from jaato client)
        self._model_name = self._jaato.model_name or model_name
        self._model_provider = self._jaato.provider_name

        # Initialize plugin registry
        self.registry = PluginRegistry(model_name=model_name)
        self.registry.discover()

        # We'll configure the todo reporter after display is created
        # For now, use memory storage
        # Note: clarification and permission channels use "queue" type for TUI integration
        plugin_configs = {
            "todo": {
                "reporter_type": "console",  # Temporary, will be replaced
                "storage_type": "memory",
            },
            "references": {
                "channel_type": "queue",
                # Callbacks will be set after display is created
            },
            "clarification": {
                "channel_type": "queue",
                # Callbacks will be set after display is created
            },
        }
        self.registry.expose_all(plugin_configs)
        self.todo_plugin = self.registry.get_plugin("todo")

        # Wire up artifact tracker with registry for cross-plugin access (LSP integration)
        artifact_tracker = self.registry.get_plugin("artifact_tracker")
        if artifact_tracker and hasattr(artifact_tracker, 'set_plugin_registry'):
            artifact_tracker.set_plugin_registry(self.registry)
            self.log(f"[plugin] artifact_tracker wired with registry (LSP integration enabled)")
        else:
            self.log(f"[plugin] artifact_tracker not found or missing set_plugin_registry - LSP integration disabled")

        # Initialize permission plugin with queue channel for TUI integration
        self.permission_plugin = PermissionPlugin()
        self.permission_plugin.initialize({
            "channel_type": "queue",
            "channel_config": {
                "use_colors": True,  # Enable ANSI colors for diff coloring
            },
            "policy": {
                "defaultPolicy": "ask",
                "whitelist": {"tools": [], "patterns": []},
                "blacklist": {"tools": [], "patterns": []},
            }
        })

        # Configure tools
        self._backend.configure_tools(self.registry, self.permission_plugin, self.ledger)

        # Load GC configuration from .jaato/gc.json if present
        gc_result = load_gc_from_file(agent_name="main")
        if gc_result:
            gc_plugin, gc_config = gc_result
            self._backend.set_gc_plugin(gc_plugin, gc_config)
            # Store threshold and strategy for status bar display
            self._gc_threshold = gc_config.threshold_percent
            # Get strategy name from plugin (e.g., "gc_truncate" -> "truncate")
            plugin_name = getattr(gc_plugin, 'name', 'gc')
            self._gc_strategy = plugin_name.replace('gc_', '') if plugin_name.startswith('gc_') else plugin_name

        # Setup session plugin
        self._setup_session_plugin()

        # Register plugin commands for completion
        self._register_plugin_commands()

        return True

    def _setup_live_reporter(self) -> None:
        """Set up the live plan reporter after display is created."""
        if not self.todo_plugin or not self._display:
            return

        # Create callbacks that map agent_name to registry agent_id
        # The reporter passes (plan_data, agent_name) where agent_name is the
        # TodoPlugin's _agent_name (e.g., None for main, "skill-code-020..." for subagent)
        #
        # Registry agent_id format:
        #   - Main agent: "main"
        #   - Subagents: "subagent_N" (e.g., "subagent_1", "subagent_2")
        #
        # We need to look up the actual agent_id by name since subagent IDs
        # don't follow a predictable pattern from the profile name.
        def update_callback(plan_data, agent_name):
            if agent_name is None or agent_name == "main":
                target_id = "main"
            else:
                # Look up the agent_id by profile name
                target_id = self._agent_registry.find_agent_id_by_name(agent_name)
                if not target_id:
                    # Fallback: try direct match (shouldn't happen normally)
                    self._trace(f"Plan update: agent_name '{agent_name}' not found in registry")
                    target_id = agent_name
            self._display.update_plan(plan_data, agent_id=target_id)

        def clear_callback(agent_name):
            if agent_name is None or agent_name == "main":
                target_id = "main"
            else:
                target_id = self._agent_registry.find_agent_id_by_name(agent_name)
                if not target_id:
                    target_id = agent_name
            self._display.clear_plan(agent_id=target_id)

        # Create live reporter with callbacks to display
        live_reporter = create_live_reporter(
            update_callback=update_callback,
            clear_callback=clear_callback,
            output_callback=self._create_output_callback(),
        )

        # Replace the todo plugin's reporter
        if hasattr(self.todo_plugin, '_reporter'):
            self.todo_plugin._reporter = live_reporter

        # Also set reporter on subagent plugin so subagent TodoPlugins use it
        if self.registry:
            subagent_plugin = self.registry.get_plugin("subagent")
            if subagent_plugin and hasattr(subagent_plugin, 'set_plan_reporter'):
                subagent_plugin.set_plan_reporter(live_reporter)
                self._trace("Plan reporter configured for subagent plugin")

    def _trace(self, msg: str) -> None:
        """Write trace message to file for debugging."""
        import datetime
        trace_path = os.environ.get("JAATO_TRACE_LOG")
        if not trace_path:
            return  # Tracing disabled
        with open(trace_path, "a") as f:
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[{ts}] {msg}\n")
            f.flush()

    def _setup_queue_channels(self) -> None:
        """Set up queue-based channels for permission, clarification, and references.

        Queue channels display prompts in the output panel and receive user
        input via a shared queue. This avoids terminal mode switching issues
        that occur when the model runs in a background thread.
        """
        if not self._display:
            return

        def on_prompt_state_change(waiting: bool):
            """Called when channel starts/stops waiting for input."""
            self._waiting_for_channel_input = waiting
            self._trace(f"prompt_callback: waiting={waiting}")
            if self._display:
                # Pass the stored response options (from permission plugin)
                # This makes the permission plugin the single source of truth
                # for valid completion options
                response_options = self._pending_response_options if waiting else None
                self._display.set_waiting_for_channel_input(waiting, response_options)
                if waiting:
                    # Channel waiting for user input - stop spinner
                    self._display.stop_spinner()
                else:
                    # Channel finished - start spinner while model continues
                    self._display.start_spinner()

        # Create a cancel token provider that checks the session's current token
        def get_cancel_token():
            """Get the current cancel token from the session, if any."""
            if self._backend:
                session = self._backend.get_session()
                if session and hasattr(session, '_cancel_token'):
                    return session._cancel_token
            return None

        # Wrapper that acts like a CancelToken but checks the current session token
        class CancelTokenProxy:
            @property
            def is_cancelled(self):
                token = get_cancel_token()
                return token.is_cancelled if token else False

        cancel_token_proxy = CancelTokenProxy()

        # Set callbacks on clarification plugin channel
        if self.registry:
            clarification_plugin = self.registry.get_plugin("clarification")
            if clarification_plugin and hasattr(clarification_plugin, '_channel'):
                channel = clarification_plugin._channel
                if hasattr(channel, 'set_callbacks'):
                    channel.set_callbacks(
                        output_callback=self._create_output_callback(),
                        input_queue=self._channel_input_queue,
                        prompt_callback=on_prompt_state_change,
                        cancel_token=cancel_token_proxy,
                    )
                    self._trace("Clarification channel callbacks set (queue)")

            # Set callbacks on references plugin channel
            references_plugin = self.registry.get_plugin("references")
            if references_plugin and hasattr(references_plugin, '_channel'):
                channel = references_plugin._channel
                if hasattr(channel, 'set_callbacks'):
                    channel.set_callbacks(
                        output_callback=self._create_output_callback(),
                        input_queue=self._channel_input_queue,
                        prompt_callback=on_prompt_state_change,
                    )
                    self._trace("References channel callbacks set (queue)")

        # Set callbacks on permission plugin channel
        # Suppress "permission" source output since it's now shown in the tool tree
        if self.permission_plugin and hasattr(self.permission_plugin, '_channel'):
            channel = self.permission_plugin._channel
            self._trace(f"Permission channel type: {type(channel).__name__}")
            if channel and hasattr(channel, 'set_callbacks'):
                channel.set_callbacks(
                    output_callback=self._create_output_callback(suppress_sources={"permission"}),
                    input_queue=self._channel_input_queue,
                    prompt_callback=on_prompt_state_change,
                    cancel_token=cancel_token_proxy,
                )
                self._trace("Permission channel callbacks set (queue)")

    def _setup_retry_callback(self) -> None:
        """Set up retry callback to route rate limit messages to output panel.

        Routes retry notifications through the output callback system instead
        of printing directly to console, so they appear in the output panel.
        """
        if not self._backend:
            return

        # Create output callback for retry messages
        output_callback = self._create_output_callback()

        def on_retry(message: str, attempt: int, max_attempts: int, delay: float) -> None:
            """Route retry messages to output panel."""
            # Use source="retry" so UI can style appropriately
            output_callback("retry", message, "write")

        self._backend.set_retry_callback(on_retry)
        self._trace("Retry callback configured for output panel")

        # Also set callback on subagent plugin so subagent sessions use it
        if self.registry:
            subagent_plugin = self.registry.get_plugin("subagent")
            if subagent_plugin and hasattr(subagent_plugin, 'set_retry_callback'):
                subagent_plugin.set_retry_callback(on_retry)
                self._trace("Retry callback configured for subagent plugin")

    def _setup_code_validation_formatter(self) -> None:
        """Set up code validation formatter for LSP diagnostics on output code blocks.

        Creates the code validation formatter, wires it with the LSP plugin,
        and registers it with the display's formatter pipeline.

        Note: Registers the formatter even if no LSP servers are connected yet,
        since servers may connect asynchronously. The formatter checks dynamically.
        """
        self._trace("_setup_code_validation_formatter: starting")
        if not self._display or not self.registry:
            self._trace("_setup_code_validation_formatter: no display or registry")
            return

        # Get LSP plugin from registry
        lsp_plugin = self.registry.get_plugin("lsp")
        self._trace(f"_setup_code_validation_formatter: lsp_plugin={lsp_plugin is not None}")
        if not lsp_plugin:
            self._trace("Code validation formatter: LSP plugin not available")
            return

        # Create code validation formatter (register regardless of current LSP state)
        code_validator = create_code_validation_formatter()
        code_validator.set_lsp_plugin(lsp_plugin)
        code_validator.initialize({
            "enabled": True,
            "max_errors_per_block": 5,
            "max_warnings_per_block": 3,
        })
        self._trace(f"_setup_code_validation_formatter: code_validator created, name={code_validator.name}, priority={code_validator.priority}")

        # Set up feedback callback for model self-correction
        # When validation issues are found, inject them into the conversation
        def on_validation_feedback(feedback: str) -> None:
            """Inject validation feedback into the conversation."""
            if self._display and feedback:
                # Show feedback in output panel as a system message
                self._display.add_system_message(
                    f"[Code Validation] Issues detected in output code blocks",
                    style="yellow"
                )
                self._trace(f"Code validation feedback: {len(feedback)} chars")

        code_validator.set_feedback_callback(on_validation_feedback)

        # Register with display's formatter pipeline
        self._display.register_formatter(code_validator)

        # Log current state (servers may connect later)
        connected_servers = getattr(lsp_plugin, '_connected_servers', set())
        self._trace(f"Code validation formatter registered (current LSP servers: {connected_servers or 'none yet'})")

        # Store reference for debugging
        self._code_validator = code_validator

        # Show visible feedback about code validation status
        if connected_servers:
            self.log(f"[plugin] Code validation enabled for: {', '.join(connected_servers)}")
        else:
            self.log(f"[plugin] Code validation formatter registered (LSP servers will be checked dynamically)")

    def _setup_agent_hooks(self) -> None:
        """Set up agent lifecycle hooks for UI integration."""
        if not self._backend or not self._agent_registry:
            return

        # Import the protocol
        from shared.plugins.subagent.ui_hooks import AgentUIHooks

        # Create hooks implementation
        registry = self._agent_registry
        display = self._display
        trace_fn = self._trace  # Capture trace function for use in hooks

        class RichClientHooks:
            """UI hooks implementation for rich client."""

            def on_agent_created(self, agent_id, agent_name, agent_type, profile_name,
                               parent_agent_id, icon_lines, created_at):
                registry.create_agent(
                    agent_id=agent_id,
                    name=agent_name,
                    agent_type=agent_type,
                    profile_name=profile_name,
                    parent_agent_id=parent_agent_id,
                    icon_lines=icon_lines,
                    created_at=created_at
                )

            def on_agent_output(self, agent_id, source, text, mode):
                buffer = registry.get_buffer(agent_id)
                if buffer:
                    # Note: We do NOT stop the spinner on model output.
                    # The spinner should remain active during the entire turn
                    # to show "thinking..." when the model pauses between chunks.
                    # The spinner is stopped when status changes to "done".
                    buffer.append(source, text, mode)
                    # Auto-scroll to bottom and refresh display
                    buffer.scroll_to_bottom()
                    if display:
                        display.refresh()

            def on_agent_status_changed(self, agent_id, status, error=None):
                registry.update_status(agent_id, status)
                # Auto-select the active agent so status bar shows its context
                if status == "active":
                    registry.select_agent(agent_id)
                # Start/stop spinner for this agent's buffer based on status
                buffer = registry.get_buffer(agent_id)
                if buffer:
                    if status == "active":
                        buffer.start_spinner()
                        if display:
                            # Ensure the spinner animation timer is running
                            display.ensure_spinner_timer_running()
                            display.refresh()
                    elif status in ("done", "error"):
                        buffer.stop_spinner()
                        if display:
                            display.refresh()

            def on_agent_completed(self, agent_id, completed_at, success,
                                  token_usage=None, turns_used=None):
                registry.mark_completed(agent_id, completed_at)

            def on_agent_turn_completed(self, agent_id, turn_number, prompt_tokens,
                                       output_tokens, total_tokens, duration_seconds,
                                       function_calls):
                registry.update_turn_accounting(
                    agent_id, turn_number, prompt_tokens, output_tokens,
                    total_tokens, duration_seconds, function_calls
                )

            def on_agent_context_updated(self, agent_id, total_tokens, prompt_tokens,
                                        output_tokens, turns, percent_used):
                registry.update_context_usage(
                    agent_id, total_tokens, prompt_tokens,
                    output_tokens, turns, percent_used
                )

            def on_agent_gc_config(self, agent_id, threshold, strategy):
                trace_fn(f"[on_agent_gc_config] agent_id={agent_id}, threshold={threshold}, strategy={strategy}")
                registry.update_gc_config(agent_id, threshold, strategy)
                if display:
                    display.refresh()

            def on_agent_history_updated(self, agent_id, history):
                registry.update_history(agent_id, history)

            def on_tool_call_start(self, agent_id, tool_name, tool_args, call_id=None):
                buffer = registry.get_buffer(agent_id)
                if buffer:
                    buffer.add_active_tool(tool_name, tool_args, call_id=call_id)
                    buffer.scroll_to_bottom()  # Auto-scroll when tool tree grows
                    if display:
                        display.refresh()

            def on_tool_call_end(self, agent_id, tool_name, success, duration_seconds,
                                  error_message=None, call_id=None):
                buffer = registry.get_buffer(agent_id)
                if buffer:
                    buffer.mark_tool_completed(
                        tool_name, success, duration_seconds, error_message, call_id=call_id
                    )
                    buffer.scroll_to_bottom()  # Auto-scroll when tool tree updates
                    if display:
                        display.refresh()

            def on_tool_output(self, agent_id, call_id, chunk):
                buffer = registry.get_buffer(agent_id)
                if buffer and call_id:
                    buffer.append_tool_output(call_id, chunk)
                    if display:
                        display.refresh()

        hooks = RichClientHooks()

        # Store hooks reference for direct calls (e.g., when user sends input)
        self._ui_hooks = hooks

        # Register hooks with backend (main agent)
        if self._backend:
            self._backend.set_ui_hooks(hooks)

        # Register hooks with SubagentPlugin if present (direct mode only)
        if self.registry:
            subagent_plugin = self.registry.get_plugin("subagent")
            if subagent_plugin and hasattr(subagent_plugin, 'set_ui_hooks'):
                subagent_plugin.set_ui_hooks(hooks)

    def _setup_permission_hooks(self) -> None:
        """Set up permission lifecycle hooks for UI integration.

        These hooks update the tool call tree to show permission status
        inline under each tool that requires permission.

        Also stores the response_options from the permission plugin so they
        can be passed to the input completer for context-aware autocompletion.
        """
        if not self.permission_plugin or not self._agent_registry:
            return

        registry = self._agent_registry
        display = self._display

        def on_permission_requested(tool_name: str, request_id: str, tool_args: dict, response_options: list):
            """Called when permission prompt is shown.

            Args:
                tool_name: Name of the tool requesting permission.
                request_id: Unique identifier for this request.
                tool_args: Arguments passed to the tool (client formats display).
                response_options: List of PermissionResponseOption objects
                    that define valid responses for autocompletion.
            """
            self._trace(f"on_permission_requested: tool={tool_name}, request_id={request_id}")
            # Store the response options for the prompt_callback to use
            # This makes the permission plugin the single source of truth
            self._pending_response_options = response_options

            # Get formatted prompt lines from permission plugin (with diff for file edits)
            prompt_lines = None
            format_hint = None
            if hasattr(self.permission_plugin, 'get_formatted_prompt'):
                try:
                    prompt_lines, format_hint = self.permission_plugin.get_formatted_prompt(
                        tool_name, tool_args, "queue"
                    )
                except Exception:
                    pass  # Fall back to basic formatting

            # Fallback to basic formatting if plugin formatting failed
            if not prompt_lines:
                from shared.ui_utils import build_permission_prompt_lines
                prompt_lines = build_permission_prompt_lines(
                    tool_args=tool_args,
                    response_options=response_options,
                    include_tool_name=True,  # Direct mode includes tool name
                    tool_name=tool_name,
                )
            self._trace(f"on_permission_requested: built {len(prompt_lines)} prompt lines, format_hint={format_hint}")

            # Format prompt lines through the pipeline (for diff coloring, etc.)
            # For diffs, we need to join lines and format as a block so the diff
            # formatter can detect the complete diff pattern and apply proper rendering
            if format_hint == "diff":
                combined = "\n".join(prompt_lines)
                formatted = registry.format_text(combined, format_hint=format_hint)
                formatted_lines = formatted.split("\n")
            else:
                formatted_lines = [
                    registry.format_text(line, format_hint=format_hint) for line in prompt_lines
                ]

            # Update the tool in the main agent's buffer
            buffer = registry.get_buffer("main")
            self._trace(f"on_permission_requested: buffer={buffer}, active_tools={len(buffer._active_tools) if buffer else 0}")
            if buffer:
                buffer.set_tool_permission_pending(tool_name, formatted_lines)
                self._trace(f"on_permission_requested: set_tool_permission_pending called")
                if display:
                    display.refresh()
                    self._trace(f"on_permission_requested: display.refresh() called")

        def on_permission_resolved(tool_name: str, request_id: str, granted: bool, method: str):
            """Called when permission is resolved."""
            self._trace(f"on_permission_resolved: tool={tool_name}, granted={granted}, method={method}")
            # Clear pending options
            self._pending_response_options = None

            # Update the tool in the main agent's buffer
            buffer = registry.get_buffer("main")
            if buffer:
                buffer.set_tool_permission_resolved(tool_name, granted, method)
                if display:
                    display.refresh()

        self.permission_plugin.set_permission_hooks(
            on_requested=on_permission_requested,
            on_resolved=on_permission_resolved
        )

    def _setup_clarification_hooks(self) -> None:
        """Set up clarification lifecycle hooks for UI integration.

        These hooks update the tool call tree to show clarification status
        inline under the request_clarification tool, with questions shown
        one at a time.
        """
        if not self.registry or not self._agent_registry:
            return

        clarification_plugin = self.registry.get_plugin("clarification")
        if not clarification_plugin or not hasattr(clarification_plugin, 'set_clarification_hooks'):
            return

        registry = self._agent_registry
        display = self._display

        def on_clarification_requested(tool_name: str, prompt_lines: list):
            """Called when clarification session starts (context only)."""
            buffer = registry.get_buffer("main")
            if buffer:
                buffer.set_tool_clarification_pending(tool_name, prompt_lines)
                if display:
                    display.refresh()

        def on_clarification_resolved(tool_name: str, qa_pairs: list):
            """Called when all clarification questions are answered."""
            buffer = registry.get_buffer("main")
            if buffer:
                buffer.set_tool_clarification_resolved(tool_name, qa_pairs)
                if display:
                    display.refresh()

        def on_question_displayed(tool_name: str, question_index: int, total_questions: int, question_lines: list):
            """Called when each question is shown."""
            buffer = registry.get_buffer("main")
            if buffer:
                buffer.set_tool_clarification_question(tool_name, question_index, total_questions, question_lines)
                if display:
                    display.refresh()

        def on_question_answered(tool_name: str, question_index: int, answer_summary: str):
            """Called when user answers a question."""
            buffer = registry.get_buffer("main")
            if buffer:
                buffer.set_tool_question_answered(tool_name, question_index, answer_summary)
                if display:
                    display.refresh()

        clarification_plugin.set_clarification_hooks(
            on_requested=on_clarification_requested,
            on_resolved=on_clarification_resolved,
            on_question_displayed=on_question_displayed,
            on_question_answered=on_question_answered
        )

    def _setup_session_plugin(self) -> None:
        """Set up session persistence plugin."""
        if not self._backend:
            return

        try:
            session_config = load_session_config()
            session_plugin = create_session_plugin()
            session_plugin.initialize({'storage_path': session_config.storage_path})
            self._backend.set_session_plugin(session_plugin, session_config)

            if self.registry:
                self.registry.register_plugin(session_plugin, enrichment_only=True)

            if self.permission_plugin and hasattr(session_plugin, 'get_auto_approved_tools'):
                auto_approved = session_plugin.get_auto_approved_tools()
                if auto_approved:
                    self.permission_plugin.add_whitelist_tools(auto_approved)

        except Exception as e:
            pass  # Session plugin is optional

    def _get_plugin_commands_by_plugin(self) -> Dict[str, list]:
        """Collect plugin commands grouped by plugin name."""
        commands_by_plugin: Dict[str, list] = {}

        if self.registry:
            for plugin_name in self.registry.list_exposed():
                plugin = self.registry.get_plugin(plugin_name)
                if plugin and hasattr(plugin, 'get_user_commands'):
                    commands = plugin.get_user_commands()
                    if commands:
                        commands_by_plugin[plugin_name] = commands

        if self.permission_plugin and hasattr(self.permission_plugin, 'get_user_commands'):
            commands = self.permission_plugin.get_user_commands()
            if commands:
                commands_by_plugin[self.permission_plugin.name] = commands

        if self._backend:
            user_commands = self._run_async(self._backend.get_user_commands())
            session_cmds = [cmd for name, cmd in user_commands.items()
                           if name in ('save', 'resume', 'sessions', 'delete-session', 'backtoturn')]
            if session_cmds:
                commands_by_plugin['session'] = session_cmds

        return commands_by_plugin

    def _register_plugin_commands(self) -> None:
        """Register plugin commands for autocompletion."""
        if not self._backend:
            return

        user_commands = self._run_async(self._backend.get_user_commands())
        if not user_commands:
            return

        completer_cmds = [(cmd.name, cmd.description) for cmd in user_commands.values()]
        self._input_handler.add_commands(completer_cmds)

        # Session provider - direct mode only (uses plugin directly)
        session_plugin = self._backend.get_session_plugin()
        if session_plugin and hasattr(session_plugin, 'list_sessions'):
            self._input_handler.set_session_provider(session_plugin.list_sessions)

        # Set up plugin command argument completion
        self._setup_command_completion_provider()

    def _setup_command_completion_provider(self) -> None:
        """Set up the provider for plugin command argument completions."""
        if not self.registry:
            return

        command_to_plugin: dict = {}

        for plugin_name in self.registry.list_exposed():
            plugin = self.registry.get_plugin(plugin_name)
            if plugin and hasattr(plugin, 'get_command_completions'):
                if hasattr(plugin, 'get_user_commands'):
                    for cmd in plugin.get_user_commands():
                        command_to_plugin[cmd.name] = plugin

        session_plugin = self._backend.get_session_plugin() if self._backend else None
        if session_plugin and hasattr(session_plugin, 'get_command_completions'):
            if hasattr(session_plugin, 'get_user_commands'):
                for cmd in session_plugin.get_user_commands():
                    command_to_plugin[cmd.name] = session_plugin

        if self.permission_plugin and hasattr(self.permission_plugin, 'get_command_completions'):
            if hasattr(self.permission_plugin, 'get_user_commands'):
                for cmd in self.permission_plugin.get_user_commands():
                    command_to_plugin[cmd.name] = self.permission_plugin

        # Built-in commands with special completion handling
        commands_with_completions = set(command_to_plugin.keys())
        commands_with_completions.add("model")  # Built-in model command
        commands_with_completions.add("tools enable")  # Tool management
        commands_with_completions.add("tools disable")  # Tool management

        def completion_provider(command: str, args: list) -> list:
            # Handle built-in model command
            if command == "model" and self._backend:
                return self._run_async(self._backend.get_model_completions(args))

            # Handle tools enable/disable completions
            if command == 'tools enable':
                # Show disabled tools (that can be enabled) + 'all'
                disabled = self.registry.list_disabled_tools() if self.registry else []
                completions = [('all', 'Enable all tools')]
                for tool in sorted(disabled):
                    completions.append((tool, 'disabled'))
                return completions
            elif command == 'tools disable':
                # Show enabled tools (that can be disabled) + 'all'
                if self.registry:
                    all_tools = self.registry.get_all_tool_names()
                    enabled = [t for t in all_tools if self.registry.is_tool_enabled(t)]
                else:
                    enabled = []
                completions = [('all', 'Disable all tools')]
                for tool in sorted(enabled):
                    completions.append((tool, 'enabled'))
                return completions

            # Handle plugin commands
            plugin = command_to_plugin.get(command)
            if plugin and hasattr(plugin, 'get_command_completions'):
                return plugin.get_command_completions(command, args)
            return []

        self._input_handler.set_command_completion_provider(
            completion_provider,
            commands_with_completions
        )

    def run_prompt(self, prompt: str) -> str:
        """Execute a prompt synchronously and return the response.

        This is used for single-prompt (non-interactive) mode only.
        For interactive mode, use _start_model_thread instead.
        """
        if not self._backend:
            return "Error: Client not initialized"

        try:
            response = self._run_async(
                self._backend.send_message(prompt, on_output=lambda s, t, m: print(f"[{s}] {t}"))
            )
            return response if response else "(No response)"
        except Exception as e:
            return f"Error: {e}"

    def _start_model_thread(self, prompt: str) -> None:
        """Start the model call in a background thread.

        This allows the prompt_toolkit event loop to continue running,
        which is necessary for handling permission/clarification prompts.
        The model thread will update the display via callbacks.
        """
        if not self._backend:
            if self._display:
                self._display.add_system_message("Error: Client not initialized", style="red")
            return

        if self._model_running:
            if self._display:
                self._display.add_system_message("Model is already running", style="yellow")
            return

        self._trace("_start_model_thread starting")

        # Spinner is already started by on_agent_status_changed hook
        # (called from _handle_input before this method)

        # Create callback - spinner stays active during entire turn
        # to show "thinking..." when model pauses between streaming chunks
        output_callback = self._create_output_callback()

        # Track maximum token usage seen during this turn (to avoid jumping backwards)
        # Initialize from REGISTRY values (not turn_accounting) to maintain continuity
        # across turns. This prevents the status bar from jumping down at the start
        # of a new turn when the first streaming chunk arrives with values that may
        # be lower than the previous turn's final values.
        if self._agent_registry:
            agent_id = self._agent_registry.get_selected_agent_id()
            prev_usage = self._agent_registry.get_selected_context_usage() if agent_id else {}
            max_tokens_seen = {
                'prompt': prev_usage.get('prompt_tokens', 0),
                'output': prev_usage.get('output_tokens', 0),
                'total': prev_usage.get('total_tokens', 0)
            }
        else:
            max_tokens_seen = {'prompt': 0, 'output': 0, 'total': 0}

        # Create usage update callback for real-time token accounting
        def usage_update_callback(usage) -> None:
            """Update status bar with real-time token usage during streaming."""
            # Skip zero values (initialization chunks, not real data)
            if usage.total_tokens == 0:
                return

            # Write to provider trace for debugging (same file as provider uses)
            import datetime
            trace_path = os.environ.get(
                "JAATO_PROVIDER_TRACE",
                os.path.join(tempfile.gettempdir(), "provider_trace.log")
            )
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] [rich_client_callback] received: prompt={usage.prompt_tokens} output={usage.output_tokens} total={usage.total_tokens}\n")
                    f.flush()
            except Exception as e:
                pass  # Ignore trace errors
            self._trace(f"[usage_callback] received: prompt={usage.prompt_tokens} output={usage.output_tokens} total={usage.total_tokens}")

            # Only update if we see HIGHER values (prevents backwards jumping)
            # The prompt_tokens is the most reliable indicator of context size
            # since it includes the full conversation history
            if usage.prompt_tokens >= max_tokens_seen['prompt']:
                max_tokens_seen['prompt'] = usage.prompt_tokens
                max_tokens_seen['output'] = max(max_tokens_seen['output'], usage.output_tokens)
                max_tokens_seen['total'] = max_tokens_seen['prompt'] + max_tokens_seen['output']
            else:
                # Skip update if prompt_tokens dropped (new API call with different context)
                self._trace(f"[usage_callback] skipping update: prompt {usage.prompt_tokens} < max {max_tokens_seen['prompt']}")
                return

            # Trace the condition check
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] [rich_client_callback] display={self._display is not None} backend={self._backend is not None}\n")
                    f.flush()
            except Exception:
                pass
            if self._display and self._backend:
                # Get context limit for percentage calculation
                context_limit = self._run_async(self._backend.get_context_limit())
                total_tokens = max_tokens_seen['total']
                percent_used = (total_tokens / context_limit * 100) if context_limit > 0 else 0
                tokens_remaining = max(0, context_limit - total_tokens)

                # Build usage dict for display update
                usage_dict = {
                    'total_tokens': total_tokens,
                    'prompt_tokens': max_tokens_seen['prompt'],
                    'output_tokens': max_tokens_seen['output'],
                    'context_limit': context_limit,
                    'percent_used': percent_used,
                    'tokens_remaining': tokens_remaining,
                }
                self._trace(f"[usage_callback] updating display: {percent_used:.1f}% used, {total_tokens} tokens")

                # Update agent registry if available (status bar reads from here)
                if self._agent_registry:
                    agent_id = self._agent_registry.get_selected_agent_id()
                    if agent_id:
                        self._agent_registry.update_context_usage(
                            agent_id=agent_id,
                            total_tokens=total_tokens,
                            prompt_tokens=max_tokens_seen['prompt'],
                            output_tokens=max_tokens_seen['output'],
                            turns=0,  # Don't know turn count during streaming
                            percent_used=percent_used
                        )

                # Also update display directly (fallback if no registry)
                self._display.update_context_usage(usage_dict)

                # Trace after update
                try:
                    with open(trace_path, "a") as f:
                        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        f.write(f"[{ts}] [rich_client_callback] update_context_usage called, percent={percent_used:.1f}%\n")
                        f.flush()
                except Exception:
                    pass

        def gc_threshold_callback(percent_used: float, threshold: float) -> None:
            """Handle GC threshold crossing notification."""
            self._trace(f"[gc_threshold] Context threshold crossed: {percent_used:.1f}% >= {threshold}%")
            # Show warning in output (optional - GC will happen automatically after turn)
            if self._display:
                self._display.add_system_message(
                    f"⚠ Context usage ({percent_used:.1f}%) exceeds threshold ({threshold}%). GC will run after this turn.",
                    style="yellow"
                )

        def model_thread():
            self._trace("[model_thread] started")
            self._model_running = True
            try:
                self._trace("[model_thread] calling send_message...")
                # For direct mode, get the underlying sync client
                client = self._backend.get_client() if self._backend else None
                if client and hasattr(client, 'send_message'):
                    client.send_message(
                        prompt,
                        on_output=output_callback,
                        on_usage_update=usage_update_callback,
                        on_gc_threshold=gc_threshold_callback
                    )
                self._trace(f"[model_thread] send_message returned")

                # Update context usage in status bar and registry
                # Important: Update REGISTRY too so it reflects post-GC values
                # when garbage collection runs during the turn
                if self._display and self._backend:
                    usage = self._run_async(self._backend.get_context_usage())
                    self._display.update_context_usage(usage)
                    # Also update registry so next turn's max_tokens_seen initializes correctly
                    if self._agent_registry:
                        agent_id = self._agent_registry.get_selected_agent_id()
                        if agent_id:
                            self._agent_registry.update_context_usage(
                                agent_id=agent_id,
                                total_tokens=usage.get('total_tokens', 0),
                                prompt_tokens=usage.get('prompt_tokens', 0),
                                output_tokens=usage.get('output_tokens', 0),
                                turns=usage.get('turns', 0),
                                percent_used=usage.get('percent_used', 0)
                            )

                # Add separator after model finishes
                # (response content is already shown via the callback)
                if self._display:
                    self._display.add_system_message("─" * 40, style="dim")
                    self._display.add_system_message("", style="dim")

            except KeyboardInterrupt:
                self._trace("[model_thread] KeyboardInterrupt")
                if self._display:
                    self._display.add_system_message("[Interrupted]", style="yellow")
            except Exception as e:
                self._trace(f"[model_thread] Exception: {e}")
                if self._display:
                    self._display.add_system_message(f"Error: {e}", style="red")
            finally:
                self._model_running = False
                self._model_thread = None
                # Signal that main agent is done - this stops the spinner
                if self._ui_hooks:
                    self._ui_hooks.on_agent_status_changed(
                        agent_id="main",
                        status="done"
                    )
                elif self._display:
                    # Fallback: directly stop spinner if no hooks
                    self._display.stop_spinner()
                self._trace("[model_thread] finished")

        # Start model call in background thread
        self._model_thread = threading.Thread(target=model_thread, daemon=True)
        self._model_thread.start()
        self._trace("model thread started")

    def clear_history(self) -> None:
        """Clear conversation history."""
        if self._backend:
            self._run_async(self._backend.reset_session())
        self._original_inputs = []
        if self._display:
            self._display.clear_output()
            self._display.clear_plan()

    def _handle_input(self, user_input: str) -> None:
        """Handle user input from the prompt_toolkit input loop.

        Args:
            user_input: The text entered by the user.
        """
        # Check if pager is active - handle pager input first
        if self._display.pager_active:
            self._display.handle_pager_input(user_input)
            return

        # Route input to channel queue if waiting for permission/clarification
        if self._waiting_for_channel_input:
            # Check for 'v' to view full prompt in pager (only if truncated)
            if user_input.lower() == 'v':
                # Use get_selected_buffer() for IPC compatibility (agent ID may not be "main")
                buffer = self._agent_registry.get_selected_buffer()
                if buffer and buffer.has_truncated_pending_prompt():
                    prompt_data = buffer.get_pending_prompt_for_pager()
                    if prompt_data:
                        prompt_type, prompt_lines = prompt_data
                        # Show full prompt in pager (omit options line - it's in the original view)
                        title = "Permission Request" if prompt_type == "permission" else "Clarification Request"
                        lines = [(f"─── {title} ───", "bold cyan")]
                        for line in prompt_lines:
                            # Skip the options line (e.g. "[y]es [n]o [a]lways...")
                            if '[y]es' in line.lower() or '(type' in line.lower():
                                continue
                            # Color diff lines
                            if line.startswith('+') and not line.startswith('+++'):
                                lines.append((line, "green"))
                            elif line.startswith('-') and not line.startswith('---'):
                                lines.append((line, "red"))
                            elif line.startswith('@@'):
                                lines.append((line, "cyan"))
                            else:
                                lines.append((line, ""))
                        lines.append(("─" * 40, "dim"))
                        lines.append(("Press 'q' to close, Enter/Space for next page", "dim italic"))
                        self._display.show_lines(lines)
                        return
            # Don't echo answer - it's shown inline in the tool tree
            # Note: Input is proactively filtered in PTDisplay._on_input_changed()
            # so only valid permission responses can be typed
            self._channel_input_queue.put(user_input)
            self._trace(f"Input routed to channel queue: {user_input}")
            # Don't start spinner here - the channel may have more questions.
            # Spinner will be started when prompt_callback(False) indicates
            # the channel is done and model continues.
            return

        if not user_input:
            return

        if user_input.lower() in ('quit', 'exit', 'q'):
            self._display.add_system_message("Goodbye!", style="bold")
            self._display.stop()
            return

        if user_input.lower() == 'help':
            self._show_help()
            return

        if user_input.lower().startswith('tools'):
            self._handle_tools_command(user_input)
            return

        if user_input.lower() == 'plugins':
            self._show_plugins()
            return

        if user_input.lower() == 'reset':
            self.clear_history()
            self._display.show_lines([("[History cleared]", "yellow")])
            return

        if user_input.lower() == 'clear':
            self._display.clear_output()
            return

        if user_input.lower().startswith('keybindings'):
            self._handle_keybindings_command(user_input)
            return

        if user_input.lower() == 'history':
            self._show_history()
            return

        if user_input.lower() == 'context':
            self._show_context()
            return

        if user_input.lower().startswith('export'):
            parts = user_input.split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "session_export.yaml"
            if filename.startswith('@'):
                filename = filename[1:]
            self._export_session(filename)
            return

        # Check for plugin commands
        plugin_result = self._try_execute_plugin_command(user_input)
        if plugin_result is not None:
            self._original_inputs.append({"text": user_input, "local": True})
            return

        # Track input
        self._original_inputs.append({"text": user_input, "local": False})

        # Show user input in output immediately
        self._display.append_output("user", user_input, "write")

        # Signal that main agent is now active (processing input)
        # This starts the spinner via the hooks
        if self._ui_hooks:
            self._ui_hooks.on_agent_status_changed(
                agent_id="main",
                status="active"
            )

        # Expand file references
        expanded_prompt = self._input_handler.expand_file_references(user_input)

        # Start model in background thread (non-blocking)
        # This allows the event loop to continue running for permission prompts
        self._start_model_thread(expanded_prompt)

    def run_interactive(self, initial_prompt: Optional[str] = None) -> None:
        """Run the interactive TUI loop.

        Args:
            initial_prompt: Optional prompt to run before entering interactive mode.
        """
        # Create the display with input handler, agent registry, and keybindings
        keybinding_config = load_keybindings()
        self._display = PTDisplay(
            input_handler=self._input_handler,
            agent_registry=self._agent_registry,
            keybinding_config=keybinding_config
        )

        # Set model info in status bar
        self._display.set_model_info(self._model_provider, self._model_name)

        # Set GC threshold and strategy in status bar (if configured)
        if self._gc_threshold is not None:
            self._display.set_gc_threshold(self._gc_threshold, self._gc_strategy)

        # Set up stop callbacks for Ctrl-C handling
        self._display.set_stop_callbacks(
            stop_callback=lambda: self._run_async(self._backend.stop()) if self._backend else False,
            is_running_callback=lambda: self._backend.is_processing if self._backend else False
        )

        # Set up the live reporter and queue channels
        self._setup_live_reporter()
        self._setup_queue_channels()

        # Set up retry callback to route rate limit messages to output panel
        self._setup_retry_callback()

        # Set up code validation formatter for LSP diagnostics on output code blocks
        self._setup_code_validation_formatter()

        # Register UI hooks with jaato client and subagent plugin
        # This will create the main agent in the registry via set_ui_hooks()
        self._setup_agent_hooks()

        # Set GC config on main agent in registry (for per-agent status bar display)
        if self._gc_threshold is not None and self._agent_registry:
            self._agent_registry.update_gc_config("main", self._gc_threshold, self._gc_strategy)

        # Set up permission hooks for inline permission display in tool tree
        self._setup_permission_hooks()

        # Set up clarification hooks for inline clarification display in tool tree
        self._setup_clarification_hooks()

        # Load release name from file
        release_name = "Jaato Rich TUI Client"
        release_file = pathlib.Path(__file__).parent / "release_name.txt"
        if release_file.exists():
            release_name = release_file.read_text().strip()

        # Add welcome messages
        self._display.add_system_message(
            release_name,
            style="bold cyan"
        )
        if self._input_handler.has_completion:
            self._display.add_system_message(
                "Tab completion enabled. Use @file to reference files, /command for slash commands.",
                style="dim"
            )
        self._display.add_system_message(
            "Type 'help' for commands, 'quit' to exit, Esc+Esc to clear input",
            style="dim"
        )
        self._display.add_system_message("", style="dim")

        # Validate TTY and start display
        self._display.start()

        try:
            # Run the prompt_toolkit input loop
            # Initial prompt (if provided) is auto-submitted once event loop starts
            self._display.run_input_loop(self._handle_input, initial_prompt=initial_prompt)
        except (EOFError, KeyboardInterrupt):
            pass

        print("Goodbye!")

    def _get_all_tool_schemas(self) -> list:
        """Get all tool schemas from registry and plugins."""
        all_decls = []
        if self.registry:
            all_decls.extend(self.registry.get_exposed_tool_schemas())
        if self.permission_plugin:
            all_decls.extend(self.permission_plugin.get_tool_schemas())
        session_plugin = self._backend.get_session_plugin() if self._backend else None
        if session_plugin and hasattr(session_plugin, 'get_tool_schemas'):
            all_decls.extend(session_plugin.get_tool_schemas())
        return all_decls

    def _handle_tools_command(self, user_input: str) -> None:
        """Handle the tools command with subcommands.

        Subcommands:
            tools / tools list  - List all tools with enabled/disabled status
            tools enable <name> - Enable a specific tool
            tools disable <name> - Disable a specific tool
            tools enable all    - Enable all tools
            tools disable all   - Disable all tools

        Args:
            user_input: The full user input string starting with 'tools'.
        """
        if not self._display:
            return

        parts = user_input.strip().split()
        subcommand = parts[1].lower() if len(parts) > 1 else "list"
        args = parts[2:] if len(parts) > 2 else []

        if subcommand == "list" or subcommand == "tools":
            self._show_tools_with_status()
            return

        if subcommand == "enable":
            if not args:
                self._display.show_lines([
                    ("[Error: Specify a tool name or 'all']", "red"),
                    ("  Usage: tools enable <tool_name>", "dim"),
                    ("         tools enable all", "dim"),
                ])
                return
            self._tools_enable(args[0])
            return

        if subcommand == "disable":
            if not args:
                self._display.show_lines([
                    ("[Error: Specify a tool name or 'all']", "red"),
                    ("  Usage: tools disable <tool_name>", "dim"),
                    ("         tools disable all", "dim"),
                ])
                return
            self._tools_disable(args[0])
            return

        # Unknown subcommand - show help
        self._display.show_lines([
            (f"[Unknown subcommand: {subcommand}]", "yellow"),
            ("  Available subcommands:", ""),
            ("    tools list         - List all tools with status", "dim"),
            ("    tools enable <n>   - Enable a tool (or 'all')", "dim"),
            ("    tools disable <n>  - Disable a tool (or 'all')", "dim"),
        ])

    def _handle_keybindings_command(self, user_input: str) -> None:
        """Handle the keybindings command with subcommands.

        Delegates to the shared implementation in ui_utils.

        Args:
            user_input: The full user input string starting with 'keybindings'.
        """
        from shared.ui_utils import handle_keybindings_command
        handle_keybindings_command(user_input, self._display)

    def _get_tool_status(self) -> list:
        """Get status of all tools including enabled/disabled state."""
        status = []

        # Get registry tools with status
        if self.registry:
            status.extend(self.registry.get_tool_status())

        # Add permission plugin tools (always enabled)
        if self.permission_plugin:
            for schema in self.permission_plugin.get_tool_schemas():
                status.append({
                    'name': schema.name,
                    'description': schema.description,
                    'enabled': True,
                    'plugin': 'permission',
                })

        # Add session plugin tools (always enabled)
        session_plugin = self._backend.get_session_plugin() if self._backend else None
        if session_plugin and hasattr(session_plugin, 'get_tool_schemas'):
            for schema in session_plugin.get_tool_schemas():
                status.append({
                    'name': schema.name,
                    'description': schema.description,
                    'enabled': True,
                    'plugin': 'session',
                })

        return status

    def _show_tools_with_status(self) -> None:
        """Show tools with enabled/disabled status in output panel."""
        if not self._display:
            return

        tool_status = self._get_tool_status()

        if not tool_status:
            self._display.show_lines([("No tools available.", "yellow")])
            return

        # Group tools by plugin
        by_plugin: dict = {}
        for tool in tool_status:
            plugin = tool.get('plugin', 'unknown')
            if plugin not in by_plugin:
                by_plugin[plugin] = []
            by_plugin[plugin].append(tool)

        # Count enabled/disabled
        enabled_count = sum(1 for t in tool_status if t.get('enabled', True))
        disabled_count = len(tool_status) - enabled_count

        lines = [
            (f"Tools ({enabled_count} enabled, {disabled_count} disabled):", "bold"),
            ("  Use 'tools enable <name>' or 'tools disable <name>' to toggle", "dim"),
            ("", ""),
        ]

        for plugin_name in sorted(by_plugin.keys()):
            tools = by_plugin[plugin_name]
            lines.append((f"  [{plugin_name}]", "cyan"))

            for tool in sorted(tools, key=lambda t: t['name']):
                name = tool['name']
                desc = tool.get('description', '')
                enabled = tool.get('enabled', True)

                # Status indicator
                status = "✓" if enabled else "○"
                status_style = "green" if enabled else "red"

                # Truncate description if too long
                max_desc = 50
                if len(desc) > max_desc:
                    desc = desc[:max_desc - 3] + "..."

                lines.append((f"    {status} {name}: {desc}", status_style if not enabled else ""))

            lines.append(("", ""))

        lines.append(("  Legend: ✓ = enabled, ○ = disabled", "dim"))
        self._display.show_lines(lines)

    def _tools_enable(self, name: str) -> None:
        """Enable a tool or all tools.

        Args:
            name: Tool name or 'all'.
        """
        if not self._display:
            return

        if not self.registry:
            self._display.show_lines([("[Error: Registry not available]", "red")])
            return

        if name.lower() == "all":
            count = self.registry.enable_all_tools()
            self._refresh_session_tools()
            self._display.show_lines([(f"[Enabled all {count} tools]", "green")])
            return

        if self.registry.enable_tool(name):
            self._refresh_session_tools()
            self._display.show_lines([(f"[Enabled tool: {name}]", "green")])
        else:
            lines = [(f"[Error: Tool '{name}' not found]", "red")]
            available = self.registry.get_all_tool_names()
            if available:
                preview = ', '.join(sorted(available)[:10])
                lines.append((f"  Available tools: {preview}", "dim"))
                if len(available) > 10:
                    lines.append((f"  ... and {len(available) - 10} more", "dim"))
            self._display.show_lines(lines)

    def _tools_disable(self, name: str) -> None:
        """Disable a tool or all tools.

        Args:
            name: Tool name or 'all'.
        """
        if not self._display:
            return

        if not self.registry:
            self._display.show_lines([("[Error: Registry not available]", "red")])
            return

        if name.lower() == "all":
            count = self.registry.disable_all_tools()
            self._refresh_session_tools()
            self._display.show_lines([
                (f"[Disabled all {count} tools]", "yellow"),
                ("  Note: Permission and session tools remain available", "dim"),
            ])
            return

        if self.registry.disable_tool(name):
            self._refresh_session_tools()
            self._display.show_lines([(f"[Disabled tool: {name}]", "yellow")])
        else:
            lines = [(f"[Error: Tool '{name}' not found]", "red")]
            available = self.registry.get_all_tool_names()
            if available:
                preview = ', '.join(sorted(available)[:10])
                lines.append((f"  Available tools: {preview}", "dim"))
                if len(available) > 10:
                    lines.append((f"  ... and {len(available) - 10} more", "dim"))
            self._display.show_lines(lines)

    def _refresh_session_tools(self) -> None:
        """Refresh the session's tools after enabling/disabling."""
        if self._backend:
            self._run_async(self._backend.refresh_tools())
            self.log("[client] Session tools refreshed")

    def _show_plugins(self) -> None:
        """Show available plugins with status and descriptions."""
        if not self._display or not self.registry:
            return

        available = self.registry.list_available()
        exposed = set(self.registry.list_exposed())
        skipped = self.registry.list_skipped_plugins()

        lines = [("Available Plugins:", "bold")]
        lines.append(("", ""))

        for name in sorted(available):
            plugin = self.registry.get_plugin(name)
            if not plugin:
                continue

            # Determine status
            if name in exposed:
                status = "✓ enabled"
                status_style = "green"
                skip_reason = None
            elif name in skipped:
                status = "⊘ skipped"
                status_style = "yellow"
                skip_reason = skipped[name]  # List of required patterns
            else:
                status = "○ available"
                status_style = "dim"
                skip_reason = None

            # Get description from plugin
            description = self._get_plugin_description(plugin)

            # Format output
            lines.append((f"  {name}", "cyan"))
            lines.append((f"    Status: {status}", status_style))
            if skip_reason:
                patterns = ", ".join(skip_reason)
                lines.append((f"    Requires model: {patterns}", "yellow"))
            if description:
                # Wrap long descriptions
                if len(description) > 70:
                    description = description[:67] + "..."
                lines.append((f"    {description}", "dim"))
            lines.append(("", ""))

        # Add summary
        lines.append(("─" * 40, "dim"))
        lines.append((f"  Total: {len(available)} plugins ({len(exposed)} enabled, {len(skipped)} skipped)", "dim"))

        self._display.show_lines(lines)

    def _get_plugin_description(self, plugin) -> str:
        """Get a description for a plugin.

        Tries multiple sources:
        1. plugin.description property/attribute
        2. plugin.__doc__ (class docstring)
        3. First tool's description (if plugin has tools)
        4. Empty string as fallback
        """
        # Try description property
        if hasattr(plugin, 'description'):
            desc = plugin.description
            if callable(desc):
                desc = desc()
            if desc:
                return str(desc).strip().split('\n')[0]

        # Try class docstring
        if plugin.__doc__:
            # Take first non-empty line of docstring
            for line in plugin.__doc__.strip().split('\n'):
                line = line.strip()
                if line:
                    return line

        # Try first tool description
        if hasattr(plugin, 'get_tool_schemas'):
            try:
                schemas = plugin.get_tool_schemas()
                if schemas and schemas[0].description:
                    return f"Provides: {schemas[0].name}"
            except Exception:
                pass

        return ""

    def _show_history(self) -> None:
        """Show conversation history for SELECTED agent.

        Uses selected agent's history from the agent registry.
        """
        if not self._display:
            return

        # Get selected agent's history and accounting
        selected_agent = self._agent_registry.get_selected_agent()
        if not selected_agent:
            return

        history = selected_agent.history
        turn_accounting = selected_agent.turn_accounting

        # For main agent, also get turn boundaries
        turn_boundaries = []
        if selected_agent.agent_id == "main" and self._backend:
            turn_boundaries = self._run_async(self._backend.get_turn_boundaries())

        count = len(history)
        total_turns = len(turn_accounting) if turn_accounting else len(turn_boundaries)

        lines = [
            ("=" * 60, ""),
            (f"  Conversation History: {selected_agent.name}", "bold"),
            (f"  Agent: {selected_agent.agent_id} ({selected_agent.agent_type})", "dim"),
            (f"  Messages: {count}, Turns: {total_turns}", "dim"),
            ("  Tip: Use 'backtoturn <turn_id>' to revert to a specific turn (main agent only)", "dim"),
            ("=" * 60, ""),
        ]

        if count == 0:
            lines.append(("  (empty)", "dim"))
            lines.append(("", ""))
            self._display.show_lines(lines)
            return

        current_turn = 0
        turn_index = 0

        for i, content in enumerate(history):
            role = getattr(content, 'role', None) or 'unknown'
            parts = getattr(content, 'parts', None) or []

            is_user_text = (role == 'user' and parts and
                           hasattr(parts[0], 'text') and parts[0].text)

            # Print turn header if this starts a new turn
            if is_user_text:
                current_turn += 1
                lines.append(("", ""))
                lines.append(("─" * 60, ""))
                # Show timestamp in turn header if available
                turn_idx = current_turn - 1
                if turn_idx < len(turn_accounting) and 'start_time' in turn_accounting[turn_idx]:
                    start_time = turn_accounting[turn_idx]['start_time']
                    try:
                        dt = datetime.fromisoformat(start_time)
                        time_str = dt.strftime('%H:%M:%S')
                        lines.append((f"  ▶ TURN {current_turn}  [{time_str}]", "cyan"))
                    except (ValueError, TypeError):
                        lines.append((f"  ▶ TURN {current_turn}", "cyan"))
                else:
                    lines.append((f"  ▶ TURN {current_turn}", "cyan"))
                lines.append(("─" * 60, ""))

            role_label = "USER" if role == 'user' else "MODEL" if role == 'model' else role.upper()
            lines.append(("", ""))
            lines.append((f"  [{role_label}]", "bold"))

            if not parts:
                lines.append(("  (no content)", "dim"))
            else:
                for part in parts:
                    self._format_part(part, lines)

            # Show token accounting at end of turn
            is_last = (i == len(history) - 1)
            next_is_user_text = False
            if not is_last:
                next_content = history[i + 1]
                next_role = getattr(next_content, 'role', None) or 'unknown'
                next_parts = getattr(next_content, 'parts', None) or []
                next_is_user_text = (next_role == 'user' and next_parts and
                                    hasattr(next_parts[0], 'text') and next_parts[0].text)

            if (is_last or next_is_user_text) and turn_index < len(turn_accounting):
                turn = turn_accounting[turn_index]
                lines.append((f"  ─── tokens: {turn['prompt']} in / {turn['output']} out / {turn['total']} total", "dim"))
                if 'duration_seconds' in turn and turn['duration_seconds'] is not None:
                    duration = turn['duration_seconds']
                    lines.append((f"  ─── duration: {duration:.2f}s", "dim"))
                    func_calls = turn.get('function_calls', [])
                    if func_calls:
                        fc_total = sum(fc['duration_seconds'] for fc in func_calls)
                        model_time = duration - fc_total
                        lines.append((f"      model: {model_time:.2f}s, tools: {fc_total:.2f}s ({len(func_calls)} call(s))", "dim"))
                        for fc in func_calls:
                            lines.append((f"        - {fc['name']}: {fc['duration_seconds']:.2f}s", "dim"))
                turn_index += 1

        # Print totals
        if turn_accounting:
            total_prompt = sum(t['prompt'] for t in turn_accounting)
            total_output = sum(t['output'] for t in turn_accounting)
            total_all = sum(t['total'] for t in turn_accounting)
            total_duration = sum(t.get('duration_seconds', 0) or 0 for t in turn_accounting)
            total_fc_time = sum(
                sum(fc['duration_seconds'] for fc in t.get('function_calls', []))
                for t in turn_accounting
            )
            lines.append(("", ""))
            lines.append(("=" * 60, ""))
            lines.append((f"  Total: {total_prompt} in / {total_output} out / {total_all} total ({total_turns} turns)", "bold"))
            if total_duration > 0:
                total_model_time = total_duration - total_fc_time
                lines.append((f"  Time:  {total_duration:.2f}s total (model: {total_model_time:.2f}s, tools: {total_fc_time:.2f}s)", ""))
            lines.append(("=" * 60, ""))

        lines.append(("", ""))
        self._display.show_lines(lines)

    def _format_part(self, part: Any, lines: List[tuple]) -> None:
        """Format a single content part for history display.

        Args:
            part: A content part (text, function_call, or function_response).
            lines: List to append formatted lines to.
        """
        # Text content - use 'is not None' to properly handle empty strings
        # (which can occur when SDK returns parts we don't fully recognize)
        if hasattr(part, 'text') and part.text is not None:
            text = part.text
            if not text:
                # Empty text part - skip display (don't show as "unknown")
                return
            if len(text) > 500:
                text = text[:500] + f"... [{len(part.text)} chars total]"
            lines.append((f"  {text}", ""))

        # Function call
        elif hasattr(part, 'function_call') and part.function_call:
            fc = part.function_call
            name = getattr(fc, 'name', 'unknown')
            args = getattr(fc, 'args', {})
            args_str = str(args)
            if len(args_str) > 200:
                args_str = args_str[:200] + "..."
            lines.append((f"  📤 CALL: {name}({args_str})", "yellow"))

        # Function response
        elif hasattr(part, 'function_response') and part.function_response:
            fr = part.function_response
            name = getattr(fr, 'name', 'unknown')
            # ToolResult uses 'result' attribute, not 'response'
            response = getattr(fr, 'result', None) or getattr(fr, 'response', {})

            # Extract and display permission info first
            if isinstance(response, dict):
                perm = response.get('_permission')
                if perm:
                    decision = perm.get('decision', '?')
                    reason = perm.get('reason', '')
                    method = perm.get('method', '')
                    icon = '✓' if decision == 'allowed' else '✗'
                    style = "green" if decision == 'allowed' else "red"
                    lines.append((f"  {icon} Permission: {decision} via {method}", style))
                    if reason:
                        lines.append((f"    Reason: {reason}", "dim"))

            # Filter out _permission from display response
            if isinstance(response, dict):
                display_response = {k: v for k, v in response.items() if k != '_permission'}
            else:
                display_response = response

            resp_str = str(display_response)
            if len(resp_str) > 300:
                resp_str = resp_str[:300] + "..."
            lines.append((f"  📥 RESULT: {name} → {resp_str}", "green"))

        # Inline data (images, etc.)
        elif hasattr(part, 'inline_data') and part.inline_data:
            mime_type = part.inline_data.get('mime_type', 'unknown')
            data = part.inline_data.get('data')
            size = len(data) if data else 0
            lines.append((f"  📎 INLINE DATA: {mime_type} ({size} bytes)", "cyan"))

        # Thought/reasoning part (Gemini thinking mode)
        elif hasattr(part, 'thought') and part.thought:
            thought = part.thought
            if len(thought) > 500:
                thought = thought[:500] + f"... [{len(part.thought)} chars total]"
            lines.append((f"  💭 THOUGHT: {thought}", "dim"))

        # Executable code part
        elif hasattr(part, 'executable_code') and part.executable_code:
            code = part.executable_code
            if len(code) > 300:
                code = code[:300] + "..."
            lines.append((f"  💻 CODE: {code}", "cyan"))

        # Code execution result part
        elif hasattr(part, 'code_execution_result') and part.code_execution_result:
            output = part.code_execution_result
            if len(output) > 300:
                output = output[:300] + "..."
            lines.append((f"  📋 EXEC RESULT: {output}", "green"))

        else:
            # Unknown part type - show diagnostic info like simple client
            part_type = type(part).__name__
            # Show available attributes to help debugging
            attrs = [a for a in dir(part) if not a.startswith('_')]
            attr_preview = ', '.join(attrs[:5])
            if len(attrs) > 5:
                attr_preview += f", ... (+{len(attrs) - 5} more)"
            lines.append((f"  (unknown part: {part_type}, attrs: [{attr_preview}])", "yellow"))

    def _show_context(self) -> None:
        """Show context/token usage for SELECTED agent."""
        if not self._display:
            return

        # Get selected agent's context usage
        selected_agent = self._agent_registry.get_selected_agent()
        if not selected_agent:
            self._display.show_lines([("Context tracking not available", "yellow")])
            return

        usage = selected_agent.context_usage

        lines = [
            ("─" * 50, "dim"),
            (f"Context Usage: {selected_agent.name}", "bold"),
            (f"  Agent: {selected_agent.agent_id}", "dim"),
            (f"  Total tokens: {usage.get('total_tokens', 0)}", "dim"),
            (f"  Prompt tokens: {usage.get('prompt_tokens', 0)}", "dim"),
            (f"  Output tokens: {usage.get('output_tokens', 0)}", "dim"),
            (f"  Turns: {usage.get('turns', 0)}", "dim"),
            (f"  Percent used: {usage.get('percent_used', 0):.1f}%", "dim"),
            ("─" * 50, "dim"),
        ]

        self._display.show_lines(lines)

    def _export_session(self, filename: str) -> None:
        """Export session to YAML file."""
        if not self._display or not self._backend:
            return

        try:
            from session_exporter import SessionExporter
            exporter = SessionExporter()
            history = self._run_async(self._backend.get_history())
            result = exporter.export_to_yaml(history, self._original_inputs, filename)

            if result.get('success'):
                self._display.show_lines([
                    (f"Session exported to: {result['filename']}", "green")
                ])
            else:
                self._display.show_lines([
                    (f"Export failed: {result.get('error', 'Unknown error')}", "red")
                ])
        except ImportError:
            self._display.show_lines([
                ("Session exporter not available", "yellow")
            ])
        except Exception as e:
            self._display.show_lines([
                (f"Export error: {e}", "red")
            ])

    def _show_help(self) -> None:
        """Show help in output panel with pagination."""
        if not self._display:
            return

        # Import shared help builders
        from shared.client_commands import (
            build_permission_help_text,
            build_file_reference_help_text,
            build_slash_command_help_text,
            build_keyboard_shortcuts_help_text,
        )

        # Direct mode has more detailed command help (keybindings, plugins, etc.)
        help_lines = [
            ("Commands (auto-complete as you type):", "bold"),
            ("  help              - Show this help message", "dim"),
            ("  tools [subcmd]    - Manage tools available to the model", "dim"),
            ("                        tools list          - List all tools with status", "dim"),
            ("                        tools enable <n>    - Enable a tool (or 'all')", "dim"),
            ("                        tools disable <n>   - Disable a tool (or 'all')", "dim"),
            ("  keybindings [sub] - Manage keyboard shortcuts", "dim"),
            ("                        keybindings list    - Show current keybindings", "dim"),
            ("                        keybindings set     - Set a keybinding", "dim"),
            ("                        keybindings profile - Show/switch terminal profiles", "dim"),
            ("                        keybindings reload  - Reload from config files", "dim"),
            ("  plugins           - List available plugins with status", "dim"),
            ("  reset             - Clear conversation history", "dim"),
            ("  history           - Show full conversation history", "dim"),
            ("  context           - Show context window usage", "dim"),
            ("  export [file]     - Export session to YAML (default: session_export.yaml)", "dim"),
            ("  clear             - Clear output panel", "dim"),
            ("  quit              - Exit the client", "dim"),
            ("", "dim"),
        ]

        # Add plugin commands grouped by plugin
        commands_by_plugin = self._get_plugin_commands_by_plugin()
        if commands_by_plugin:
            help_lines.append(("Plugin-provided user commands:", "bold"))
            for plugin_name, commands in sorted(commands_by_plugin.items()):
                help_lines.append((f"  [{plugin_name}]", "cyan"))
                for cmd in commands:
                    padding = max(2, 14 - len(cmd.name))
                    shared_marker = " [shared with model]" if cmd.share_with_model else ""
                    help_lines.append((f"    {cmd.name}{' ' * padding}- {cmd.description}{shared_marker}", "dim"))
            help_lines.append(("", "dim"))

        # Use shared help sections for common content
        help_lines.extend(build_permission_help_text())
        help_lines.extend(build_file_reference_help_text())
        # Add extra file reference detail for direct mode
        help_lines.insert(-1, ("  Completions appear automatically as you type after @.", "dim"))

        help_lines.extend(build_slash_command_help_text())
        # Add extra slash command detail for direct mode
        help_lines.insert(-1, ("  - Pass arguments after the command name: /review file.py", "dim"))

        # Direct mode extra sections
        help_lines.extend([
            ("Multi-turn conversation:", "bold"),
            ("  The model remembers previous exchanges in this session.", "dim"),
            ("  Use 'reset' to start a fresh conversation.", "dim"),
            ("", "dim"),
        ])

        help_lines.extend(build_keyboard_shortcuts_help_text())
        # Add extra keyboard shortcut for direct mode
        help_lines.insert(-1, ("  Ctrl+A/E  - Jump to start/end of line", "dim"))

        help_lines.extend([
            ("", "dim"),
            ("Display:", "bold"),
            ("  The plan panel at top shows current plan status.", "dim"),
            ("  Model output scrolls in the panel below.", "dim"),
        ])

        # Show help with auto-pagination if needed
        self._display.show_lines(help_lines)

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.registry:
            self.registry.unexpose_all()
        if self.permission_plugin:
            self.permission_plugin.shutdown()


# =============================================================================
# IPC Client Mode
# =============================================================================

async def run_ipc_mode(socket_path: str, auto_start: bool = True, env_file: str = ".env",
                       initial_prompt: Optional[str] = None, single_prompt: Optional[str] = None,
                       new_session: bool = False):
    """Run the client in IPC mode, connecting to a server.

    Uses full PTDisplay for rich TUI experience with plan panel, scrolling output,
    and integrated input prompt.

    Args:
        socket_path: Path to the Unix domain socket.
        auto_start: Whether to auto-start the server if not running.
        env_file: Path to .env file for auto-started server.
        initial_prompt: Optional initial prompt to send.
        single_prompt: Optional single prompt (non-interactive mode).
        new_session: Whether to start a new session instead of resuming default.
    """
    import asyncio
    from ipc_client import IPCClient
    from server.events import (
        Event,
        EventType,
        AgentOutputEvent,
        AgentCreatedEvent,
        AgentStatusChangedEvent,
        AgentCompletedEvent,
        PermissionRequestedEvent,
        PermissionResolvedEvent,
        ClarificationRequestedEvent,
        ClarificationQuestionEvent,
        ClarificationResolvedEvent,
        ReferenceSelectionRequestedEvent,
        ReferenceSelectionResolvedEvent,
        PlanUpdatedEvent,
        PlanClearedEvent,
        ToolCallStartEvent,
        ToolCallEndEvent,
        ToolOutputEvent,
        ContextUpdatedEvent,
        TurnCompletedEvent,
        SystemMessageEvent,
        InitProgressEvent,
        ErrorEvent,
        SessionListEvent,
        SessionInfoEvent,
        SessionDescriptionUpdatedEvent,
        CommandListEvent,
        ToolStatusEvent,
        HistoryEvent,
        WorkspaceMismatchRequestedEvent,
        WorkspaceMismatchResponseRequest,
        MidTurnPromptQueuedEvent,
        MidTurnPromptInjectedEvent,
    )

    # Load keybindings
    keybindings = load_keybindings()

    # Create agent registry for multi-agent support
    agent_registry = AgentRegistry()

    # Create input handler for completions - use default commands like direct mode
    # Server/plugin commands are added dynamically when CommandListEvent is received
    input_handler = InputHandler()

    # Session provider will be set after state variables are defined (below)

    # Create display with full features
    # server_formatted=True because server handles syntax highlighting and code validation
    display = PTDisplay(
        keybinding_config=keybindings,
        agent_registry=agent_registry,
        input_handler=input_handler,
        server_formatted=True,
    )

    # Create IPC client
    client = IPCClient(
        socket_path=socket_path,
        auto_start=auto_start,
        env_file=env_file,
    )

    # State tracking
    pending_permission_request: Optional[dict] = None
    pending_clarification_request: Optional[dict] = None
    pending_reference_selection_request: Optional[dict] = None
    pending_workspace_mismatch_request: Optional[dict] = None
    model_running = False
    should_exit = False
    server_commands: list = []  # Commands from server for help display
    available_sessions: list = []  # Sessions from server for completion
    available_tools: list = []  # Tools from server for completion
    available_models: list = []  # Models from server for completion

    # Queue for input from PTDisplay to async handler
    input_queue: asyncio.Queue[str] = asyncio.Queue()

    def get_sessions_for_completion():
        """Provider for session ID completion."""
        # Return session objects with session_id and description attributes
        # Prefer description (model-generated) over name for display
        class SessionInfo:
            def __init__(self, session_id, description=""):
                self.session_id = session_id
                self.description = description
        return [SessionInfo(s.get('id', ''), s.get('description', '') or s.get('name', '')) for s in available_sessions]

    # Set up session provider for completion
    input_handler.set_session_provider(get_sessions_for_completion)

    # Set up command completion provider for model command
    def model_completion_provider(command: str, args: list) -> list:
        """Provide completions for model command."""
        if command == "model":
            return [(model, "") for model in available_models]
        return []

    input_handler.set_command_completion_provider(
        model_completion_provider,
        {"model"}  # Commands that need subcommand completion
    )

    def on_input(text: str) -> None:
        """Callback when user submits input in PTDisplay."""
        try:
            # Schedule putting the text in the queue
            asyncio.get_event_loop().call_soon_threadsafe(
                lambda: input_queue.put_nowait(text)
            )
        except Exception:
            pass

    # Set up stop callback for Ctrl-C handling
    def on_stop() -> bool:
        """Handle stop request from display."""
        if model_running:
            try:
                asyncio.get_event_loop().call_soon_threadsafe(
                    lambda: asyncio.create_task(client.stop())
                )
            except Exception:
                pass
            return True
        return False

    def is_running() -> bool:
        """Check if model is currently running."""
        return model_running

    display.set_stop_callbacks(on_stop, is_running)

    # Connect to server before starting display
    print(f"Connecting to server at {socket_path}...")

    try:
        connected = await client.connect()
        if not connected:
            print("Connection failed: Server did not respond with handshake")
            return
        print("Connected!")

    except ConnectionError as e:
        print(f"Connection failed: {e}")
        return

    # Load release name for welcome message (shown when main agent is created)
    release_name = "Jaato Rich TUI Client"
    release_file = pathlib.Path(__file__).parent / "release_name.txt"
    if release_file.exists():
        release_name = release_file.read_text().strip()

    # IPC event tracing - use JAATO_TRACE_LOG if set
    from datetime import datetime as dt
    trace_file = os.environ.get("JAATO_TRACE_LOG")
    def ipc_trace(msg: str):
        if not trace_file:
            return  # Tracing disabled
        with open(trace_file, "a") as f:
            ts = dt.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[{ts}] [IPC] {msg}\n")

    # Track initialization progress for formatted display
    init_shown_header = False
    init_step_max_len = 30  # Fixed width for step names
    init_current_step = None  # Track current step for in-place updates

    async def handle_events():
        """Handle events from the server."""
        nonlocal pending_permission_request, pending_clarification_request, pending_reference_selection_request
        nonlocal pending_workspace_mismatch_request
        nonlocal model_running, should_exit
        nonlocal init_shown_header, init_current_step

        ipc_trace("Event handler starting")
        event_count = 0
        async for event in client.events():
            event_count += 1
            ipc_trace(f"<- [{event_count}] {type(event).__name__}")
            if should_exit:
                ipc_trace("  should_exit=True, breaking")
                break

            if isinstance(event, InitProgressEvent):
                # Handle initialization progress with in-place updates
                step_name = event.step
                status = event.status

                # Show header once
                if not init_shown_header:
                    display.add_system_message("Initializing session:", style="dim")
                    init_shown_header = True

                # Format step name with fixed width
                padded_name = step_name.ljust(init_step_max_len)

                if status == "running":
                    # Show step in progress
                    display.add_system_message(f"   {padded_name} ...", style="dim italic")
                    init_current_step = step_name
                elif status == "done":
                    # Update the same line to show completion
                    if init_current_step == step_name:
                        # Update in place
                        display.update_last_system_message(f"   {padded_name} OK", style="dim")
                    else:
                        # Step mismatch (shouldn't happen), add new line
                        display.add_system_message(f"   {padded_name} OK", style="dim")
                    init_current_step = None
                elif status == "error":
                    # Show error
                    msg = event.message or "ERROR"
                    if init_current_step == step_name:
                        display.update_last_system_message(f"   {padded_name} {msg}", style="dim red")
                    else:
                        display.add_system_message(f"   {padded_name} {msg}", style="dim red")
                    init_current_step = None

            elif isinstance(event, AgentOutputEvent):
                # Route output to the correct agent's buffer
                ipc_trace(f"  AgentOutputEvent: agent={event.agent_id}, source={event.source}, mode={event.mode}, len={len(event.text)}")
                buffer = agent_registry.get_buffer(event.agent_id)
                if buffer:
                    buffer.append(event.source, event.text, event.mode)
                    display.refresh()
                else:
                    # Agent not yet created - queue output for later
                    # This handles race condition where AgentOutputEvent arrives before AgentCreatedEvent
                    ipc_trace(f"  Queuing output for unknown agent: {event.agent_id}")
                    agent_registry.queue_output(event.agent_id, event.source, event.text, event.mode)

            elif isinstance(event, AgentCreatedEvent):
                # Register new agent
                agent_registry.create_agent(
                    agent_id=event.agent_id,
                    agent_type=event.agent_type,
                    name=event.agent_name,
                    profile_name=event.profile_name,
                    parent_agent_id=event.parent_agent_id,
                    icon_lines=event.icon_lines,
                )
                # Show welcome messages when main agent is created (now in correct buffer)
                if event.agent_id == "main":
                    display.add_system_message(release_name, style="bold cyan")
                    if input_handler.has_completion:
                        display.add_system_message(
                            "Tab completion enabled. Use @file to reference files, /command for slash commands.",
                            style="dim"
                        )
                    display.add_system_message(
                        "Type 'help' for commands, 'quit' to exit",
                        style="dim"
                    )
                display.refresh()

            elif isinstance(event, AgentStatusChangedEvent):
                ipc_trace(f"  AgentStatusChangedEvent: status={event.status}")
                if event.status == "active":
                    model_running = True
                    agent_registry.update_status(event.agent_id, "active")
                    # Auto-select the active agent so status bar shows its context
                    agent_registry.select_agent(event.agent_id)
                    # Start spinner on agent's buffer
                    buffer = agent_registry.get_buffer(event.agent_id)
                    if buffer:
                        buffer.start_spinner()
                        display.ensure_spinner_timer_running()
                elif event.status in ("done", "error"):
                    model_running = False
                    agent_registry.update_status(event.agent_id, event.status)
                    # Stop spinner on agent's buffer
                    buffer = agent_registry.get_buffer(event.agent_id)
                    if buffer:
                        buffer.stop_spinner()
                ipc_trace("  calling display.refresh()...")
                display.refresh()
                ipc_trace("  display.refresh() done, continuing loop...")

            elif isinstance(event, AgentCompletedEvent):
                agent_registry.mark_completed(event.agent_id)
                display.refresh()

            elif isinstance(event, PermissionRequestedEvent):
                ipc_trace(f"  PermissionRequestedEvent: tool={event.tool_name}, id={event.request_id}")
                # Show permission request
                pending_permission_request = {
                    "request_id": event.request_id,
                    "options": event.response_options,
                }

                # Use pre-formatted prompt lines from server if available (includes diff)
                if event.prompt_lines:
                    prompt_lines = event.prompt_lines
                else:
                    # Fall back to building prompt lines locally
                    from shared.ui_utils import build_permission_prompt_lines
                    prompt_lines = build_permission_prompt_lines(
                        tool_args=event.tool_args,
                        response_options=event.response_options,
                        include_tool_name=False,  # Tool name already shown in tool tree
                    )

                # Format prompt lines through the pipeline (for diff coloring, etc.)
                # For diffs, join lines and format as a block so the diff formatter
                # can detect the complete diff pattern and apply proper rendering
                if event.format_hint == "diff":
                    combined = "\n".join(prompt_lines)
                    formatted = agent_registry.format_text(combined, format_hint=event.format_hint)
                    formatted_lines = formatted.split("\n")
                else:
                    formatted_lines = [
                        agent_registry.format_text(line, format_hint=event.format_hint)
                        for line in prompt_lines
                    ]

                # Integrate into tool tree (same as direct mode)
                # Route to the agent that requested permission, not the selected agent
                buffer = agent_registry.get_buffer(event.agent_id) if event.agent_id else agent_registry.get_selected_buffer()
                if buffer:
                    buffer.set_tool_permission_pending(
                        event.tool_name,
                        formatted_lines
                    )
                display.refresh()
                # Enable permission input mode
                display.set_waiting_for_channel_input(True, event.response_options)

            elif isinstance(event, PermissionResolvedEvent):
                pending_permission_request = None
                display.set_waiting_for_channel_input(False)
                # Update tool tree with permission result
                # Route to the agent whose permission was resolved, not the selected agent
                buffer = agent_registry.get_buffer(event.agent_id) if event.agent_id else agent_registry.get_selected_buffer()
                if buffer:
                    buffer.set_tool_permission_resolved(event.tool_name, event.granted, event.method)
                    display.refresh()

            elif isinstance(event, ClarificationRequestedEvent):
                ipc_trace(f"  ClarificationRequestedEvent: tool={event.tool_name}, id={event.request_id}")
                # Initialize clarification in tool tree (same as direct mode)
                pending_clarification_request = {
                    "request_id": event.request_id,
                    "tool_name": event.tool_name,
                    "agent_id": event.agent_id,  # Track which agent requested clarification
                }
                # Route to the agent that requested clarification, not the selected agent
                buffer = agent_registry.get_buffer(event.agent_id) if event.agent_id else agent_registry.get_selected_buffer()
                if buffer:
                    buffer.set_tool_clarification_pending(event.tool_name, event.context_lines)
                display.refresh()

            elif isinstance(event, ClarificationQuestionEvent):
                ipc_trace(f"  ClarificationQuestionEvent: q{event.question_index}/{event.total_questions}")
                # Show clarification question in tool tree (same as direct mode)
                if not pending_clarification_request:
                    pending_clarification_request = {"request_id": event.request_id, "agent_id": event.agent_id}
                pending_clarification_request["current_question"] = event.question_index
                pending_clarification_request["total_questions"] = event.total_questions

                # Update tool tree with current question
                tool_name = pending_clarification_request.get("tool_name", "clarification")
                # Route to the agent that requested clarification, not the selected agent
                agent_id = event.agent_id or pending_clarification_request.get("agent_id")
                buffer = agent_registry.get_buffer(agent_id) if agent_id else agent_registry.get_selected_buffer()
                if buffer:
                    question_lines = event.question_text.split("\n") if event.question_text else []
                    buffer.set_tool_clarification_question(
                        tool_name,
                        event.question_index,
                        event.total_questions,
                        question_lines
                    )
                display.refresh()
                # Enable clarification input mode
                display.set_waiting_for_channel_input(True)

            elif isinstance(event, ClarificationResolvedEvent):
                ipc_trace(f"  ClarificationResolvedEvent: tool={event.tool_name}, qa_pairs={len(event.qa_pairs)}")
                # Update tool tree with resolution (same as direct mode)
                # Route to the agent whose clarification was resolved, not the selected agent
                buffer = agent_registry.get_buffer(event.agent_id) if event.agent_id else agent_registry.get_selected_buffer()
                if buffer:
                    # Convert [[q, a], ...] back to [(q, a), ...] for compatibility
                    qa_pairs = [(q, a) for q, a in event.qa_pairs] if event.qa_pairs else None
                    buffer.set_tool_clarification_resolved(event.tool_name, qa_pairs)
                    display.refresh()
                pending_clarification_request = None
                display.set_waiting_for_channel_input(False)

            elif isinstance(event, ReferenceSelectionRequestedEvent):
                ipc_trace(f"  ReferenceSelectionRequestedEvent: tool={event.tool_name}, id={event.request_id}")
                # Show reference selection prompt
                pending_reference_selection_request = {
                    "request_id": event.request_id,
                    "tool_name": event.tool_name,
                    "agent_id": event.agent_id,  # Track which agent requested selection
                }
                # Display the prompt lines in the output
                # Route to the agent that requested selection, not the selected agent
                buffer = agent_registry.get_buffer(event.agent_id) if event.agent_id else agent_registry.get_selected_buffer()
                if buffer:
                    for line in event.prompt_lines:
                        buffer.append("references", line + "\n", "append")
                    display.refresh()
                # Enable input mode for selection
                display.set_waiting_for_channel_input(True)

            elif isinstance(event, ReferenceSelectionResolvedEvent):
                ipc_trace(f"  ReferenceSelectionResolvedEvent: tool={event.tool_name}, selected={event.selected_ids}")
                pending_reference_selection_request = None
                display.set_waiting_for_channel_input(False)

            elif isinstance(event, PlanUpdatedEvent):
                # Update plan display - convert event steps to dict format expected by PlanPanel
                # Calculate progress from step statuses
                total_steps = len(event.steps)
                completed_steps = sum(
                    1 for step in event.steps
                    if step.get("status") == "completed"
                )
                percent = (completed_steps / total_steps * 100) if total_steps > 0 else 0

                plan_data = {
                    "title": event.plan_name or "Plan",
                    "steps": [
                        {
                            "description": step.get("content", ""),
                            "status": step.get("status", "pending"),
                            "active_form": step.get("active_form"),
                            "sequence": i + 1,  # 1-based for display
                        }
                        for i, step in enumerate(event.steps)
                    ],
                    "progress": {
                        "total": total_steps,
                        "completed": completed_steps,
                        "percent": round(percent, 1),
                    },
                }
                agent_id = getattr(event, 'agent_id', None)
                display.update_plan(plan_data, agent_id)

            elif isinstance(event, PlanClearedEvent):
                agent_id = getattr(event, 'agent_id', None)
                display.clear_plan(agent_id)

            elif isinstance(event, ToolCallStartEvent):
                # Use tool tree visualization (same as direct mode)
                buffer = agent_registry.get_buffer(event.agent_id)
                if buffer:
                    buffer.add_active_tool(event.tool_name, event.tool_args, call_id=event.call_id)
                    buffer.scroll_to_bottom()  # Auto-scroll when tool tree grows
                    display.refresh()
                else:
                    # Agent not yet created - queue event for later
                    ipc_trace(f"  Queuing tool start for unknown agent: {event.agent_id}")
                    agent_registry.queue_tool_start(event.agent_id, event.tool_name, event.tool_args, event.call_id)

            elif isinstance(event, ToolCallEndEvent):
                # Use tool tree visualization (same as direct mode)
                buffer = agent_registry.get_buffer(event.agent_id)
                if buffer:
                    buffer.mark_tool_completed(
                        event.tool_name,
                        event.success,
                        event.duration_seconds,
                        event.error_message,
                        call_id=event.call_id
                    )
                    buffer.scroll_to_bottom()  # Auto-scroll when tool tree updates
                    display.refresh()
                else:
                    # Agent not yet created - queue event for later
                    ipc_trace(f"  Queuing tool end for unknown agent: {event.agent_id}")
                    agent_registry.queue_tool_end(
                        event.agent_id, event.tool_name, event.success,
                        event.duration_seconds, event.error_message, event.call_id
                    )

            elif isinstance(event, ToolOutputEvent):
                # Live output chunk from running tool (tail -f style preview)
                buffer = agent_registry.get_buffer(event.agent_id)
                if buffer and event.call_id:
                    buffer.append_tool_output(event.call_id, event.chunk)
                    display.refresh()
                elif event.call_id:
                    # Agent not yet created - queue event for later
                    ipc_trace(f"  Queuing tool output for unknown agent: {event.agent_id}")
                    agent_registry.queue_tool_output(event.agent_id, event.call_id, event.chunk)

            elif isinstance(event, ContextUpdatedEvent):
                # Update context usage in agent registry (status bar reads from here)
                agent_id = event.agent_id or agent_registry.get_selected_agent_id()
                if agent_id:
                    agent_registry.update_context_usage(
                        agent_id=agent_id,
                        total_tokens=event.total_tokens,
                        prompt_tokens=event.prompt_tokens,
                        output_tokens=event.output_tokens,
                        turns=event.turns,
                        percent_used=event.percent_used,
                    )
                    # Update GC config if present in event
                    if event.gc_threshold is not None:
                        agent_registry.update_gc_config(agent_id, event.gc_threshold, event.gc_strategy)
                # Also update display (fallback if no registry)
                usage = {
                    "prompt_tokens": event.prompt_tokens,
                    "output_tokens": event.output_tokens,
                    "total_tokens": event.total_tokens,
                    "context_size": event.context_limit,
                    "percent_used": event.percent_used,
                }
                display.update_context_usage(usage)

            elif isinstance(event, TurnCompletedEvent):
                model_running = False
                display.refresh()

            elif isinstance(event, SystemMessageEvent):
                # Map style to actual prompt_toolkit style
                style = event.style if event.style else ""
                if style == "error":
                    style = "bold red"
                elif style == "warning":
                    style = "yellow"
                elif style == "success":
                    style = "green"
                elif style == "info":
                    style = "cyan"
                display.add_system_message(event.message, style=style)

            elif isinstance(event, ErrorEvent):
                display.add_system_message(
                    f"Error: {event.error_type}: {event.error}",
                    style="bold red"
                )

            elif isinstance(event, MidTurnPromptQueuedEvent):
                # Add to pending prompts bar above input
                display.add_pending_prompt(event.text)

            elif isinstance(event, MidTurnPromptInjectedEvent):
                # Remove from pending prompts bar when processed
                display.remove_pending_prompt(event.text)

            elif isinstance(event, SessionListEvent):
                # Store sessions for completion AND display
                nonlocal available_sessions
                available_sessions = event.sessions

                # Format session list for display with pager
                sessions = event.sessions

                if not sessions:
                    display.show_lines([
                        ("No sessions available.", "yellow"),
                        ("Use 'session new' to create one.", "dim"),
                    ])
                else:
                    lines = [
                        ("Sessions:", "bold"),
                        ("  Use 'session attach <id>' to switch sessions", "dim"),
                        ("", ""),
                    ]

                    for s in sessions:
                        is_current = s.get('is_current', False)
                        is_loaded = s.get('is_loaded', False)
                        # Use arrow for current session, bullet for loaded, circle for unloaded
                        if is_current:
                            status = "▶"
                        elif is_loaded:
                            status = "●"
                        else:
                            status = "○"
                        sid = s.get('id', 'unknown')
                        # Prefer description (model-generated) over name
                        desc = s.get('description', '') or s.get('name', '')
                        desc_part = f" - {desc}" if desc else ""
                        provider = s.get('model_provider', '')
                        model = s.get('model_name', '')
                        model_part = f" [{provider}/{model}]" if provider else ""
                        clients = s.get('client_count', 0)
                        clients_part = f", {clients} client(s)" if clients else ""
                        turns = s.get('turn_count', 0)
                        turns_part = f", {turns} turns" if turns else ""
                        workspace = s.get('workspace_path', '')

                        # Highlight current session
                        if is_current:
                            status_style = "bold cyan"
                        elif is_loaded:
                            status_style = "green"
                        else:
                            status_style = "dim"
                        lines.append((f"  {status} {sid}{desc_part}{model_part}{clients_part}{turns_part}", status_style))
                        # Show workspace on second line if available
                        if workspace:
                            # Shorten home directory to ~
                            import os
                            home = os.path.expanduser("~")
                            if workspace.startswith(home):
                                workspace = "~" + workspace[len(home):]
                            lines.append((f"      {workspace}", "dim"))

                    # Add legend
                    lines.append(("", ""))
                    lines.append(("  ▶ current  ● loaded  ○ on disk", "dim"))

                    display.show_lines(lines)

            elif isinstance(event, SessionInfoEvent):
                # Store state snapshot for local use (completion, display)
                # Note: available_sessions already declared nonlocal in SessionListEvent handler
                nonlocal available_tools, available_models
                if event.sessions:
                    available_sessions = event.sessions
                if event.tools:
                    available_tools = event.tools
                if event.models:
                    available_models = event.models
                # Update status bar with model info
                display.set_model_info(event.model_provider, event.model_name)
                # Update session bar with current session info
                current_session = next(
                    (s for s in event.sessions if s.get('id') == event.session_id),
                    None
                )
                if current_session:
                    display.set_session_info(
                        session_id=event.session_id,
                        description=current_session.get('description', ''),
                        workspace=current_session.get('workspace_path', ''),
                    )
                else:
                    display.set_session_info(session_id=event.session_id)
                display.refresh()

            elif isinstance(event, SessionDescriptionUpdatedEvent):
                # Update session description in local cache
                for s in available_sessions:
                    if s.get('id') == event.session_id:
                        s['description'] = event.description
                        # Update session bar if this is the current session
                        if display._session_id == event.session_id:
                            display.set_session_info(
                                session_id=event.session_id,
                                description=event.description,
                                workspace=display._session_workspace,
                            )
                        break

            elif isinstance(event, WorkspaceMismatchRequestedEvent):
                ipc_trace(f"  WorkspaceMismatchRequestedEvent: session={event.session_id}")
                # Store pending request
                pending_workspace_mismatch_request = {
                    "request_id": event.request_id,
                    "session_id": event.session_id,
                    "options": event.response_options,
                }
                # Show the prompt in output panel (not pager)
                prompt_text = "\n".join(event.prompt_lines)
                display.append_output("system", prompt_text, "write")
                display.refresh()
                # Enable input mode for response
                display.set_waiting_for_channel_input(True, event.response_options)

            elif isinstance(event, CommandListEvent):
                # Register server/plugin commands for tab completion
                nonlocal server_commands
                ipc_trace(f"  CommandListEvent: {len(event.commands)} commands")
                server_commands = event.commands  # Store for help display
                cmd_tuples = [
                    (cmd.get("name", ""), cmd.get("description", ""))
                    for cmd in event.commands
                ]
                if cmd_tuples:
                    input_handler.add_commands(cmd_tuples)
                    ipc_trace(f"    Registered {len(cmd_tuples)} commands")

            elif isinstance(event, ToolStatusEvent):
                # Format tools list for display with pager
                tool_status = event.tools
                ipc_trace(f"  ToolStatusEvent: {len(tool_status)} tools")

                if not tool_status:
                    display.show_lines([("No tools available.", "yellow")])
                else:
                    # Group tools by plugin
                    by_plugin = {}
                    for tool in tool_status:
                        plugin = tool.get('plugin', 'unknown')
                        if plugin not in by_plugin:
                            by_plugin[plugin] = []
                        by_plugin[plugin].append(tool)

                    # Count enabled/disabled
                    enabled_count = sum(1 for t in tool_status if t.get('enabled', True))
                    disabled_count = len(tool_status) - enabled_count

                    lines = [
                        (f"Tools ({enabled_count} enabled, {disabled_count} disabled):", "bold"),
                        ("  Use 'tools enable <name>' or 'tools disable <name>' to toggle", "dim"),
                        ("", ""),
                    ]

                    # Show result message if present
                    if event.message:
                        lines.insert(0, (event.message, "green"))
                        lines.insert(1, ("", ""))

                    for plugin_name in sorted(by_plugin.keys()):
                        tools = by_plugin[plugin_name]
                        lines.append((f"  [{plugin_name}]", "cyan"))

                        for tool in sorted(tools, key=lambda t: t['name']):
                            name = tool['name']
                            desc = tool.get('description', '')
                            enabled = tool.get('enabled', True)
                            status = "✓" if enabled else "○"
                            status_style = "green" if enabled else "red"
                            lines.append((f"    {status} {name}: {desc}", status_style if not enabled else "dim"))

                    display.show_lines(lines)

            elif isinstance(event, HistoryEvent):
                # Format and display conversation history
                ipc_trace(f"  HistoryEvent: {len(event.history)} messages")

                if not event.history:
                    display.show_lines([("No conversation history.", "yellow")])
                else:
                    turn_accounting = event.turn_accounting or []
                    lines = [
                        (f"Conversation History ({len(event.history)} messages, {len(turn_accounting)} turns):", "bold"),
                        ("", ""),
                    ]

                    turn_index = 0
                    for i, msg in enumerate(event.history):
                        role = msg.get('role', 'unknown')
                        parts = msg.get('parts', [])

                        # Format role header
                        if role == 'user':
                            lines.append((f"[User]", "cyan bold"))
                        elif role == 'model':
                            lines.append((f"[Model]", "green bold"))
                        else:
                            lines.append((f"[{role}]", "yellow bold"))

                        # Format parts
                        for part in parts:
                            part_type = part.get('type', 'unknown')
                            if part_type == 'text':
                                text = part.get('text', '')
                                # Truncate long text
                                if len(text) > 500:
                                    text = text[:500] + "..."
                                lines.append((f"  {text}", ""))
                            elif part_type == 'function_call':
                                name = part.get('name', 'unknown')
                                lines.append((f"  [Function Call: {name}]", "magenta"))
                            elif part_type == 'function_response':
                                name = part.get('name', 'unknown')
                                lines.append((f"  [Function Response: {name}]", "blue"))

                        # Show turn accounting at end of model turn
                        is_model = role == 'model'
                        is_last = (i == len(event.history) - 1)
                        next_is_user = (not is_last and
                                       event.history[i + 1].get('role') == 'user')

                        if is_model and (is_last or next_is_user):
                            if turn_index < len(turn_accounting):
                                acc = turn_accounting[turn_index]
                                prompt = acc.get('prompt', 0)
                                output = acc.get('output', 0)
                                total = acc.get('total', prompt + output)
                                lines.append((f"  --- Turn {turn_index + 1}: {total:,} tokens (in: {prompt:,}, out: {output:,}) ---", "dim"))
                                turn_index += 1

                        lines.append(("", ""))

                    display.show_lines(lines)

    async def handle_input():
        """Handle user input from the queue."""
        nonlocal pending_permission_request, pending_clarification_request, pending_reference_selection_request
        nonlocal pending_workspace_mismatch_request
        nonlocal model_running, should_exit

        # Yield control to let handle_events() start listening before we trigger session init
        await asyncio.sleep(0)

        # Request session - new or default
        if new_session:
            await client.create_session()
        else:
            await client.get_default_session()

        # Request available commands for tab completion
        await client.request_command_list()

        # Note: Session data (sessions, tools, models) is received via
        # SessionInfoEvent on connect - no separate request needed

        # Handle single prompt mode
        if single_prompt:
            model_running = True
            await client.send_message(single_prompt)
            # Wait for completion
            while model_running and not should_exit:
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.5)  # Wait for final events
            should_exit = True
            display.stop()
            return

        # Handle initial prompt
        if initial_prompt:
            model_running = True
            display.add_to_history(initial_prompt)  # Add to command history
            await client.send_message(initial_prompt)

        # Main input loop - get input from queue
        while not should_exit:
            try:
                # Wait for input with timeout to allow checking should_exit
                try:
                    text = await asyncio.wait_for(input_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                text = text.strip()
                if not text:
                    continue

                # Handle permission response
                if pending_permission_request:
                    ipc_trace(f"Sending permission response: {text} for {pending_permission_request['request_id']}")
                    await client.respond_to_permission(
                        pending_permission_request["request_id"],
                        text
                    )
                    continue

                # Handle workspace mismatch response
                if pending_workspace_mismatch_request:
                    ipc_trace(f"Sending workspace mismatch response: {text} for {pending_workspace_mismatch_request['request_id']}")
                    await client._send_event(WorkspaceMismatchResponseRequest(
                        request_id=pending_workspace_mismatch_request["request_id"],
                        response=text,
                    ))
                    pending_workspace_mismatch_request = None
                    display.set_waiting_for_channel_input(False)
                    continue

                # Handle clarification response
                if pending_clarification_request:
                    await client.respond_to_clarification(
                        pending_clarification_request["request_id"],
                        text
                    )
                    continue

                # Handle reference selection response
                if pending_reference_selection_request:
                    ipc_trace(f"Sending reference selection response: {text} for {pending_reference_selection_request['request_id']}")
                    await client.respond_to_reference_selection(
                        pending_reference_selection_request["request_id"],
                        text
                    )
                    continue

                # Handle commands
                text_lower = text.lower()
                cmd_parts = text.split()
                cmd = cmd_parts[0].lower() if cmd_parts else ""
                args = cmd_parts[1:] if len(cmd_parts) > 1 else []

                # Client-only commands
                if text_lower in ("exit", "quit", "q"):
                    should_exit = True
                    display.stop()
                    break
                elif text_lower == "stop":
                    await client.stop()
                    continue
                elif text_lower == "clear":
                    display.clear_output()
                    continue
                elif text_lower == "help":
                    # Show full help with pager using shared help text
                    from shared.client_commands import build_full_help_text
                    help_lines = build_full_help_text(server_commands)
                    display.show_lines(help_lines)
                    continue
                elif text_lower == "context":
                    # Show context usage (client-side, from agent registry)
                    selected_agent = agent_registry.get_selected_agent()
                    if not selected_agent:
                        display.show_lines([("Context tracking not available", "yellow")])
                    else:
                        usage = selected_agent.context_usage
                        lines = [
                            ("─" * 50, "dim"),
                            (f"Context Usage: {selected_agent.name}", "bold"),
                            (f"  Agent: {selected_agent.agent_id}", "dim"),
                            (f"  Total tokens: {usage.get('total_tokens', 0)}", "dim"),
                            (f"  Prompt tokens: {usage.get('prompt_tokens', 0)}", "dim"),
                            (f"  Output tokens: {usage.get('output_tokens', 0)}", "dim"),
                            (f"  Turns: {usage.get('turns', 0)}", "dim"),
                            (f"  Percent used: {usage.get('percent_used', 0):.1f}%", "dim"),
                            ("─" * 50, "dim"),
                        ]
                        display.show_lines(lines)
                    continue

                # Tools command - forward to server
                elif cmd == "tools":
                    subcmd = args[0] if args else "list"
                    subargs = args[1:] if len(args) > 1 else []
                    await client.execute_command(f"tools.{subcmd}", subargs)
                    continue

                # Session subcommands - forward to server as session.<subcommand>
                elif cmd == "session":
                    subcmd = args[0] if args else "list"
                    subargs = args[1:] if len(args) > 1 else []
                    await client.execute_command(f"session.{subcmd}", subargs)
                    continue

                # History command - request from server
                elif cmd == "history":
                    await client.request_history()
                    continue

                # Keybindings command - handle locally using shared function
                elif cmd == "keybindings":
                    from shared.ui_utils import handle_keybindings_command
                    handle_keybindings_command(text, display)
                    continue

                # Other server commands (reset, plugin commands) - forward directly
                elif cmd in ("reset",):
                    await client.execute_command(cmd, args)
                    continue

                # Check if input matches any server/plugin command
                # (mcp, permissions, model, save, resume, etc.)
                # Server expects base command name + args (e.g., "model" + ["list"])
                matched_base_command = None
                command_args = []
                if server_commands:
                    input_lower = text.lower()
                    input_parts = text.split()

                    # Try to match input against known commands
                    # Commands are like "model", "model list", "mcp status", etc.
                    for srv_cmd in server_commands:
                        cmd_name = srv_cmd.get("name", "").lower()
                        cmd_parts = cmd_name.split()

                        if not cmd_parts:
                            continue

                        base_cmd = cmd_parts[0]  # e.g., "model", "mcp"

                        # Check if input starts with this base command
                        if input_lower == base_cmd or input_lower.startswith(base_cmd + " "):
                            matched_base_command = base_cmd
                            # All parts after base command are args
                            if len(input_parts) > 1:
                                command_args = input_parts[1:]
                            else:
                                command_args = []
                            break

                if matched_base_command:
                    # Forward to server as base command + args
                    await client.execute_command(matched_base_command, command_args)
                    continue

                # Send message to model
                model_running = True
                display.add_to_history(text)
                await client.send_message(text)

            except asyncio.CancelledError:
                break
            except Exception as e:
                display.add_system_message(f"Error: {e}", style="bold red")

    # Run everything concurrently
    try:
        # Start event handler
        event_task = asyncio.create_task(handle_events())

        # Start input handler
        input_task = asyncio.create_task(handle_input())

        # Run PTDisplay (this is the main UI loop)
        # Use run_input_loop_async which returns when display.stop() is called
        await display.run_input_loop_async(on_input, initial_prompt=None)

        # Clean up tasks
        should_exit = True
        input_task.cancel()
        event_task.cancel()

        try:
            await input_task
        except asyncio.CancelledError:
            pass
        try:
            await event_task
        except asyncio.CancelledError:
            pass

    finally:
        await client.disconnect()


def main():
    import argparse

    # Configure UTF-8 encoding for Windows console (before any output)
    from shared.console_encoding import configure_utf8_output
    configure_utf8_output()

    parser = argparse.ArgumentParser(
        description="Rich TUI client for Jaato AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Server Mode:
  To run the server separately, use:
    python -m server --ipc-socket /tmp/jaato.sock

  Then connect with:
    python rich_client.py --connect /tmp/jaato.sock

  Or let the client auto-start the server (default behavior).
        """,
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce verbose output"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Run a single prompt and exit (non-interactive mode)"
    )
    parser.add_argument(
        "--initial-prompt", "-i",
        type=str,
        help="Start with this prompt, then continue interactively"
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Model provider to use (e.g., 'google_genai', 'github_models'). "
             "Overrides JAATO_PROVIDER env var."
    )

    # Server connection arguments
    parser.add_argument(
        "--connect",
        metavar="SOCKET_PATH",
        type=str,
        help="Connect to an existing server via IPC socket. "
             "If not specified, runs in legacy mode (embedded client)."
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Don't auto-start server if not running (only with --connect)"
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Start with a new session instead of resuming the default (only with --connect)"
    )

    args = parser.parse_args()

    # Check TTY before proceeding (except for single prompt mode)
    if not sys.stdout.isatty() and not args.prompt:
        sys.exit(
            "Error: rich-client requires an interactive terminal.\n"
            "Use simple-client for non-TTY environments."
        )

    # Connection mode: connect to server via IPC
    if args.connect:
        import asyncio
        asyncio.run(run_ipc_mode(
            socket_path=args.connect,
            auto_start=not args.no_auto_start,
            env_file=args.env_file,
            initial_prompt=args.initial_prompt,
            single_prompt=args.prompt,
            new_session=args.new_session,
        ))
        return

    # Legacy mode: embedded JaatoClient (current behavior)
    client = RichClient(
        env_file=args.env_file,
        verbose=not args.quiet,
        provider=args.provider
    )

    if not client.initialize():
        sys.exit(1)

    try:
        if args.prompt:
            # Single prompt mode - run and exit (no TUI)
            response = client.run_prompt(args.prompt)
            print(response)
        elif args.initial_prompt:
            # Initial prompt mode - run prompt first, then continue interactively
            client.run_interactive(initial_prompt=args.initial_prompt)
        else:
            # Interactive mode
            client.run_interactive()
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
