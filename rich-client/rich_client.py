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
from shared.plugins.vision_capture import (
    VisionCapture,
    VisionCaptureFormatter,
    CaptureConfig,
    CaptureContext,
    CaptureFormat,
    create_formatter as create_vision_formatter,
)

# Reuse input handling from simple-client
from input_handler import InputHandler

# Rich TUI components
from pt_display import PTDisplay
from plan_reporter import create_live_reporter
from agent_registry import AgentRegistry
from keybindings import load_keybindings, detect_terminal, list_available_profiles
from theme import load_theme, list_available_themes

# Backend abstraction for mode-agnostic operation
from backend import Backend, DirectBackend, IPCBackend


def _capture_vision(
    buffer,
    vision_capture: VisionCapture,
    display_height: int,
    display_width: int,
    terminal_theme,
    context: CaptureContext,
    turn_index: int,
    agent_id: Optional[str],
):
    """Core vision capture logic shared between direct and IPC modes.

    Args:
        buffer: Output buffer to render.
        vision_capture: VisionCapture instance.
        display_height: Terminal height for 1:1 capture.
        display_width: Terminal width for 1:1 capture.
        terminal_theme: Theme for export styling.
        context: What triggered the capture.
        turn_index: Current turn index.
        agent_id: Selected agent ID.

    Returns:
        CaptureResult on success, None on failure.
    """
    panel = buffer.render_panel(height=display_height, width=display_width)
    return vision_capture.capture(
        panel,
        context=context,
        turn_index=turn_index,
        agent_id=agent_id,
        terminal_theme=terminal_theme,
    )


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
        self._input_handler.set_available_themes(list_available_themes())

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

        # Pending system hints to inject into next user message (e.g., screenshot paths)
        self._pending_system_hints: list[str] = []

        # Background model thread tracking
        self._model_thread: Optional[threading.Thread] = None
        self._model_running: bool = False
        self._pending_exit_confirmation: bool = False

        # Model info for status bar
        self._model_provider: str = ""
        self._model_name: str = ""

        # GC info for status bar (set during initialization if GC is configured)
        self._gc_threshold: Optional[float] = None
        self._gc_strategy: Optional[str] = None
        self._gc_target_percent: Optional[float] = None
        self._gc_continuous_mode: bool = False

        # UI hooks reference for signaling agent status changes
        self._ui_hooks: Optional[Any] = None

        # Vision capture for TUI screenshots
        self._vision_capture: Optional[VisionCapture] = None
        self._vision_formatter: Optional[VisionCaptureFormatter] = None

    def log(self, msg: str) -> None:
        """Log message to output panel."""
        if self.verbose and self._display:
            self._display.add_system_message(msg, style="system_highlight")

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

            # Update permission status in session bar if permissions command was executed
            if command.name.lower() == "permissions":
                self._update_permission_status_display()

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
                self._display.show_lines([(f"Error: {e}", "system_error")])
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

        # Load env vars for client-side components (OutputBuffer tracing, etc.)
        load_dotenv(self.env_file)

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

        # Note: Plugins with set_plugin_registry() are auto-wired during expose_all()
        # No manual wiring needed for artifact_tracker, file_edit, cli, references, etc.

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

        # Verify authentication before loading tools
        # For providers that support interactive login (like Anthropic OAuth),
        # this will trigger the login flow if credentials are not found
        self._trace(f"[auth] Starting verify_auth for provider: {self._model_provider}")

        def auth_message(msg: str) -> None:
            self._trace(f"[auth] {msg}")
            print(msg, flush=True)

        try:
            if not self._backend.verify_auth(allow_interactive=True, on_message=auth_message):
                print("Error: Authentication failed or was cancelled", flush=True)
                return False
        except Exception as e:
            print(f"Error: Authentication failed: {e}", flush=True)
            return False

        self._trace("[auth] verify_auth completed successfully")

        # Configure tools (only after auth is verified)
        self._backend.configure_tools(self.registry, self.permission_plugin, self.ledger)

        # Load GC configuration from .jaato/gc.json if present
        gc_result = load_gc_from_file(agent_name="main")
        if gc_result:
            gc_plugin, gc_config = gc_result
            self._backend.set_gc_plugin(gc_plugin, gc_config)
            # Store threshold and strategy for status bar display
            self._gc_threshold = gc_config.threshold_percent
            self._gc_target_percent = gc_config.target_percent
            self._gc_continuous_mode = gc_config.continuous_mode
            # Get strategy name from plugin (e.g., "gc_truncate" -> "truncate")
            plugin_name = getattr(gc_plugin, 'name', 'gc')
            self._gc_strategy = plugin_name.replace('gc_', '') if plugin_name.startswith('gc_') else plugin_name

        # Note: Plugins with set_session() are auto-wired during session.configure()
        # This includes thinking plugin, etc. No manual wiring needed.

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
                    style="system_warning"
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

            def on_agent_gc_config(self, agent_id, threshold, strategy, target_percent=None, continuous_mode=False):
                trace_fn(f"[on_agent_gc_config] agent_id={agent_id}, threshold={threshold}, strategy={strategy}, target={target_percent}, continuous={continuous_mode}")
                registry.update_gc_config(agent_id, threshold, strategy, target_percent, continuous_mode)
                if display:
                    display.refresh()

            def on_agent_history_updated(self, agent_id, history):
                registry.update_history(agent_id, history)

            def on_tool_call_start(self, agent_id, tool_name, tool_args, call_id=None):
                buffer = registry.get_buffer(agent_id)
                if buffer:
                    # Extract intent args (message, summary, etc.) and display as model text
                    # This shows the model's intent before the tool block, not collapsed in it
                    intent_arg_names = ("message", "summary", "intent", "rationale")
                    if tool_args:
                        for arg_name in intent_arg_names:
                            if arg_name in tool_args:
                                val = tool_args[arg_name]
                                if val and isinstance(val, str) and val.strip():
                                    buffer.append("model", val.strip(), "write")
                                    break  # Use first found intent arg
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

    def _update_permission_status_display(self) -> None:
        """Update the session bar permission status from the permission plugin.

        Should be called:
        - After initial setup
        - After any permission response that may change status (t/i/a responses)
        - After permission commands (suspend/resume/default)
        """
        if not self.permission_plugin or not self._display:
            return

        status = self.permission_plugin.get_permission_status()
        self._display.set_permission_status(
            effective_default=status.get("effective_default", "ask"),
            suspension_scope=status.get("suspension_scope"),
        )

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
        update_status = self._update_permission_status_display

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
            formatted_lines = [
                registry.format_text(line, format_hint=format_hint) for line in prompt_lines
            ]

            # Update the tool in the main agent's buffer
            buffer = registry.get_buffer("main")
            self._trace(f"on_permission_requested: buffer={buffer}, active_tools={len(buffer._active_tools) if buffer else 0}")
            if buffer:
                buffer.set_tool_permission_pending(tool_name, formatted_lines, format_hint)
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

            # Update permission status in session bar (may have changed due to t/i/a responses)
            update_status()

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

        # Subscribe to prompt library changes for dynamic tool refresh
        self._setup_prompt_library_subscription()

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

    def _setup_prompt_library_subscription(self) -> None:
        """Subscribe to prompt library changes for dynamic tool refresh.

        When new prompts are fetched, this ensures:
        1. Command completions are refreshed to include new prompts
        2. Tool schemas are available for model calls
        """
        if not self.registry:
            return

        prompt_library = self.registry.get_plugin('prompt_library')
        if not prompt_library or not hasattr(prompt_library, 'set_on_tools_changed'):
            return

        def on_prompts_changed(new_tools: list) -> None:
            """Handle notification that new prompts were fetched."""
            self._trace(f"Prompt library tools changed: {new_tools}")

            # Refresh command completion provider to pick up new prompts
            self._setup_command_completion_provider()

            # Log for user visibility
            if self._output_buffer and new_tools:
                tool_names = [t.replace('prompt.', '') for t in new_tools]
                self._output_buffer.append(
                    "system",
                    f"New prompt(s) available: {', '.join(tool_names)}",
                    "info"
                )

        prompt_library.set_on_tools_changed(on_prompts_changed)
        self._trace("Subscribed to prompt library tool changes")

        # Set up prompt provider for %prompt completion
        def get_prompts_for_completion():
            """Return list of prompts for completion dropdown."""
            try:
                prompts = prompt_library._discover_prompts()
                return list(prompts.values())
            except Exception:
                return []

        self._input_handler.set_prompt_provider(get_prompts_for_completion)

        # Set up prompt expander for %prompt reference processing
        def expand_prompt(name: str, params: dict) -> str:
            """Expand a prompt reference to its content."""
            try:
                result = prompt_library._execute_prompt_tool(name, params)
                if 'content' in result:
                    return result['content']
                return None
            except Exception:
                return None

        self._input_handler.set_prompt_expander(expand_prompt)

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
                self._display.add_system_message("Error: Client not initialized", style="system_error")
            return

        if self._model_running:
            if self._display:
                self._display.add_system_message("Model is already running", style="system_warning")
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
                    style="system_warning"
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
                    self._display.add_system_message("─" * 40, style="separator")
                    self._display.add_system_message("", style="system_info")

            except KeyboardInterrupt:
                self._trace("[model_thread] KeyboardInterrupt")
                if self._display:
                    self._display.add_system_message("[Interrupted]", style="system_warning")
            except Exception as e:
                self._trace(f"[model_thread] Exception: {e}")
                if self._display:
                    self._display.add_system_message(f"Error: {e}", style="system_error")
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
                                lines.append((line, "system_success"))
                            elif line.startswith('-') and not line.startswith('---'):
                                lines.append((line, "system_error"))
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

        # Handle exit confirmation response
        if self._pending_exit_confirmation:
            self._pending_exit_confirmation = False
            self._display.set_prompt(None)  # Restore default prompt
            choice = user_input.strip().lower()
            if choice == "c":
                # Cancel task and exit
                if self._backend:
                    client = self._backend.get_client()
                    if client and hasattr(client, 'stop'):
                        client.stop()
                self._display.add_system_message("Task cancelled.", style="system_warning")
                self._display.add_system_message("Goodbye!", style="system_info")
                self._display.stop()
                return
            else:
                # Return to session (includes 'r' and any other input)
                self._display.add_system_message("Returning to session.", style="hint")
                return

        if user_input.lower() in ('quit', 'exit', 'q'):
            # Check if a task is running
            if self._model_running:
                # Show confirmation dialog (direct mode - no detach option)
                self._display.add_system_message("", style="hint")
                self._display.add_system_message("Task in progress. Exiting will cancel the task.", style="system_warning")
                self._display.add_system_message("  [c] Cancel task and exit", style="hint")
                self._display.add_system_message("  [r] Return to session", style="hint")
                self._display.add_system_message("", style="hint")
                self._display.set_prompt("Choice [c/r]: ")
                self._pending_exit_confirmation = True
                return
            else:
                # No task running, exit immediately
                self._display.add_system_message("Goodbye!", style="system_info")
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

        if user_input.lower().startswith('theme'):
            self._handle_theme_command(user_input)
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

        if user_input.lower().startswith('screenshot'):
            self._handle_screenshot_command(user_input)
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

        # Expand file references (@file) and prompt references (%prompt)
        expanded_prompt = self._input_handler.expand_file_references(user_input)
        expanded_prompt = self._input_handler.expand_prompt_references(expanded_prompt)

        # Inject any pending system hints (e.g., screenshot paths) - hidden from user
        if self._pending_system_hints:
            hints = "\n".join(self._pending_system_hints)
            expanded_prompt = f"{hints}\n\n{expanded_prompt}"
            self._pending_system_hints.clear()

        # Start model in background thread (non-blocking)
        # This allows the event loop to continue running for permission prompts
        self._start_model_thread(expanded_prompt)

    def run_interactive(self, initial_prompt: Optional[str] = None) -> None:
        """Run the interactive TUI loop.

        Args:
            initial_prompt: Optional prompt to run before entering interactive mode.
        """
        # Create the display with input handler, agent registry, keybindings, and theme
        keybinding_config = load_keybindings()
        theme_config = load_theme()
        self._display = PTDisplay(
            input_handler=self._input_handler,
            agent_registry=self._agent_registry,
            keybinding_config=keybinding_config,
            theme_config=theme_config
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
        if (self._gc_threshold is not None or self._gc_continuous_mode) and self._agent_registry:
            self._agent_registry.update_gc_config(
                "main",
                self._gc_threshold,
                self._gc_strategy,
                self._gc_target_percent,
                self._gc_continuous_mode,
            )

        # Set up permission hooks for inline permission display in tool tree
        self._setup_permission_hooks()

        # Set initial permission status in session bar
        self._update_permission_status_display()

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
            style="system_version"
        )
        if self._input_handler.has_completion:
            self._display.add_system_message(
                "Tab completion enabled. Use @file for files, %prompt for skills.",
                style="system_info"
            )
        self._display.add_system_message(
            "Type 'help' for commands, Ctrl+G for editor, Ctrl+F for search.",
            style="system_info"
        )
        self._display.add_system_message("", style="system_info")

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

        if subcommand == "help":
            self._display.show_lines([
                ("Tools Command", "bold"),
                ("", ""),
                ("Manage tools available to the model. Tools can be enabled or disabled", ""),
                ("to control what capabilities the model has access to.", ""),
                ("", ""),
                ("USAGE", "bold"),
                ("    tools [subcommand] [args]", ""),
                ("", ""),
                ("SUBCOMMANDS", "bold"),
                ("    list              List all tools with their enabled/disabled status", "dim"),
                ("                      (this is the default when no subcommand is given)", "dim"),
                ("", ""),
                ("    enable <name>     Enable a specific tool by name", "dim"),
                ("    enable all        Enable all tools at once", "dim"),
                ("", ""),
                ("    disable <name>    Disable a specific tool by name", "dim"),
                ("    disable all       Disable all tools at once", "dim"),
                ("", ""),
                ("    help              Show this help message", "dim"),
                ("", ""),
                ("EXAMPLES", "bold"),
                ("    tools                    Show all tools and their status", "dim"),
                ("    tools list               Same as above", "dim"),
                ("    tools enable Bash        Enable the Bash tool", "dim"),
                ("    tools disable web_search Disable web search", "dim"),
                ("    tools enable all         Enable all tools", "dim"),
                ("", ""),
                ("NOTES", "bold"),
                ("    - Tool names are case-sensitive", "dim"),
                ("    - Disabled tools will not be available for the model to use", "dim"),
                ("    - Use 'tools list' to see available tool names", "dim"),
            ])
            return

        if subcommand == "enable":
            if not args:
                self._display.show_lines([
                    ("[Error: Specify a tool name or 'all']", "system_error"),
                    ("  Usage: tools enable <tool_name>", "dim"),
                    ("         tools enable all", "dim"),
                ])
                return
            self._tools_enable(args[0])
            return

        if subcommand == "disable":
            if not args:
                self._display.show_lines([
                    ("[Error: Specify a tool name or 'all']", "system_error"),
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

    def _init_vision_capture(self) -> None:
        """Initialize vision capture plugin and register formatter with pipeline.

        Sets up:
        - VisionCapture: Core capture utility for rendering panels to images
        - VisionCaptureFormatter: Pipeline plugin for observing output and auto-capture

        Format priority: Environment variable > Saved preference > Default (svg)
        Output directory: {cwd}/.jaato/vision
        """
        if self._vision_capture:
            return  # Already initialized

        # Use current working directory as workspace
        output_dir = os.path.join(os.getcwd(), '.jaato', 'vision')

        # Determine format: env var takes priority, then saved preference, then default
        format_map = {
            'svg': CaptureFormat.SVG,
            'png': CaptureFormat.PNG,
            'html': CaptureFormat.HTML,
        }

        env_format = os.environ.get('JAATO_VISION_FORMAT', '').lower()
        if env_format and env_format in format_map:
            format_str = env_format
        else:
            # Load saved preference
            from preferences import load_preference
            format_str = load_preference('vision_format', 'svg')

        capture_format = format_map.get(format_str, CaptureFormat.SVG)

        config = CaptureConfig(
            output_dir=output_dir,
            format=capture_format,
            width=self._display._width if self._display else 120,
        )

        # Initialize core capture utility
        self._vision_capture = VisionCapture()
        self._vision_capture.initialize(config)

        # Create formatter and register with pipeline
        self._vision_formatter = create_vision_formatter(
            capture_callback=self._on_vision_capture_triggered,
            auto_capture_on_turn_end=False,  # Manual only by default
        )

        # Register formatter with display's pipeline
        if self._display:
            self._display.register_formatter(self._vision_formatter)

    def _on_vision_capture_triggered(self, context: CaptureContext, turn_index: int) -> None:
        """Callback for formatter-triggered captures (e.g., auto-capture on turn end).

        Args:
            context: What triggered the capture.
            turn_index: Current turn index from the formatter.
        """
        result = self._do_vision_capture(context)
        if result and result.success:
            self._display.add_system_message(
                f"Auto-captured: {result.path}",
                style="hint"
            )

    def _do_vision_capture(self, context: CaptureContext):
        """Perform a vision capture of the current TUI state.

        Args:
            context: What triggered the capture.

        Returns:
            CaptureResult on success, None on failure.
        """
        if not self._display:
            return None

        # Initialize if needed
        self._init_vision_capture()

        # Get the selected agent's output buffer
        buffer = self._agent_registry.get_selected_buffer()
        if not buffer:
            buffer = self._display._output_buffer

        try:
            return _capture_vision(
                buffer=buffer,
                vision_capture=self._vision_capture,
                display_height=self._display._height,
                display_width=self._display._width,
                terminal_theme=self._display._theme.to_terminal_theme(),
                context=context,
                turn_index=len(self._original_inputs),
                agent_id=self._agent_registry._selected_agent_id,
            )
        except Exception as e:
            self._display.show_lines([
                ("[Screenshot failed]", "system_error"),
                (f"  Error: {e}", "dim"),
            ])
            return None

    def _handle_screenshot_command(self, user_input: str) -> None:
        """Handle the screenshot command for TUI vision capture.

        The command itself is intercepted client-side. By default, a system hint
        is injected to inform the model about the capture path.

        Usage:
            screenshot           - Capture TUI state (hint sent via plugin)
            screenshot format F  - Set output format (svg, png, html)
            screenshot auto      - Toggle auto-capture on turn end
            screenshot interval N - Set periodic capture interval (ms)
            screenshot help      - Show help

        Args:
            user_input: The full user input string starting with 'screenshot'.
        """
        parts = user_input.lower().split()
        subcommand = parts[1] if len(parts) > 1 else ""

        if subcommand == 'help':
            self._display.show_lines([
                ("Screenshot Command", "bold"),
                ("", ""),
                ("Capture the TUI state as an image for vision analysis or debugging.", ""),
                ("Captures can be sent to the model as hints or saved for later use.", ""),
                ("", ""),
                ("USAGE", "bold"),
                ("    screenshot [subcommand] [args]", ""),
                ("", ""),
                ("SUBCOMMANDS", "bold"),
                ("    (none)            Capture TUI and send path hint to model", "dim"),
                ("                      Model receives the path to read the image", "dim"),
                ("", ""),
                ("    nosend            Capture TUI without sending hint to model", "dim"),
                ("                      Useful for manual inspection", "dim"),
                ("", ""),
                ("    copy              Capture and copy to clipboard as PNG", "dim"),
                ("                      Requires clipboard support (xclip/pbcopy)", "dim"),
                ("", ""),
                ("    format [F]        Show or set output format", "dim"),
                ("                      Available: svg, png, html", "dim"),
                ("", ""),
                ("    auto              Toggle auto-capture on turn end", "dim"),
                ("                      Automatically captures after each model turn", "dim"),
                ("", ""),
                ("    interval <N>      Set periodic capture interval in ms", "dim"),
                ("                      Use 0 to disable (default)", "dim"),
                ("", ""),
                ("    delay <N>         Capture once after N seconds", "dim"),
                ("                      Default: 5 seconds", "dim"),
                ("", ""),
                ("    help              Show this help message", "dim"),
                ("", ""),
                ("EXAMPLES", "bold"),
                ("    screenshot                    Capture and hint model", "dim"),
                ("    screenshot nosend             Capture without hint", "dim"),
                ("    screenshot copy               Capture to clipboard", "dim"),
                ("    screenshot format png         Switch to PNG output", "dim"),
                ("    screenshot auto               Toggle auto-capture", "dim"),
                ("    screenshot interval 5000      Capture every 5 seconds", "dim"),
                ("    screenshot delay 3            Capture in 3 seconds", "dim"),
                ("", ""),
                ("OUTPUT FORMATS", "bold"),
                ("    svg               Scalable vector (default, best quality)", "dim"),
                ("    png               Raster image (requires cairosvg)", "dim"),
                ("    html              HTML with embedded styles", "dim"),
                ("", ""),
                ("OUTPUT DIRECTORY", "bold"),
                ("    Captures are saved to $JAATO_VISION_DIR", ""),
                ("    Default: /tmp/jaato_vision", "dim"),
                ("", ""),
                ("NOTES", "bold"),
                ("    - SVG format preserves text and is searchable", "dim"),
                ("    - PNG requires cairosvg package for conversion", "dim"),
                ("    - Auto-capture helps debug streaming output", "dim"),
                ("    - Hint includes <tui-screenshot> tag for model", "dim"),
            ])
            return

        if subcommand == 'format':
            # Set output format
            self._init_vision_capture()
            format_str = parts[2] if len(parts) > 2 else ""

            if not format_str:
                # Show current format
                current = self._vision_capture._config.format.value if self._vision_capture else "svg"
                self._display.add_system_message(f"Current format: {current}", "cyan")
                self._display.add_system_message("  Available: svg, png, html", "dim")
                return

            from shared.plugins.vision_capture.protocol import CaptureFormat
            format_map = {
                'svg': CaptureFormat.SVG,
                'png': CaptureFormat.PNG,
                'html': CaptureFormat.HTML,
            }

            if format_str not in format_map:
                self._display.add_system_message(f"[Invalid format: {format_str}]", "system_error")
                self._display.add_system_message("  Available: svg, png, html", "dim")
                return

            new_format = format_map[format_str]

            # Warn if PNG selected but cairosvg not available
            if new_format == CaptureFormat.PNG:
                try:
                    import cairosvg  # noqa: F401
                except ImportError:
                    self._display.add_system_message("[Warning: cairosvg not installed]", "yellow")
                    self._display.add_system_message("  PNG format requires cairosvg for SVG to PNG conversion.", "dim")
                    self._display.add_system_message("  Install with: pip install cairosvg", "dim")
                    self._display.add_system_message("  (also requires system libcairo2-dev)", "dim")
                    self._display.add_system_message("")
                    self._display.add_system_message("  Falling back to SVG format.", "dim")
                    new_format = CaptureFormat.SVG

            if self._vision_capture:
                self._vision_capture._config.format = new_format
                # Save preference for future sessions
                from preferences import save_preference
                save_preference('vision_format', new_format.value)
                self._display.add_system_message(f"Screenshot format set to: {new_format.value}", "cyan")
            return

        if subcommand == 'auto':
            # Toggle auto-capture mode
            self._init_vision_capture()
            if self._vision_formatter:
                current = self._vision_formatter._auto_capture_on_turn_end
                self._vision_formatter.set_auto_capture(not current)
                state = "enabled" if not current else "disabled"
                self._display.add_system_message(f"Auto-capture on turn end: {state}", "cyan")
            return

        if subcommand == 'interval':
            # Set periodic capture interval
            self._init_vision_capture()
            interval_str = parts[2] if len(parts) > 2 else ""
            try:
                interval_ms = int(interval_str) if interval_str else 0
                if self._vision_formatter:
                    self._vision_formatter.set_capture_interval(interval_ms)
                    if interval_ms > 0:
                        self._display.add_system_message(f"Periodic capture: every {interval_ms}ms during streaming", "cyan")
                    else:
                        self._display.add_system_message("Periodic capture: disabled", "cyan")
            except ValueError:
                self._display.add_system_message(f"[Invalid interval: {interval_str}]", "system_error")
                self._display.add_system_message("  Usage: screenshot interval <milliseconds>", "dim")
            return

        if subcommand == 'delay':
            # One-shot delayed capture
            self._init_vision_capture()
            delay_str = parts[2] if len(parts) > 2 else ""
            try:
                delay_sec = float(delay_str) if delay_str else 5.0
                if delay_sec <= 0:
                    self._display.add_system_message("[Delay must be positive]", "system_error")
                    return

                import threading
                from shared.plugins.vision_capture.protocol import CaptureContext

                def delayed_capture():
                    result = self._do_vision_capture(CaptureContext.USER_REQUESTED)
                    if result and result.success:
                        self._display.add_system_message(f"Delayed screenshot captured: {result.path}", "cyan")
                    elif result and not result.success:
                        self._display.add_system_message(f"[Delayed screenshot failed: {result.error}]", "system_error")

                timer = threading.Timer(delay_sec, delayed_capture)
                timer.daemon = True
                timer.start()
                self._display.add_system_message(f"Screenshot scheduled in {delay_sec}s", "cyan")
            except ValueError:
                self._display.add_system_message(f"[Invalid delay: {delay_str}]", "system_error")
                self._display.add_system_message("  Usage: screenshot delay <seconds>", "dim")
            return

        if subcommand == 'nosend':
            # Capture without sending hint to model
            result = self._do_vision_capture(CaptureContext.USER_REQUESTED)
            if result and result.success:
                self._display.add_system_message("Screenshot captured:", "system_success")
                self._display.add_system_message(f"  {result.path}", "cyan")
            elif result and not result.success:
                self._display.add_system_message("[Screenshot failed]", "system_error")
                self._display.add_system_message(f"  Error: {result.error}", "dim")
            return

        if subcommand == 'copy':
            # Capture and copy to clipboard (requires PNG format)
            from shared.plugins.vision_capture.protocol import CaptureFormat
            from clipboard import copy_image_to_clipboard

            self._init_vision_capture()

            # Save current format and temporarily switch to PNG for clipboard
            original_format = None
            if self._vision_capture:
                original_format = self._vision_capture._config.format
                if original_format != CaptureFormat.PNG:
                    self._vision_capture._config.format = CaptureFormat.PNG

            result = self._do_vision_capture(CaptureContext.USER_REQUESTED)

            # Restore original format
            if self._vision_capture and original_format is not None:
                self._vision_capture._config.format = original_format

            if result and result.success:
                # Copy to clipboard
                success, error_msg = copy_image_to_clipboard(result.path)
                if success:
                    self._display.add_system_message("Screenshot copied to clipboard:", "system_success")
                    self._display.add_system_message(f"  {result.path}", "cyan")
                else:
                    self._display.add_system_message("Screenshot captured but clipboard copy failed:", "system_warning")
                    self._display.add_system_message(f"  {result.path}", "cyan")
                    self._display.add_system_message(f"  ({error_msg})", "dim")
            elif result and not result.success:
                self._display.add_system_message("[Screenshot failed]", "system_error")
                self._display.add_system_message(f"  Error: {result.error}", "dim")
            return

        # Default: capture and send hint to model
        result = self._do_vision_capture(CaptureContext.USER_REQUESTED)
        if result and result.success:
            self._display.add_system_message("Screenshot captured:", "system_success")
            self._display.add_system_message(f"  {result.path}", "cyan")
            # Send hint to model as normal user message (queued if model is busy)
            # Use cwd as workspace root for relative paths
            hint = result.to_user_message(workspace_root=os.getcwd())
            self._start_model_thread(hint)
        elif result and not result.success:
            self._display.add_system_message("[Screenshot failed]", "system_error")
            self._display.add_system_message(f"  Error: {result.error}", "dim")

    def _handle_theme_command(self, user_input: str) -> None:
        """Handle the theme command with subcommands.

        Subcommands:
            theme           - Show current theme info
            theme reload    - Reload theme from config files
            theme <preset>  - Switch to an available theme

        Args:
            user_input: The full user input string starting with 'theme'.
        """
        from theme import load_theme, BUILTIN_THEMES, save_theme_preference, list_available_themes

        parts = user_input.strip().split(maxsplit=1)
        subcommand = parts[1].lower() if len(parts) > 1 else ""
        available = list_available_themes()

        if not subcommand:
            # Show current theme info
            theme = self._display.theme
            self._display.add_system_message(f"Current theme: {theme.name}", "system_info")
            self._display.add_system_message(f"Source: {theme.source_path}", "hint")
            self._display.add_system_message("")
            self._display.add_system_message("Base colors:", "system_info")
            for name in ["primary", "secondary", "success", "warning", "error", "muted"]:
                color = theme.get_color(name)
                self._display.add_system_message(f"  {name}: {color}", "hint")
            self._display.add_system_message("")
            self._display.add_system_message("Commands:", "system_info")
            self._display.add_system_message("  theme reload           - Reload from config files", "hint")
            self._display.add_system_message(f"  theme <preset>         - Switch preset ({', '.join(sorted(available))})", "hint")
            return

        if subcommand == "reload":
            new_theme = load_theme()
            self._display.set_theme(new_theme)
            # Refresh available themes list for completions
            self._input_handler.set_available_themes(list_available_themes())
            self._display.add_system_message(f"Theme reloaded: {new_theme.name}", "system_success")
            self._display.add_system_message(f"Source: {new_theme.source_path}", "hint")
            return

        if subcommand in BUILTIN_THEMES:
            new_theme = BUILTIN_THEMES[subcommand].copy()
            self._display.set_theme(new_theme)
            save_theme_preference(subcommand)  # Persist the selection
            self._display.add_system_message(f"Switched to '{subcommand}' theme", "system_success")
            return

        # Unknown subcommand
        self._display.add_system_message(f"Unknown theme command: {subcommand}", "system_warning")
        self._display.add_system_message(f"Available: reload, {', '.join(sorted(available))}", "hint")

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
            self._display.show_lines([("[Error: Registry not available]", "system_error")])
            return

        if name.lower() == "all":
            count = self.registry.enable_all_tools()
            self._refresh_session_tools()
            self._display.show_lines([(f"[Enabled all {count} tools]", "system_success")])
            return

        if self.registry.enable_tool(name):
            self._refresh_session_tools()
            self._display.show_lines([(f"[Enabled tool: {name}]", "system_success")])
        else:
            lines = [(f"[Error: Tool '{name}' not found]", "system_error")]
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
            self._display.show_lines([("[Error: Registry not available]", "system_error")])
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
            lines = [(f"[Error: Tool '{name}' not found]", "system_error")]
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
            lines.append((f"  📥 RESULT: {name} → {resp_str}", "system_success"))

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
            lines.append((f"  📋 EXEC RESULT: {output}", "system_success"))

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
                    (f"Session exported to: {result['filename']}", "system_success")
                ])
            else:
                self._display.show_lines([
                    (f"Export failed: {result.get('error', 'Unknown error')}", "system_error")
                ])
        except ImportError:
            self._display.show_lines([
                ("Session exporter not available", "yellow")
            ])
        except Exception as e:
            self._display.show_lines([
                (f"Export error: {e}", "system_error")
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
            ("  screenshot        - Capture TUI and send hint to model (nosend to skip)", "dim"),
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

def _get_ipc_vision_state(display):
    """Get or create vision capture state for IPC mode.

    State is stored on the display object to persist across calls.
    Format priority: Environment variable > Saved preference > Default (svg)
    Output directory: {cwd}/.jaato/vision
    """
    import os
    from shared.plugins.vision_capture import VisionCapture, VisionCaptureFormatter
    from shared.plugins.vision_capture.protocol import CaptureConfig, CaptureFormat

    if not hasattr(display, '_vision_capture'):
        # Use current working directory as workspace (same as what client sends to server)
        output_dir = os.path.join(os.getcwd(), '.jaato', 'vision')

        # Determine format: env var takes priority, then saved preference, then default
        format_map = {
            'svg': CaptureFormat.SVG,
            'png': CaptureFormat.PNG,
            'html': CaptureFormat.HTML,
        }

        env_format = os.environ.get('JAATO_VISION_FORMAT', '').lower()
        if env_format and env_format in format_map:
            format_str = env_format
        else:
            # Load saved preference
            from preferences import load_preference
            format_str = load_preference('vision_format', 'svg')

        capture_format = format_map.get(format_str, CaptureFormat.SVG)

        config = CaptureConfig(output_dir=output_dir, format=capture_format)
        display._vision_capture = VisionCapture()
        display._vision_capture.initialize(config)

    if not hasattr(display, '_vision_formatter'):
        display._vision_formatter = VisionCaptureFormatter()
        display.register_formatter(display._vision_formatter)

    return display._vision_capture, display._vision_formatter


def _queue_ipc_system_hint(display, hint: str) -> None:
    """Queue a system hint for injection into the next user message (IPC mode).

    Hints are stored on the display object to persist across calls.
    """
    if not hasattr(display, '_pending_system_hints'):
        display._pending_system_hints = []
    display._pending_system_hints.append(hint)


def _pop_ipc_system_hints(display) -> list:
    """Get and clear pending system hints (IPC mode).

    Returns:
        List of pending hint strings, or empty list if none.
    """
    if not hasattr(display, '_pending_system_hints'):
        return []
    hints = display._pending_system_hints
    display._pending_system_hints = []
    return hints


async def handle_screenshot_command_ipc(user_input: str, display, agent_registry, ipc_client) -> None:
    """Handle the screenshot command in IPC mode (client-side only).

    Args:
        user_input: The full user input string starting with 'screenshot'.
        display: The PTDisplay instance.
        agent_registry: The AgentRegistry for getting output buffer.
        ipc_client: The IPCClient for sending hints to model.
    """
    from shared.plugins.vision_capture.protocol import CaptureContext

    parts = user_input.lower().split()
    subcommand = parts[1] if len(parts) > 1 else ""

    if subcommand == 'help':
        display.show_lines([
            ("Screenshot Command", "bold"),
            ("", ""),
            ("Capture the TUI state as an image for vision analysis or debugging.", ""),
            ("Captures can be sent to the model as hints or saved for later use.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    screenshot [subcommand] [args]", ""),
            ("", ""),
            ("SUBCOMMANDS", "bold"),
            ("    (none)            Capture TUI and send path hint to model", "dim"),
            ("                      Model receives the path to read the image", "dim"),
            ("", ""),
            ("    nosend            Capture TUI without sending hint to model", "dim"),
            ("                      Useful for manual inspection", "dim"),
            ("", ""),
            ("    copy              Capture and copy to clipboard as PNG", "dim"),
            ("                      Requires clipboard support (xclip/pbcopy)", "dim"),
            ("", ""),
            ("    format [F]        Show or set output format", "dim"),
            ("                      Available: svg, png, html", "dim"),
            ("", ""),
            ("    auto              Toggle auto-capture on turn end", "dim"),
            ("                      Automatically captures after each model turn", "dim"),
            ("", ""),
            ("    interval <N>      Set periodic capture interval in ms", "dim"),
            ("                      Use 0 to disable (default)", "dim"),
            ("", ""),
            ("    delay <N>         Capture once after N seconds", "dim"),
            ("                      Default: 5 seconds", "dim"),
            ("", ""),
            ("    help              Show this help message", "dim"),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    screenshot                    Capture and hint model", "dim"),
            ("    screenshot nosend             Capture without hint", "dim"),
            ("    screenshot copy               Capture to clipboard", "dim"),
            ("    screenshot format png         Switch to PNG output", "dim"),
            ("    screenshot auto               Toggle auto-capture", "dim"),
            ("    screenshot interval 5000      Capture every 5 seconds", "dim"),
            ("    screenshot delay 3            Capture in 3 seconds", "dim"),
            ("", ""),
            ("OUTPUT FORMATS", "bold"),
            ("    svg               Scalable vector (default, best quality)", "dim"),
            ("    png               Raster image (requires cairosvg)", "dim"),
            ("    html              HTML with embedded styles", "dim"),
            ("", ""),
            ("OUTPUT DIRECTORY", "bold"),
            ("    Captures are saved to $JAATO_VISION_DIR", ""),
            ("    Default: /tmp/jaato_vision", "dim"),
            ("", ""),
            ("NOTES", "bold"),
            ("    - SVG format preserves text and is searchable", "dim"),
            ("    - PNG requires cairosvg package for conversion", "dim"),
            ("    - Auto-capture helps debug streaming output", "dim"),
            ("    - Hint includes <tui-screenshot> tag for model", "dim"),
        ])
        return

    if subcommand == 'format':
        # Set output format
        vision_capture, _ = _get_ipc_vision_state(display)
        format_str = parts[2] if len(parts) > 2 else ""

        if not format_str:
            # Show current format
            current = vision_capture._config.format.value
            display.add_system_message(f"Current format: {current}", "cyan")
            display.add_system_message("  Available: svg, png, html", "dim")
            return

        from shared.plugins.vision_capture.protocol import CaptureFormat
        format_map = {
            'svg': CaptureFormat.SVG,
            'png': CaptureFormat.PNG,
            'html': CaptureFormat.HTML,
        }

        if format_str not in format_map:
            display.add_system_message(f"[Invalid format: {format_str}]", "system_error")
            display.add_system_message("  Available: svg, png, html", "dim")
            return

        new_format = format_map[format_str]

        # Warn if PNG selected but cairosvg not available
        if new_format == CaptureFormat.PNG:
            try:
                import cairosvg  # noqa: F401
            except ImportError:
                display.add_system_message("[Warning: cairosvg not installed]", "yellow")
                display.add_system_message("  PNG format requires cairosvg for SVG to PNG conversion.", "dim")
                display.add_system_message("  Install with: pip install cairosvg", "dim")
                display.add_system_message("  (also requires system libcairo2-dev)", "dim")
                display.add_system_message("")
                display.add_system_message("  Falling back to SVG format.", "dim")
                new_format = CaptureFormat.SVG

        vision_capture._config.format = new_format
        # Save preference for future sessions
        from preferences import save_preference
        save_preference('vision_format', new_format.value)
        display.add_system_message(f"Screenshot format set to: {new_format.value}", "cyan")
        return

    if subcommand == 'auto':
        # Toggle auto-capture mode
        _, formatter = _get_ipc_vision_state(display)
        current = formatter._auto_capture_on_turn_end
        formatter.set_auto_capture(not current)

        # Set up capture callback if not already done
        if not formatter._capture_callback:
            def on_capture(context, turn_index):
                _do_vision_capture_ipc(display, agent_registry, context)
            formatter.set_capture_callback(on_capture)

        state = "enabled" if not current else "disabled"
        display.add_system_message(f"Auto-capture on turn end: {state}", "cyan")
        return

    if subcommand == 'interval':
        # Set periodic capture interval
        _, formatter = _get_ipc_vision_state(display)
        interval_str = parts[2] if len(parts) > 2 else ""
        try:
            interval_ms = int(interval_str) if interval_str else 0
            formatter.set_capture_interval(interval_ms)

            # Set up capture callback if not already done
            if not formatter._capture_callback:
                def on_capture(context, turn_index):
                    _do_vision_capture_ipc(display, agent_registry, context)
                formatter.set_capture_callback(on_capture)

            if interval_ms > 0:
                display.add_system_message(f"Periodic capture: every {interval_ms}ms during streaming", "cyan")
            else:
                display.add_system_message("Periodic capture: disabled", "cyan")
        except ValueError:
            display.add_system_message(f"[Invalid interval: {interval_str}]", "system_error")
            display.add_system_message("  Usage: screenshot interval <milliseconds>", "dim")
        return

    if subcommand == 'delay':
        # One-shot delayed capture
        delay_str = parts[2] if len(parts) > 2 else ""
        try:
            delay_sec = float(delay_str) if delay_str else 5.0
            if delay_sec <= 0:
                display.add_system_message("[Delay must be positive]", "system_error")
                return

            import threading

            def delayed_capture():
                result = _do_vision_capture_ipc(display, agent_registry, CaptureContext.USER_REQUESTED)
                if result and result.success:
                    display.add_system_message(f"Delayed screenshot captured: {result.path}", "cyan")
                elif result and not result.success:
                    display.add_system_message(f"[Delayed screenshot failed: {result.error}]", "system_error")

            timer = threading.Timer(delay_sec, delayed_capture)
            timer.daemon = True
            timer.start()
            display.add_system_message(f"Screenshot scheduled in {delay_sec}s", "cyan")
        except ValueError:
            display.add_system_message(f"[Invalid delay: {delay_str}]", "system_error")
            display.add_system_message("  Usage: screenshot delay <seconds>", "dim")
        return

    if subcommand == 'nosend':
        # Capture without sending hint to model
        result = _do_vision_capture_ipc(display, agent_registry, CaptureContext.USER_REQUESTED)
        if result and result.success:
            display.add_system_message("Screenshot captured:", "system_success")
            display.add_system_message(f"  {result.path}", "cyan")
        elif result and not result.success:
            display.add_system_message("[Screenshot failed]", "system_error")
            display.add_system_message(f"  Error: {result.error}", "dim")
        return

    if subcommand == 'copy':
        # Capture and copy to clipboard (requires PNG format)
        from shared.plugins.vision_capture.protocol import CaptureFormat
        from clipboard import copy_image_to_clipboard

        vision_capture, _ = _get_ipc_vision_state(display)

        # Save current format and temporarily switch to PNG for clipboard
        original_format = vision_capture._config.format
        if original_format != CaptureFormat.PNG:
            vision_capture._config.format = CaptureFormat.PNG

        result = _do_vision_capture_ipc(display, agent_registry, CaptureContext.USER_REQUESTED)

        # Restore original format
        vision_capture._config.format = original_format

        if result and result.success:
            # Copy to clipboard
            success, error_msg = copy_image_to_clipboard(result.path)
            if success:
                display.add_system_message("Screenshot copied to clipboard:", "system_success")
                display.add_system_message(f"  {result.path}", "cyan")
            else:
                display.add_system_message("Screenshot captured but clipboard copy failed:", "system_warning")
                display.add_system_message(f"  {result.path}", "cyan")
                display.add_system_message(f"  ({error_msg})", "dim")
        elif result and not result.success:
            display.add_system_message("[Screenshot failed]", "system_error")
            display.add_system_message(f"  Error: {result.error}", "dim")
        return

    # Default: capture and send hint to model
    result = _do_vision_capture_ipc(display, agent_registry, CaptureContext.USER_REQUESTED)
    if result and result.success:
        display.add_system_message("Screenshot captured:", "system_success")
        display.add_system_message(f"  {result.path}", "cyan")
        # Send hint to model as normal user message (queued if model is busy)
        # Use cwd as workspace root for relative paths
        hint = result.to_user_message(workspace_root=os.getcwd())
        await ipc_client.send_message(hint)
    elif result and not result.success:
        display.add_system_message("[Screenshot failed]", "system_error")
        display.add_system_message(f"  Error: {result.error}", "dim")


def _do_vision_capture_ipc(display, agent_registry, context):
    """Perform a vision capture in IPC mode."""
    try:
        vision_capture, _ = _get_ipc_vision_state(display)

        # Get the selected agent's output buffer
        buffer = agent_registry.get_selected_buffer()
        if not buffer:
            display.show_lines([
                ("[Screenshot failed]", "system_error"),
                ("  No output buffer available", "dim"),
            ])
            return None

        # Get terminal theme if available
        terminal_theme = None
        if hasattr(display, '_theme') and display._theme:
            terminal_theme = display._theme.to_terminal_theme()

        result = _capture_vision(
            buffer=buffer,
            vision_capture=vision_capture,
            display_height=getattr(display, '_height', 50),
            display_width=getattr(display, '_width', 120),
            terminal_theme=terminal_theme,
            context=context,
            turn_index=0,
            agent_id=agent_registry.get_selected_agent_id(),
        )

        # For auto/periodic captures, just show a brief message
        if context in (CaptureContext.TURN_END, CaptureContext.PERIODIC) and result and result.success:
            display.add_system_message(f"Auto-captured: {result.path}", style="hint")

        return result

    except Exception as e:
        display.show_lines([
            ("[Screenshot failed]", "system_error"),
            (f"  Error: {e}", "dim"),
        ])
        return None


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
    # Load env vars for client-side components (OutputBuffer tracing, etc.)
    load_dotenv(env_file)

    # Configure logging to NOT output to stderr (breaks TUI).
    # Redirect to trace file if JAATO_TRACE_LOG is set, otherwise suppress.
    import logging
    trace_log_path = os.environ.get("JAATO_TRACE_LOG")
    if trace_log_path:
        # Redirect logs to trace file
        file_handler = logging.FileHandler(trace_log_path)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        # Configure root logger to use file handler only
        root_logger = logging.getLogger()
        root_logger.handlers = [file_handler]
        root_logger.setLevel(logging.DEBUG)
    else:
        # Suppress all logging to console (would break TUI)
        logging.getLogger().handlers = [logging.NullHandler()]
        logging.getLogger().setLevel(logging.CRITICAL + 1)  # Suppress all

    import asyncio
    from pathlib import Path
    from ipc_client import IPCClient
    from ipc_recovery import (
        IPCRecoveryClient,
        ConnectionState,
        ConnectionStatus,
        ReconnectingError,
        ConnectionClosedError,
    )
    from client_config import get_recovery_config
    from server.events import (
        Event,
        EventType,
        AgentOutputEvent,
        AgentCreatedEvent,
        AgentStatusChangedEvent,
        AgentCompletedEvent,
        PermissionInputModeEvent,
        PermissionResolvedEvent,
        ClarificationInputModeEvent,
        ClarificationResolvedEvent,
        ReferenceSelectionRequestedEvent,
        ReferenceSelectionResolvedEvent,
        PlanUpdatedEvent,
        PlanClearedEvent,
        ToolCallStartEvent,
        ToolCallEndEvent,
        ToolOutputEvent,
        ContextUpdatedEvent,
        InstructionBudgetEvent,
        TurnCompletedEvent,
        TurnProgressEvent,
        SystemMessageEvent,
        HelpTextEvent,
        InitProgressEvent,
        ErrorEvent,
        RetryEvent,
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
        MidTurnInterruptEvent,
    )

    # Load keybindings and theme
    keybindings = load_keybindings()
    theme_config = load_theme()

    # Create agent registry for multi-agent support
    agent_registry = AgentRegistry()

    # Create input handler for completions - use default commands like direct mode
    # Server/plugin commands are added dynamically when CommandListEvent is received
    input_handler = InputHandler()
    input_handler.set_available_themes(list_available_themes())

    # Session provider will be set after state variables are defined (below)

    # Create display with full features
    # server_formatted=True because server handles syntax highlighting and code validation
    display = PTDisplay(
        keybinding_config=keybindings,
        theme_config=theme_config,
        agent_registry=agent_registry,
        input_handler=input_handler,
        server_formatted=True,
    )

    # Load recovery config
    workspace_path = Path.cwd()
    recovery_config = get_recovery_config(workspace_path)

    # Connection status tracking for UI
    connection_status_message: Optional[str] = None
    is_reconnecting: bool = False  # Track if we're in a reconnection (to suppress init messages)
    pending_history_request: bool = False  # Request history after reconnect completes

    def on_connection_status(status: ConnectionStatus):
        """Handle connection status changes from recovery client."""
        nonlocal connection_status_message, is_reconnecting, pending_history_request

        if status.state == ConnectionState.RECONNECTING:
            is_reconnecting = True
            if status.next_retry_in is not None:
                msg = f"Connection lost. Reconnecting in {status.next_retry_in:.0f}s... (attempt {status.attempt}/{status.max_attempts})"
            else:
                msg = f"Reconnecting... (attempt {status.attempt}/{status.max_attempts})"
            connection_status_message = msg
            # Update display if available
            try:
                display.set_connection_status(msg, style="warning")
            except Exception:
                pass  # Display may not be ready yet

        elif status.state == ConnectionState.CONNECTED:
            # Only show "Reestablishing session..." if we were reconnecting AND
            # there's actually a session to restore. If session_id is None, this
            # is effectively a fresh connection (nothing to reestablish).
            if is_reconnecting and status.session_id:
                # Show "Reestablishing session..." while waiting for full restoration
                connection_status_message = "Reestablishing session..."
                try:
                    display.set_connection_status("Reestablishing session...", style="warning")
                except Exception:
                    pass
            else:
                # Initial connection or no session to restore - clear status
                connection_status_message = None
                is_reconnecting = False  # Reset - nothing to reestablish
                try:
                    display.set_connection_status(None)
                except Exception:
                    pass

        elif status.state == ConnectionState.DISCONNECTED:
            # Disconnected without reconnection (e.g., auto-reconnect disabled)
            connection_status_message = "Disconnected"
            try:
                display.set_connection_status("Disconnected", style="warning")
            except Exception:
                pass

        elif status.state == ConnectionState.CLOSED:
            msg = f"Connection lost permanently: {status.last_error or 'Max retries exceeded'}"
            connection_status_message = msg
            try:
                display.add_system_message(msg, style="system_error_bold")
                display.set_connection_status("Disconnected", style="error")
            except Exception:
                pass

    # Create IPC client with recovery support
    client: IPCRecoveryClient = IPCRecoveryClient(
        socket_path=socket_path,
        config=recovery_config,
        auto_start=auto_start,
        env_file=env_file,
        workspace_path=workspace_path,
        on_status_change=on_connection_status,
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

    # Set up prompt provider for %prompt completion (local prompt discovery)
    try:
        from shared.plugins.prompt_library.plugin import PromptLibraryPlugin
        _prompt_lib = PromptLibraryPlugin()
        _prompt_lib.set_workspace_path(str(workspace_path))

        def get_prompts_for_completion():
            """Return list of prompts for completion dropdown."""
            try:
                prompts = _prompt_lib._discover_prompts()
                return list(prompts.values())
            except Exception:
                return []

        def expand_prompt(name: str, params: dict) -> str:
            """Expand a prompt reference to its content."""
            try:
                result = _prompt_lib._execute_prompt_tool(name, params)
                if 'content' in result:
                    return result['content']
                return None
            except Exception:
                return None

        input_handler.set_prompt_provider(get_prompts_for_completion)
        input_handler.set_prompt_expander(expand_prompt)
    except ImportError:
        pass  # Prompt library not available

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
        nonlocal model_running, should_exit, is_reconnecting
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
                # Suppress init progress messages during reconnection
                # The session is being restored, not created fresh - don't spam the output
                if is_reconnecting:
                    continue

                # Handle initialization progress with in-place updates
                step_name = event.step
                status = event.status

                # Show header once
                if not init_shown_header:
                    display.add_system_message("Initializing session:", style="system_info")
                    init_shown_header = True

                # Format step name with fixed width
                padded_name = step_name.ljust(init_step_max_len)

                # Prefix for matching the "running" message when updating
                step_prefix = f"   {padded_name}"

                if status == "running":
                    # Show step in progress
                    display.add_system_message(f"{step_prefix} ...", style="system_progress")
                    init_current_step = step_name
                elif status == "done":
                    # Update the same line to show completion
                    if init_current_step == step_name:
                        # Update in place - use prefix to find the correct message
                        # even if other system messages were added in between
                        display.update_last_system_message(
                            f"{step_prefix} OK", style="system_info", prefix=step_prefix
                        )
                    else:
                        # Step mismatch (shouldn't happen), add new line
                        display.add_system_message(f"{step_prefix} OK", style="system_info")
                    init_current_step = None
                elif status == "error":
                    # Show error
                    msg = event.message or "ERROR"
                    if init_current_step == step_name:
                        display.update_last_system_message(
                            f"{step_prefix} {msg}", style="system_init_error", prefix=step_prefix
                        )
                    else:
                        display.add_system_message(f"{step_prefix} {msg}", style="system_init_error")
                    init_current_step = None
                elif status == "pending":
                    # Show pending status (e.g., waiting for auth)
                    # Always add new line - don't update in place because other messages
                    # may have been added between "running" and "pending" (e.g., auth instructions)
                    msg = event.message or "PENDING"
                    display.add_system_message(f"   {padded_name} {msg}", style="system_warning")
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
                # Skip during reconnection - these are init-only messages
                if event.agent_id == "main" and not is_reconnecting:
                    display.add_system_message(release_name, style="system_version")
                    if input_handler.has_completion:
                        display.add_system_message(
                            "Tab completion enabled. Use @file for files, %prompt for skills.",
                            style="system_info"
                        )
                    display.add_system_message(
                        "Type 'help' for commands, Ctrl+G for editor, Ctrl+F for search.",
                        style="system_info"
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
                elif event.status == "idle":
                    # Subagent finished its turn but remains available for more prompts
                    # Stop spinner but keep model_running unchanged (main agent may still be active)
                    agent_registry.update_status(event.agent_id, event.status)
                    buffer = agent_registry.get_buffer(event.agent_id)
                    if buffer:
                        buffer.stop_spinner()
                ipc_trace("  calling display.refresh()...")
                display.refresh()
                ipc_trace("  display.refresh() done, continuing loop...")

            elif isinstance(event, AgentCompletedEvent):
                agent_registry.mark_completed(event.agent_id)
                display.refresh()

            elif isinstance(event, PermissionInputModeEvent):
                # New unified flow: content already emitted via AgentOutputEvent,
                # this event just signals input mode and updates tool tree status
                ipc_trace(f"  PermissionInputModeEvent: tool={event.tool_name}, id={event.request_id}, call_id={event.call_id}")
                pending_permission_request = {
                    "request_id": event.request_id,
                    "options": event.response_options,
                }
                # Update tool tree to show simple "awaiting approval" status
                buffer = agent_registry.get_buffer(event.agent_id) if event.agent_id else agent_registry.get_selected_buffer()
                if buffer:
                    buffer.set_tool_awaiting_approval(event.tool_name, call_id=event.call_id)
                display.set_waiting_for_channel_input(True, event.response_options)
                display.refresh()

            elif isinstance(event, PermissionResolvedEvent):
                # Check if this is a "session restored" clear event
                if event.method == "session_restored":
                    # Only show message if we actually had a pending request
                    if pending_permission_request:
                        ipc_trace("Clearing stale permission request after session restore")
                        display.set_waiting_for_channel_input(False)
                        display.add_system_message(
                            "Tool execution interrupted due to session recovery. "
                            "Send a message to continue.",
                            style="system_warning"
                        )
                    pending_permission_request = None
                else:
                    pending_permission_request = None
                    display.set_waiting_for_channel_input(False)
                    # Update tool tree with permission result
                    # Route to the agent whose permission was resolved, not the selected agent
                    buffer = agent_registry.get_buffer(event.agent_id) if event.agent_id else agent_registry.get_selected_buffer()
                    if buffer:
                        buffer.set_tool_permission_resolved(event.tool_name, event.granted, event.method)
                        display.refresh()

            elif isinstance(event, ClarificationInputModeEvent):
                # New unified flow: content already emitted via AgentOutputEvent,
                # this event just signals input mode
                ipc_trace(f"  ClarificationInputModeEvent: tool={event.tool_name}, q{event.question_index}/{event.total_questions}")
                if not pending_clarification_request:
                    pending_clarification_request = {"request_id": event.request_id, "agent_id": event.agent_id}
                pending_clarification_request["current_question"] = event.question_index
                pending_clarification_request["total_questions"] = event.total_questions
                pending_clarification_request["tool_name"] = event.tool_name
                # Update tool tree with simple "awaiting input" status
                buffer = agent_registry.get_buffer(event.agent_id) if event.agent_id else agent_registry.get_selected_buffer()
                if buffer:
                    buffer.set_tool_awaiting_clarification(event.tool_name, event.question_index, event.total_questions)
                display.set_waiting_for_channel_input(True)
                display.refresh()

            elif isinstance(event, ClarificationResolvedEvent):
                ipc_trace(f"  ClarificationResolvedEvent: tool={event.tool_name}, qa_pairs={len(event.qa_pairs)}")
                # Check if this is a "session restored" clear event (empty request_id and tool_name)
                if not event.request_id and not event.tool_name:
                    # Only show message if we actually had a pending request
                    if pending_clarification_request:
                        ipc_trace("Clearing stale clarification request after session restore")
                        display.set_waiting_for_channel_input(False)
                        display.add_system_message(
                            "Tool execution interrupted due to session recovery. "
                            "Send a message to continue.",
                            style="system_warning"
                        )
                    pending_clarification_request = None
                else:
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
                # Check if this is a "session restored" clear event (empty request_id and tool_name)
                if not event.request_id and not event.tool_name:
                    # Only show message if we actually had a pending request
                    if pending_reference_selection_request:
                        ipc_trace("Clearing stale reference selection request after session restore")
                        display.set_waiting_for_channel_input(False)
                        display.add_system_message(
                            "Tool execution interrupted due to session recovery. "
                            "Send a message to continue.",
                            style="system_warning"
                        )
                    pending_reference_selection_request = None
                else:
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

                # Build step dicts, including cross-agent dependency info
                plan_steps = []
                for i, step in enumerate(event.steps):
                    step_data = {
                        "description": step.get("content", ""),
                        "status": step.get("status", "pending"),
                        "active_form": step.get("active_form"),
                        "sequence": i + 1,  # 1-based for display
                    }
                    # Include cross-agent dependency fields if present
                    if step.get("blocked_by"):
                        step_data["blocked_by"] = step["blocked_by"]
                    if step.get("depends_on"):
                        step_data["depends_on"] = step["depends_on"]
                    if step.get("received_outputs"):
                        step_data["received_outputs"] = step["received_outputs"]
                    plan_steps.append(step_data)

                plan_data = {
                    "title": event.plan_name or "Plan",
                    "steps": plan_steps,
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
                    # add_active_tool() calls scroll_to_show_tool_tree() internally
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
                    if event.gc_threshold is not None or event.gc_continuous_mode:
                        agent_registry.update_gc_config(
                            agent_id,
                            event.gc_threshold,
                            event.gc_strategy,
                            event.gc_target_percent,
                            event.gc_continuous_mode,
                        )
                # Also update display (fallback if no registry)
                usage = {
                    "prompt_tokens": event.prompt_tokens,
                    "output_tokens": event.output_tokens,
                    "total_tokens": event.total_tokens,
                    "context_size": event.context_limit,
                    "percent_used": event.percent_used,
                }
                display.update_context_usage(usage)

            elif isinstance(event, InstructionBudgetEvent):
                # Update budget panel with new budget data
                if hasattr(display, 'update_instruction_budget'):
                    display.update_instruction_budget(event.agent_id, event.budget_snapshot)
                # Also derive toolbar context usage from budget snapshot
                # This ensures the toolbar shows accurate context consumption at startup
                # (before any ContextUpdatedEvent is emitted)
                budget = event.budget_snapshot
                if budget and agent_registry:
                    agent_registry.update_context_usage(
                        agent_id=event.agent_id,
                        total_tokens=budget.get('total_tokens', 0),
                        prompt_tokens=0,  # Not tracked in budget, but not needed for toolbar %
                        output_tokens=0,  # Not tracked in budget, but not needed for toolbar %
                        turns=0,  # Not tracked in budget
                        percent_used=budget.get('utilization_percent', 0),
                    )
                # Also update display directly (fallback if no registry)
                if budget:
                    usage = {
                        "total_tokens": budget.get('total_tokens', 0),
                        "context_size": budget.get('context_limit', 0),
                        "percent_used": budget.get('utilization_percent', 0),
                    }
                    display.update_context_usage(usage)

            elif isinstance(event, TurnProgressEvent):
                # Update context usage with incremental progress during turn
                agent_id = event.agent_id or agent_registry.get_selected_agent_id()
                if agent_id and agent_registry:
                    agent_registry.update_context_usage(
                        agent_id=agent_id,
                        total_tokens=event.total_tokens,
                        prompt_tokens=event.prompt_tokens,
                        output_tokens=event.output_tokens,
                        turns=0,  # Not updated during turn
                        percent_used=event.percent_used,
                    )
                # Update display status bar
                usage = {
                    "prompt_tokens": event.prompt_tokens,
                    "output_tokens": event.output_tokens,
                    "total_tokens": event.total_tokens,
                    "context_size": event.context_limit,
                    "percent_used": event.percent_used,
                }
                display.update_context_usage(usage)

            elif isinstance(event, TurnCompletedEvent):
                # Flush the output buffer to ensure all pending content from this turn
                # is rendered before the next turn starts. This prevents late-arriving
                # chunks from being concatenated with chunks from a new turn.
                buffer = agent_registry.get_buffer(event.agent_id)
                if buffer:
                    buffer.flush()
                model_running = False
                display.refresh()

            elif isinstance(event, SystemMessageEvent):
                # Map style to semantic style names
                style = event.style if event.style else "system_info"
                if style == "error":
                    style = "system_error_bold"
                elif style == "warning":
                    style = "system_warning"
                elif style == "success":
                    style = "system_success"
                elif style == "info":
                    style = "system_highlight"
                display.add_system_message(event.message, style=style)

            elif isinstance(event, HelpTextEvent):
                # Display help text using the pager
                # Lines are (text, style) tuples
                display.show_lines(event.lines)

            elif isinstance(event, ErrorEvent):
                display.add_system_message(
                    f"Error: {event.error_type}: {event.error}",
                    style="system_error_bold"
                )

            elif isinstance(event, RetryEvent):
                # Show retry notification with countdown
                display.add_system_message(
                    f"[Retry {event.attempt}/{event.max_attempts}] {event.error_type}: waiting {event.delay:.1f}s before retry...",
                    style="system_warning"
                )

            elif isinstance(event, MidTurnPromptQueuedEvent):
                # Add to pending prompts bar above input
                display.add_pending_prompt(event.text)

            elif isinstance(event, MidTurnPromptInjectedEvent):
                # Remove from pending prompts bar when processed
                display.remove_pending_prompt(event.text)

            elif isinstance(event, MidTurnInterruptEvent):
                # Streaming was interrupted to process user prompt
                # Show a brief notification that the model is pivoting to user input
                ipc_trace(f"  MidTurnInterruptEvent: partial={event.partial_response_chars} chars")
                display.add_system_message(
                    f"[Pivoting to your input...]",
                    style="system_info"
                )

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

                # Track session ID for recovery reattachment
                if event.session_id:
                    client.set_session_id(event.session_id)

                # If we were reconnecting, session attachment is now complete
                if is_reconnecting:
                    is_reconnecting = False
                    connection_status_message = None
                    try:
                        display.set_connection_status(None)  # Clear "Reestablishing session..."
                        display.add_system_message("Session restored!", style="system_success")
                    except Exception:
                        pass

                # Restore command history for prompt up/down arrow navigation
                if event.user_inputs:
                    for user_input in event.user_inputs:
                        display.add_to_history(user_input)
                    ipc_trace(f"  Restored {len(event.user_inputs)} inputs to prompt history")

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
                        lines.insert(0, (event.message, "system_success"))
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
        pending_exit_confirmation = False

        ipc_trace("Input handler starting")

        # Yield control to let handle_events() start listening before we trigger session init
        await asyncio.sleep(0)

        ipc_trace("Input handler: requesting session")
        # Request session - new or default
        try:
            if new_session:
                await client.create_session()
            else:
                await client.get_default_session()
            ipc_trace("Input handler: session requested")
        except Exception as e:
            ipc_trace(f"Input handler: session request failed: {e}")
            raise

        # Request available commands for tab completion
        ipc_trace("Input handler: requesting command list")
        await client.request_command_list()
        ipc_trace("Input handler: command list requested")

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
                # Allow empty input for clarification (optional answers without default)
                if not text and not pending_clarification_request:
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

                # Handle exit confirmation response
                if pending_exit_confirmation:
                    choice = text_lower
                    pending_exit_confirmation = False
                    display.set_prompt(None)  # Restore default prompt
                    display.set_waiting_for_channel_input(False)

                    session_id = client.session_id or "unknown"
                    socket_path = client.socket_path

                    if choice == "c" and model_running:
                        # Cancel task but keep session
                        await client.stop()
                        display.add_system_message("Task cancelled. Session preserved.", style="system_warning")
                        display.add_system_message("", style="hint")
                        display.add_system_message("To reconnect:", style="system_info")
                        display.add_system_message(f"  python rich_client.py --connect {socket_path}", style="system_info")
                        display.add_system_message(f"Session ID: {session_id}", style="hint")
                        should_exit = True
                        display.stop()
                        break
                    elif choice == "d":
                        # Detach - keep session alive
                        display.add_system_message("", style="hint")
                        if model_running:
                            display.add_system_message("Task will continue running on the server.", style="system_success")
                        else:
                            display.add_system_message("Session preserved on server.", style="system_success")
                        display.add_system_message("", style="hint")
                        display.add_system_message("To reconnect:", style="system_info")
                        display.add_system_message(f"  python rich_client.py --connect {socket_path}", style="system_info")
                        display.add_system_message("", style="hint")
                        display.add_system_message(f"Session ID: {session_id}", style="hint")
                        display.add_system_message("", style="hint")
                        should_exit = True
                        display.stop()
                        break
                    elif choice == "e":
                        # End session - delete from server
                        if model_running:
                            await client.stop()
                        # TODO: Add client.delete_session() when available
                        display.add_system_message("Session ended.", style="system_warning")
                        should_exit = True
                        display.stop()
                        break
                    else:
                        # Return to session (includes 'r' and any other input)
                        display.add_system_message("Returning to session.", style="hint")
                        continue

                # Client-only commands
                if text_lower in ("exit", "quit", "q"):
                    # Show confirmation dialog for session lifecycle
                    pending_exit_confirmation = True

                    display.add_system_message("", style="hint")
                    if model_running:
                        display.add_system_message("Task in progress. What would you like to do?", style="system_warning")
                        display.add_system_message("  [c] Cancel task and exit (session preserved)", style="hint")
                        display.add_system_message("  [d] Detach (task continues in background)", style="hint")
                        display.add_system_message("  [e] End session (cancel task and delete session)", style="hint")
                        display.add_system_message("  [r] Return to session", style="hint")
                        display.add_system_message("", style="hint")
                        display.set_prompt("Choice [c/d/e/r]: ")
                        # Create simple response options for input filtering
                        exit_options = [
                            type('Option', (), {'short': 'c', 'full': 'cancel', 'description': 'Cancel task'})(),
                            type('Option', (), {'short': 'd', 'full': 'detach', 'description': 'Detach'})(),
                            type('Option', (), {'short': 'e', 'full': 'end', 'description': 'End session'})(),
                            type('Option', (), {'short': 'r', 'full': 'return', 'description': 'Return'})(),
                        ]
                    else:
                        display.add_system_message("Exit options:", style="system_info")
                        display.add_system_message("  [d] Detach (keep session, can reconnect later)", style="hint")
                        display.add_system_message("  [e] End session (delete session from server)", style="hint")
                        display.add_system_message("  [r] Return to session", style="hint")
                        display.add_system_message("", style="hint")
                        display.set_prompt("Choice [d/e/r]: ")
                        exit_options = [
                            type('Option', (), {'short': 'd', 'full': 'detach', 'description': 'Detach'})(),
                            type('Option', (), {'short': 'e', 'full': 'end', 'description': 'End session'})(),
                            type('Option', (), {'short': 'r', 'full': 'return', 'description': 'Return'})(),
                        ]
                    display.set_waiting_for_channel_input(True, exit_options)
                    continue
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

                # Screenshot command - handle locally (client-side only)
                elif cmd == "screenshot":
                    await handle_screenshot_command_ipc(text, display, agent_registry, client)
                    continue

                # Theme command - handle locally
                elif cmd == "theme":
                    from theme import load_theme, BUILTIN_THEMES, save_theme_preference, list_available_themes
                    subcmd = args[0].lower() if args else ""
                    available = list_available_themes()

                    if not subcmd:
                        # Show current theme info
                        theme = display.theme
                        display.add_system_message(f"Current theme: {theme.name}", "system_info")
                        display.add_system_message(f"Source: {theme.source_path}", "hint")
                        display.add_system_message("")
                        display.add_system_message("Base colors:", "system_info")
                        for name in ["primary", "secondary", "success", "warning", "error", "muted"]:
                            color = theme.get_color(name)
                            display.add_system_message(f"  {name}: {color}", "hint")
                        display.add_system_message("")
                        display.add_system_message("Commands:", "system_info")
                        display.add_system_message("  theme reload           - Reload from config files", "hint")
                        display.add_system_message(f"  theme <preset>         - Switch preset ({', '.join(sorted(available))})", "hint")
                    elif subcmd == "reload":
                        new_theme = load_theme()
                        display.set_theme(new_theme)
                        # Refresh available themes list for completions
                        input_handler.set_available_themes(list_available_themes())
                        display.add_system_message(f"Theme reloaded: {new_theme.name}", "system_success")
                        display.add_system_message(f"Source: {new_theme.source_path}", "hint")
                    elif subcmd == "help":
                        display.show_lines([
                            ("Theme Command", "bold"),
                            ("", ""),
                            ("Manage the visual theme of the client. Themes control colors,", ""),
                            ("styles, and the overall appearance of the interface.", ""),
                            ("", ""),
                            ("USAGE", "bold"),
                            ("    theme [subcommand]", ""),
                            ("", ""),
                            ("SUBCOMMANDS", "bold"),
                            ("    (none)            Show current theme info and available commands", "dim"),
                            ("", ""),
                            ("    reload            Reload theme from configuration files", "dim"),
                            ("                      Picks up changes from theme.json", "dim"),
                            ("", ""),
                            (f"    <preset>          Switch to a built-in theme preset", "dim"),
                            (f"                      Available: {', '.join(sorted(available))}", "dim"),
                            ("", ""),
                            ("    help              Show this help message", "dim"),
                            ("", ""),
                            ("EXAMPLES", "bold"),
                            ("    theme                   Show current theme and colors", "dim"),
                            ("    theme dark              Switch to dark theme", "dim"),
                            ("    theme light             Switch to light theme", "dim"),
                            ("    theme high-contrast     Switch to high-contrast theme", "dim"),
                            ("    theme reload            Reload from config files", "dim"),
                            ("", ""),
                            ("CUSTOM THEMES", "bold"),
                            ("    Create a theme.json file in .jaato/ or ~/.jaato/ with:", ""),
                            ("", ""),
                            ('    {', "dim"),
                            ('      "colors": {', "dim"),
                            ('        "primary": "#007ACC",', "dim"),
                            ('        "secondary": "#6C757D",', "dim"),
                            ('        "success": "#28A745",', "dim"),
                            ('        "warning": "#FFC107",', "dim"),
                            ('        "error": "#DC3545",', "dim"),
                            ('        "muted": "#6C757D",', "dim"),
                            ('        "background": "#1E1E1E",', "dim"),
                            ('        "surface": "#252526",', "dim"),
                            ('        "text": "#D4D4D4",', "dim"),
                            ('        "text_muted": "#808080"', "dim"),
                            ('      }', "dim"),
                            ('    }', "dim"),
                            ("", ""),
                            ("CONFIGURATION FILES", "bold"),
                            ("    .jaato/theme.json       Project-level theme", "dim"),
                            ("    ~/.jaato/theme.json     User-level theme", "dim"),
                        ])
                    elif subcmd in BUILTIN_THEMES:
                        new_theme = BUILTIN_THEMES[subcmd].copy()
                        display.set_theme(new_theme)
                        save_theme_preference(subcmd)  # Persist the selection
                        display.add_system_message(f"Switched to '{subcmd}' theme", "system_success")
                    else:
                        display.add_system_message(f"Unknown theme command: {subcmd}", "system_warning")
                        display.add_system_message(f"Available: reload, {', '.join(sorted(available))}", "hint")
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

                # Inject any pending system hints (hidden from display)
                pending_hints = _pop_ipc_system_hints(display)
                if pending_hints:
                    hints_text = "\n".join(pending_hints)
                    text = f"{hints_text}\n\n{text}"

                await client.send_message(text)

            except asyncio.CancelledError:
                break
            except ReconnectingError:
                display.add_system_message(
                    "Cannot send message while reconnecting. Please wait...",
                    style="system_warning"
                )
            except ConnectionClosedError:
                display.add_system_message(
                    "Connection is closed. Please restart the client.",
                    style="system_error_bold"
                )
            except Exception as e:
                display.add_system_message(f"Error: {e}", style="system_error_bold")

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
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode with file output (requires --connect and --prompt). "
             "Output goes to {workspace}/jaato-headless-client-agents/"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="Workspace directory for headless output (default: current directory)"
    )

    args = parser.parse_args()

    # Validate headless mode requirements
    if args.headless:
        if not args.connect:
            sys.exit("Error: --headless requires --connect")
        if not args.prompt and not args.initial_prompt:
            sys.exit("Error: --headless requires --prompt or --initial-prompt")

    # Check TTY before proceeding (except for single prompt mode or headless)
    if not sys.stdout.isatty() and not args.prompt and not args.headless:
        sys.exit(
            "Error: rich-client requires an interactive terminal.\n"
            "Use --headless for non-TTY environments."
        )

    # Headless mode: file-based output, auto-approve permissions
    if args.headless:
        import asyncio
        from headless_mode import run_headless_mode
        workspace = pathlib.Path(args.workspace) if args.workspace else pathlib.Path.cwd()
        asyncio.run(run_headless_mode(
            socket_path=args.connect,
            prompt=args.prompt or args.initial_prompt,
            workspace=workspace,
            auto_start=not args.no_auto_start,
            env_file=args.env_file,
            new_session=args.new_session,
        ))
        return

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
