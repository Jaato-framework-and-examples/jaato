#!/usr/bin/env python3
"""Rich TUI client with sticky plan display.

This client provides a terminal UI experience with:
- Sticky plan panel at the top showing current plan status
- Scrolling output panel below for model responses and tool output
- Full-screen alternate buffer for immersive experience

Connects to a jaato server daemon via IPC (auto-starting it if needed).

Requires an interactive TTY. For non-TTY environments, use simple-client.
"""

import asyncio
import os
import sys
import pathlib
import tempfile
import threading
from typing import Any, Callable, Dict, List, Optional

# Add project root to path for imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Add package directory to path for bare sibling imports (e.g. pt_display, backend)
PKG_DIR = str(pathlib.Path(__file__).resolve().parent)
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

# Add simple-client to path for reusable components
SIMPLE_CLIENT = ROOT / "simple-client"
if str(SIMPLE_CLIENT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_CLIENT))

from dotenv import load_dotenv


from renderers.vision_capture import (
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
from agent_registry import AgentRegistry
from keybindings import load_keybindings, detect_terminal, list_available_profiles
from theme import load_theme, list_available_themes

# Backend abstraction for mode-agnostic operation
from backend import Backend, IPCBackend


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

        # Backend abstraction (IPC mode only — direct mode removed)
        self._backend: Optional[Backend] = backend
        self._is_ipc_mode = backend is not None and isinstance(backend, IPCBackend)

        # Async event loop for backend calls from threads
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None

        # Agent registry for tracking agents and their state
        self._agent_registry = AgentRegistry()

        # Rich TUI display (prompt_toolkit-based)
        self._display: Optional[PTDisplay] = None

        # Input handler (for file expansion, history, completions)
        self._input_handler = InputHandler()
        self._input_handler.set_available_themes(list_available_themes())

        # Model info for status bar
        self._model_provider: str = ""
        self._model_name: str = ""

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

    def _create_edit_callback(self) -> Callable[[Dict[str, Any], Any], Optional[Dict[str, Any]]]:
        """Create callback for editing tool content in external editor.

        This callback is invoked when user selects 'e' (edit) at a permission
        prompt for a tool that has editable content. It runs from the model
        thread (background), so we schedule the TUI suspension on the app's
        event loop via run_in_terminal.

        Returns:
            Callback function that takes (arguments, editable_metadata) and
            returns edited arguments dict, or None if edit was cancelled.
        """
        from editor_utils import edit_tool_content
        import asyncio

        def edit_callback(arguments: Dict[str, Any], editable: Any) -> Optional[Dict[str, Any]]:
            """Open external editor for tool content.

            Args:
                arguments: Current tool arguments.
                editable: EditableContent metadata from the tool schema.

            Returns:
                Edited arguments dict, or None if cancelled/error.
            """
            self._trace(f"edit_callback: editing {len(arguments)} arguments")

            # Get session directory for edit history (if available)
            session_dir = None
            if hasattr(self, '_session_dir') and self._session_dir:
                session_dir = self._session_dir

            result = None

            def run_editor():
                nonlocal result
                try:
                    result = edit_tool_content(arguments, editable, session_dir)
                except Exception as e:
                    from editor_utils import EditResult
                    result = EditResult(
                        success=False, arguments=arguments, was_modified=False, error=str(e)
                    )

            # Use prompt_toolkit's run_in_terminal to properly suspend the TUI.
            # This callback runs from the model thread, so schedule onto the
            # app's event loop and block until complete.
            app = getattr(self._display, '_app', None) if self._display else None
            if app and getattr(app, 'loop', None):
                from prompt_toolkit.application import run_in_terminal
                future = asyncio.run_coroutine_threadsafe(
                    run_in_terminal(run_editor, in_executor=False),
                    app.loop,
                )
                future.result(timeout=600)  # Block until editor completes
            else:
                run_editor()

            if result and result.success:
                if result.was_modified:
                    self._trace(f"edit_callback: content was modified")
                    return result.arguments
                else:
                    self._trace(f"edit_callback: content unchanged")
                    return arguments
            else:
                error_msg = result.error if result else "Unknown error"
                self._trace(f"edit_callback: edit failed: {error_msg}")
                if self._display:
                    self._display.add_system_message(
                        f"Edit failed: {error_msg}",
                        style="system_error"
                    )
                return None

        return edit_callback

    def initialize(self) -> bool:
        """Initialize the client.

        Requires an IPC backend (connection to a running server daemon).
        """
        if not self._is_ipc_mode:
            raise RuntimeError(
                "RichClient requires an IPC backend. "
                "Use run_ipc_mode() or pass an IPCBackend to the constructor."
            )
        return self._initialize_ipc_mode()

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

    def _trace(self, msg: str) -> None:
        """Write trace message to file for debugging."""
        from jaato_sdk.trace import trace
        trace("rich_client", msg)


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
    from renderers.vision_capture import VisionCapture, VisionCaptureFormatter
    from renderers.vision_capture.protocol import CaptureConfig, CaptureFormat

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
    from renderers.vision_capture.protocol import CaptureContext

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

        from renderers.vision_capture.protocol import CaptureFormat
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
            except (ImportError, OSError):
                display.add_system_message("[Warning: cairosvg not available]", "yellow")
                display.add_system_message("  PNG format requires cairosvg for SVG to PNG conversion.", "dim")
                display.add_system_message("  Install with: pip install cairosvg", "dim")
                display.add_system_message("  (also requires system Cairo library:", "dim")
                display.add_system_message("   Linux: apt install libcairo2-dev", "dim")
                display.add_system_message("   Windows: install GTK3 runtime or use conda install cairo)", "dim")
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
        from renderers.vision_capture.protocol import CaptureFormat
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


async def _handle_client_side_edit(
    pending_request: dict,
    display: Any,
) -> Optional[Dict[str, Any]]:
    """Handle editing of tool content on the client side (IPC mode).

    Opens the external editor with the tool arguments and returns the
    edited arguments, or None if the edit was cancelled.

    Uses prompt_toolkit's run_in_terminal() to temporarily exit the TUI,
    run the editor, and restore the display automatically.

    Args:
        pending_request: The pending permission request dict with tool_args and editable_metadata.
        display: The PTDisplay instance (must have _app for run_in_terminal).

    Returns:
        Edited arguments dict, or None if cancelled.
    """
    from editor_utils import edit_tool_content
    from prompt_toolkit.application import run_in_terminal

    tool_args = pending_request.get("tool_args")
    editable_meta = pending_request.get("editable_metadata")

    if not tool_args or not editable_meta:
        return None

    # Reconstruct an EditableContent-like object from metadata
    class _EditableProxy:
        def __init__(self, meta: dict):
            self.parameters = meta.get("parameters", [])
            self.format = meta.get("format", "yaml")
            self.template = meta.get("template")

    editable = _EditableProxy(editable_meta)

    result = None

    def run_editor():
        nonlocal result
        try:
            result = edit_tool_content(tool_args, editable)
        except Exception as e:
            from editor_utils import EditResult
            result = EditResult(
                success=False, arguments=tool_args, was_modified=False, error=str(e)
            )

    # run_in_terminal properly suspends the TUI, runs the editor,
    # and restores the display. It's already an awaitable.
    await run_in_terminal(run_editor, in_executor=False)

    if result and result.success:
        return result.arguments if result.was_modified else tool_args
    else:
        if result and result.error and display:
            display.add_system_message(
                f"Edit failed: {result.error}",
                style="system_error"
            )
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
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(os.path.abspath(trace_log_path)), exist_ok=True)
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
    from jaato_sdk.client.ipc import IPCClient, IncompatibleServerError
    from jaato_sdk.client.recovery import (
        IPCRecoveryClient,
        ConnectionState,
        ConnectionStatus,
        ReconnectingError,
        ConnectionClosedError,
    )
    from jaato_sdk.client.config import get_recovery_config
    from jaato_sdk.events import (
        Event,
        EventType,
        AgentOutputEvent,
        AgentCreatedEvent,
        AgentStatusChangedEvent,
        AgentCompletedEvent,
        PermissionInputModeEvent,
        PermissionResolvedEvent,
        PermissionStatusEvent,
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
        MemoryListEvent,
        SandboxPathsEvent,
        ServiceListEvent,
        CommandListEvent,
        ToolStatusEvent,
        HistoryEvent,
        WorkspaceMismatchRequestedEvent,
        WorkspaceMismatchResponseRequest,
        PostAuthSetupEvent,
        MidTurnPromptQueuedEvent,
        MidTurnPromptInjectedEvent,
        MidTurnInterruptEvent,
        WorkspaceFilesChangedEvent,
        WorkspaceFilesSnapshotEvent,
    )

    # Minimum server version this TUI release requires.
    # Bump this when the TUI starts depending on a new server feature.
    MIN_SERVER_VERSION = "0.2.27"

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
    display = PTDisplay(
        keybinding_config=keybindings,
        theme_config=theme_config,
        agent_registry=agent_registry,
        input_handler=input_handler,
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
    pending_post_auth_setup: Optional[dict] = None
    model_running = False
    should_exit = False
    server_commands: list = []  # Commands from server for help display
    available_sessions: list = []  # Sessions from server for completion
    available_tools: list = []  # Tools from server for completion
    available_models: list = []  # Models from server for completion
    available_memories: list = []  # Memories from server for completion
    available_services: list = []  # Services from server for completion

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

    # Set up command completion provider for model, memory, and services commands
    # Note: subcommands (list, remove, edit, show, etc.) are already provided
    # by CommandCompleter via CommandListEvent. This provider only handles
    # argument-level completions (memory IDs, service names, HTTP methods)
    # that CommandCompleter can't provide.
    def command_completion_provider(command: str, args: list) -> list:
        """Provide completions for dynamic argument-level completions."""
        if command == "model":
            return [(model, "") for model in available_models]
        elif command in ("memory remove", "memory edit"):
            # Memory ID completions for remove/edit subcommands
            partial = args[0].lower() if args else ""
            return [
                (m["id"], m.get("description", "")[:40])
                for m in available_memories
                if m["id"].lower().startswith(partial)
            ]
        elif command in ("services show", "services endpoints", "services auth", "services remove"):
            if not args or (len(args) == 1 and args[0]):
                # First arg: service name completion
                partial = args[0].lower() if args else ""
                return [
                    (s["name"], "Service")
                    for s in available_services
                    if s["name"].lower().startswith(partial)
                ]
            elif command == "services endpoints" and len(args) >= 2:
                # Second arg for endpoints: HTTP method completion
                service_name = args[0] if len(args) >= 1 else ""
                partial = args[-1].upper() if len(args) >= 2 and args[-1] else ""
                # Find methods for this service
                methods = []
                for s in available_services:
                    if s["name"] == service_name:
                        methods = s.get("methods", [])
                        break
                if not methods:
                    # Fallback to common HTTP methods
                    methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
                return [
                    (m, "HTTP method")
                    for m in methods
                    if m.upper().startswith(partial)
                ]
        return []

    input_handler.set_command_completion_provider(
        command_completion_provider,
        {
            "model", "memory remove", "memory edit",
            "services show", "services endpoints", "services auth", "services remove",
        }
    )

    # Set up sandbox path provider for @@ completion (server mode)
    # Initial defaults; refreshed dynamically via SandboxPathsEvent and SessionInfoEvent
    _server_sandbox_paths: list[tuple[str, str]] = []
    if workspace_path:
        _server_sandbox_paths.append((str(workspace_path), "workspace"))
    _server_sandbox_paths.append(("/tmp", "system temp"))

    def _update_sandbox_paths(paths_data: list[dict]) -> None:
        """Update the mutable sandbox paths cache from event data."""
        _server_sandbox_paths.clear()
        for entry in paths_data:
            _server_sandbox_paths.append((entry.get("path", ""), entry.get("description", "")))

    def get_sandbox_paths_for_completion():
        """Return sandbox-allowed root paths for @@ completion."""
        return list(_server_sandbox_paths)

    input_handler.set_sandbox_path_provider(get_sandbox_paths_for_completion)

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

        # Check server version against our minimum requirement
        sv = client.server_version
        if sv is not None:
            def _parse_version(v: str) -> tuple:
                return tuple(int(x) for x in v.split("."))
            if _parse_version(sv) < _parse_version(MIN_SERVER_VERSION):
                raise IncompatibleServerError(sv, MIN_SERVER_VERSION)

        print("Connected!")

    except IncompatibleServerError as e:
        print(f"Error: {e}")
        return

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
        nonlocal pending_workspace_mismatch_request, pending_post_auth_setup
        nonlocal model_running, should_exit, is_reconnecting
        nonlocal init_shown_header, init_current_step

        ipc_trace("Event handler starting")
        event_count = 0
        # Periodic yielding: when the IPC StreamReader has buffered data,
        # readexactly() returns immediately without suspending, which means
        # the event loop never yields to prompt_toolkit for keyboard handling.
        # We track time and force a yield every ~16ms to keep the UI responsive.
        last_yield_time = asyncio.get_event_loop().time()
        YIELD_INTERVAL = 0.016  # ~60fps, yield every 16ms

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
                    # Sub-step detail (e.g., plugin name) shown inline
                    detail = f" ({event.message})" if event.message else ""
                    line_text = f"{step_prefix} ...{detail}"
                    if init_current_step == step_name:
                        # Same step firing again — update line in-place
                        display.update_last_system_message(
                            line_text, style="system_progress", prefix=step_prefix
                        )
                    else:
                        # First "running" for this step — add new line
                        display.add_system_message(line_text, style="system_progress")
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
                # Register agent name in budget panel for display
                if hasattr(display, 'register_agent_name'):
                    display.register_agent_name(event.agent_id, event.agent_name)
                # Show welcome messages when main agent is created (now in correct buffer)
                # Skip during reconnection - these are init-only messages
                if event.agent_id == "main" and not is_reconnecting:
                    display.add_system_message(release_name, style="system_version")
                    if input_handler.has_completion:
                        display.add_system_message(
                            "Tab completion enabled. Use @file for files, @@path for sandbox paths, %prompt for skills.",
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
                    "tool_args": getattr(event, 'tool_args', None),
                    "editable_metadata": getattr(event, 'editable_metadata', None),
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
                    # Only clear pending permission state if this resolution matches
                    # the request we're waiting on. Auto-approved permissions from
                    # subagents (e.g. readFile) must not cancel the main agent's
                    # pending interactive permission prompt.
                    is_pending_resolved = (
                        pending_permission_request
                        and event.request_id
                        and event.request_id == pending_permission_request.get("request_id")
                    )
                    if is_pending_resolved:
                        pending_permission_request = None
                        display.set_waiting_for_channel_input(False)
                    # Always update tool tree with permission result regardless
                    # Route to the agent whose permission was resolved, not the selected agent
                    buffer = agent_registry.get_buffer(event.agent_id) if event.agent_id else agent_registry.get_selected_buffer()
                    if buffer:
                        buffer.set_tool_permission_resolved(event.tool_name, event.granted, event.method)
                        display.refresh()

            elif isinstance(event, PermissionStatusEvent):
                # Update toolbar with current permission status from server
                display.set_permission_status(
                    effective_default=event.effective_default,
                    suspension_scope=event.suspension_scope,
                )

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

                # Build step_id → description mapping so tool args display
                # human-readable task descriptions instead of raw UUIDs
                step_id_map = {}
                for i, step in enumerate(event.steps):
                    sid = step.get("step_id")
                    if sid:
                        desc = step.get("content", "")
                        step_id_map[sid] = desc if desc else f"Step #{i + 1}"
                target_agent = agent_id or "main"
                buffer = agent_registry.get_buffer(target_agent)
                if buffer:
                    buffer.update_step_id_map(step_id_map)

                display.update_plan(plan_data, agent_id)

            elif isinstance(event, PlanClearedEvent):
                agent_id = getattr(event, 'agent_id', None)
                display.clear_plan(agent_id)

            elif isinstance(event, WorkspaceFilesChangedEvent):
                display.update_workspace_files(event.changes)

            elif isinstance(event, WorkspaceFilesSnapshotEvent):
                display.set_workspace_snapshot(event.files)

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
                        call_id=event.call_id,
                        backgrounded=event.backgrounded,
                        continuation_id=event.continuation_id,
                        show_output=event.show_output,
                        show_popup=event.show_popup,
                    )
                    buffer.scroll_to_bottom()  # Auto-scroll when tool tree updates
                    display.refresh()
                else:
                    # Agent not yet created - queue event for later
                    ipc_trace(f"  Queuing tool end for unknown agent: {event.agent_id}")
                    agent_registry.queue_tool_end(
                        event.agent_id, event.tool_name, event.success,
                        event.duration_seconds, event.error_message, event.call_id,
                        continuation_id=event.continuation_id,
                        show_output=event.show_output,
                        show_popup=event.show_popup,
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
                    # Show turn summary in output buffer
                    if event.total_tokens > 0:
                        buffer.add_system_message(
                            f"─── tokens: {event.prompt_tokens:,} in / {event.output_tokens:,} out / {event.total_tokens:,} total",
                            "dim",
                        )
                        if event.duration_seconds:
                            buffer.add_system_message(
                                f"─── duration: {event.duration_seconds:.2f}s",
                                "dim",
                            )
                        if event.cache_read_tokens and event.prompt_tokens > 0:
                            hit_pct = event.cache_read_tokens / event.prompt_tokens * 100
                            buffer.add_system_message(
                                f"─── cache hit: {hit_pct:.0f}%",
                                "dim",
                            )
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
                display.refresh()

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
                ipc_trace(f"  MidTurnInterruptEvent: partial={event.partial_response_chars} chars")

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

            elif isinstance(event, MemoryListEvent):
                # Store memories for completion cache
                nonlocal available_memories
                available_memories = event.memories

            elif isinstance(event, SandboxPathsEvent):
                # Refresh sandbox paths for @@ completion cache
                _update_sandbox_paths(event.paths)

            elif isinstance(event, ServiceListEvent):
                # Store services for completion cache
                nonlocal available_services
                available_services = event.services

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
                if event.memories:
                    available_memories = event.memories
                if event.sandbox_paths:
                    _update_sandbox_paths(event.sandbox_paths)
                if event.services:
                    available_services = event.services

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
                # Update session bar only when the event carries session data.
                # Partial updates (e.g. after model-change) only carry model
                # info with empty session_id/sessions — touching the session
                # bar would blank out the session ID, description, and workspace.
                if event.sessions or event.session_id:
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

            elif isinstance(event, PostAuthSetupEvent):
                ipc_trace(f"  PostAuthSetupEvent: provider={event.provider_name}")
                # Store the full event data for the multi-step wizard
                pending_post_auth_setup = {
                    "request_id": event.request_id,
                    "provider_name": event.provider_name,
                    "provider_display_name": event.provider_display_name,
                    "models": event.available_models,
                    "has_active_session": event.has_active_session,
                    "current_provider": event.current_provider,
                    "current_model": event.current_model,
                    "workspace_path": event.workspace_path,
                    "step": "connect",  # First step: ask to connect
                }
                # Render the first prompt
                display.append_output("system", "", "write")
                if event.has_active_session:
                    display.append_output(
                        "system",
                        f"Switch to {event.provider_display_name}? "
                        f"(currently: {event.current_provider}/{event.current_model}) [y/n]",
                        "write",
                    )
                else:
                    display.append_output(
                        "system",
                        f"Connect to {event.provider_display_name} now? [y/n]",
                        "write",
                    )
                display.refresh()
                display.set_waiting_for_channel_input(True)

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
                                turn_line = f"  --- Turn {turn_index + 1}: {total:,} tokens (in: {prompt:,}, out: {output:,})"
                                cache_read = acc.get('cache_read')
                                if cache_read and prompt > 0:
                                    hit_pct = cache_read / prompt * 100
                                    turn_line += f", cache hit: {hit_pct:.0f}%"
                                turn_line += " ---"
                                lines.append((turn_line, "dim"))
                                turn_index += 1

                        lines.append(("", ""))

                    display.show_lines(lines)

            # Periodically yield to the event loop to keep the UI responsive.
            # When the IPC StreamReader has buffered data, readexactly() returns
            # immediately without suspending, starving prompt_toolkit of CPU time.
            now = asyncio.get_event_loop().time()
            if now - last_yield_time >= YIELD_INTERVAL:
                await asyncio.sleep(0)
                last_yield_time = asyncio.get_event_loop().time()

    async def handle_input():
        """Handle user input from the queue."""
        nonlocal pending_permission_request, pending_clarification_request, pending_reference_selection_request
        nonlocal pending_workspace_mismatch_request, pending_post_auth_setup
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
                    # Check if user wants to edit content (client-side editing)
                    if text.lower() in ("e", "edit") and pending_permission_request.get("tool_args") is not None:
                        ipc_trace(f"Edit requested for {pending_permission_request['request_id']}, opening editor locally")
                        edited_args = await _handle_client_side_edit(
                            pending_permission_request, display,
                        )
                        if edited_args is not None:
                            # Send edit response with edited arguments
                            ipc_trace(f"Sending edit response with edited args")
                            await client.respond_to_permission(
                                pending_permission_request["request_id"],
                                "e",
                                edited_arguments=edited_args,
                            )
                        else:
                            # Edit was cancelled - don't send anything, re-show prompt
                            ipc_trace(f"Edit cancelled, re-showing prompt")
                            display.set_waiting_for_channel_input(True, pending_permission_request.get("options"))
                            display.refresh()
                        continue

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

                # Handle post-auth setup wizard (multi-step)
                if pending_post_auth_setup:
                    step = pending_post_auth_setup.get("step", "")
                    answer = text.lower().strip()

                    if step == "connect":
                        if answer in ("y", "yes"):
                            # Show model selection
                            models = pending_post_auth_setup["models"]
                            display.append_output("system", "", "write")
                            display.append_output("system", "Select a model:", "write")
                            for i, m in enumerate(models, 1):
                                desc = m.get("description", "")
                                display.append_output(
                                    "system",
                                    f"  [{i}] {m['name']}"
                                    + (f" — {desc}" if desc else ""),
                                    "write",
                                )
                            display.refresh()
                            pending_post_auth_setup["step"] = "model"
                            display.set_waiting_for_channel_input(True)
                        else:
                            # User declined — send empty response
                            await client.respond_to_post_auth_setup(
                                request_id=pending_post_auth_setup["request_id"],
                                connect=False,
                            )
                            pending_post_auth_setup = None
                            display.set_waiting_for_channel_input(False)
                        continue

                    elif step == "model":
                        models = pending_post_auth_setup["models"]
                        selected_model = ""
                        # Accept number or name
                        if answer.isdigit():
                            idx = int(answer) - 1
                            if 0 <= idx < len(models):
                                selected_model = models[idx]["name"]
                        else:
                            # Try matching by name
                            for m in models:
                                if answer in m["name"].lower():
                                    selected_model = m["name"]
                                    break

                        if not selected_model:
                            display.append_output("system", f"Invalid selection. Enter 1-{len(models)} or model name.", "write")
                            display.refresh()
                            display.set_waiting_for_channel_input(True)
                            continue

                        pending_post_auth_setup["selected_model"] = selected_model

                        # Ask about .env persistence
                        workspace = pending_post_auth_setup.get("workspace_path", "")
                        if workspace:
                            display.append_output("system", "", "write")
                            display.append_output(
                                "system",
                                f"Save provider/model to {workspace}/.env? [y/n]",
                                "write",
                            )
                            display.refresh()
                            pending_post_auth_setup["step"] = "persist"
                            display.set_waiting_for_channel_input(True)
                        else:
                            # No workspace — send response without persist
                            await client.respond_to_post_auth_setup(
                                request_id=pending_post_auth_setup["request_id"],
                                connect=True,
                                model_name=selected_model,
                                persist_env=False,
                            )
                            pending_post_auth_setup = None
                            display.set_waiting_for_channel_input(False)
                        continue

                    elif step == "persist":
                        persist = answer in ("y", "yes")
                        await client.respond_to_post_auth_setup(
                            request_id=pending_post_auth_setup["request_id"],
                            connect=True,
                            model_name=pending_post_auth_setup["selected_model"],
                            persist_env=persist,
                        )
                        pending_post_auth_setup = None
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

                # Handle commands using shared parser
                from client_commands import parse_user_input, CommandAction

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

                # ==================== TUI-specific commands (not in shared parser) ====================
                # Keybindings command - handle locally using shared function
                if cmd == "keybindings":
                    from ui_utils import handle_keybindings_command
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

                # ==================== Use shared parser for all other commands ====================
                parsed = parse_user_input(text, server_commands)

                if parsed.action == CommandAction.EXIT:
                    # TUI-specific: Show confirmation dialog for session lifecycle
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

                elif parsed.action == CommandAction.STOP:
                    await client.stop()

                elif parsed.action == CommandAction.CLEAR:
                    display.clear_output()

                elif parsed.action == CommandAction.HELP:
                    from client_commands import build_full_help_text
                    help_lines = build_full_help_text(server_commands)
                    display.show_lines(help_lines)

                elif parsed.action == CommandAction.CONTEXT:
                    # TUI-specific: Use local agent_registry for context
                    selected_agent = agent_registry.get_selected_agent()
                    if not selected_agent:
                        display.show_lines([("Context tracking not available", "yellow")])
                    else:
                        usage = selected_agent.context_usage
                        lines = [
                            ("─" * 50, "dim"),
                            (f"Context Usage: {selected_agent.name}", "bold"),
                            (f"  Agent: {selected_agent.name}", "dim"),
                            (f"  Total tokens: {usage.get('total_tokens', 0)}", "dim"),
                            (f"  Prompt tokens: {usage.get('prompt_tokens', 0)}", "dim"),
                            (f"  Output tokens: {usage.get('output_tokens', 0)}", "dim"),
                            (f"  Turns: {usage.get('turns', 0)}", "dim"),
                            (f"  Percent used: {usage.get('percent_used', 0):.1f}%", "dim"),
                            ("─" * 50, "dim"),
                        ]
                        display.show_lines(lines)

                elif parsed.action == CommandAction.HISTORY:
                    await client.request_history()

                elif parsed.action == CommandAction.SERVER_COMMAND:
                    await client.execute_command(parsed.command, parsed.args or [])

                elif parsed.action == CommandAction.SEND_MESSAGE:
                    if parsed.text:
                        model_running = True
                        display.add_to_history(text)

                        # Expand file references (@file) and prompt references (%prompt)
                        message_text = input_handler.expand_file_references(parsed.text)
                        message_text = input_handler.expand_prompt_references(message_text)

                        # Inject any pending system hints (hidden from display)
                        pending_hints = _pop_ipc_system_hints(display)
                        if pending_hints:
                            hints_text = "\n".join(pending_hints)
                            message_text = f"{hints_text}\n\n{message_text}"

                        await client.send_message(message_text)

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


def _run_init(target: str = ".jaato") -> None:
    """Scaffold project configuration: ``.jaato/`` directory and ``.env.example``.

    Copies the bundled example configuration directory into the current
    working directory as ``.jaato/`` (or the path given by *target*), and
    copies the bundled ``.env.example`` alongside it so the user can
    ``cp .env.example .env`` and fill in credentials.

    Exits with an error if the target directory already exists.  The
    ``.env.example`` copy is skipped (with a note) if the file already
    exists in the working directory.
    """
    import shutil

    pkg_dir = pathlib.Path(__file__).resolve().parent
    source = pkg_dir / ".jaato.example"
    dest = pathlib.Path(target)

    if dest.exists():
        sys.exit(
            f"Error: {dest} already exists.\n"
            f"Remove or rename it first, or copy individual files from:\n"
            f"  {source}"
        )

    if not source.exists():
        sys.exit(
            "Error: .jaato.example not found in the installed package.\n"
            "Try reinstalling jaato-tui: pip install --force-reinstall jaato-tui"
        )

    shutil.copytree(source, dest)
    print(f"Initialized {dest}/ with example configuration files.")
    print(f"Edit the files to customize your setup. See {dest}/README.md for details.")

    # Copy .env.example if bundled and not already present
    env_example_src = pkg_dir / ".env.example"
    env_example_dst = pathlib.Path(".env.example")
    if env_example_src.exists():
        if env_example_dst.exists():
            print(f"Note: .env.example already exists, skipping copy.")
        else:
            shutil.copy2(env_example_src, env_example_dst)
            print(f"Copied .env.example — run 'cp .env.example .env' and fill in your credentials.")


def main():
    import argparse

    # Configure UTF-8 encoding for Windows console (before any output)
    from console_encoding import configure_utf8_output
    configure_utf8_output()

    parser = argparse.ArgumentParser(
        description="Rich TUI client for Jaato AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
The client auto-starts the server daemon if not already running.
To run the server separately: python -m server --ipc-socket /tmp/jaato.sock
To connect to a specific server: jaato --connect /path/to/socket
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
        help="Connect to server via IPC socket (default: auto-detect)."
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Don't auto-start server if not running"
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Start with a new session instead of resuming the default"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode with file output (requires --prompt). "
             "Output goes to {workspace}/jaato-headless-client-agents/"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="Workspace directory for headless output (default: current directory)"
    )
    parser.add_argument(
        "--session",
        type=str,
        metavar="SESSION_ID",
        help="Attach to a specific session (use with --cmd to send commands to headless sessions)"
    )
    parser.add_argument(
        "--cmd",
        type=str,
        metavar="COMMAND",
        help="Send a command to the session and exit (e.g., 'stop', 'reset', 'permissions default deny'). "
             "Requires --session."
    )
    parser.add_argument(
        "--init",
        nargs="?",
        const=".jaato",
        metavar="PATH",
        help="Initialize .jaato/ config directory with example files and exit (default: .jaato)"
    )

    args = parser.parse_args()

    # Init mode: scaffold config directory and exit
    if args.init is not None:
        _run_init(args.init)
        return

    # Resolve socket path: explicit --connect or default
    from jaato_sdk.client.ipc import DEFAULT_SOCKET_PATH
    socket_path = args.connect or DEFAULT_SOCKET_PATH

    # Validate --cmd requirements
    if args.cmd:
        if not args.session:
            sys.exit("Error: --cmd requires --session to specify which session to send the command to")

    # Validate headless mode requirements
    if args.headless:
        if not args.prompt and not args.initial_prompt:
            sys.exit("Error: --headless requires --prompt or --initial-prompt")

    # Command mode: send a command to a session and exit
    if args.cmd:
        import asyncio
        from command_mode import run_command_mode
        asyncio.run(run_command_mode(
            socket_path=socket_path,
            session_id=args.session,
            command=args.cmd,
            auto_start=not args.no_auto_start,
            env_file=args.env_file,
        ))
        return

    # Check TTY before proceeding (except for single prompt mode, headless, or command mode)
    if not sys.stdout.isatty() and not args.prompt and not args.headless and not args.cmd:
        sys.exit(
            "Error: jaato-tui requires an interactive terminal.\n"
            "Use --headless for non-TTY environments."
        )

    # Headless mode: file-based output, auto-approve permissions
    if args.headless:
        import asyncio
        from headless_mode import run_headless_mode
        workspace = pathlib.Path(args.workspace) if args.workspace else pathlib.Path.cwd()
        asyncio.run(run_headless_mode(
            socket_path=socket_path,
            prompt=args.prompt or args.initial_prompt,
            workspace=workspace,
            auto_start=not args.no_auto_start,
            env_file=args.env_file,
            new_session=args.new_session,
        ))
        return

    import asyncio
    asyncio.run(run_ipc_mode(
        socket_path=socket_path,
        auto_start=not args.no_auto_start,
        env_file=args.env_file,
        initial_prompt=args.initial_prompt,
        single_prompt=args.prompt,
        new_session=args.new_session,
    ))


if __name__ == "__main__":
    main()
