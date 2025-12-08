"""Console presenter for rendering output to the terminal.

Handles all presentation logic for the interactive client, including
history display, plan status, context usage, and help text.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from terminal_ui import TerminalUI

if TYPE_CHECKING:
    from shared.plugins.model_provider.types import ToolSchema
    from shared.plugins.registry import UserCommand


class ConsolePresenter:
    """Presenter class for console output rendering."""

    def __init__(self, ui: Optional[TerminalUI] = None):
        """Initialize the presenter.

        Args:
            ui: TerminalUI instance for formatting. Creates one if not provided.
        """
        self.ui = ui or TerminalUI()

    def print_banner(self, has_completion: bool = False) -> None:
        """Print the interactive client welcome banner.

        Args:
            has_completion: Whether command completion is available.
        """
        print("\n" + "=" * 60)
        print("  Simple Interactive Client with Permission Prompts")
        print("=" * 60)
        print("\nEnter task descriptions for the model to execute.")
        print("Tool calls will prompt for your approval.")
        print("Use ↑/↓ arrows to navigate prompt history.")
        if has_completion:
            print("Commands auto-complete as you type (help, tools, reset, etc.).")
            print("Use @path/to/file to reference files (completions appear as you type).")
            print("Use /command to invoke slash commands (from .jaato/commands/).")
        print("Type 'quit' or 'exit' to stop, 'help' for guidance.\n")

    def print_help(self, plugin_commands: Optional[Dict[str, List['UserCommand']]] = None) -> None:
        """Print help information.

        Args:
            plugin_commands: Dict mapping plugin names to their user commands.
        """
        print("""
Commands (auto-complete as you type):
  help          - Show this help message
  tools         - List tools available to the model
  reset         - Clear conversation history
  history       - Show full conversation history
  context       - Show context window usage
  export [file] - Export session to YAML for replay (default: session_export.yaml)
  quit          - Exit the client
  exit          - Exit the client""")

        # Print plugin-contributed commands
        if plugin_commands:
            self._print_plugin_commands(plugin_commands)

        print("""
When the model tries to use a tool, you'll see a permission prompt:
  [y]es     - Allow this execution
  [n]o      - Deny this execution
  [a]lways  - Allow and remember for this session
  [never]   - Deny and block for this session
  [once]    - Allow just this once

File references:
  Use @path/to/file to include file contents in your prompt.
  - @src/main.py      - Reference a file (contents included)
  - @./config.json    - Reference with explicit relative path
  - @~/documents/     - Reference with home directory
  - @/absolute/path   - Absolute path reference
  Completions appear automatically as you type after @.
  Use ↑/↓ to navigate the dropdown, Enter or TAB to accept.

Slash commands:
  Use /command_name [args...] to invoke slash commands from .jaato/commands/.
  - Type / to see available commands with descriptions
  - Pass arguments after the command name: /review file.py
  - Command files use {{$1}}, {{$2}} for parameter substitution
  - Use {{$1:default}} for optional parameters with defaults
  Example: "/summarize" or "/review src/main.py"

Example prompts:
  - "List files in the current directory"
  - "Show me the git status"
  - "Review @src/utils.py for issues"
  - "Explain what @./README.md describes"
  - "/review src/main.py" - Invoke slash command with argument

Multi-turn conversation:
  The model remembers previous exchanges in this session.
  Use 'reset' to start a fresh conversation.

Keyboard shortcuts:
  ↑/↓       - Navigate prompt history (or completion menu)
  ←/→       - Move cursor within line
  Ctrl+A/E  - Jump to start/end of line
  TAB/Enter - Accept selected completion
  Escape    - Dismiss completion menu
""")

    def _print_plugin_commands(self, commands_by_plugin: Dict[str, List['UserCommand']]) -> None:
        """Print plugin-contributed user commands grouped by plugin.

        Args:
            commands_by_plugin: Dict mapping plugin names to their commands.
        """
        if not commands_by_plugin:
            return

        print("\nPlugin-provided user commands:")
        for plugin_name, commands in sorted(commands_by_plugin.items()):
            print(f"  [{plugin_name}]")
            for cmd in commands:
                # Calculate padding for alignment
                padding = max(2, 16 - len(cmd.name))
                shared_marker = " [shared with model]" if cmd.share_with_model else ""
                print(f"    {cmd.name}{' ' * padding}- {cmd.description}{shared_marker}")

    def print_tools(self, tool_schemas: List['ToolSchema']) -> None:
        """Print available tools.

        Args:
            tool_schemas: List of tool schema declarations.
        """
        print("\nAvailable tools:")
        for decl in tool_schemas:
            print(f"  - {decl.name}: {decl.description}")
        print()

    def print_context(self, usage: Dict[str, Any]) -> None:
        """Print context window usage statistics.

        Args:
            usage: Dict with context usage info (model, context_limit, percent_used,
                   total_tokens, prompt_tokens, output_tokens, tokens_remaining, turns).
        """
        print(f"\n{'=' * 50}")
        print("  Context Window Usage")
        print(f"{'=' * 50}")
        print(f"  Model: {usage['model']}")
        print(f"  Context limit: {usage['context_limit']:,} tokens")
        print()

        # Visual progress bar
        bar = self.ui.progress_bar(usage['percent_used'])
        print(f"  {bar}")
        print()

        # Token breakdown
        print(f"  Tokens used:     {usage['total_tokens']:,}")
        print(f"    - Prompt:      {usage['prompt_tokens']:,}")
        print(f"    - Output:      {usage['output_tokens']:,}")
        print(f"  Tokens remaining: {usage['tokens_remaining']:,}")
        print(f"  Turns: {usage['turns']}")
        print(f"{'=' * 50}")
        print()

    def print_history(
        self,
        history: List[Any],
        turn_accounting: List[Dict[str, Any]],
        turn_boundaries: List[Any]
    ) -> None:
        """Print full conversation history with token accounting and turn numbers.

        Args:
            history: List of conversation content objects.
            turn_accounting: List of token accounting dicts per turn.
            turn_boundaries: List of turn boundary markers.
        """
        count = len(history)
        total_turns = len(turn_boundaries)

        print(f"\n{'=' * 60}")
        print(f"  Conversation History: {count} message(s), {total_turns} turn(s)")
        print("  Tip: Use 'backtoturn <turn_id>' to revert to a specific turn")
        print(f"{'=' * 60}")

        if count == 0:
            print("  (empty)")
            print()
            return

        # Track which turn we're in
        current_turn = 0
        turn_index = 0

        for i, content in enumerate(history):
            role = getattr(content, 'role', None) or 'unknown'
            parts = getattr(content, 'parts', None) or []

            # Check if this is a new user turn
            is_user_text = (role == 'user' and parts and
                           hasattr(parts[0], 'text') and parts[0].text)

            # Print turn header if this starts a new turn
            if is_user_text:
                current_turn += 1
                print(f"\n{'─' * 60}")
                # Show timestamp in turn header if available
                turn_idx = current_turn - 1
                if turn_idx < len(turn_accounting) and 'start_time' in turn_accounting[turn_idx]:
                    start_time = turn_accounting[turn_idx]['start_time']
                    try:
                        dt = datetime.fromisoformat(start_time)
                        time_str = dt.strftime('%H:%M:%S')
                        print(f"  ▶ TURN {current_turn}  [{time_str}]")
                    except (ValueError, TypeError):
                        print(f"  ▶ TURN {current_turn}")
                else:
                    print(f"  ▶ TURN {current_turn}")
                print(f"{'─' * 60}")

            # Print the message
            role_label = "USER" if role == 'user' else "MODEL" if role == 'model' else role.upper()
            print(f"\n  [{role_label}]")

            if not parts:
                print("  (no content)")
            else:
                for part in parts:
                    self._print_part(part)

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
                print(f"  ─── tokens: {turn['prompt']} in / {turn['output']} out / {turn['total']} total")
                if 'duration_seconds' in turn and turn['duration_seconds'] is not None:
                    duration = turn['duration_seconds']
                    print(f"  ─── duration: {duration:.2f}s")
                    func_calls = turn.get('function_calls', [])
                    if func_calls:
                        fc_total = sum(fc['duration_seconds'] for fc in func_calls)
                        model_time = duration - fc_total
                        print(f"      model: {model_time:.2f}s, tools: {fc_total:.2f}s ({len(func_calls)} call(s))")
                        for fc in func_calls:
                            print(f"        - {fc['name']}: {fc['duration_seconds']:.2f}s")
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
            print(f"\n{'=' * 60}")
            print(f"  Total: {total_prompt} in / {total_output} out / {total_all} total ({total_turns} turns)")
            if total_duration > 0:
                total_model_time = total_duration - total_fc_time
                print(f"  Time:  {total_duration:.2f}s total (model: {total_model_time:.2f}s, tools: {total_fc_time:.2f}s)")
            print(f"{'=' * 60}")

        print()

    def _print_part(self, part: Any) -> None:
        """Print a single content part.

        Args:
            part: A content part (text, function_call, or function_response).
        """
        # Text content
        if hasattr(part, 'text') and part.text:
            text = part.text
            if len(text) > 500:
                text = text[:500] + f"... [{len(part.text)} chars total]"
            print(f"  {text}")

        # Function call
        elif hasattr(part, 'function_call') and part.function_call:
            fc = part.function_call
            name = getattr(fc, 'name', 'unknown')
            args = getattr(fc, 'args', {})
            args_str = str(args)
            if len(args_str) > 200:
                args_str = args_str[:200] + "..."
            print(f"  📤 CALL: {name}({args_str})")

        # Function response
        elif hasattr(part, 'function_response') and part.function_response:
            fr = part.function_response
            name = getattr(fr, 'name', 'unknown')
            response = getattr(fr, 'response', {})

            # Extract and display permission info first
            if isinstance(response, dict):
                perm = response.get('_permission')
                if perm:
                    decision = perm.get('decision', '?')
                    reason = perm.get('reason', '')
                    method = perm.get('method', '')
                    icon = '✓' if decision == 'allowed' else '✗'
                    print(f"  {icon} Permission: {decision} via {method}")
                    if reason:
                        print(f"    Reason: {reason}")

            # Filter out _permission from display response
            if isinstance(response, dict):
                display_response = {k: v for k, v in response.items() if k != '_permission'}
            else:
                display_response = response

            resp_str = str(display_response)
            if len(resp_str) > 300:
                resp_str = resp_str[:300] + "..."
            print(f"  📥 RESULT: {name} → {resp_str}")

        else:
            print(f"  (unknown part: {type(part).__name__})")

    def display_command_result(
        self,
        command_name: str,
        result: Any,
        shared: bool
    ) -> None:
        """Display the result of a plugin command to the user.

        Args:
            command_name: Name of the executed command.
            result: The command's return value.
            shared: Whether the result was shared with the model.
        """
        # Special formatting for plan command
        if command_name == "plan" and isinstance(result, dict):
            self.display_plan_result(result)
            if shared:
                print("  [Result shared with model]")
            return

        print(f"\n[{command_name}]")

        if isinstance(result, dict):
            for key, value in result.items():
                if key.startswith('_'):
                    continue
                if isinstance(value, (list, dict)):
                    print(f"  {key}:")
                    if isinstance(value, list):
                        for item in value[:20]:
                            if isinstance(item, dict):
                                item_str = ", ".join(f"{k}: {v}" for k, v in item.items())
                                print(f"    - {item_str}")
                            else:
                                print(f"    - {item}")
                        if len(value) > 20:
                            print(f"    ... and {len(value) - 20} more")
                    else:
                        print(f"    {json.dumps(value, indent=2)}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {result}")

        if shared:
            print("  [Result shared with model]")

    def display_plan_result(self, result: Dict[str, Any]) -> None:
        """Display plan status in a user-friendly format.

        Args:
            result: The plan status dict from getPlanStatus.
        """
        if "error" in result:
            print(f"\n[plan] {result['error']}")
            return

        status = result.get("status", "unknown")
        title = result.get("title", "Untitled Plan")

        status_display = {
            "pending": "📋 PENDING",
            "in_progress": "🔄 IN PROGRESS",
            "completed": "✅ COMPLETED",
            "failed": "❌ FAILED",
            "cancelled": "⚠️  CANCELLED",
        }.get(status, status.upper())

        print(f"\n{'=' * 60}")
        print(f"  {status_display}: {title}")
        print(f"{'=' * 60}")

        # Progress bar
        progress = result.get("progress", {})
        total = progress.get("total", 0)
        completed = progress.get("completed", 0)
        failed = progress.get("failed", 0)
        in_prog = progress.get("in_progress", 0)
        pending = progress.get("pending", 0)
        percent = progress.get("percent", 0)

        if total > 0:
            bar = self.ui.progress_bar(percent)
            print(f"\n  Progress: {bar}")
            print(f"  Steps: {completed} completed, {in_prog} in progress, {pending} pending, {failed} failed")

        summary = result.get("summary")
        if summary:
            print(f"\n  Summary: {summary}")

        steps = result.get("steps", [])
        if steps:
            print("\n  Steps:")
            print(f"  {'-' * 56}")

            for step in sorted(steps, key=lambda s: s.get("sequence", 0)):
                seq = step.get("sequence", "?")
                desc = step.get("description", "")
                step_status = step.get("status", "pending")

                indicator = {
                    "pending": "○",
                    "in_progress": "◐",
                    "completed": "●",
                    "failed": "✗",
                    "skipped": "○",
                }.get(step_status, "?")

                max_desc_len = 50
                if len(desc) > max_desc_len:
                    desc = desc[:max_desc_len - 3] + "..."

                print(f"  {indicator} {seq}. {desc}")

                step_result = step.get("result")
                step_error = step.get("error")
                if step_result and step_status == "completed":
                    if len(step_result) > 60:
                        step_result = step_result[:57] + "..."
                    print(f"      └─ {step_result}")
                elif step_error and step_status == "failed":
                    if len(step_error) > 60:
                        step_error = step_error[:57] + "..."
                    print(f"      └─ Error: {step_error}")

        print(f"\n{'=' * 60}")

    def format_model_output(self, text: str) -> str:
        """Format model output with prefix and wrapping.

        Args:
            text: The model's output text.

        Returns:
            Formatted output string with Model> prefix.
        """
        model_prefix = self.ui.colorize('Model>', 'bold') + ' '
        continuation_indent = "       "  # 7 spaces to match "Model> " width
        wrapped = self.ui.wrap_text(text, prefix=continuation_indent, initial_prefix="")
        return f"\n{model_prefix}{wrapped}"

    def format_prompt(self) -> str:
        """Get the formatted user prompt string.

        Returns:
            Formatted prompt string like "You> ".
        """
        return f"\n{self.ui.colorize('You>', 'green')} "
