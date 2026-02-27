#!/usr/bin/env python3
"""Simple test client for Jaato WebSocket Server.

This is a minimal CLI client for testing the WebSocket server.
It connects to the server, displays events, and allows sending messages.

Usage:
    # Start server in one terminal:
    python rich_client.py --headless --port 8080

    # Run test client in another terminal:
    python -m server.test_client --port 8080

    # Or with a specific message:
    python -m server.test_client --port 8080 --message "Hello, world!"
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from typing import Optional

try:
    import websockets
    from websockets import ClientConnection
except ImportError:
    print("Error: websockets package required. Install with: pip install websockets")
    sys.exit(1)


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def colorize(text: str, color: str) -> str:
    """Apply ANSI color to text."""
    return f"{color}{text}{Colors.RESET}"


def format_event(event: dict) -> str:
    """Format an event for display."""
    event_type = event.get("type", "unknown")
    timestamp = event.get("timestamp", "")

    # Extract time portion for compact display
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            time_str = dt.strftime("%H:%M:%S")
        except ValueError:
            time_str = timestamp[:8]
    else:
        time_str = datetime.now().strftime("%H:%M:%S")

    # Color based on event type
    if event_type.startswith("agent.output"):
        source = event.get("source", "?")
        text = event.get("text", "")
        mode = event.get("mode", "write")
        agent_id = event.get("agent_id", "main")

        if source == "model":
            return f"{colorize(f'[{time_str}]', Colors.DIM)} {colorize(f'[{agent_id}]', Colors.CYAN)} {text}"
        elif source == "user":
            return f"{colorize(f'[{time_str}]', Colors.DIM)} {colorize('[you]', Colors.GREEN)} {text}"
        else:
            return f"{colorize(f'[{time_str}]', Colors.DIM)} {colorize(f'[{source}]', Colors.YELLOW)} {text}"

    elif event_type == "agent.status_changed":
        agent_id = event.get("agent_id", "?")
        status = event.get("status", "?")
        status_color = Colors.GREEN if status == "done" else Colors.YELLOW if status == "active" else Colors.RED
        return f"{colorize(f'[{time_str}]', Colors.DIM)} {colorize(f'● {agent_id}: {status}', status_color)}"

    elif event_type == "tool.call_start":
        tool_name = event.get("tool_name", "?")
        return f"{colorize(f'[{time_str}]', Colors.DIM)} {colorize(f'→ {tool_name}', Colors.MAGENTA)} ..."

    elif event_type == "tool.call_end":
        tool_name = event.get("tool_name", "?")
        success = event.get("success", True)
        duration = event.get("duration_seconds", 0)
        icon = "✓" if success else "✗"
        color = Colors.GREEN if success else Colors.RED
        return f"{colorize(f'[{time_str}]', Colors.DIM)} {colorize(f'{icon} {tool_name}', color)} ({duration:.2f}s)"

    elif event_type == "permission.requested":
        tool_name = event.get("tool_name", "?")
        options = event.get("response_options", [])
        prompt_lines = event.get("prompt_lines", [])

        lines = [f"{colorize(f'[{time_str}]', Colors.DIM)} {colorize('⚠ Permission required:', Colors.YELLOW)} {tool_name}"]
        for line in prompt_lines[:5]:  # Show first 5 lines
            lines.append(f"    {line}")
        if len(prompt_lines) > 5:
            lines.append(f"    ... ({len(prompt_lines) - 5} more lines)")

        option_str = " ".join(f"[{o.get('key', '?')}]{o.get('label', '?')}" for o in options[:4])
        lines.append(f"    {colorize('Options:', Colors.BOLD)} {option_str}")
        return "\n".join(lines)

    elif event_type == "context.updated":
        total = event.get("total_tokens", 0)
        percent = event.get("percent_used", 0)
        return f"{colorize(f'[{time_str}]', Colors.DIM)} {colorize('📊', Colors.BLUE)} Context: {total} tokens ({percent:.1f}%)"

    elif event_type == "plan.updated":
        plan_name = event.get("plan_name", "Plan")
        steps = event.get("steps", [])
        lines = [f"{colorize(f'[{time_str}]', Colors.DIM)} {colorize(f'📋 {plan_name}', Colors.CYAN)}"]
        for step in steps:
            status = step.get("status", "pending")
            content = step.get("content", "?")
            icon = "✓" if status == "completed" else "●" if status == "in_progress" else "○"
            color = Colors.GREEN if status == "completed" else Colors.YELLOW if status == "in_progress" else Colors.DIM
            lines.append(f"    {colorize(f'{icon} {content}', color)}")
        return "\n".join(lines)

    elif event_type == "system.message":
        message = event.get("message", "")
        style = event.get("style", "info")
        color = Colors.YELLOW if style == "warning" else Colors.RED if style == "error" else Colors.CYAN
        return f"{colorize(f'[{time_str}]', Colors.DIM)} {colorize(f'ℹ {message}', color)}"

    elif event_type == "error":
        error = event.get("error", "Unknown error")
        return f"{colorize(f'[{time_str}]', Colors.DIM)} {colorize(f'✗ Error: {error}', Colors.RED)}"

    elif event_type == "connected":
        info = event.get("server_info", {})
        model = info.get("model_name", "?")
        provider = info.get("model_provider", "?")
        return f"{colorize(f'[{time_str}]', Colors.DIM)} {colorize(f'✓ Connected to {provider}/{model}', Colors.GREEN)}"

    else:
        # Generic event display
        return f"{colorize(f'[{time_str}]', Colors.DIM)} {colorize(event_type, Colors.BLUE)}: {json.dumps(event, default=str)[:100]}"


async def receive_events(websocket: ClientConnection) -> None:
    """Receive and display events from the server."""
    try:
        async for message in websocket:
            try:
                event = json.loads(message)
                formatted = format_event(event)
                print(formatted)
            except json.JSONDecodeError:
                print(f"[Invalid JSON] {message[:100]}")
    except websockets.ConnectionClosed:
        print(colorize("\n[Disconnected from server]", Colors.YELLOW))


async def send_messages(websocket: ClientConnection) -> None:
    """Read user input and send messages to the server."""
    print(colorize("\nType a message and press Enter. Type 'quit' to exit.\n", Colors.DIM))

    loop = asyncio.get_event_loop()

    while True:
        try:
            # Read input in executor to not block
            user_input = await loop.run_in_executor(None, lambda: input(colorize("> ", Colors.GREEN)))

            if user_input.lower() in ("quit", "exit", "q"):
                break

            if not user_input.strip():
                continue

            # Check for special commands
            if user_input.startswith("/"):
                parts = user_input[1:].split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1].split() if len(parts) > 1 else []

                if cmd == "stop":
                    request = {
                        "type": "session.stop",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                elif cmd == "perm":
                    # Quick permission response: /perm request_id y
                    if len(args) >= 2:
                        request = {
                            "type": "permission.response",
                            "request_id": args[0],
                            "response": args[1],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    else:
                        print(colorize("Usage: /perm <request_id> <response>", Colors.YELLOW))
                        continue
                else:
                    request = {
                        "type": "command.execute",
                        "command": cmd,
                        "args": args,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
            else:
                # Regular message
                request = {
                    "type": "message.send",
                    "text": user_input,
                    "attachments": [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            await websocket.send(json.dumps(request))

        except EOFError:
            break
        except Exception as e:
            print(colorize(f"Error: {e}", Colors.RED))


async def run_client(
    host: str,
    port: int,
    message: Optional[str] = None,
    interactive: bool = True,
) -> None:
    """Run the test client.

    Args:
        host: Server host.
        port: Server port.
        message: Optional single message to send (non-interactive).
        interactive: Whether to run in interactive mode.
    """
    uri = f"ws://{host}:{port}"
    print(colorize(f"Connecting to {uri}...", Colors.DIM))

    try:
        async with websockets.connect(uri) as websocket:
            if message:
                # Single message mode
                request = {
                    "type": "message.send",
                    "text": message,
                    "attachments": [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await websocket.send(json.dumps(request))

                # Receive events until model completes
                async for msg in websocket:
                    event = json.loads(msg)
                    print(format_event(event))

                    # Exit after model finishes
                    if event.get("type") == "agent.status_changed":
                        if event.get("status") == "done":
                            break

            elif interactive:
                # Interactive mode - run send and receive concurrently
                receive_task = asyncio.create_task(receive_events(websocket))
                send_task = asyncio.create_task(send_messages(websocket))

                # Wait for send task (user quits) or receive task (connection closed)
                done, pending = await asyncio.wait(
                    [receive_task, send_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            else:
                # Just receive events (monitoring mode)
                await receive_events(websocket)

    except ConnectionRefusedError:
        print(colorize(f"Error: Could not connect to {uri}", Colors.RED))
        print(colorize("Make sure the server is running:", Colors.DIM))
        print(colorize(f"  python rich_client.py --headless --port {port}", Colors.DIM))
        sys.exit(1)
    except Exception as e:
        print(colorize(f"Error: {e}", Colors.RED))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Test client for Jaato WebSocket Server"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Server port (default: 8080)"
    )
    parser.add_argument(
        "--message", "-m",
        type=str,
        help="Send a single message and exit"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor mode - only receive events, don't send"
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_client(
            host=args.host,
            port=args.port,
            message=args.message,
            interactive=not args.monitor and not args.message,
        ))
    except KeyboardInterrupt:
        print(colorize("\nGoodbye!", Colors.DIM))


if __name__ == "__main__":
    main()
