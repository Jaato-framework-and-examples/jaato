# rich-client/command_mode.py
"""Command mode for sending commands to jaato sessions.

Allows controlling headless sessions from another terminal:
    python rich_client.py --connect /tmp/jaato.sock --session <id> --cmd stop
    python rich_client.py --connect /tmp/jaato.sock --session <id> --cmd "permissions default deny"
"""

import asyncio
import sys
from typing import Optional

from dotenv import load_dotenv


async def run_command_mode(
    socket_path: str,
    session_id: str,
    command: str,
    auto_start: bool = True,
    env_file: str = ".env",
):
    """Connect to a session and send a command.

    Args:
        socket_path: Path to the Unix domain socket.
        session_id: Session ID to attach to.
        command: Command to send (e.g., 'stop', 'reset', 'permissions default deny').
        auto_start: Whether to auto-start the server if not running.
        env_file: Path to .env file for auto-started server.
    """
    load_dotenv(env_file)

    from ipc_recovery import IPCRecoveryClient
    from server.events import (
        SystemMessageEvent,
        ErrorEvent,
        ToolStatusEvent,
        HelpTextEvent,
    )

    client = IPCRecoveryClient(
        socket_path=socket_path,
        auto_start=auto_start,
        env_file=env_file,
    )

    try:
        connected = await client.connect()
        if not connected:
            print(f"Error: Failed to connect to server at {socket_path}", file=sys.stderr)
            return

        # Attach to the specified session
        attached = await client.attach_session(session_id)
        if not attached:
            print(f"Error: Failed to attach to session '{session_id}'", file=sys.stderr)
            print("Use 'session list' to see available sessions.", file=sys.stderr)
            await client.disconnect()
            return

        # Parse command - split into command name and args
        parts = command.split()
        cmd_name = parts[0] if parts else ""
        cmd_args = parts[1:] if len(parts) > 1 else []

        # Handle special commands that need different formatting
        if cmd_name == "stop":
            # Stop is a direct method, not a command
            await client.stop()
            print(f"Sent stop signal to session '{session_id}'")
            await client.disconnect()
            return

        # For other commands, use execute_command
        # Format command with subcommand if present (e.g., "permissions default allow" -> "permissions", ["default", "allow"])
        await client.execute_command(cmd_name, cmd_args)

        # Wait briefly for response events
        response_received = False
        timeout = 5.0  # seconds
        start_time = asyncio.get_event_loop().time()

        async for event in client.events():
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                break

            if isinstance(event, SystemMessageEvent):
                print(event.message)
                response_received = True
                break

            elif isinstance(event, ErrorEvent):
                print(f"Error: {event.message}", file=sys.stderr)
                if event.details:
                    print(f"Details: {event.details}", file=sys.stderr)
                response_received = True
                break

            elif isinstance(event, ToolStatusEvent):
                if event.message:
                    print(event.message)
                response_received = True
                break

            elif isinstance(event, HelpTextEvent):
                for line, style in event.lines:
                    print(line)
                response_received = True
                break

        if not response_received:
            print(f"Command '{command}' sent to session '{session_id}'")

    except ConnectionError as e:
        print(f"Connection error: {e}", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

    finally:
        await client.disconnect()
