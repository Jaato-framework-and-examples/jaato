# rich-client/command_mode.py
"""Command mode for sending commands/messages to jaato sessions.

Allows controlling headless sessions from another terminal:
    python rich_client.py --connect /tmp/jaato.sock --session <id> --cmd stop
    python rich_client.py --connect /tmp/jaato.sock --session <id> --cmd "permissions default deny"
    python rich_client.py --connect /tmp/jaato.sock --session <id> --cmd "please summarize"

Commands are processed using the same routing logic as the TUI.
"""

import asyncio
import sys

from dotenv import load_dotenv


async def run_command_mode(
    socket_path: str,
    session_id: str,
    command: str,
    auto_start: bool = True,
    env_file: str = ".env",
):
    """Connect to a session and send a command/message.

    Uses shared command parsing logic from shared.client_commands.

    Args:
        socket_path: Path to the Unix domain socket.
        session_id: Session ID to attach to.
        command: Command or message to send.
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
        SessionListEvent,
    )
    from shared.client_commands import parse_user_input, CommandAction

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

        # Parse using shared logic
        parsed = parse_user_input(command)

        # Track if we need to wait for response (commands that return data)
        wait_for_response = False

        # Execute based on action type
        if parsed.action == CommandAction.EXIT:
            await client.stop()
            print(f"Sent stop signal to session '{session_id}'")

        elif parsed.action == CommandAction.STOP:
            await client.stop()
            print(f"Sent stop signal to session '{session_id}'")

        elif parsed.action == CommandAction.CLEAR:
            print("'clear' is a display-only command, not applicable in command mode")

        elif parsed.action == CommandAction.HELP:
            await client.request_command_list()
            wait_for_response = True

        elif parsed.action == CommandAction.CONTEXT:
            print("'context' requires display state, not available in command mode")

        elif parsed.action == CommandAction.HISTORY:
            await client.request_history()
            wait_for_response = True

        elif parsed.action == CommandAction.SERVER_COMMAND:
            await client.execute_command(parsed.command, parsed.args or [])
            wait_for_response = True

        elif parsed.action == CommandAction.SEND_MESSAGE:
            if parsed.text:
                print(f"Sent message to session '{session_id}'")
                await client.send_message(parsed.text)
                # Fire and forget - the session handles the response
                # Don't wait for turn completion as this would hijack the session
            else:
                print("Empty message, nothing to send")

        # Wait for response events if needed (only for commands that return data)
        if wait_for_response:
            timeout = 5.0
            start_time = asyncio.get_event_loop().time()

            async for event in client.events():
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    break

                if isinstance(event, SystemMessageEvent):
                    print(event.message)
                    break

                elif isinstance(event, ErrorEvent):
                    print(f"Error: {event.error}", file=sys.stderr)
                    if event.error_type:
                        print(f"Type: {event.error_type}", file=sys.stderr)
                    break

                elif isinstance(event, ToolStatusEvent):
                    if event.message:
                        print(event.message)
                    break

                elif isinstance(event, HelpTextEvent):
                    for line, style in event.lines:
                        print(line)
                    break

                elif isinstance(event, SessionListEvent):
                    print("Available sessions:")
                    for s in event.sessions:
                        status = "loaded" if s.get("is_loaded") else "saved"
                        print(f"  {s.get('id', '?')} - {s.get('name', '')} [{status}]")
                    break

    except ConnectionError as e:
        print(f"Connection error: {e}", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

    finally:
        await client.disconnect()
