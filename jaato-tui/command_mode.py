# jaato-tui/command_mode.py
"""Command mode for sending commands/messages to jaato sessions.

Allows controlling headless sessions from another terminal:
    python rich_client.py --connect /tmp/jaato.sock --session <id> --cmd stop
    python rich_client.py --connect /tmp/jaato.sock --session <id> --cmd "permissions default deny"
    python rich_client.py --connect /tmp/jaato.sock --session <id> --cmd "please summarize"

And sending a DAEMON-level command, which needs no session:
    python rich_client.py --connect /tmp/jaato.sock --cmd "pool resize 6"
    python rich_client.py --connect /tmp/jaato.sock --cmd "session list"

Without ``--session`` the client attaches to nothing, never starts a daemon
(a command for the daemon is meaningless to a daemon started to receive
it), and accepts only server commands: a plain message has no session to
go to, and ``stop`` / ``exit`` / ``history`` name the session it does not
have.

Commands are processed using the same routing logic as the TUI.
"""

import asyncio
import sys
from typing import Optional

from dotenv import load_dotenv


async def run_command_mode(
    socket_path: str,
    session_id: Optional[str],
    command: str,
    auto_start: bool = True,
    env_file: str = ".env",
):
    """Connect to a session (or only to the daemon) and send a command/message.

    Uses shared command parsing logic from shared.client_commands.

    Args:
        socket_path: Path to the Unix domain socket.
        session_id: Session ID to attach to, or ``None`` for a daemon-level
            command (see the module docstring): then nothing is attached,
            ``auto_start`` is ignored, and only a server command is sent.
        command: Command or message to send.
        auto_start: Whether to auto-start the server if not running.
        env_file: Path to .env file for auto-started server.
    """
    load_dotenv(env_file)
    from client_commands import parse_user_input, CommandAction
    if session_id is None:
        if parse_user_input(command).action != CommandAction.SERVER_COMMAND:
            print(f"Error: {command!r} needs a session; pass --session <id>, "
                  f"or send a daemon command such as 'pool status' or "
                  f"'session list'", file=sys.stderr)
            return
        auto_start = False

    from jaato_sdk.client.recovery import IPCRecoveryClient
    from jaato_sdk.events import (
        SystemMessageEvent,
        ErrorEvent,
        WorkspaceMismatchRequestedEvent,
        WorkspaceMismatchResponseRequest,
    )

    from jaato_sdk.events import ClientType
    client = IPCRecoveryClient(
        socket_path=socket_path,
        client_type=ClientType.TERMINAL,
        auto_start=auto_start,
        env_file=env_file,
    )

    try:
        connected = await client.connect()
        if not connected:
            print(f"Error: Failed to connect to server at {socket_path}", file=sys.stderr)
            return

        # A daemon-level command attaches to nothing.
        if session_id is None:
            parsed = parse_user_input(command)
            await client.execute_command(parsed.command, parsed.args or [])
            await _print_response(client)
            return

        # Attach to the specified session
        attached = await client.attach_session(session_id)
        if not attached:
            print(f"Error: Failed to attach to session '{session_id}'", file=sys.stderr)
            print("Use 'session list' to see available sessions.", file=sys.stderr)
            await client.disconnect()
            return

        # Wait for server to confirm attachment before sending messages.
        # This ensures the server has registered us with the session.
        # In command mode we auto-accept workspace mismatches (switch to the
        # session's workspace) because the user explicitly targeted --session.
        timeout = 5.0
        start = asyncio.get_event_loop().time()
        async for event in client.events():
            if asyncio.get_event_loop().time() - start > timeout:
                print("Warning: Attach confirmation timeout", file=sys.stderr)
                break
            if isinstance(event, WorkspaceMismatchRequestedEvent):
                # Auto-switch to session workspace in command mode
                await client._client._send_event(WorkspaceMismatchResponseRequest(
                    request_id=event.request_id,
                    response="switch",
                ))
                # Reset timeout — server will now proceed with the attach
                start = asyncio.get_event_loop().time()
            elif isinstance(event, SystemMessageEvent):
                # Server sends "Attached to session: <id>" on successful attach
                if "Attached to session" in event.message:
                    break
            elif isinstance(event, ErrorEvent):
                print(f"Error: {event.error}", file=sys.stderr)
                await client.disconnect()
                return

        # Parse using shared logic
        parsed = parse_user_input(command)

        # Track if we need to wait for response (commands that return data)
        wait_for_response = False

        # Execute based on action type
        if parsed.action == CommandAction.EXIT:
            # End session - stop agent and delete from server
            await client.delete_session(session_id)
            print(f"Session '{session_id}' ended")

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
            await _print_response(client)

    except ConnectionError as e:
        print(f"Connection error: {e}", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

    finally:
        await client.disconnect()


async def _print_response(client, timeout: float = 5.0) -> None:
    """Print the first answer the daemon sends, then return.

    A ``PoolStatusEvent`` is skipped rather than printed: the daemon sends
    the readable ``SystemMessageEvent`` line right behind it, for a person.
    """
    from jaato_sdk.events import (
        ErrorEvent,
        HelpTextEvent,
        SessionListEvent,
        SystemMessageEvent,
        ToolStatusEvent,
    )
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
