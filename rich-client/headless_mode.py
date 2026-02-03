# rich-client/headless_mode.py
"""Headless mode for jaato client.

Provides non-interactive, file-based output for automation and scripting.
All permissions are auto-approved.
Output goes to per-agent files in {workspace}/jaato-headless-client-agents/
"""

import asyncio
import os
import pathlib
import sys
from typing import Optional

from dotenv import load_dotenv

from renderers.headless import HeadlessFileRenderer


async def run_headless_mode(
    socket_path: str,
    prompt: str,
    workspace: Optional[pathlib.Path] = None,
    auto_start: bool = True,
    env_file: str = ".env",
    new_session: bool = False,
):
    """Run the client in headless mode with file output.

    All output goes to files in {workspace}/jaato-headless-client-agents/.

    Permission handling:
    - Sets session default policy to "allow" via `permissions default allow`
    - This auto-approves all tools not in the blacklist
    - If a prompt still occurs (blacklisted tool), responds with "y" (once)

    Clarification handling:
    - Sets main agent's channel to "auto" via `clarification channel auto`
    - Main agent clarifications are auto-responded (no user to ask)
    - Subagent clarifications still forward to parent agent (ParentBridgedChannel)

    Session isolation:
    - Use --new-session to create an isolated session for headless mode
    - Without --new-session, may attach to existing session (shared permission state)

    Args:
        socket_path: Path to the Unix domain socket.
        prompt: The prompt to send.
        workspace: Workspace root directory (default: current directory).
        auto_start: Whether to auto-start the server if not running.
        env_file: Path to .env file for auto-started server.
        new_session: Whether to start a new session instead of resuming default.
    """
    # Load env vars
    load_dotenv(env_file)

    from ipc_recovery import IPCRecoveryClient
    from server.events import (
        AgentOutputEvent,
        AgentCreatedEvent,
        AgentStatusChangedEvent,
        AgentCompletedEvent,
        PermissionInputModeEvent,
        PermissionResolvedEvent,
        ClarificationInputModeEvent,
        ClarificationResolvedEvent,
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
        RetryEvent,
    )

    # Determine workspace
    if workspace is None:
        workspace = pathlib.Path.cwd()

    # Create renderer
    renderer = HeadlessFileRenderer(workspace=workspace, flush_immediately=True)
    renderer.start()

    # Create IPC client with recovery support
    client = IPCRecoveryClient(
        socket_path=socket_path,
        auto_start=auto_start,
        env_file=env_file,
        workspace_path=workspace,
    )

    # State tracking
    model_running = False
    should_exit = False
    turn_completed = False

    # Connect to server
    print(f"[headless] Connecting to server at {socket_path}...", file=sys.stderr)

    try:
        connected = await client.connect()
        if not connected:
            print("[headless] Connection failed: Server did not respond with handshake", file=sys.stderr)
            renderer.shutdown()
            return
        print("[headless] Connected!", file=sys.stderr)

    except ConnectionError as e:
        print(f"[headless] Connection failed: {e}", file=sys.stderr)
        renderer.shutdown()
        return

    # Request new session if specified (recommended for headless to ensure isolation)
    if new_session:
        await client.create_session()

    # Set default permission policy to "allow" for headless mode
    # This auto-approves all tools not in blacklist, avoiding per-prompt responses
    print("[headless] Setting permission policy to auto-approve...", file=sys.stderr)
    await client.execute_command("permissions", ["default", "allow"])

    # Disable clarification tool - no user to answer questions in headless mode
    # Uses direct registry call (no response events to consume)
    print("[headless] Disabling clarification tool...", file=sys.stderr)
    await client.disable_tool("clarification")

    async def handle_events():
        """Handle events from the server."""
        nonlocal model_running, should_exit, turn_completed

        async for event in client.events():
            if should_exit:
                break

            # ==================== Init Progress ====================
            if isinstance(event, InitProgressEvent):
                status_map = {"running": "...", "done": "OK", "error": "ERROR", "pending": "PENDING"}
                status_text = status_map.get(event.status, event.status)
                renderer.on_system_message(
                    f"  {event.step}: {status_text}",
                    style="system_progress" if event.status == "running" else "system_info"
                )

            # ==================== Agent Events ====================
            elif isinstance(event, AgentCreatedEvent):
                renderer.on_agent_created(
                    agent_id=event.agent_id,
                    agent_type=event.agent_type,
                    name=event.agent_name,
                    profile_name=event.profile_name,
                    parent_agent_id=event.parent_agent_id,
                )

            elif isinstance(event, AgentStatusChangedEvent):
                model_running = event.status == "active"
                renderer.on_agent_status_changed(event.agent_id, event.status)
                # Don't exit on status changes - wait for AgentCompletedEvent

            elif isinstance(event, AgentCompletedEvent):
                renderer.on_agent_completed(event.agent_id)
                # Exit when main agent completes
                if event.agent_id == "main":
                    should_exit = True
                    break

            elif isinstance(event, AgentOutputEvent):
                renderer.on_agent_output(
                    agent_id=event.agent_id,
                    source=event.source,
                    text=event.text,
                    mode=event.mode,
                )

            # ==================== Tool Events ====================
            elif isinstance(event, ToolCallStartEvent):
                renderer.on_tool_start(
                    agent_id=event.agent_id,
                    tool_name=event.tool_name,
                    tool_args=event.tool_args or {},
                    call_id=event.call_id,
                )

            elif isinstance(event, ToolCallEndEvent):
                renderer.on_tool_end(
                    agent_id=event.agent_id,
                    tool_name=event.tool_name,
                    success=event.success,
                    duration_seconds=event.duration_seconds or 0.0,
                    error_message=event.error_message,
                    call_id=event.call_id,
                )

            elif isinstance(event, ToolOutputEvent):
                if event.call_id:
                    renderer.on_tool_output(
                        agent_id=event.agent_id,
                        call_id=event.call_id,
                        chunk=event.chunk,
                    )

            # ==================== Permission Events ====================
            elif isinstance(event, PermissionInputModeEvent):
                # With "permissions default allow" policy, this shouldn't happen
                # But if it does (e.g., blacklisted tool), respond with "y" (once)
                renderer.on_permission_requested(
                    agent_id=event.agent_id or "main",
                    request_id=event.request_id,
                    tool_name=event.tool_name,
                    call_id=event.call_id,
                    response_options=event.response_options,
                )
                # Respond with "y" (once) - safer than "a" (always) for edge cases
                await client.respond_to_permission(event.request_id, "y")

            elif isinstance(event, PermissionResolvedEvent):
                renderer.on_permission_resolved(
                    agent_id=event.agent_id or "main",
                    tool_name=event.tool_name,
                    granted=event.granted,
                    method=event.method or "auto",
                )

            # ==================== Clarification Events ====================
            elif isinstance(event, ClarificationInputModeEvent):
                # With AutoChannel, this shouldn't happen for main agent
                # But handle as fallback (e.g., if channel wasn't set correctly)
                renderer.on_system_message(
                    f"[headless] Clarification requested for {event.tool_name}, auto-skipping",
                    style="system_warning"
                )
                await client.respond_to_clarification(event.request_id, "")

            # ==================== Plan Events ====================
            elif isinstance(event, PlanUpdatedEvent):
                # Convert to plan_data dict
                total_steps = len(event.steps)
                completed_steps = sum(1 for s in event.steps if s.get("status") == "completed")
                percent = (completed_steps / total_steps * 100) if total_steps > 0 else 0

                plan_steps = []
                for i, step in enumerate(event.steps):
                    step_data = {
                        "description": step.get("content", ""),
                        "status": step.get("status", "pending"),
                        "active_form": step.get("active_form"),
                        "sequence": i + 1,
                    }
                    if step.get("blocked_by"):
                        step_data["blocked_by"] = step["blocked_by"]
                    if step.get("depends_on"):
                        step_data["depends_on"] = step["depends_on"]
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
                renderer.on_plan_updated(agent_id, plan_data)

            elif isinstance(event, PlanClearedEvent):
                agent_id = getattr(event, 'agent_id', None)
                renderer.on_plan_cleared(agent_id)

            # ==================== Context Events ====================
            elif isinstance(event, ContextUpdatedEvent):
                renderer.on_context_updated(
                    agent_id=event.agent_id or "main",
                    total_tokens=event.total_tokens or 0,
                    prompt_tokens=event.prompt_tokens or 0,
                    output_tokens=event.output_tokens or 0,
                    turns=event.turns or 0,
                    percent_used=event.percent_used or 0.0,
                )

            # ==================== Error/Retry Events ====================
            elif isinstance(event, ErrorEvent):
                renderer.on_error(event.error, event.error_type or None)

            elif isinstance(event, RetryEvent):
                renderer.on_retry(
                    attempt=event.attempt,
                    max_attempts=event.max_attempts,
                    reason=event.reason,
                    delay_seconds=event.delay_seconds,
                )

            # ==================== System Messages ====================
            elif isinstance(event, SystemMessageEvent):
                renderer.on_system_message(event.message, event.style or "system_info")

            # ==================== Turn Completion ====================
            elif isinstance(event, TurnCompletedEvent):
                turn_completed = True
                # Don't exit on turn completion alone - wait for agent to complete
                # Turn completion just means one request-response cycle finished

    # Send the prompt
    print(f"[headless] Sending prompt...", file=sys.stderr)
    await client.send_message(prompt)

    # Wait for events until turn completes
    try:
        await handle_events()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[headless] Error: {e}", file=sys.stderr)

    # Cleanup - use close() for permanent shutdown (stops event stream)
    renderer.shutdown()
    await client.close()

    print(f"[headless] Output written to: {renderer.output_dir}", file=sys.stderr)
