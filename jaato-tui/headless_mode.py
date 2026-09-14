# jaato-tui/headless_mode.py
"""Headless mode for jaato client.

Provides non-interactive, file-based output for automation and scripting.
All permissions are auto-approved.
Output goes to per-agent files in {workspace}/jaato-headless-client-agents/
"""

import asyncio
import logging
import os
import pathlib
import sys
from typing import Optional

from dotenv import load_dotenv

from renderers.headless import HeadlessFileRenderer

logger = logging.getLogger(__name__)


async def _auto_skip_clarification(client, renderer, event) -> None:
    """Answer a clarification nobody is here to answer.

    Headless runs disable the clarification tool and put the main agent on
    the auto channel, so this is a fallback for the cases that slip past
    both.  It has to exist because an unanswered clarification is not a
    missing prompt but a stuck turn: the tool call blocks until it gets a
    reply (#704).

    Both shapes of request are answered the same way — every question
    skipped:

    * ``ClarificationInputModeEvent`` — one empty answer for the question
      currently being asked; the daemon drives the next one.
    * ``ClarificationBatchEvent`` — one empty answer per question, sent
      as a single batch.  ``batch_only`` batches (runner-tier sessions)
      get nothing else, so this reply is the only thing that unblocks
      them; a preview batch is left to the per-question flow that
      follows it.
    """
    from jaato_sdk.events import ClarificationBatchEvent

    if not isinstance(event, ClarificationBatchEvent):
        renderer.on_system_message(
            f"[headless] Clarification requested for {event.tool_name}, auto-skipping",
            style="system_warning"
        )
        await client.respond_to_clarification(event.request_id, "")
        return

    if not event.batch_only:
        return
    questions = event.questions or []
    renderer.on_system_message(
        f"[headless] Clarification requested for {event.tool_name} "
        f"({len(questions)} questions), auto-skipping",
        style="system_warning"
    )
    await client.respond_to_clarification_batch(
        event.request_id, [""] * len(questions)
    )


async def run_headless_mode(
    socket_path: str,
    prompt: str,
    workspace: Optional[pathlib.Path] = None,
    auto_start: bool = True,
    env_file: str = ".env",
    new_session: bool = False,
    profile: Optional[str] = None,
    agent: Optional[str] = None,
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
        profile: Optional runtime profile name from .jaato/profiles/.
        agent: Optional agent name from .jaato/agents/.
    """
    # Load env vars
    load_dotenv(env_file)

    # Configure logging - redirect to JAATO_TRACE_LOG if set
    trace_log_path = os.environ.get("JAATO_TRACE_LOG")
    if trace_log_path:
        os.makedirs(os.path.dirname(os.path.abspath(trace_log_path)), exist_ok=True)
        file_handler = logging.FileHandler(trace_log_path)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        root_logger = logging.getLogger()
        root_logger.handlers = [file_handler]
        root_logger.setLevel(logging.DEBUG)

    from jaato_sdk.client.recovery import IPCRecoveryClient
    from jaato_sdk.client.errors import SessionCreateFailed
    from jaato_sdk.events import (
        AgentOutputEvent,
        AgentCreatedEvent,
        AgentStatusChangedEvent,
        AgentCompletedEvent,
        PermissionInputModeEvent,
        PermissionResolvedEvent,
        ClarificationBatchEvent,
        ClarificationInputModeEvent,
        ClarificationResolvedEvent,
        PlanUpdatedEvent,
        PlanClearedEvent,
        ToolCallStartEvent,
        ToolCallEndEvent,
        ToolOutputEvent,
        ContextUpdatedEvent,
        InstructionBudgetEvent,
        TurnCompletedEvent,
        SystemMessageEvent,
        InitProgressEvent,
        ErrorEvent,
        RetryEvent,
        SessionInfoEvent,
    )

    # Determine workspace
    if workspace is None:
        workspace = pathlib.Path.cwd()
    else:
        # A caller may hand us a relative path; resolve it HERE, where the
        # cwd it is relative to actually lives.  The daemon refuses a
        # relative workspace rather than resolving it against its own cwd
        # (issue #742), so an unresolved value would fail the connect.
        workspace = pathlib.Path(workspace).expanduser().resolve()

    # Create renderer
    renderer = HeadlessFileRenderer(workspace=workspace, flush_immediately=True)
    renderer.start()

    # Create IPC client with recovery support.  ``--headless`` mode is
    # for non-TTY automation (CI, scripting, batch runs); declare
    # ClientType.API so the server's interactive-root filter does NOT
    # strip ``signal_completion`` — headless agents need it to terminate
    # cleanly.
    from jaato_sdk.events import ClientType
    client = IPCRecoveryClient(
        socket_path=socket_path,
        client_type=ClientType.API,
        auto_start=auto_start,
        env_file=env_file,
        workspace_path=workspace,
    )

    # State tracking
    model_running = False
    should_exit = False
    turn_completed = False
    turn_count = 0
    active_subagents: set = set()  # Track running subagent IDs

    # Budget tracking per agent (agent_id -> snapshot)
    budget_snapshots: dict = {}

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
    #
    # The return value used to be DISCARDED.  A create failure was invisible
    # here: headless mode went on to set the permission policy, disable
    # clarification, and send the prompt -- against no session -- and the
    # operator saw a confusing downstream failure instead of "the session was
    # never created".  There is no interactive user to notice, which is what
    # made it worth handling rather than merely logging.
    if new_session or profile or agent:
        try:
            await client.create_session(profile=profile, agent=agent)
        except SessionCreateFailed as exc:
            print(f"[headless] Session creation failed: {exc}", file=sys.stderr)
            if exc.may_exist:
                # Do NOT retry blind: session.new has no idempotency key, so a
                # second attempt is a second session with its own runner.
                print("[headless] A session may have been created despite "
                      "this — check `session.list` before retrying.",
                      file=sys.stderr)
            renderer.shutdown()
            return

    # Set default permission policy to "allow" for headless mode
    # This auto-approves all tools not in blacklist, avoiding per-prompt responses
    print("[headless] Setting permission policy to auto-approve...", file=sys.stderr)
    await client.set_default_policy("allow")

    # Disable clarification tool - no user to answer questions in headless mode
    # Uses direct registry call (no response events to consume)
    print("[headless] Disabling clarification tool...", file=sys.stderr)
    await client.disable_tool("clarification")

    # Track if session_id has been printed (print once when SessionInfoEvent arrives)
    session_id_printed = False
    # Track if main agent has been activated (for initial "working on prompt" message)
    main_agent_activated = False
    print("[headless] Waiting for session acquisition...", file=sys.stderr)

    async def handle_events():
        """Handle events from the server."""
        nonlocal model_running, should_exit, turn_completed, turn_count, session_id_printed, main_agent_activated, active_subagents

        async for event in client.events():
            if should_exit:
                break

            # ==================== Session Info ====================
            # Print session_id once when acquired, then "Sending prompt..."
            if isinstance(event, SessionInfoEvent):
                if not session_id_printed and event.session_id:
                    print(f"[headless] Session ID: {event.session_id}", file=sys.stderr)
                    print("[headless] Sending prompt...", file=sys.stderr)
                    session_id_printed = True
                continue

            # ==================== Init Progress ====================
            elif isinstance(event, InitProgressEvent):
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
                if event.agent_id != "main":
                    active_subagents.add(event.agent_id)

            elif isinstance(event, AgentStatusChangedEvent):
                model_running = event.status == "active"
                renderer.on_agent_status_changed(event.agent_id, event.status)
                # Print initial activation message for main agent
                if event.agent_id == "main" and event.status == "active" and not main_agent_activated:
                    print("[headless] Main agent working on input prompt...", file=sys.stderr)
                    main_agent_activated = True
                # Print budget when agent becomes idle or done
                if event.status in ("idle", "done"):
                    agent_id = event.agent_id or "main"
                    snapshot = budget_snapshots.get(agent_id)
                    if snapshot:
                        total = snapshot.get("total_tokens", 0)
                        limit = snapshot.get("context_limit", 0)
                        pct = snapshot.get("utilization_percent", 0)
                        entries = snapshot.get("entries", {})
                        categories = []
                        for source_name, entry in entries.items():
                            tokens = entry.get("tokens", 0)
                            if tokens > 0:
                                categories.append(f"{source_name}:{tokens:,}")
                        category_str = " | ".join(categories) if categories else "no data"
                        renderer.on_system_message(
                            f"Budget: {total:,}/{limit:,} tokens ({pct:.1f}%) | {category_str}",
                            style="system_info",
                            agent_id=agent_id,
                        )
                # Exit when main agent is truly finished.
                # The server emits "done" only when there is nothing left
                # to do (no pending channel input, no active subagents).
                # While subagents are still running or the agent awaits
                # user input, the server emits "idle" instead.
                if event.agent_id == "main" and event.status == "done":
                    should_exit = True
                    break

            elif isinstance(event, AgentCompletedEvent):
                renderer.on_agent_completed(event.agent_id)
                active_subagents.discard(event.agent_id)

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
                        stream_id=event.stream_id,
                        sequence=event.sequence,
                        mime_type=event.mime_type,
                        data_b64=event.data_b64,
                        final=event.final,
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
            elif isinstance(event, (ClarificationInputModeEvent, ClarificationBatchEvent)):
                # With AutoChannel, this shouldn't happen for the main agent,
                # but answer it anyway: an unanswered clarification blocks the
                # tool call and the turn behind it forever (#704), and there
                # is nobody here to notice.
                await _auto_skip_clarification(client, renderer, event)

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
                        "step_id": step.get("step_id", ""),
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
                    total_tokens=event.usage.total_tokens or 0,
                    prompt_tokens=event.usage.prompt_tokens or 0,
                    output_tokens=event.usage.output_tokens or 0,
                    turns=event.turns or 0,
                    percent_used=event.percent_used or 0.0,
                )

            elif isinstance(event, InstructionBudgetEvent):
                # Track budget per agent
                agent_id = event.agent_id or "main"
                logger.debug(f"InstructionBudgetEvent: agent_id={agent_id}, snapshot keys={list(event.budget_snapshot.keys()) if event.budget_snapshot else 'None'}")
                if event.budget_snapshot:
                    budget_snapshots[agent_id] = event.budget_snapshot
                    logger.debug(f"budget_snapshots[{agent_id}] updated: {event.budget_snapshot.get('total_tokens', 'N/A')} tokens")

            # ==================== Error/Retry Events ====================
            elif isinstance(event, ErrorEvent):
                renderer.on_error(event.error, event.error_type or None)

            elif isinstance(event, RetryEvent):
                renderer.on_retry(
                    attempt=event.attempt,
                    max_attempts=event.max_attempts,
                    reason=event.message,
                    delay_seconds=event.delay,
                )

            # ==================== System Messages ====================
            elif isinstance(event, SystemMessageEvent):
                # Check for session termination signal
                if event.message == "[SESSION_TERMINATED]":
                    print("[headless] Session terminated", file=sys.stderr)
                    should_exit = True
                    break
                renderer.on_system_message(event.message, event.style or "system_info")

            # ==================== Turn Completion ====================
            elif isinstance(event, TurnCompletedEvent):
                turn_count += 1
                turn_completed = True
                logger.debug(f"TurnCompletedEvent received: turn={turn_count}")
                # Budget is printed on status change (idle/done), not here
                # TurnCompletedEvent depends on model token reporting which isn't reliable
                # Turn completion just means one request-response cycle finished

    # Send the prompt (message printed in event handler after session_id)
    await client.send_message(prompt)

    # Wait for events until turn completes
    try:
        await handle_events()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[headless] Error: {e}", file=sys.stderr)

    # Terminate the session — the "exit" command. This stops the agent
    # (if still running) and signals any other attached clients to exit
    # via the [SESSION_TERMINATED] broadcast.
    try:
        session_id = client.session_id
        if session_id:
            await client.delete_session(session_id)
        else:
            await client.end_session()
        print("[headless] Session ended", file=sys.stderr)
    except Exception as e:
        # Best-effort: connection may already be closing
        logger.debug(f"session.delete failed (non-fatal): {e}")

    # Cleanup - use close() for permanent shutdown (stops event stream)
    renderer.shutdown()
    await client.close()

    print(f"[headless] Output written to: {renderer.output_dir}", file=sys.stderr)
