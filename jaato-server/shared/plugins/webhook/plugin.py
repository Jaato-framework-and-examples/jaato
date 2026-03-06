"""Webhook ingress plugin for daemon agent sessions.

Starts a per-instance HTTP server that receives webhooks and publishes
them to the TaskEventBus. Provides tools for the model to subscribe to
and poll for events.

Lifecycle:
    1. ``initialize(config)`` — load and merge config, server not started yet.
    2. ``set_workspace_path(path)`` — re-resolve workspace-relative config.
    3. First ``webhook_subscribe`` call — HTTP server starts lazily.
    4. ``shutdown()`` — stop HTTP server, clean up subscriptions.
"""

import logging
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional

from jaato_sdk.plugins.base import UserCommand
from jaato_sdk.plugins.model_provider.types import ToolSchema

from .config import WebhookConfig, load_config
from .http_server import WebhookHTTPServer

logger = logging.getLogger(__name__)

# Maximum events buffered per subscription before FIFO eviction
_MAX_BUFFER_SIZE = 1000


class WebhookPlugin:
    """Webhook ingress plugin for daemon agent sessions.

    Provides three tools:
    - ``webhook_subscribe`` — subscribe to incoming webhook events
    - ``webhook_poll`` — long-poll for events on a subscription
    - ``webhook_status`` — check listener status and event stats

    The HTTP server starts lazily on first subscribe call to avoid
    binding ports for sessions that don't use webhooks.

    State:
        _config: Merged WebhookConfig from all precedence layers.
        _explicit_config: Config dict passed via initialize() (profile override).
        _workspace_path: Workspace root for config file resolution.
        _http_server: The WebhookHTTPServer instance (None until first subscribe).
        _subscriptions: Per-subscription event buffers keyed by subscription ID.
        _subscription_events: Threading events for long-poll wakeup.
        _total_events_published: Running counter of all events published.
    """

    PLUGIN_KIND = "tool"

    def __init__(self):
        self._config: Optional[WebhookConfig] = None
        self._explicit_config: Optional[Dict[str, Any]] = None
        self._workspace_path: Optional[str] = None
        self._http_server: Optional[WebhookHTTPServer] = None
        self._initialized = False

        # Per-subscription event buffers: sub_id → deque of event dicts
        self._subscriptions: Dict[str, Deque[Dict[str, Any]]] = {}
        # Per-subscription source filters: sub_id → set of source names (empty = all)
        self._subscription_filters: Dict[str, set] = {}
        # Per-subscription threading events for long-poll wakeup
        self._subscription_events: Dict[str, threading.Event] = {}
        # Lock for subscription state
        self._sub_lock = threading.Lock()
        # Total events published counter
        self._total_events_published = 0

    @property
    def name(self) -> str:
        return "webhook"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Load config with standard precedence.

        The explicit config (from profile ``plugin_configs.webhook``) is stored
        and merged with workspace/user configs during ``_reload_config()``.

        Args:
            config: Optional dict from profile plugin_configs.webhook.
        """
        self._explicit_config = config
        self._reload_config()
        self._initialized = True

    def set_workspace_path(self, path: str) -> None:
        """Re-resolve config with workspace path for .jaato/webhook.json lookup.

        Called by PluginRegistry after ``initialize()`` and when workspace changes.

        Args:
            path: Absolute path to workspace root.
        """
        self._workspace_path = path
        self._reload_config()

    def _reload_config(self) -> None:
        """Reload config from all precedence layers."""
        self._config = load_config(
            explicit_config=self._explicit_config,
            workspace_path=self._workspace_path,
        )

    def shutdown(self) -> None:
        """Stop HTTP server and clean up all subscriptions."""
        if self._http_server:
            self._http_server.stop()
            self._http_server = None

        with self._sub_lock:
            # Wake up any blocked pollers
            for event in self._subscription_events.values():
                event.set()
            self._subscriptions.clear()
            self._subscription_filters.clear()
            self._subscription_events.clear()

        self._initialized = False
        logger.info("Webhook plugin shut down")

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for webhook subscribe, poll, and status."""
        return [
            ToolSchema(
                name='webhook_subscribe',
                description=(
                    'Subscribe to incoming webhook events. Returns a '
                    'subscription ID for polling. Call this once at session start.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Filter by source names (route keys from config). "
                                "Empty = all sources."
                            ),
                        },
                    },
                    "required": [],
                },
                category="integration",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='webhook_poll',
                description=(
                    'Fallback: poll for webhook events directly. Prefer '
                    'pollForTasks(event_types=["external_event"]) for event '
                    'delivery via the shared bus. Blocks up to timeout seconds.'
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "subscription_id": {
                            "type": "string",
                            "description": "Subscription ID from webhook_subscribe",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Max seconds to wait (1-30, default 15)",
                        },
                    },
                    "required": ["subscription_id"],
                },
                category="integration",
                discoverability="discoverable",
            ),
            ToolSchema(
                name='webhook_status',
                description=(
                    'Show webhook listener status, active routes, and event statistics.'
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                category="integration",
                discoverability="discoverable",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return tool executor mapping."""
        return {
            'webhook_subscribe': self._execute_subscribe,
            'webhook_poll': self._execute_poll,
            'webhook_status': self._execute_status,
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions describing webhook tools."""
        return (
            "You have access to webhook tools for receiving external HTTP events.\n\n"
            "Webhook events are published to the shared TaskEventBus as "
            "`external_event` events with `source_agent=\"webhook:<route>\"`.\n\n"
            "To start processing webhooks:\n"
            "1. Call `webhook_subscribe` to start the HTTP listener\n"
            "2. Use `pollForTasks(event_types=[\"external_event\"])` to receive "
            "events via the shared event bus\n"
            "3. Process each event and poll again\n\n"
            "`webhook_poll` is available as a fallback for direct polling.\n"
            "Use `webhook_status` to check listener health and statistics."
        )

    def get_auto_approved_tools(self) -> List[str]:
        """Status is read-only and safe to auto-approve."""
        return ['webhook_status']

    def get_user_commands(self) -> List[UserCommand]:
        """No user commands — model tools only."""
        return []

    # ------------------------------------------------------------------
    # Tool executors
    # ------------------------------------------------------------------

    def _execute_subscribe(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute webhook_subscribe — create subscription and start server.

        Starts the HTTP server lazily on first call. Returns a subscription ID
        and the endpoints the external service should POST to.

        Args:
            args: Tool args with optional 'sources' list.

        Returns:
            Dict with subscription_id, message, and endpoints.
        """
        if not self._config:
            return {"error": "Webhook plugin not configured"}

        # Start HTTP server lazily
        if not self._http_server or not self._http_server.is_running:
            try:
                self._http_server = WebhookHTTPServer(
                    self._config, self._on_webhook_received
                )
                self._http_server.start()
            except OSError as e:
                return {
                    "error": f"Failed to start HTTP server on "
                    f"{self._config.host}:{self._config.port}: {e}",
                    "hint": "The port may already be in use. Check webhook config.",
                }

        sources = set(args.get("sources", []))
        sub_id = str(uuid.uuid4())

        with self._sub_lock:
            self._subscriptions[sub_id] = deque(maxlen=_MAX_BUFFER_SIZE)
            self._subscription_filters[sub_id] = sources
            self._subscription_events[sub_id] = threading.Event()

        # Build endpoint list
        endpoints = []
        for name, route in self._config.routes.items():
            if not sources or name in sources:
                url = f"http://{self._config.host}:{self._config.port}{route.path}"
                endpoints.append({"source": name, "url": url})

        source_desc = list(sources) if sources else "all"
        return {
            "subscription_id": sub_id,
            "message": f"Subscribed to webhook events from sources: {source_desc}",
            "endpoints": endpoints,
        }

    def _execute_poll(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute webhook_poll — drain buffered events or block waiting.

        If events are already buffered, returns them immediately. Otherwise
        blocks up to ``timeout`` seconds for new events to arrive.

        Args:
            args: Tool args with 'subscription_id' and optional 'timeout'.

        Returns:
            Dict with 'events' list and 'cursor' (last event_id).
        """
        sub_id = args.get("subscription_id", "")
        timeout = min(max(args.get("timeout", 15), 1), 30)

        with self._sub_lock:
            if sub_id not in self._subscriptions:
                return {
                    "error": "Subscription not found. Call webhook_subscribe first.",
                }
            buffer = self._subscriptions[sub_id]
            wake_event = self._subscription_events[sub_id]

        # Fast path: events already buffered
        events = self._drain_buffer(buffer)
        if events:
            return self._poll_response(events)

        # Slow path: block until event arrives or timeout
        wake_event.clear()
        wake_event.wait(timeout=timeout)

        events = self._drain_buffer(buffer)
        return self._poll_response(events)

    def _execute_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute webhook_status — return server and subscription stats.

        Args:
            args: Tool args (unused).

        Returns:
            Dict with listening status, routes, subscriptions, and totals.
        """
        if self._http_server:
            stats = self._http_server.get_stats()
        else:
            stats = {
                "listening": False,
                "host": self._config.host if self._config else "127.0.0.1",
                "port": self._config.port if self._config else 9100,
                "routes": [],
                "total_events_received": 0,
            }

        with self._sub_lock:
            stats["active_subscriptions"] = len(self._subscriptions)

        stats["total_events_published"] = self._total_events_published
        return stats

    # ------------------------------------------------------------------
    # Internal event routing
    # ------------------------------------------------------------------

    def _on_webhook_received(
        self,
        route_name: str,
        event_type: str,
        headers: Dict[str, str],
        payload: Any,
    ) -> None:
        """Called by HTTP server when a valid webhook POST is received.

        Builds an event dict and appends it to all matching subscription
        buffers. Wakes up any blocked pollers.

        Thread safety: called from the HTTP server thread. The subscription
        lock protects buffer access.

        Args:
            route_name: Name of the matched route (e.g., 'github').
            event_type: Event type from header (e.g., 'push').
            headers: Request headers dict.
            payload: Parsed JSON payload.
        """
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat() + "Z"

        event = {
            "event_id": event_id,
            "source": route_name,
            "event_type": event_type,
            "timestamp": timestamp,
            "headers": headers,
            "payload": payload,
        }

        self._total_events_published += 1

        # Publish to shared TaskEventBus (primary delivery mechanism)
        self._publish_to_event_bus(
            event_id, route_name, event_type, timestamp, headers, payload
        )

        # Also buffer for direct webhook_poll (fallback mechanism)
        with self._sub_lock:
            for sub_id, buffer in self._subscriptions.items():
                sources = self._subscription_filters.get(sub_id, set())
                if not sources or route_name in sources:
                    buffer.append(event)
                    # Wake up blocked pollers
                    wake_event = self._subscription_events.get(sub_id)
                    if wake_event:
                        wake_event.set()

    def _publish_to_event_bus(
        self,
        event_id: str,
        route_name: str,
        event_type: str,
        timestamp: str,
        headers: Dict[str, str],
        payload: Any,
    ) -> None:
        """Publish a webhook event to the shared TaskEventBus.

        This is the primary delivery mechanism. Events appear as
        ``EXTERNAL_EVENT`` with ``source_agent="webhook:<route>"``,
        retrievable via ``pollForTasks(event_types=["external_event"])``.

        Args:
            event_id: Unique event identifier.
            route_name: Name of the matched route (e.g., 'github').
            event_type: Event type from header (e.g., 'push').
            timestamp: ISO 8601 timestamp string.
            headers: Request headers dict.
            payload: Parsed JSON payload.
        """
        try:
            from shared.plugins.todo.event_bus import get_event_bus
            from jaato_sdk.plugins.todo.models import TaskEvent, TaskEventType
        except Exception:
            logger.debug("TaskEventBus not available, skipping bus publish")
            return

        bus = get_event_bus()
        task_event = TaskEvent(
            event_id=event_id,
            event_type=TaskEventType.EXTERNAL_EVENT,
            timestamp=timestamp,
            source_agent=f"webhook:{route_name}",
            source_plan_id=route_name,
            source_plan_title=f"webhook:{route_name}",
            source_step_id=event_id,
            source_step_description=event_type,
            payload={
                "source": route_name,
                "event_type": event_type,
                "headers": headers,
                "payload": payload,
            },
        )
        bus.publish(task_event)

    @staticmethod
    def _drain_buffer(buffer: Deque[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drain all events from a buffer.

        Args:
            buffer: The subscription's event deque.

        Returns:
            List of event dicts (may be empty).
        """
        events = []
        while buffer:
            try:
                events.append(buffer.popleft())
            except IndexError:
                break
        return events

    @staticmethod
    def _poll_response(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a poll response dict.

        Args:
            events: List of event dicts from the buffer.

        Returns:
            Dict with 'events' and 'cursor'.
        """
        cursor = events[-1]["event_id"] if events else None
        return {
            "events": events,
            "cursor": cursor,
        }


def create_plugin() -> WebhookPlugin:
    """Factory function to create the webhook plugin instance."""
    return WebhookPlugin()
