"""Tests for EventBusTools (shared event bus subscription tools).

These tests were moved from the todo plugin tests when the subscription
tools (subscribeToEvents, getEvents, listSubscriptions, unsubscribe)
were extracted from the todo plugin into session-level core infrastructure.
"""

import threading
import time
from unittest.mock import Mock

from shared.event_bus import EventBus
from shared.event_bus_tools import EventBusTools
from jaato_sdk.event_bus import EventType
from jaato_sdk.plugins.todo.models import (
    TaskEvent, TaskEventType, TodoPlan,
)


def _make_tools(bus: EventBus) -> EventBusTools:
    """Create EventBusTools with a mock session wired to the given bus."""
    runtime = Mock()
    runtime.event_bus = bus
    session = Mock()
    session.agent_id = "test-agent"
    session._runtime = runtime
    tools = EventBusTools(session)
    return tools


def _publish_event(bus: EventBus, agent_id: str = "sub") -> TaskEvent:
    """Publish a dummy event and return it."""
    plan = TodoPlan.create("P", ["S"])
    event = TaskEvent.create(TaskEventType.PLAN_CREATED, agent_id, plan)
    bus.publish(event)
    return event


class TestGetEventsExecutor:
    """Tests for getEvents executor with long-poll support."""

    def setup_method(self):
        self.bus = EventBus()

    def test_basic_get_events(self):
        """getEvents returns events without timeout."""
        tools = _make_tools(self.bus)
        event = _publish_event(self.bus)

        result = tools._execute_get_events({})

        assert result["count"] == 1
        assert result["events"][0]["event_id"] == event.event_id

    def test_after_event_filters_old_events(self):
        """after_event parameter excludes events up to and including the cursor."""
        tools = _make_tools(self.bus)

        event1 = _publish_event(self.bus, "a")
        event2 = _publish_event(self.bus, "b")

        result = tools._execute_get_events({"after_event": event1.event_id})

        assert result["count"] == 1
        assert result["events"][0]["event_id"] == event2.event_id

    def test_last_event_id_in_result(self):
        """Result includes last_event_id for cursor-based consumption."""
        tools = _make_tools(self.bus)

        event1 = _publish_event(self.bus)
        event2 = _publish_event(self.bus)

        result = tools._execute_get_events({})

        assert result["last_event_id"] == event2.event_id

    def test_no_last_event_id_when_empty(self):
        """Result has no last_event_id key when there are no events."""
        tools = _make_tools(self.bus)

        result = tools._execute_get_events({})

        assert result["count"] == 0
        assert "last_event_id" not in result

    def test_timeout_returns_immediately_with_events(self):
        """timeout does not delay when events already exist."""
        tools = _make_tools(self.bus)
        _publish_event(self.bus, "sub")

        start = time.monotonic()
        result = tools._execute_get_events({
            "timeout": 5,
            "agent_id": "sub",
        })
        elapsed = time.monotonic() - start

        assert result["count"] == 1
        assert elapsed < 1.0

    def test_timeout_blocks_until_event(self):
        """timeout blocks and returns when a new event is published."""
        tools = _make_tools(self.bus)

        result_holder: list = []

        def call_tool():
            r = tools._execute_get_events({
                "timeout": 10,
                "agent_id": "sub",
            })
            result_holder.append(r)

        t = threading.Thread(target=call_tool)
        t.start()

        # Give the tool time to enter the wait.
        time.sleep(0.3)

        event = _publish_event(self.bus, "sub")
        t.join(timeout=5)

        assert len(result_holder) == 1
        assert result_holder[0]["count"] == 1
        assert result_holder[0]["events"][0]["event_id"] == event.event_id

    def test_timeout_returns_empty(self):
        """timeout returns empty after timeout with no events."""
        tools = _make_tools(self.bus)

        start = time.monotonic()
        result = tools._execute_get_events({
            "timeout": 0.3,
            "agent_id": "nobody",
        })
        elapsed = time.monotonic() - start

        assert result["count"] == 0
        assert elapsed >= 0.25

    def test_timeout_with_after_event(self):
        """timeout combined with after_event for incremental consumption."""
        tools = _make_tools(self.bus)

        event1 = _publish_event(self.bus)

        result_holder: list = []

        def call_tool():
            r = tools._execute_get_events({
                "timeout": 10,
                "after_event": event1.event_id,
            })
            result_holder.append(r)

        t = threading.Thread(target=call_tool)
        t.start()
        time.sleep(0.3)

        event2 = _publish_event(self.bus)
        t.join(timeout=5)

        assert len(result_holder) == 1
        assert result_holder[0]["count"] == 1
        assert result_holder[0]["events"][0]["event_id"] == event2.event_id

    def test_timeout_clamped(self):
        """timeout is clamped to [0, 30]."""
        tools = _make_tools(self.bus)

        # Negative → treated as 0 (immediate return)
        start = time.monotonic()
        result = tools._execute_get_events({"timeout": -5})
        elapsed = time.monotonic() - start
        assert elapsed < 0.5

        # Invalid type → treated as 0
        result = tools._execute_get_events({"timeout": "not_a_number"})
        assert result["count"] == 0

    def test_timeout_requires_narrowing(self):
        """timeout without any filters or cursor returns an error."""
        tools = _make_tools(self.bus)

        result = tools._execute_get_events({"timeout": 5})
        assert "error" in result
        assert "timeout requires" in result["error"]

    def test_timeout_accepted_with_agent_id(self):
        """timeout is accepted when agent_id narrows the query."""
        tools = _make_tools(self.bus)

        # Should not error — agent_id provides narrowing.
        start = time.monotonic()
        result = tools._execute_get_events({
            "timeout": 0.2,
            "agent_id": "sub",
        })
        elapsed = time.monotonic() - start

        assert "error" not in result
        # Should have waited (no events from "sub").
        assert elapsed >= 0.15

    def test_timeout_accepted_with_event_types(self):
        """timeout is accepted when event_types narrows the query."""
        tools = _make_tools(self.bus)

        start = time.monotonic()
        result = tools._execute_get_events({
            "timeout": 0.2,
            "event_types": ["step_completed"],
        })
        elapsed = time.monotonic() - start

        assert "error" not in result
        assert elapsed >= 0.15

    def test_timeout_accepted_with_after_event(self):
        """timeout is accepted when after_event provides a cursor."""
        tools = _make_tools(self.bus)
        event = _publish_event(self.bus)

        start = time.monotonic()
        result = tools._execute_get_events({
            "timeout": 0.2,
            "after_event": event.event_id,
        })
        elapsed = time.monotonic() - start

        assert "error" not in result
        assert elapsed >= 0.15


class TestSubscribeExecutor:
    """Tests for subscribeToEvents executor."""

    def setup_method(self):
        self.bus = EventBus()

    def test_subscribe_returns_subscription_id(self):
        tools = _make_tools(self.bus)
        result = tools._execute_subscribe({
            "event_types": ["plan_created"],
        })
        assert "subscription_id" in result
        assert "error" not in result

    def test_subscribe_requires_event_types(self):
        tools = _make_tools(self.bus)
        result = tools._execute_subscribe({})
        assert "error" in result

    def test_subscribe_invalid_event_type(self):
        tools = _make_tools(self.bus)
        result = tools._execute_subscribe({
            "event_types": ["nonexistent_type"],
        })
        assert "error" in result


class TestListSubscriptionsExecutor:
    """Tests for listSubscriptions executor."""

    def setup_method(self):
        self.bus = EventBus()

    def test_list_empty(self):
        tools = _make_tools(self.bus)
        result = tools._execute_list_subscriptions({})
        assert result["count"] == 0

    def test_list_after_subscribe(self):
        tools = _make_tools(self.bus)
        tools._execute_subscribe({"event_types": ["plan_created"]})
        result = tools._execute_list_subscriptions({})
        assert result["count"] >= 1


class TestUnsubscribeExecutor:
    """Tests for unsubscribe executor."""

    def setup_method(self):
        self.bus = EventBus()

    def test_unsubscribe_success(self):
        tools = _make_tools(self.bus)
        sub = tools._execute_subscribe({"event_types": ["plan_created"]})
        result = tools._execute_unsubscribe({
            "subscription_id": sub["subscription_id"],
        })
        assert result["success"] is True

    def test_unsubscribe_not_found(self):
        tools = _make_tools(self.bus)
        result = tools._execute_unsubscribe({
            "subscription_id": "nonexistent",
        })
        assert result["success"] is False

    def test_unsubscribe_requires_id(self):
        tools = _make_tools(self.bus)
        result = tools._execute_unsubscribe({})
        assert "error" in result


class TestBackwardCompatAliases:
    """Tests that backward-compatible tool name aliases work."""

    def setup_method(self):
        self.bus = EventBus()

    def test_get_task_events_alias(self):
        """getTaskEvents alias invokes the same method as getEvents."""
        tools = _make_tools(self.bus)
        executors = tools.get_executors()
        assert "getTaskEvents" in executors
        # Both point to the same underlying method
        assert executors["getTaskEvents"].__func__ is executors["getEvents"].__func__

    def test_subscribe_to_tasks_alias(self):
        """subscribeToTasks alias invokes the same method as subscribeToEvents."""
        tools = _make_tools(self.bus)
        executors = tools.get_executors()
        assert "subscribeToTasks" in executors
        assert executors["subscribeToTasks"].__func__ is executors["subscribeToEvents"].__func__


class TestToolSchemas:
    """Tests for EventBusTools tool schemas."""

    def test_schema_names(self):
        tools = _make_tools(EventBus())
        schemas = tools.get_tool_schemas()
        names = {s.name for s in schemas}
        assert names == {"subscribeToEvents", "getEvents", "listSubscriptions", "unsubscribe"}

    def test_auto_approved(self):
        tools = _make_tools(EventBus())
        auto = tools.get_auto_approved_tools()
        assert "listSubscriptions" in auto


class TestShutdown:
    """Tests for EventBusTools.shutdown()."""

    def setup_method(self):
        self.bus = EventBus()

    def test_shutdown_cleans_subscriptions(self):
        tools = _make_tools(self.bus)
        tools._execute_subscribe({"event_types": ["plan_created"]})
        assert len(tools._subscriptions) == 1
        tools.shutdown()
        assert len(tools._subscriptions) == 0
