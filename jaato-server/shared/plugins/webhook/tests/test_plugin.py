"""Tests for the WebhookPlugin tool plugin."""

import json
import socket
import pytest
import time
import urllib.request

from shared.plugins.webhook.plugin import WebhookPlugin, create_plugin


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _post(port, path, payload, headers=None):
    """Send a POST to localhost."""
    url = f"http://127.0.0.1:{port}{path}"
    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestWebhookPluginBasics:
    """Tests for plugin protocol compliance."""

    def test_create_plugin(self):
        plugin = create_plugin()
        assert isinstance(plugin, WebhookPlugin)

    def test_name(self):
        plugin = create_plugin()
        assert plugin.name == "webhook"

    def test_tool_schemas(self):
        plugin = create_plugin()
        plugin.initialize()
        schemas = plugin.get_tool_schemas()
        names = {s.name for s in schemas}
        assert names == {"webhook_subscribe", "webhook_poll", "webhook_status"}

    def test_executors_match_schemas(self):
        plugin = create_plugin()
        plugin.initialize()
        schemas = plugin.get_tool_schemas()
        executors = plugin.get_executors()
        for schema in schemas:
            assert schema.name in executors

    def test_discoverable_tools(self):
        plugin = create_plugin()
        plugin.initialize()
        for schema in plugin.get_tool_schemas():
            assert schema.discoverability == "discoverable"

    def test_auto_approved_tools(self):
        plugin = create_plugin()
        assert "webhook_status" in plugin.get_auto_approved_tools()

    def test_no_user_commands(self):
        plugin = create_plugin()
        assert plugin.get_user_commands() == []

    def test_system_instructions(self):
        plugin = create_plugin()
        instructions = plugin.get_system_instructions()
        assert instructions is not None
        assert "webhook_subscribe" in instructions

    def test_initialize_and_shutdown(self):
        plugin = create_plugin()
        plugin.initialize({"port": 9999})
        assert plugin._initialized
        plugin.shutdown()
        assert not plugin._initialized


class TestWebhookSubscribe:
    """Tests for the webhook_subscribe tool."""

    def test_subscribe_returns_id(self):
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({"port": port, "routes": {"generic": {"path": "/webhook", "allow_unauthenticated": True}}})
        try:
            result = plugin._execute_subscribe({})
            assert "subscription_id" in result
            assert "endpoints" in result
            assert len(result["endpoints"]) > 0
        finally:
            plugin.shutdown()

    def test_subscribe_starts_server(self):
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({"port": port, "routes": {"generic": {"path": "/webhook", "allow_unauthenticated": True}}})
        try:
            assert plugin._http_server is None
            plugin._execute_subscribe({})
            assert plugin._http_server is not None
            assert plugin._http_server.is_running
        finally:
            plugin.shutdown()

    def test_subscribe_with_source_filter(self):
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({
            "port": port,
            "routes": {
                "github": {"path": "/webhook/github", "allow_unauthenticated": True},
                "slack": {"path": "/webhook/slack", "allow_unauthenticated": True},
            },
        })
        try:
            result = plugin._execute_subscribe({"sources": ["github"]})
            endpoints = result["endpoints"]
            assert len(endpoints) == 1
            assert endpoints[0]["source"] == "github"
        finally:
            plugin.shutdown()

    def test_subscribe_without_config(self):
        plugin = create_plugin()
        # Don't initialize
        plugin._config = None
        result = plugin._execute_subscribe({})
        assert "error" in result


class TestWebhookPoll:
    """Tests for the webhook_poll tool."""

    def test_poll_unknown_subscription(self):
        plugin = create_plugin()
        plugin.initialize()
        result = plugin._execute_poll({"subscription_id": "nonexistent"})
        assert "error" in result

    def test_poll_returns_empty_on_timeout(self):
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({"port": port, "routes": {"generic": {"path": "/webhook", "allow_unauthenticated": True}}})
        try:
            sub = plugin._execute_subscribe({})
            sub_id = sub["subscription_id"]
            result = plugin._execute_poll({
                "subscription_id": sub_id,
                "timeout": 0.1,
            })
            assert result["events"] == []
            assert result["cursor"] is None
        finally:
            plugin.shutdown()

    def test_poll_receives_posted_event(self):
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({"port": port, "routes": {"generic": {"path": "/webhook", "allow_unauthenticated": True}}})
        try:
            sub = plugin._execute_subscribe({})
            sub_id = sub["subscription_id"]

            # POST a webhook
            _post(port, "/webhook", {"action": "test"})
            # Small delay for server thread
            time.sleep(0.1)

            result = plugin._execute_poll({
                "subscription_id": sub_id,
                "timeout": 1,
            })
            assert len(result["events"]) == 1
            assert result["events"][0]["source"] == "generic"
            assert result["events"][0]["payload"] == {"action": "test"}
            assert result["cursor"] is not None
        finally:
            plugin.shutdown()

    def test_poll_drains_buffer(self):
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({"port": port, "routes": {"generic": {"path": "/webhook", "allow_unauthenticated": True}}})
        try:
            sub = plugin._execute_subscribe({})
            sub_id = sub["subscription_id"]

            # POST multiple webhooks
            _post(port, "/webhook", {"n": 1})
            _post(port, "/webhook", {"n": 2})
            _post(port, "/webhook", {"n": 3})
            time.sleep(0.1)

            result = plugin._execute_poll({
                "subscription_id": sub_id,
                "timeout": 1,
            })
            assert len(result["events"]) == 3

            # Second poll should be empty
            result2 = plugin._execute_poll({
                "subscription_id": sub_id,
                "timeout": 0.1,
            })
            assert len(result2["events"]) == 0
        finally:
            plugin.shutdown()

    def test_source_filtering(self):
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({
            "port": port,
            "routes": {
                "github": {"path": "/webhook/github", "allow_unauthenticated": True},
                "slack": {"path": "/webhook/slack", "allow_unauthenticated": True},
            },
        })
        try:
            # Subscribe to github only
            sub = plugin._execute_subscribe({"sources": ["github"]})
            sub_id = sub["subscription_id"]

            _post(port, "/webhook/github", {"from": "github"})
            _post(port, "/webhook/slack", {"from": "slack"})
            time.sleep(0.1)

            result = plugin._execute_poll({
                "subscription_id": sub_id,
                "timeout": 1,
            })
            assert len(result["events"]) == 1
            assert result["events"][0]["source"] == "github"
        finally:
            plugin.shutdown()


class TestWebhookStatus:
    """Tests for the webhook_status tool."""

    def test_status_before_subscribe(self):
        plugin = create_plugin()
        plugin.initialize()
        result = plugin._execute_status({})
        assert result["listening"] is False
        assert result["active_subscriptions"] == 0

    def test_status_after_subscribe(self):
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({"port": port, "routes": {"generic": {"path": "/webhook", "allow_unauthenticated": True}}})
        try:
            plugin._execute_subscribe({})
            result = plugin._execute_status({})
            assert result["listening"] is True
            assert result["active_subscriptions"] == 1
            assert result["port"] == port
        finally:
            plugin.shutdown()

    def test_status_counts_events(self):
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({"port": port, "routes": {"generic": {"path": "/webhook", "allow_unauthenticated": True}}})
        try:
            plugin._execute_subscribe({})
            _post(port, "/webhook", {"a": 1})
            _post(port, "/webhook", {"b": 2})
            time.sleep(0.1)

            result = plugin._execute_status({})
            assert result["total_events_published"] == 2
        finally:
            plugin.shutdown()


class TestWebhookPluginLifecycle:
    """Tests for plugin lifecycle (shutdown, re-init, etc.)."""

    def test_shutdown_stops_server(self):
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({"port": port, "routes": {"generic": {"path": "/webhook", "allow_unauthenticated": True}}})
        plugin._execute_subscribe({})
        assert plugin._http_server.is_running
        plugin.shutdown()
        assert plugin._http_server is None

    def test_shutdown_clears_subscriptions(self):
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({"port": port, "routes": {"generic": {"path": "/webhook", "allow_unauthenticated": True}}})
        plugin._execute_subscribe({})
        plugin._execute_subscribe({})
        assert len(plugin._subscriptions) == 2
        plugin.shutdown()
        assert len(plugin._subscriptions) == 0

    def test_workspace_path_reloads_config(self):
        plugin = create_plugin()
        plugin.initialize({"port": 9999})
        assert plugin._config.port == 9999
        # set_workspace_path triggers reload but explicit config still wins
        plugin.set_workspace_path("/tmp/test_workspace")
        assert plugin._config.port == 9999


class TestEventBusBridge:
    """Tests for webhook → TaskEventBus bridge."""

    def _make_bus_and_session(self):
        """Create a fresh EventBus, TaskEventBus, and mock session."""
        from shared.event_bus import EventBus
        from shared.plugins.todo.event_bus import TaskEventBus
        from unittest.mock import Mock
        event_bus = EventBus()
        task_bus = TaskEventBus(event_bus)
        session = Mock()
        session._runtime = Mock()
        session._runtime.event_bus = event_bus
        return task_bus, session

    def test_webhook_publishes_to_bus(self):
        from jaato_sdk.plugins.todo.models import TaskEventType
        bus, session = self._make_bus_and_session()
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({"port": port, "routes": {"generic": {"path": "/webhook", "allow_unauthenticated": True}}})
        plugin.set_session(session)
        try:
            plugin._execute_subscribe({})
            _post(port, "/webhook", {"action": "test"})
            time.sleep(0.1)

            events = bus.get_recent_events(
                event_types=[TaskEventType.EXTERNAL_EVENT]
            )
            assert len(events) == 1
            assert events[0].event_type == TaskEventType.EXTERNAL_EVENT
            assert events[0].payload["payload"] == {"action": "test"}
        finally:
            plugin.shutdown()

    def test_bus_event_carries_route_and_type(self):
        from jaato_sdk.plugins.todo.models import TaskEventType
        bus, session = self._make_bus_and_session()
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({
            "port": port,
            "routes": {
                "github": {
                    "path": "/webhook/github",
                    "event_type_header": "X-GitHub-Event",
                    "allow_unauthenticated": True,
                },
            },
        })
        plugin.set_session(session)
        try:
            plugin._execute_subscribe({})
            _post(port, "/webhook/github", {"ref": "main"},
                  headers={"X-GitHub-Event": "push"})
            time.sleep(0.1)

            events = bus.get_recent_events(
                event_types=[TaskEventType.EXTERNAL_EVENT]
            )
            assert len(events) == 1
            assert events[0].source_agent == "webhook:github"
            assert events[0].payload["source"] == "github"
            assert events[0].payload["event_type"] == "push"
        finally:
            plugin.shutdown()

    def test_bus_events_retrievable_via_wait(self):
        from jaato_sdk.plugins.todo.models import TaskEventType
        bus, session = self._make_bus_and_session()
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({"port": port, "routes": {"generic": {"path": "/webhook", "allow_unauthenticated": True}}})
        plugin.set_session(session)
        try:
            plugin._execute_subscribe({})
            _post(port, "/webhook", {"n": 1})
            time.sleep(0.1)

            events = bus.wait_for_events(
                timeout=1.0,
                event_types=[TaskEventType.EXTERNAL_EVENT],
            )
            assert len(events) >= 1
        finally:
            plugin.shutdown()

    def test_bus_events_filterable_by_agent(self):
        from jaato_sdk.plugins.todo.models import TaskEventType
        bus, session = self._make_bus_and_session()
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({
            "port": port,
            "routes": {
                "github": {"path": "/webhook/github", "allow_unauthenticated": True},
                "slack": {"path": "/webhook/slack", "allow_unauthenticated": True},
            },
        })
        plugin.set_session(session)
        try:
            plugin._execute_subscribe({})
            _post(port, "/webhook/github", {"from": "gh"})
            _post(port, "/webhook/slack", {"from": "sl"})
            time.sleep(0.1)

            gh_events = bus.get_recent_events(agent_id="webhook:github")
            sl_events = bus.get_recent_events(agent_id="webhook:slack")
            assert len(gh_events) == 1
            assert len(sl_events) == 1
            assert gh_events[0].payload["source"] == "github"
            assert sl_events[0].payload["source"] == "slack"
        finally:
            plugin.shutdown()

    def test_multiple_webhooks_publish_multiple_events(self):
        from jaato_sdk.plugins.todo.models import TaskEventType
        bus, session = self._make_bus_and_session()
        port = _find_free_port()
        plugin = create_plugin()
        plugin.initialize({"port": port, "routes": {"generic": {"path": "/webhook", "allow_unauthenticated": True}}})
        plugin.set_session(session)
        try:
            plugin._execute_subscribe({})
            _post(port, "/webhook", {"n": 1})
            _post(port, "/webhook", {"n": 2})
            _post(port, "/webhook", {"n": 3})
            time.sleep(0.1)

            events = bus.get_recent_events(
                event_types=[TaskEventType.EXTERNAL_EVENT]
            )
            assert len(events) == 3
        finally:
            plugin.shutdown()
