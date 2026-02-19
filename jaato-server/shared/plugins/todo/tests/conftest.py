"""Pytest fixtures for TODO plugin tests."""

import pytest
from shared.plugins.todo import plugin as todo_plugin


@pytest.fixture(autouse=True)
def cleanup_thread_local():
    """Clean up thread-local storage before and after each test.

    The TODO plugin uses thread-local storage for session and agent_name
    references to prevent subagents from overwriting parent's context.
    This can cause test pollution if sessions persist between tests.
    """
    # Clear before test
    if hasattr(todo_plugin._thread_local, 'session'):
        todo_plugin._thread_local.session = None
    if hasattr(todo_plugin._thread_local, 'agent_name'):
        todo_plugin._thread_local.agent_name = None
    yield
    # Clear after test
    if hasattr(todo_plugin._thread_local, 'session'):
        todo_plugin._thread_local.session = None
    if hasattr(todo_plugin._thread_local, 'agent_name'):
        todo_plugin._thread_local.agent_name = None
