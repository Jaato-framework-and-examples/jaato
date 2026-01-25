"""Pytest fixtures for environment plugin tests."""

import pytest
from shared.plugins.environment import plugin as env_plugin


@pytest.fixture(autouse=True)
def cleanup_thread_local():
    """Clean up thread-local storage before and after each test.

    The environment plugin uses thread-local storage for session references
    to prevent subagents from overwriting parent's session. This can cause
    test pollution if sessions persist between tests.
    """
    # Clear before test
    if hasattr(env_plugin._thread_local, 'session'):
        env_plugin._thread_local.session = None
    yield
    # Clear after test
    if hasattr(env_plugin._thread_local, 'session'):
        env_plugin._thread_local.session = None
