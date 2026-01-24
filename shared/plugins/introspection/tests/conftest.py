"""Pytest fixtures for introspection plugin tests."""

import pytest
from shared.plugins.introspection import plugin as introspection_plugin


@pytest.fixture(autouse=True)
def cleanup_thread_local():
    """Clean up thread-local storage before and after each test.

    The introspection plugin uses thread-local storage for session references
    to prevent subagents from overwriting parent's session. This can cause
    test pollution if sessions persist between tests.
    """
    # Clear before test
    if hasattr(introspection_plugin._thread_local, 'session'):
        introspection_plugin._thread_local.session = None
    yield
    # Clear after test
    if hasattr(introspection_plugin._thread_local, 'session'):
        introspection_plugin._thread_local.session = None
