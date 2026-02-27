"""Tests for per-agent provider trace routing.

Verifies that provider_trace() writes to agent-specific files when a
trace agent context is active, and falls back to the base file otherwise.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest

from jaato_sdk.trace import (
    _agent_trace_path,
    _trace_agent_id,
    clear_trace_agent_context,
    provider_trace,
    set_trace_agent_context,
)


class TestSetTraceAgentContext:
    """Tests for set_trace_agent_context / clear_trace_agent_context."""

    def test_default_is_none(self):
        """ContextVar defaults to None (main agent)."""
        clear_trace_agent_context()
        assert _trace_agent_id.get() is None

    def test_set_and_get(self):
        """Setting agent ID is readable from the ContextVar."""
        set_trace_agent_context("subagent_1")
        try:
            assert _trace_agent_id.get() == "subagent_1"
        finally:
            clear_trace_agent_context()

    def test_clear_resets_to_none(self):
        """Clearing restores the default None value."""
        set_trace_agent_context("subagent_2")
        clear_trace_agent_context()
        assert _trace_agent_id.get() is None


class TestAgentTracePath:
    """Tests for _agent_trace_path derivation."""

    def test_none_base_returns_none(self):
        """None base path is returned as-is."""
        clear_trace_agent_context()
        assert _agent_trace_path(None) is None

    def test_main_agent_returns_base(self):
        """Main agent (or no context) returns the base path unchanged."""
        clear_trace_agent_context()
        assert _agent_trace_path("/tmp/provider_trace.log") == "/tmp/provider_trace.log"

    def test_explicit_main_returns_base(self):
        """Explicitly setting 'main' returns the base path unchanged."""
        set_trace_agent_context("main")
        try:
            assert _agent_trace_path("/tmp/provider_trace.log") == "/tmp/provider_trace.log"
        finally:
            clear_trace_agent_context()

    def test_subagent_inserts_id(self):
        """Subagent context inserts agent ID before the extension."""
        set_trace_agent_context("subagent_1")
        try:
            result = _agent_trace_path("/tmp/provider_trace.log")
            assert result == "/tmp/provider_trace_subagent_1.log"
        finally:
            clear_trace_agent_context()

    def test_subagent_no_extension(self):
        """Paths without extension get the agent ID appended."""
        set_trace_agent_context("subagent_2")
        try:
            result = _agent_trace_path("/tmp/provider_trace")
            assert result == "/tmp/provider_trace_subagent_2"
        finally:
            clear_trace_agent_context()

    def test_subagent_nested_path(self):
        """Works with nested directory paths."""
        set_trace_agent_context("subagent_3")
        try:
            result = _agent_trace_path("/var/log/jaato/provider_trace.log")
            assert result == "/var/log/jaato/provider_trace_subagent_3.log"
        finally:
            clear_trace_agent_context()


class TestProviderTracePerAgent:
    """Tests that provider_trace() writes to the correct per-agent file."""

    def test_main_writes_to_base_file(self, tmp_path):
        """Main agent writes to the base provider_trace.log."""
        base = str(tmp_path / "provider_trace.log")
        clear_trace_agent_context()

        with patch.dict(os.environ, {"JAATO_PROVIDER_TRACE": base}):
            provider_trace("test", "main message")

        assert os.path.exists(base)
        content = open(base).read()
        assert "main message" in content

    def test_subagent_writes_to_own_file(self, tmp_path):
        """Subagent writes to provider_trace_subagent_1.log."""
        base = str(tmp_path / "provider_trace.log")
        expected = str(tmp_path / "provider_trace_subagent_1.log")

        set_trace_agent_context("subagent_1")
        try:
            with patch.dict(os.environ, {"JAATO_PROVIDER_TRACE": base}):
                provider_trace("test", "subagent message")
        finally:
            clear_trace_agent_context()

        # Subagent file should exist with the message
        assert os.path.exists(expected)
        content = open(expected).read()
        assert "subagent message" in content

        # Base file should NOT have been written
        assert not os.path.exists(base)

    def test_concurrent_agents_write_to_separate_files(self, tmp_path):
        """Multiple concurrent agents write to their own files without mixing."""
        base = str(tmp_path / "provider_trace.log")
        results = {}
        barrier = threading.Barrier(3)

        def agent_work(agent_id, message):
            set_trace_agent_context(agent_id)
            try:
                # Synchronize to maximize overlap
                barrier.wait(timeout=5)
                with patch.dict(os.environ, {"JAATO_PROVIDER_TRACE": base}):
                    for i in range(10):
                        provider_trace("test", f"{message}_{i}")
            finally:
                clear_trace_agent_context()

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(agent_work, "main", "main_msg"): "main",
                pool.submit(agent_work, "subagent_1", "sub1_msg"): "subagent_1",
                pool.submit(agent_work, "subagent_2", "sub2_msg"): "subagent_2",
            }
            for f in as_completed(futures):
                f.result()  # Raise any exceptions

        # Main writes to base file
        main_content = open(base).read()
        for i in range(10):
            assert f"main_msg_{i}" in main_content
        assert "sub1_msg" not in main_content
        assert "sub2_msg" not in main_content

        # Subagent 1 writes to its own file
        sub1_file = str(tmp_path / "provider_trace_subagent_1.log")
        sub1_content = open(sub1_file).read()
        for i in range(10):
            assert f"sub1_msg_{i}" in sub1_content
        assert "main_msg" not in sub1_content

        # Subagent 2 writes to its own file
        sub2_file = str(tmp_path / "provider_trace_subagent_2.log")
        sub2_content = open(sub2_file).read()
        for i in range(10):
            assert f"sub2_msg_{i}" in sub2_content
        assert "main_msg" not in sub2_content
