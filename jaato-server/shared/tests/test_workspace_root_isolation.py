"""Tests for the per-task workspace_root / config_root ContextVars
(server 0.6.68+).

Pre-0.6.68 the active workspace was carried in ``os.environ`` —
process-global state — and concurrent overlapping sessions clobbered
each other's values, making session A's profile discovery read from
session B's workspace.

0.6.68 introduces per-task ``ContextVar``s; these tests verify the
isolation property.
"""

import asyncio
import pytest

from shared.session_context import (
    get_config_root,
    get_workspace_root,
    reset_config_root,
    reset_workspace_root,
    set_config_root,
    set_workspace_root,
)


# =====================================================================
# Single-task get/set/reset
# =====================================================================


def test_get_workspace_root_default_falls_back_to_environ(monkeypatch):
    monkeypatch.setenv("JAATO_WORKSPACE_ROOT", "/from-env")
    assert get_workspace_root() == "/from-env"


def test_get_workspace_root_default_when_unset(monkeypatch):
    monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)
    assert get_workspace_root() is None
    assert get_workspace_root("/fallback") == "/fallback"


def test_set_workspace_root_overrides_environ(monkeypatch):
    monkeypatch.setenv("JAATO_WORKSPACE_ROOT", "/from-env")
    token = set_workspace_root("/from-ctx")
    try:
        assert get_workspace_root() == "/from-ctx"
    finally:
        reset_workspace_root(token)
    assert get_workspace_root() == "/from-env"


def test_set_config_root_overrides_environ(monkeypatch):
    monkeypatch.setenv("JAATO_CONFIG_ROOT", "/etc-from-env")
    token = set_config_root("/etc-from-ctx")
    try:
        assert get_config_root() == "/etc-from-ctx"
    finally:
        reset_config_root(token)
    assert get_config_root() == "/etc-from-env"


# =====================================================================
# Concurrent-task isolation — the actual race-condition fix
# =====================================================================


@pytest.mark.asyncio
async def test_concurrent_tasks_each_see_own_workspace_root():
    """The original bug: two overlapping sessions clobbered each other
    because ``os.environ`` is process-global.  ContextVars are per-task,
    so two concurrent tasks each see their own value regardless of
    the other's set/reset cycle."""

    observations: dict[str, list] = {"A": [], "B": []}

    async def session(label: str, root: str, hold_ms: int) -> None:
        token = set_workspace_root(root)
        try:
            observations[label].append(get_workspace_root())
            # Simulate work that overlaps with the other session.
            await asyncio.sleep(hold_ms / 1000)
            # After yielding to the event loop — would the other task's
            # set_workspace_root have leaked into our view?  Not in
            # ContextVar-land.
            observations[label].append(get_workspace_root())
        finally:
            reset_workspace_root(token)

    # Two interleaved sessions setting different roots.
    await asyncio.gather(
        session("A", "/ws-A", hold_ms=20),
        session("B", "/ws-B", hold_ms=10),
    )

    assert observations["A"] == ["/ws-A", "/ws-A"], (
        "session A should see /ws-A both before and after yielding, "
        "even though session B ran in between"
    )
    assert observations["B"] == ["/ws-B", "/ws-B"], (
        "session B should see its own /ws-B regardless of session A"
    )


@pytest.mark.asyncio
async def test_concurrent_tasks_each_see_own_config_root():
    """Same isolation guarantee for config_root."""
    seen_a: list = []
    seen_b: list = []

    async def task_a():
        token = set_config_root("/cfg-a")
        try:
            seen_a.append(get_config_root())
            await asyncio.sleep(0.02)
            seen_a.append(get_config_root())
        finally:
            reset_config_root(token)

    async def task_b():
        token = set_config_root("/cfg-b")
        try:
            seen_b.append(get_config_root())
            await asyncio.sleep(0.01)
            seen_b.append(get_config_root())
        finally:
            reset_config_root(token)

    await asyncio.gather(task_a(), task_b())
    assert seen_a == ["/cfg-a", "/cfg-a"]
    assert seen_b == ["/cfg-b", "/cfg-b"]


@pytest.mark.asyncio
async def test_outer_context_unaffected_by_inner_task(monkeypatch):
    """A child task that sets/resets workspace_root must not leak its
    value to the parent context after the child exits."""
    monkeypatch.delenv("JAATO_WORKSPACE_ROOT", raising=False)

    parent_before = get_workspace_root()
    assert parent_before is None

    async def child():
        token = set_workspace_root("/child")
        try:
            assert get_workspace_root() == "/child"
        finally:
            reset_workspace_root(token)

    await child()

    parent_after = get_workspace_root()
    assert parent_after is None, (
        "parent context should be unaffected by child's set/reset cycle"
    )
