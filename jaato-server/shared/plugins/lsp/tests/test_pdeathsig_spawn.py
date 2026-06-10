"""Regression: language-server processes must be spawned with a
parent-death signal so they're reaped when their owning runner dies.

Root cause of the #284 daemon OOM (#280): jdtls (0.5-1GB each) was
spawned as a runner child with NO PR_SET_PDEATHSIG, so a runner killed
ungracefully (OOM / SIGKILL / pool recycle / mid-cascade abort) left its
language server orphaned — re-parenting to the daemon (subreaper) and
accumulating one-per-cascade-stage until the daemon itself OOM'd.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.plugins.lsp.lsp_client import (
    LSPClient,
    ServerConfig,
    _make_pdeathsig_preexec,
    _set_pdeathsig_sigkill,
)


def test_preexec_is_callable_on_linux():
    if sys.platform == "linux":
        assert callable(_set_pdeathsig_sigkill)
    # On non-linux it's None (preexec_fn=None is a harmless no-op).


def test_make_preexec_returns_none_off_linux():
    with patch("shared.plugins.lsp.lsp_client.sys") as msys:
        msys.platform = "darwin"
        assert _make_pdeathsig_preexec() is None


def test_start_spawns_with_pdeathsig_preexec():
    """start() must pass preexec_fn=_set_pdeathsig_sigkill to the spawn."""
    client = LSPClient(ServerConfig(name="java", command="jdtls", args=["-data", "/tmp/x"]))

    fake_proc = MagicMock()
    fake_proc.stdin = MagicMock()
    fake_proc.stdout = MagicMock()

    captured = {}

    async def fake_exec(*args, **kwargs):
        captured.update(kwargs)
        return fake_proc

    async def run():
        with patch(
            "asyncio.create_subprocess_exec", side_effect=fake_exec
        ), patch.object(client, "_initialize", new=AsyncMock()), patch.object(
            client, "_read_messages", new=AsyncMock()
        ):
            await client.start()

    asyncio.run(run())

    assert "preexec_fn" in captured
    assert captured["preexec_fn"] is _set_pdeathsig_sigkill
