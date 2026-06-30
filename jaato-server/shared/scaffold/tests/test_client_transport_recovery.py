"""`--transport` x `--recoverable` maps to the right SDK client class.

ipc  -> IPCClient / IPCRecoveryClient
ws   -> WSClient  / WSRecoveryClient   (the ws+recoverable path — was rejected
                                        before WSRecoveryClient shipped)

The generated client must import + construct the right class, and wire an
on_status_change callback for the recovery variants.
"""

import argparse
from pathlib import Path

from shared.scaffold import build


def _gen(tmp_path: Path, *, transport: str, recoverable: bool, url=None) -> str:
    args = argparse.Namespace(
        archetype="client",
        workspace=str(tmp_path),
        provider="anthropic",
        model="test-model",
        set=None,
        agents=None,
        force=True,
        recoverable=recoverable,
        json=False,
        transport=transport,
        url=url,
        token="tok" if url else None,
    )
    rc = build.run(args)
    assert rc == 0, f"build.run(transport={transport}, recoverable={recoverable}) -> {rc}"
    return (tmp_path / "run_client.py").read_text()


def test_ipc_default_is_ipcclient(tmp_path):
    src = _gen(tmp_path, transport="ipc", recoverable=False)
    assert "IPCClient" in src and "IPCRecoveryClient" not in src
    assert "on_status_change" not in src


def test_ipc_recoverable_is_ipcrecoveryclient(tmp_path):
    src = _gen(tmp_path, transport="ipc", recoverable=True)
    assert "IPCRecoveryClient" in src
    assert "on_status_change=_on_status" in src


def test_ws_is_wsclient(tmp_path):
    src = _gen(tmp_path, transport="ws", recoverable=False, url="wss://h:1")
    assert "WSClient" in src and "WSRecoveryClient" not in src
    assert "on_status_change" not in src


def test_ws_recoverable_is_wsrecoveryclient(tmp_path):
    src = _gen(tmp_path, transport="ws", recoverable=True, url="wss://h:1")
    assert "from jaato_sdk import WSRecoveryClient" in src
    assert "WSRecoveryClient(" in src
    assert "on_status_change=_on_status" in src
