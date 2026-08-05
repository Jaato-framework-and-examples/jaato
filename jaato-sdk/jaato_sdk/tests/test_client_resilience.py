"""Client connection-resilience fixes (B).

Covers the three behaviour fixes to ``IPCClient``:
- (a) env_file=None fails loud with a clear ValueError (not an opaque
  os.PathLike crash mid-connect).
- (b) autostart_timeout knob: cold-daemon launch gets a longer budget than the
  already-running connect timeout.
- (c) a stale pidfile whose PID was REUSED no longer blocks auto-start — the
  endpoint must actually be live before the pidfile is trusted.
"""

import asyncio

import pytest

from jaato_sdk import IPCClient, ClientType


def _client(tmp_path, **kw):
    return IPCClient(str(tmp_path / "x.sock"), client_type=ClientType.API, **kw)


# ---- (a) env_file=None fail-loud -----------------------------------------

def test_env_file_none_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="env_file"):
        IPCClient(str(tmp_path / "x.sock"), client_type=ClientType.API,
                  env_file=None)


# ---- (b) autostart_timeout knob ------------------------------------------

def test_autostart_timeout_default_and_override(tmp_path):
    assert _client(tmp_path).autostart_timeout == 120.0
    assert _client(tmp_path, autostart_timeout=200.0).autostart_timeout == 200.0


# ---- (c) stale-pidfile / PID-reuse recovery ------------------------------

def test_endpoint_is_live_false_for_dead_socket(tmp_path):
    # nothing is listening on this path → not live, regardless of any file
    assert _client(tmp_path)._endpoint_is_live() is False


def test_start_server_relaunches_when_pid_alive_but_endpoint_dead(tmp_path, monkeypatch):
    c = _client(tmp_path)
    monkeypatch.setattr(c, "_check_server_running", lambda: 99999)   # "alive" reused PID
    monkeypatch.setattr(c, "_endpoint_is_live", lambda: False)       # but socket dead
    launched = {"n": 0}
    monkeypatch.setattr("jaato_sdk.client.ipc.subprocess.run",
                        lambda cmd, check: launched.__setitem__("n", launched["n"] + 1))

    async def fake_wait(timeout=10.0):
        return True
    monkeypatch.setattr(c, "_wait_for_socket", fake_wait)

    assert asyncio.run(c._start_server()) is True
    assert launched["n"] == 1   # relaunched instead of waiting on the dead socket


def test_start_server_attaches_without_relaunch_when_endpoint_live(tmp_path, monkeypatch):
    c = _client(tmp_path)
    monkeypatch.setattr(c, "_check_server_running", lambda: 12345)
    monkeypatch.setattr(c, "_endpoint_is_live", lambda: True)        # genuinely up
    launched = {"n": 0}
    monkeypatch.setattr("jaato_sdk.client.ipc.subprocess.run",
                        lambda cmd, check: launched.__setitem__("n", launched["n"] + 1))

    async def fake_wait(timeout=10.0):
        return True
    monkeypatch.setattr(c, "_wait_for_socket", fake_wait)

    assert asyncio.run(c._start_server()) is True
    assert launched["n"] == 0   # attached to the live daemon, no relaunch
