"""`--stop` / `--status` find client-autostarted daemons via the socket.

Regression for the gap the LORA peer hit: `jaato-server --stop` reported
"not running" while a client-autostarted daemon was still alive on the socket,
because its ephemeral teardown removed the pidfile.  ``check_running`` now
falls back from the pidfile to the authoritative socket signal, so a
missing/stale pidfile no longer orphans a live daemon.
"""

import subprocess
import sys

import server.__main__ as M


def test_falls_back_to_socket_when_pidfile_missing(monkeypatch):
    monkeypatch.setattr(M, "_pid_on_socket", lambda s: 4242)
    assert M.check_running("/no/such.pid", "/tmp/whatever.sock") == 4242


def test_no_socket_means_no_fallback(monkeypatch):
    monkeypatch.setattr(M, "_pid_on_socket", lambda s: 4242)
    assert M.check_running("/no/such.pid", None) is None


def test_live_pidfile_wins_without_probing_socket(tmp_path, monkeypatch):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        pf = tmp_path / "x.pid"
        pf.write_text(str(proc.pid))
        probed = {"v": False}
        monkeypatch.setattr(
            M, "_pid_on_socket",
            lambda s: probed.__setitem__("v", True) or 999)
        assert M.check_running(str(pf), "/tmp/whatever.sock") == proc.pid
        assert probed["v"] is False          # live pidfile -> socket never probed
    finally:
        proc.terminate()
        proc.wait()


def test_pid_on_socket_none_without_listener():
    assert M._pid_on_socket("/tmp/definitely-absent-xyz.sock") is None
    assert M._pid_on_socket(None) is None
