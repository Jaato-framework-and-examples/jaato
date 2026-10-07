"""Tracing is off unless a variable names a file.

With ``JAATO_TRACE_LOG`` unset the application trace used to fall back to
``<tempdir>/rich_client_trace.log``, so every daemon, runner and client
traced by default, opening and closing that file for every line: about
11% of the daemon's CPU on a 16-session fan-out, and a per-session file
in every runner's temp directory.
"""

from __future__ import annotations

import tempfile

import importlib

trace_mod = importlib.import_module("jaato_sdk.trace")
from jaato_server.shared.tests.reversion import Reversion

REVERSIONS = [
    Reversion(
        target="jaato-sdk/jaato_sdk/trace.py",
        find=("            return value\n"
              "    return None\n"),
        replace=("            return value\n"
                 "    return os.path.join(\"/tmp\", \"rich_client_trace.log\")\n"),
        test="test_an_unset_variable_means_no_trace",
        because="tracing back on by default, to a file in /tmp",
    ),
]


def test_an_unset_variable_means_no_trace(monkeypatch, tmp_path):
    for var in ("JAATO_TRACE_LOG", "JAATO_PROVIDER_TRACE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    assert trace_mod.resolve_trace_path("JAATO_TRACE_LOG") is None
    assert trace_mod.resolve_trace_path(
        "JAATO_PROVIDER_TRACE", default_filename="provider_trace.log") is None
    trace_mod.trace("Test", "nobody asked for this")
    assert list(tmp_path.iterdir()) == []


def test_an_empty_variable_means_no_trace(monkeypatch):
    monkeypatch.setenv("JAATO_TRACE_LOG", "")
    assert trace_mod.resolve_trace_path("JAATO_TRACE_LOG") is None


def test_a_named_file_is_written(monkeypatch, tmp_path):
    target = tmp_path / "trace.log"
    monkeypatch.setenv("JAATO_TRACE_LOG", str(target))
    trace_mod.trace("Test", "asked for")
    assert "asked for" in target.read_text()
