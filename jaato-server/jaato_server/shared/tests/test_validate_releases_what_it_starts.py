"""A ``validate`` run releases the plugins it started.

``introspect.plugins()`` asks every plugin for its tool schemas, and ``lsp``
and ``mcp`` initialize themselves there: a background thread, an event loop
and a stderr pipe each.  Nothing shut them down, so every ``validate`` run
leaked about ten descriptors.  A process that validated a few dozen
workspaces (the shared test suite does) crossed 1024 and a later
``select()`` failed with ``filedescriptor out of range``, far from the cause.

MCP had a second leak under the first: its stderr ``LogCapture`` was never
closed, so even a clean ``shutdown()`` left the pipe pair open.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from jaato_server.shared.scaffold import validate as V
from jaato_server.shared.tests.reversion import Reversion

REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/introspect.py",
        find=("    try:\n"
              "        return _describe_plugins(reg)\n"
              "    finally:\n"
              "        _release(reg)\n"),
        replace="    return _describe_plugins(reg)\n",
        because="every validate run leaves lsp's and mcp's threads, event "
                "loops and pipes behind",
        test="test_repeated_validate_runs_do_not_accumulate_descriptors",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/plugins/mcp/plugin.py",
        find=("        if self._errlog is not None:\n"
              "            self._errlog.close()\n"
              "            self._errlog = None\n"),
        replace="",
        because="mcp's shutdown leaves its stderr pipe open, one pair per run",
        test="test_repeated_validate_runs_do_not_accumulate_descriptors",
    ),
]

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="counts /proc/self/fd")


def _open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_repeated_validate_runs_do_not_accumulate_descriptors(tmp_path: Path):
    prof = tmp_path / ".jaato" / "profiles"
    prof.mkdir(parents=True)
    (prof / "w.yaml").write_text(
        "name: w\ndescription: x\nplugins: [cli]\n", encoding="utf-8")

    V.validate_workspace(str(tmp_path))  # imports and caches settle here
    before = _open_fds()
    for _ in range(3):
        V.validate_workspace(str(tmp_path))
    assert _open_fds() - before <= 1
