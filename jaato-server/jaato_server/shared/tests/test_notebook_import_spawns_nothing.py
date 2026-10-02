"""Importing the notebook's subprocess backend starts no process.

``ctypes.util.find_library("c")`` at import ran ``ldconfig -p`` in every
runner; under SELinux the exec is refused and find_library falls back to
running gcc and objdump (phase 2b kernel run, ``--trace-subprocess``).
``prctl`` is taken from the process's own symbols instead, as
``shared/private_tmp.py`` already does.
"""

import importlib
import subprocess
from unittest.mock import patch

from jaato_server.shared.tests.reversion import Reversion

_MODULE = "jaato_server.shared.plugins.notebook.backends.subprocess_kernel"

REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/shared/plugins/notebook/backends/subprocess_kernel.py",
        find="    _LIBC = ctypes.CDLL(None, use_errno=True)\n",
        replace="    _LIBC = ctypes.CDLL(ctypes.util.find_library(\"c\") or \"libc.so.6\",\n"
                "                        use_errno=True)\n",
        test="test_importing_the_backend_starts_no_process",
        because="every runner would run ldconfig at import, and gcc + objdump "
                "where a confined runner may not exec ldconfig",
    ),
]


def test_importing_the_backend_starts_no_process():
    import jaato_server.shared.plugins.notebook.backends.subprocess_kernel as mod

    with patch.object(subprocess, "Popen", side_effect=AssertionError("spawned")) as popen:
        importlib.reload(mod)
    popen.assert_not_called()
    assert mod._LIBC is not None
    assert mod._LIBC.prctl is not None
