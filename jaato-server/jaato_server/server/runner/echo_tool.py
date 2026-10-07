"""Echo executor — single test tool for §8.3 RPC-overhead measurement.

Returns ``args`` verbatim wrapped in a result dict.  No subprocess, no
filesystem, no streaming — pure Python dispatch.  Used by:

- The runner package's unit tests (round-trip + cancel-while-idle
  scenarios).
- The §2.6 sister test ``test_phase2_rpc_overhead.py`` which times
  1000 echo calls to assert p50 ≤ 5 ms (§8.3 acceptance gate).

The dispatcher already handles ``method=="echo"`` directly in
``RunnerRPC._dispatch_method`` for the simplest possible path; this
module exists for the case where ``echo`` is called via the generic
``tool.execute`` route too (so the typed envelope sees the same
shape as a real tool result).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


def execute_echo(args: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Return ``args`` wrapped in a result dict.

    Result shape: ``{"echo": <args>}`` — distinct from the raw args so
    the envelope decoder can tell "what we sent" from "what came
    back".
    """
    return True, {"echo": dict(args)}
