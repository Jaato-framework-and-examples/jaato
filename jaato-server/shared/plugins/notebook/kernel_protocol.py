"""Wire protocol for the notebook subprocess kernel (design option 1c).

Length-prefixed JSON frames over a byte stream (a pipe fd pair), modelled on the
daemon↔runner ``RunnerRPC`` framing.  The runner side (``SubprocessKernelBackend``)
and the kernel side (``kernel_main``) both import this module so the frame shapes
stay in lockstep.

Frame = 4-byte big-endian unsigned length prefix + that many bytes of UTF-8 JSON.

Frame types
-----------
runner → kernel:
  - ``execute``     {type, cell_id, code}        dispatch a cell
  - ``variables``   {type}                       request the namespace summary
  - ``reset``       {type}                        clear the namespace
  - ``shutdown``    {type}                        graceful kernel exit
  - ``tool_result`` {type, call_id, ok, result|error}
                                                  answer a kernel ``tool_call``
                                                  (the cross-process tools bridge,
                                                  wired in PR 2)
kernel → runner:
  - ``ready``       {type, cwd}                   handshake after chdir + init
  - ``stream``      {type, cell_id, name, text}   streamed stdout/stderr
  - ``result``      {type, cell_id, status, value, execution_count}
  - ``error``       {type, cell_id, ename, evalue, traceback}
  - ``variables``   {type, variables}             reply to a ``variables`` request
  - ``tool_call``   {type, call_id, name, args}   notebook ``tools.X()`` → bridge
                                                  (PR 2)
"""

import json
import struct
from typing import Any, BinaryIO, Dict

# ---- runner → kernel ----
EXECUTE = "execute"
VARIABLES = "variables"
RESET = "reset"
SHUTDOWN = "shutdown"
TOOL_RESULT = "tool_result"

# ---- kernel → runner ----
READY = "ready"
STREAM = "stream"
RESULT = "result"
ERROR = "error"
TOOL_CALL = "tool_call"

_LEN = struct.Struct(">I")  # 4-byte big-endian length prefix


def write_frame(stream: BinaryIO, frame: Dict[str, Any]) -> None:
    """Serialize one frame as length-prefixed JSON, write it, and flush."""
    payload = json.dumps(frame).encode("utf-8")
    stream.write(_LEN.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def read_frame(stream: BinaryIO) -> Dict[str, Any]:
    """Read exactly one frame.  Raises ``EOFError`` on a clean stream close."""
    header = _read_exactly(stream, _LEN.size)
    (length,) = _LEN.unpack(header)
    payload = _read_exactly(stream, length)
    return json.loads(payload.decode("utf-8"))


def _read_exactly(stream: BinaryIO, n: int) -> bytes:
    """Read exactly ``n`` bytes or raise ``EOFError`` if the stream ends first."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("kernel protocol stream closed")
        buf.extend(chunk)
    return bytes(buf)
