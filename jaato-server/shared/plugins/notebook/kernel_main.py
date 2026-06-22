"""Notebook subprocess kernel entrypoint (design option 1c, PR 1).

Run as a subprocess by ``SubprocessKernelBackend`` with ``cwd=workspace_root``
(so the kernel's own ``os.getcwd()`` IS the workspace — relative paths in
notebook code resolve in-workspace, with no process-global ``os.chdir`` in the
runner).  Speaks ``kernel_protocol`` frames over two inherited fds.

Lifecycle: handshake ``ready`` → loop {execute | variables | reset | shutdown}.
Cell stdout/stderr is captured and streamed as ``stream`` frames; the last
expression's value (Jupyter-style) is returned in the ``result`` frame.

The ``tools`` object in the namespace is a STUB in PR 1 (raises a clear error);
the cross-process tools bridge lands in PR 2.

Invocation:
  python -m shared.plugins.notebook.kernel_main \
      --workspace-root <abs> --read-fd <n> --write-fd <n>
"""

import argparse
import ast
import os
import sys
import traceback
from typing import Any, Dict

from shared.plugins.notebook import kernel_protocol as proto


class _StreamWriter:
    """File-like that turns ``write()`` into ``stream`` frames for one cell."""

    def __init__(self, out, cell_id: str, name: str):
        self._out = out
        self._cell_id = cell_id
        self._name = name

    def write(self, text: str) -> int:
        if text:
            proto.write_frame(self._out, {
                "type": proto.STREAM, "cell_id": self._cell_id,
                "name": self._name, "text": text,
            })
        return len(text)

    def flush(self) -> None:
        pass


class _ToolsStub:
    """PR 1 placeholder for the notebook ``tools`` bridge — raises clearly.

    The cross-process tool bridge (kernel ``tool_call`` → runner executor →
    ``tool_result``) lands in PR 2.
    """

    def __getattr__(self, name: str):
        def _unavailable(*_a, **_k):
            raise RuntimeError(
                f"tools.{name}(): the notebook tools bridge is not available in "
                "subprocess-kernel mode yet (lands in PR 2 of the kernel "
                "re-architecture)."
            )
        return _unavailable


def _fresh_namespace() -> Dict[str, Any]:
    ns: Dict[str, Any] = {"__name__": "__main__", "__builtins__": __builtins__}
    ns["tools"] = _ToolsStub()
    return ns


def _execute(namespace: Dict[str, Any], out, cell_id: str, code: str,
             execution_count: int) -> None:
    """Exec a cell; stream stdout/stderr; send ``result`` or ``error``.

    The last top-level expression is eval'd (Jupyter ``Out[]`` semantics) and its
    ``repr`` is returned as ``result.value``.
    """
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _StreamWriter(out, cell_id, "stdout")
    sys.stderr = _StreamWriter(out, cell_id, "stderr")
    try:
        try:
            tree = ast.parse(code, filename="<cell>", mode="exec")
        except SyntaxError as exc:
            sys.stdout, sys.stderr = old_out, old_err
            proto.write_frame(out, {
                "type": proto.ERROR, "cell_id": cell_id,
                "ename": type(exc).__name__, "evalue": str(exc),
                "traceback": traceback.format_exc(),
            })
            return

        value_repr = None
        last = tree.body[-1] if tree.body else None
        if isinstance(last, ast.Expr):
            tree.body.pop()
            exec(compile(tree, "<cell>", "exec"), namespace)
            value = eval(
                compile(ast.Expression(last.value), "<cell>", "eval"), namespace)
            if value is not None:
                value_repr = repr(value)
        else:
            exec(compile(tree, "<cell>", "exec"), namespace)

        sys.stdout, sys.stderr = old_out, old_err
        proto.write_frame(out, {
            "type": proto.RESULT, "cell_id": cell_id, "status": "ok",
            "value": value_repr, "execution_count": execution_count,
        })
    except BaseException as exc:  # noqa: BLE001 — surface ANY cell error
        sys.stdout, sys.stderr = old_out, old_err
        proto.write_frame(out, {
            "type": proto.ERROR, "cell_id": cell_id,
            "ename": type(exc).__name__, "evalue": str(exc),
            "traceback": traceback.format_exc(),
        })


def _variables(namespace: Dict[str, Any]) -> Dict[str, str]:
    """Summarise user-visible variables as ``name -> type`` (skip dunder/tools)."""
    out: Dict[str, str] = {}
    for k, v in namespace.items():
        if k.startswith("__") or k == "tools":
            continue
        out[k] = type(v).__name__
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="notebook-kernel")
    ap.add_argument("--workspace-root", required=True)
    ap.add_argument("--read-fd", type=int, required=True)
    ap.add_argument("--write-fd", type=int, required=True)
    args = ap.parse_args(argv)

    # The whole point of 1c: the kernel's OWN cwd is the workspace, so relative
    # paths in notebook code resolve in-workspace — no process-global chdir in
    # the runner (which the framework forbids, core.py:915).
    os.chdir(args.workspace_root)

    rstream = os.fdopen(args.read_fd, "rb", buffering=0)
    wstream = os.fdopen(args.write_fd, "wb", buffering=0)

    proto.write_frame(wstream, {"type": proto.READY, "cwd": os.getcwd()})

    namespace = _fresh_namespace()
    execution_count = 0
    while True:
        try:
            frame = proto.read_frame(rstream)
        except EOFError:
            break
        ftype = frame.get("type")
        if ftype == proto.EXECUTE:
            execution_count += 1
            _execute(namespace, wstream, frame.get("cell_id", ""),
                     frame.get("code", ""), execution_count)
        elif ftype == proto.VARIABLES:
            proto.write_frame(wstream, {
                "type": proto.VARIABLES, "variables": _variables(namespace),
            })
        elif ftype == proto.RESET:
            namespace = _fresh_namespace()
            execution_count = 0
            proto.write_frame(wstream, {"type": proto.RESULT, "cell_id": "",
                                        "status": "reset", "value": None,
                                        "execution_count": 0})
        elif ftype == proto.SHUTDOWN:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
