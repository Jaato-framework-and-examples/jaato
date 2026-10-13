"""How long the daemon waits for a client-provided ("host") tool (#1625).

A client registers host tools with ``register_client_tools`` (or the
facade's ``client_tools=``).  When the model calls one, the daemon sends a
``ToolExecuteRequestEvent`` to the client and waits for the matching
``ToolExecuteResultEvent``.  That wait has a deadline, and on expiry the
model receives::

    {"error": "Client tool <name> timed out after <N>s"}

A client that wraps its handler in its own deadline (to turn a slow
dependency into a domain answer the model can read) must fire BEFORE the
daemon's, or the model gets the raw transport error instead.  So the limit
must be knowable, and this module is the one place it is decided:

* the knob is each tool spec's ``"timeout"`` key, in MILLISECONDS;
* absent, the budget is :data:`DEFAULT_CLIENT_TOOL_TIMEOUT_MS` (30 s);
* a value that is not a positive finite number is not honoured: the default
  applies and :attr:`ClientToolBudget.problem` says why.

The daemon's registration (IPC and WS) and the SDK clients both call
:func:`resolve_client_tool_timeout`, so the budget a client is told and the
budget the daemon enforces cannot be two answers.  The daemon also reports
the budgets back on ``ToolsRegisterClientResultEvent`` (protocol 1.39), which
``IPCClient.register_client_tools`` returns.

Stdlib-only: the daemon imports it on its registration path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

#: The deadline, in milliseconds, for a host tool whose spec names none.
DEFAULT_CLIENT_TOOL_TIMEOUT_MS = 30000

#: The tool-spec key that carries a per-tool deadline, in milliseconds.
CLIENT_TOOL_TIMEOUT_KEY = "timeout"

#: ``ClientToolBudget.source`` values.
SOURCE_CLIENT = "client"
SOURCE_DEFAULT = "default"


@dataclass(frozen=True)
class ClientToolBudget:
    """The deadline the daemon holds one host tool to.

    Attributes:
        name: The tool's model-facing name.
        timeout_ms: The budget in milliseconds.
        source: ``"client"`` when the spec's ``timeout`` was honoured,
            ``"default"`` when it was absent or not valid.
        problem: Why a declared ``timeout`` was not honoured; ``None`` when
            nothing was declared or the declared value was used.
    """

    name: str
    timeout_ms: float
    source: str
    problem: Optional[str] = None

    @property
    def timeout_s(self) -> float:
        """The budget in seconds, the unit the expiry message uses."""
        return self.timeout_ms / 1000.0

    def as_dict(self) -> Dict[str, Any]:
        """The wire shape carried on ``ToolsRegisterClientResultEvent``."""
        out: Dict[str, Any] = {
            "name": self.name,
            "timeout_ms": self.timeout_ms,
            "timeout_s": self.timeout_s,
            "source": self.source,
        }
        if self.problem:
            out["problem"] = self.problem
        return out


def resolve_client_tool_timeout(spec: Mapping[str, Any]) -> ClientToolBudget:
    """The budget the daemon applies to the host tool *spec* describes.

    ``bool`` is refused although it is an ``int``: ``timeout: True`` would
    otherwise be one millisecond, on a technicality.
    """
    name = str(spec.get("name") or "")
    if CLIENT_TOOL_TIMEOUT_KEY not in spec or spec.get(CLIENT_TOOL_TIMEOUT_KEY) is None:
        return ClientToolBudget(name, DEFAULT_CLIENT_TOOL_TIMEOUT_MS, SOURCE_DEFAULT)
    raw = spec[CLIENT_TOOL_TIMEOUT_KEY]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        problem = (f"timeout must be a number of milliseconds, got "
                   f"{type(raw).__name__} {raw!r}")
    elif not math.isfinite(raw) or raw <= 0:
        problem = f"timeout must be a positive finite number of ms, got {raw!r}"
    else:
        return ClientToolBudget(name, raw, SOURCE_CLIENT)
    return ClientToolBudget(
        name, DEFAULT_CLIENT_TOOL_TIMEOUT_MS, SOURCE_DEFAULT, problem)


def resolve_client_tool_timeouts(
    specs: Iterable[Mapping[str, Any]],
) -> List[ClientToolBudget]:
    """:func:`resolve_client_tool_timeout` for every named spec, in order."""
    return [resolve_client_tool_timeout(s) for s in specs or []
            if isinstance(s, Mapping) and s.get("name")]


def budgets_by_name(
    budgets: Iterable[ClientToolBudget],
) -> Dict[str, float]:
    """``{name: seconds}``, the shape ``register_client_tools`` returns."""
    return {b.name: b.timeout_s for b in budgets}


__all__ = [
    "CLIENT_TOOL_TIMEOUT_KEY",
    "ClientToolBudget",
    "DEFAULT_CLIENT_TOOL_TIMEOUT_MS",
    "SOURCE_CLIENT",
    "SOURCE_DEFAULT",
    "budgets_by_name",
    "resolve_client_tool_timeout",
    "resolve_client_tool_timeouts",
]
