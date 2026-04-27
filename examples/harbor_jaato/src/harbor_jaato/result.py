"""Wire schema between the in-container harness and the host BaseAgent.

The harness writes ``HarnessResult`` to ``result.json`` after every
``TurnCompletedEvent`` (atomic rename). The BaseAgent on the host reads
the same JSON back and copies the fields onto Harbor's ``AgentContext``.

Kept stdlib-only on purpose: this module is imported on both sides of
the host/container boundary and must not pull in pydantic, jaato_sdk,
or harbor itself.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HarnessResult:
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    n_cache_read_tokens: int = 0
    n_cache_creation_tokens: int = 0
    cost_usd: float | None = None
    turns: int = 0
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    finished: bool = False

    @property
    def n_cache_tokens(self) -> int:
        return self.n_cache_read_tokens + self.n_cache_creation_tokens

    def write_atomic(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=path.parent, prefix=".result.", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(asdict(self), f)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def read(cls, path: Path) -> "HarnessResult":
        return cls(**json.loads(path.read_text()))
