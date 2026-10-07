"""A grader that reads the same contract its driver did (jaato #1127).

``driver.py`` beside this file is a complete consumer of the contract on
the RUN side.  This is the other side: a ``script`` grader is handed the
same table, from the same builder, so a task can score a driver arm with
the package that drove it rather than re-deriving the run from whatever
files it happened to leave.

That is what ``$JAATO_EVAL_PYTHON`` buys, and why this file is invoked
through it rather than through ``python``.  A grader that imports
``jaato_sdk`` — to open a judging session on ``$JAATO_EVAL_SOCKET``, say
— needs the interpreter that HAS it, and ``run`` inherits the engine's
``PATH`` and nothing else.  Until #1127 the variable was empty here, the
shell tried to execute ``""``, and the adapter read the resulting exit
127 as a missing toolchain and BLOCKED the arm — every arm, while every
driver exited 0.

Exit 0 is PASS, as for any script grader.  This is a PROBE and not a
benchmark, so a missing contract row is a failure like any other: it is
the thing the probe exists to detect, and reporting it as a pass with a
note on stderr is the silent-success shape the whole engine is built
against.  A real task's grader, which is measuring a model rather than
this harness, would want the opposite.
"""
from __future__ import annotations

import os
import pathlib
import sys

#: Rows this grader reads.  Absent means the harness supplied none, never
#: that it supplied an empty value: the contract omits a variable it has no
#: value for, which is what makes ``os.environ.get`` meaningful here.
REQUIRED = ("JAATO_EVAL_WORKSPACE", "JAATO_EVAL_CASCADE_ID", "JAATO_EVAL_PYTHON")

#: Stages ``driver.py`` runs, and therefore sessions the daemon should have
#: persisted a record for in THIS arm's workspace — the same records the
#: engine attributes the arm's spend from.
STAGES = 2


def main() -> int:
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        print(f"score: contract rows absent: {', '.join(missing)}",
              file=sys.stderr)
        return 1

    workspace = pathlib.Path(os.environ["JAATO_EVAL_WORKSPACE"])
    cid = os.environ["JAATO_EVAL_CASCADE_ID"]
    word = os.environ.get("JAATO_EVAL_PARAM_WORD", "READY")

    answer = workspace / "answer.txt"
    echo = workspace / "echo.txt"
    if not answer.is_file() or answer.read_text().strip() != word:
        print(f"score: answer.txt is not {word!r}", file=sys.stderr)
        return 1
    if not echo.is_file() or echo.read_text() != answer.read_text():
        print("score: echo.txt does not match answer.txt", file=sys.stderr)
        return 1

    # The workspace is named by the contract rather than assumed to be the
    # working directory — a grader that cd's somewhere is still talking
    # about this arm.  The records under it are the arm's own sessions:
    # the daemon writes one per session into the workspace that session
    # ran in, which is how the engine tells this arm's stages from a
    # concurrent sibling's under a shared pool cid.
    records = sorted((workspace / ".jaato" / "sessions").glob("*.json"))
    if len(records) < STAGES:
        print(f"score: {len(records)} session records under {cid}; the driver "
              f"ran {STAGES} stages", file=sys.stderr)
        return 1

    print(f"score: {word} echoed, {len(records)} sessions under {cid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
