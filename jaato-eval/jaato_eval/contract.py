"""The arm, as environment — one table, two consumers.

A grader and the thing it grades must be talking about the same arm.
:mod:`jaato_eval.params` established that for the task's INPUTS, and said
why in as many words: *the whole value of the export is that a grader and
the thing it grades read the SAME variable for the same input*.  The rest
of the table was built one consumer at a time and did not follow.

``JAATO_EVAL_PYTHON`` is where that cost something.  It names the
interpreter jaato-eval itself runs under — the one that HAS ``jaato_sdk``
— and the driver-arm documentation tells an author to write ``run:
'"$JAATO_EVAL_PYTHON" -m pkg.driver'`` rather than bet on the host's
``PATH``.  A ``script`` grader written the same way, against the same
package, was handed an empty variable and asked the shell to execute
``""``: exit 127, which the adapter correctly reads as *the toolchain is
missing* and BLOCKS on.  Every arm of the first real driver task blocked
that way while the arms themselves ran to exit 0 (jaato #1127).  The
manifest had no other way to say it: ``python`` is the ``PATH`` bet #1112
removed for the driver, and an absolute path is a per-host constant in a
committed file.

So the table is built HERE, once, and both consumers call it:

* :func:`jaato_eval.driver.driver_environment` — the driver process, for
  which this has always been the contract (``JAATO_EVAL_CONTRACT``);
* :class:`jaato_eval.graders.script.ScriptGrader` — the command run
  against the workspace the arm left behind.

The grader receives the whole table rather than the interpreter alone,
because every row answers a question a grader has as legitimately as a
driver does: the workspace and the config root are what it is grading
against, the socket is the daemon the ARM ran on (a grader that opens its
own session must use that one — ``GraderContext.socket_path`` exists for
exactly that reason), and the cascade id is how it finds the arm's session
records on disk.  Values a grader cannot be given are ABSENT rather than
empty, which is the rule ``JAATO_EVAL_SOCKET`` already followed: a
variable that is there is a variable a consumer may use unconditionally.

A NEW variable is additive and does not bump :data:`CONTRACT_VERSION` — a
consumer that has never heard of one behaves exactly as it did.  Exporting
an existing variable to a SECOND consumer is additive in the same sense,
and for the same reason is not a bump either.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .params import param_env

#: The version of the contract table.  Bumped when a variable changes
#: MEANING; a new variable, or an existing one reaching a new consumer, is
#: additive and is not.
CONTRACT_VERSION = "1"


def arm_environment(*, workspace: Path, config_root: Path,
                    params: Mapping[str, Any],
                    cascade_id: Optional[str] = None,
                    socket_path: Optional[str] = None,
                    ) -> Tuple[Dict[str, str], Optional[str]]:
    """The contract, as the environment ADDED to a child process's.

    Returns ``(env, None)`` normally and ``({}, reason)`` when two
    ``params`` keys collide on one variable name — the shape
    :func:`jaato_eval.params.param_env` returns, so each caller keeps its
    own way of reporting it: the manifest parser refuses a driver task
    before anything is spent, the script grader returns BLOCKED.

    Returns only the contract variables; the caller overlays them on
    ``os.environ`` at spawn.  Kept pure so the table is testable as a
    table.

    What it deliberately does NOT carry is a credential.  The ``.env`` the
    engine writes into an arm's workspace carries ``JAATO_PROFILE_SET``
    and nothing else, so a driver's profiles resolve theirs through
    ``pass://`` / ``vault://`` or from the daemon's environment — see
    :mod:`jaato_eval.driver`.
    """
    env, collision = param_env(params)
    if collision:
        return {}, collision
    contract = {
        "JAATO_EVAL": "1",
        "JAATO_EVAL_CONTRACT": CONTRACT_VERSION,
        "JAATO_EVAL_WORKSPACE": str(workspace),
        "JAATO_EVAL_CONFIG_ROOT": str(config_root),
        **env,
    }
    # Absent rather than empty, all three.  A driver arm always has a cid
    # (the observer needs one) and a session arm has one only when its task
    # declared a pool; the socket is absent when the sweep uses the SDK
    # default; and the interpreter cannot name itself in an embedded build.
    if cascade_id:
        contract["JAATO_EVAL_CASCADE_ID"] = str(cascade_id)
    if socket_path:
        contract["JAATO_EVAL_SOCKET"] = str(socket_path)
    # The interpreter, not an interpreter: this one is where ``jaato_sdk``
    # is importable.  A driver is an SDK client, and so is any grader that
    # imports the package it is grading.
    if sys.executable:
        contract["JAATO_EVAL_PYTHON"] = sys.executable
    return contract, None
