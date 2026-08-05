"""jaato_embedded — the in-process (embedded) runtime backend.

Ships with **jaato-server**. This is the ``mode="in_process"`` backend behind the
public ``jaato.session(...)`` facade (which lives in jaato-sdk): it runs the agent
loop in your process with no daemon, no socket, no runner subprocess.

The public facade imports this lazily — ``jaato.session(mode="in_process")`` does
``from jaato_embedded import InProcessClient`` only when in-process mode is
requested, so a thin ipc/ws client needs only jaato-sdk. See
``docs/design/in-process-facade.md``.
"""

from jaato_embedded.client import InProcessClient, InProcessEventEmitter

__all__ = ["InProcessClient", "InProcessEventEmitter"]
