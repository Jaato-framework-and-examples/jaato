"""Ask a running daemon to render an `explain` topic (protocol 1.18).

``jaato-scaffold explain`` introspects the framework installed in the
CALLING process.  That is exactly right when the CLI and the daemon share a
virtualenv, and silently wrong the moment they do not — which is the normal
shape of a deployed application: ``jaato-sdk`` (and ``jaato-server``) in the
application's own ``.venv``, driving a daemon owned by a different user over
IPC.  There are then TWO installs, and the CLI was answering about the one
that is not serving the sessions.

Measured, on exactly that pair::

    $ jaato-scaffold explain reactors
    unknown explain scope 'reactors' — one of: plugins | plugin <name> | ...

``reactors`` is contributed by ``jaato-premium``, which is installed in the
DAEMON's venv.  The refusal is indistinguishable from *no such topic exists*,
so it sends a reader looking for a feature they already have.  The
entry-point seam is not the gap — it works, in the process that has the
package.  What was missing is a way to ask the other process.

**When this runs, and when it must not.**  A topic the caller's own venv can
answer is still answered locally, with no socket touched: quietly giving an
offline introspection an egress would change what running it means, which is
the argument ``explain releases`` already makes about being its own topic
rather than a facet of ``explain dependencies``.  So the daemon is consulted
on exactly two paths — ``--connect``, where the reader asked for it, and a
topic this venv does not HAVE, where the alternative is asserting a
completeness the CLI cannot claim.

Every failure here returns rather than raises: no daemon, a daemon too old
to serve the verb, a socket that refuses, a reply that never comes.  A
diagnostic that dies because a daemon is not running is worse than one that
says so, and the caller always has a local answer to fall back on.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: How long to wait for a daemon to render a topic.  Generous because
#: ``explain plugins`` runs plugin discovery on the daemon's side, and mean
#: because this sits in front of an error message a person is waiting for.
DEFAULT_TIMEOUT = 20.0


@dataclass
class RemoteAnswer:
    """What a daemon said, or why it could not be asked.

    ``reached`` is the discriminator the CLI branches on, and it is
    deliberately separate from ``ok``: *the daemon answered and does not have
    that topic either* is a different fact from *no daemon could be asked*,
    and collapsing them reproduces the confusion this module exists to
    remove — one of them means the topic does not exist, the other means
    nobody looked.
    """

    reached: bool = False
    ok: bool = False
    topic: str = ""
    text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    topics: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    server_version: str = ""
    socket_path: str = ""
    #: Why the daemon could not be asked — set only when ``reached`` is False.
    unreachable: str = ""


def default_socket_path() -> str:
    """The socket a daemon listens on when nobody said otherwise."""
    from jaato_sdk.client.ipc import DEFAULT_SOCKET_PATH
    return DEFAULT_SOCKET_PATH


def daemon_is_listening(socket_path: str) -> bool:
    """Whether *socket_path* looks like a live daemon, without connecting.

    A cheap existence test so the unknown-topic path does not pay a connect
    attempt on the overwhelmingly common single-venv machine where no daemon
    is running at all.  It is a HINT: a stale socket file passes here and
    fails at connect, which :func:`ask_daemon` reports as unreachable.
    """
    try:
        from pathlib import Path
        import stat
        st = Path(socket_path).stat()
        return stat.S_ISSOCK(st.st_mode)
    except Exception:
        # A Windows named pipe has no stat-able path; let the connect decide.
        import sys
        return sys.platform == "win32"


async def _ask(socket_path: str, topic: Optional[str], name: Optional[str],
               timeout: float) -> RemoteAnswer:
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.events import ClientType, EventType

    # `auto_start=False` is load-bearing: a diagnostic that SPAWNS a daemon as
    # a side effect of being asked a question has answered about a process it
    # created, which is a different process from the one serving the reader's
    # application — the very confusion this module exists to remove.  No
    # daemon is a fact to report, not one to fix.
    client = IPCClient(socket_path, client_type=ClientType.API,
                       auto_start=False)
    answer = RemoteAnswer(topic=topic or "", socket_path=socket_path)
    try:
        await asyncio.wait_for(client.connect(), timeout=timeout)
    except Exception as exc:
        answer.unreachable = f"could not connect to {socket_path}: {exc}"
        return answer

    try:
        try:
            await client.explain_topic(topic, name)
        except ValueError as exc:
            # The SDK refuses below the protocol floor rather than waiting out
            # a reply an older daemon will never send.  That refusal names the
            # daemon's version, which is the actionable half.
            answer.unreachable = str(exc)
            return answer

        deadline = asyncio.get_event_loop().time() + timeout
        async for event in client.events():
            if event.type == EventType.SCAFFOLD_EXPLAIN_RESULT:
                answer.reached = True
                answer.ok = bool(getattr(event, "ok", False))
                answer.text = getattr(event, "text", "") or ""
                answer.data = getattr(event, "data", {}) or {}
                answer.topics = getattr(event, "topics", []) or []
                answer.error = getattr(event, "error", "") or ""
                answer.server_version = getattr(event, "server_version", "") or ""
                return answer
            if asyncio.get_event_loop().time() > deadline:
                break
        answer.unreachable = (
            f"the daemon at {socket_path} did not answer scaffold.explain "
            f"within {timeout:g}s")
        return answer
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass


def ask_daemon(
    socket_path: Optional[str] = None,
    topic: Optional[str] = None,
    name: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> RemoteAnswer:
    """Render *topic* on the daemon at *socket_path*; never raise.

    Args:
        socket_path: The daemon's IPC socket; the SDK default when omitted.
        topic: The topic, or ``None`` for the daemon's overview + catalog.
        name: The topic's argument, when it takes one.
        timeout: Seconds for the whole exchange.

    Returns:
        A :class:`RemoteAnswer`.  Check ``reached`` before ``ok``.
    """
    path = socket_path or default_socket_path()
    try:
        return asyncio.run(_ask(path, topic, name, timeout))
    except Exception as exc:                        # pragma: no cover - defensive
        return RemoteAnswer(socket_path=path, topic=topic or "",
                            unreachable=f"could not ask {path}: {exc}")


def attribution(answer: RemoteAnswer) -> str:
    """The line that says WHOSE install produced a rendering.

    Never omitted on a remote answer.  A reader who cannot tell the daemon's
    install from their own cannot tell which one to change, and on the
    deployment this module exists for those are two different machines'
    worth of different packages.
    """
    version = answer.server_version or "unknown version"
    return (f"[answered by the daemon at {answer.socket_path} "
            f"(jaato-server {version}) — not this venv]")
