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
import json
import sys
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


def render_from_daemon(
    asked: Optional[str],
    scope: Optional[str],
    name: Optional[str],
    args,
    *,
    required: bool,
) -> "tuple[Optional[int], str]":
    """Ask a running daemon to render *scope*, and print what it says.

    The one implementation of the ``explain`` daemon fallback, shared by
    jaato-server's ``explain`` verb (a topic its venv lacks) and the SDK
    shell's ``explain`` refusal (a venv with no jaato-server at all, #1267),
    so the two cannot disagree about when a daemon is asked or whose refusal
    prints.

    ``jaato-scaffold`` introspects the framework in the CALLING process.  When
    the CLI and the daemon share a virtualenv that is the whole answer; when
    they do not — ``jaato-sdk`` in an application's own ``.venv`` driving a
    daemon owned by another user — the topic a premium extension contributes
    exists in the daemon's install and in no other, and the local refusal is
    indistinguishable from *no such topic exists*.

    Args:
        asked: The socket the reader named with ``--connect``; ``None`` on the
            fallback path, where the SDK default is used.
        scope: The topic, or ``None`` for the overview.
        name: The topic's argument, when it takes one.
        args: The parsed namespace, read for ``--json``.
        required: ``True`` when the reader asked for the daemon, so an
            unreachable one is this command's failure and is reported.
            ``False`` on the fallback, where an unreachable daemon means only
            that nobody could be asked — the caller then prints its own local
            refusal, which is the honest answer on a machine with no daemon.

    Returns:
        ``(rc, note)``.  ``rc`` is an exit code, or ``None`` when the caller
        should print its own local refusal instead; ``note`` is a line to
        print after it, empty unless there is something to add.
    """
    if not required and not daemon_is_listening(
            default_socket_path()):
        return None, ""

    answer = ask_daemon(asked if isinstance(asked, str) else None,
                                scope, name)
    if not answer.reached:
        if not required:
            return None, ""
        print(answer.unreachable, file=sys.stderr)
        return 2, ""

    if not answer.ok:
        if required:
            # The reader named this daemon, so its refusal IS the answer —
            # including its topic list, which is the one they asked about.
            print(answer.error or f"unknown explain scope {scope!r}",
                  file=sys.stderr)
            print(attribution(answer), file=sys.stderr)
            return 2, ""
        # On the FALLBACK, the daemon's refusal must not replace the local
        # one: its list is the DAEMON's topics, and a reader shown that list
        # concludes a topic their own install has does not exist.  The local
        # refusal is the one they can act on without a socket, so it prints,
        # and the note below keeps "we asked and it is not there either"
        # distinguishable from "nobody looked" — the distinction `reached`
        # exists to preserve, one layer out.
        return None, (f"(also asked the daemon at {answer.socket_path} "
                      f"(jaato-server "
                      f"{answer.server_version or 'unknown version'}): it "
                      f"does not serve that topic either)")

    if args.json:
        print(json.dumps(answer.data, indent=2, default=str))
    else:
        print(answer.text)
        print()
        print(attribution(answer))
    return 0, ""


# ------------------------------------------------------------------ validate

#: How long to wait for a daemon's ``validate``.  It runs plugin discovery and
#: resolves every profile, which takes several seconds on a warm daemon.
VALIDATE_TIMEOUT = 90.0


@dataclass
class RemoteValidation:
    """What a daemon's validator said, or why it could not be asked.

    ``reached`` is separate from ``ok`` for the reason :class:`RemoteAnswer`
    gives, and ``ok`` is separate from the findings: ``ok`` says the
    validator ran, never that the workspace is valid.
    """

    reached: bool = False
    ok: bool = False
    workspace: str = ""
    scope: str = ""
    profile_set: str = ""
    findings: List[Dict[str, Any]] = field(default_factory=list)
    errors: int = 0
    warnings: int = 0
    error: str = ""
    server_version: str = ""
    socket_path: str = ""
    #: Why the daemon could not be asked — set only when ``reached`` is False.
    unreachable: str = ""


async def _ask_validate(socket_path: str, workspace: str,
                        profile_set: Optional[str], profile: Optional[str],
                        timeout: float) -> RemoteValidation:
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.events import ClientType, EventType

    # The workspace is declared at the handshake, where the daemon refuses a
    # path the connecting account cannot reach.  `auto_start=False` for the
    # reason `_ask` gives: a validator must not start the daemon it reports on.
    client = IPCClient(socket_path, client_type=ClientType.API,
                       auto_start=False, workspace_path=workspace)
    answer = RemoteValidation(socket_path=socket_path, workspace=workspace)
    try:
        await asyncio.wait_for(client.connect(), timeout=timeout)
    except Exception as exc:
        answer.unreachable = f"could not connect to {socket_path}: {exc}"
        return answer

    try:
        try:
            await client.validate_workspace(profile_set, profile)
        except ValueError as exc:
            answer.unreachable = str(exc)
            return answer

        async def _first_answer() -> None:
            async for event in client.events():
                if event.type == EventType.SCAFFOLD_VALIDATE_RESULT:
                    answer.reached = True
                    answer.ok = bool(getattr(event, "ok", False))
                    answer.workspace = (getattr(event, "workspace", "")
                                        or workspace)
                    answer.scope = getattr(event, "scope", "") or ""
                    answer.profile_set = getattr(event, "profile_set", "") or ""
                    answer.findings = list(getattr(event, "findings", []) or [])
                    answer.errors = int(getattr(event, "errors", 0) or 0)
                    answer.warnings = int(getattr(event, "warnings", 0) or 0)
                    answer.error = getattr(event, "error", "") or ""
                    answer.server_version = (getattr(event, "server_version",
                                                     "") or "")
                    return
                if event.type == EventType.ERROR:
                    # A refused handshake path arrives as an error event,
                    # not a validate answer: nobody validated anything.
                    answer.unreachable = (getattr(event, "error", "") or
                                          "the daemon refused the request")
                    return
            answer.unreachable = (f"the daemon at {socket_path} closed the "
                                  f"connection before answering")

        # A bound on the WAIT, not only on the events: a daemon that sends
        # nothing at all must not hang the command.
        try:
            await asyncio.wait_for(_first_answer(), timeout=timeout)
        except asyncio.TimeoutError:
            answer.unreachable = (
                f"the daemon at {socket_path} did not answer "
                f"scaffold.validate within {timeout:g}s")
        return answer
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass


def ask_daemon_validate(
    socket_path: Optional[str],
    workspace: str,
    profile_set: Optional[str] = None,
    profile: Optional[str] = None,
    timeout: float = VALIDATE_TIMEOUT,
) -> RemoteValidation:
    """Validate *workspace* with the daemon at *socket_path*; never raise."""
    path = socket_path or default_socket_path()
    try:
        return asyncio.run(_ask_validate(path, workspace, profile_set,
                                         profile, timeout))
    except Exception as exc:                        # pragma: no cover - defensive
        return RemoteValidation(socket_path=path, workspace=workspace,
                                unreachable=f"could not ask {path}: {exc}")


def validation_attribution(answer: RemoteValidation) -> str:
    """The line that says WHOSE validator produced the findings."""
    version = answer.server_version or "unknown version"
    return (f"[validated by the daemon at {answer.socket_path} "
            f"(jaato-server {version}) — not this venv]")


def validate_from_daemon(
    asked: Optional[str],
    target: str,
    profile_set: Optional[str],
    profile: Optional[str],
    *,
    json_out: bool,
    required: bool,
) -> Optional[int]:
    """Run ``validate`` on a daemon and print the findings as a local run would.

    The route an SDK-only install takes (#1267, tier 3), and the one
    ``--connect`` takes anywhere.  The daemon's own validator checks the
    workspace this connection declares; there is no second implementation.

    Args:
        asked: The socket named with ``--connect``, or ``None``.
        target: A workspace directory, or a profile file inside one.
        profile_set: ``--set``, overriding one the target implies.
        profile: ``--profile``, overriding one the target implies.
        json_out: Print the findings as JSON, as ``--json`` does locally.
        required: ``True`` when the reader asked for the daemon, so an
            unreachable one is reported here.  ``False`` when the caller
            has a refusal of its own to print instead.

    Returns:
        An exit code (1 when any finding is an error), or ``None`` when no
        daemon was reached and ``required`` is ``False``.  Findings are never
        reported unless a validator actually ran.
    """
    from pathlib import Path

    from . import findings as _findings

    p = Path(target)
    if p.is_file() and not _findings.is_canonical_profile_layout(p.resolve()):
        print(f"jaato-scaffold validate: {target} is a standalone profile "
              f"file, which is validated in-process; a daemon validates a "
              f"workspace.  Install jaato-server here, or put the file under "
              f"<workspace>/.jaato/profiles/ and validate the workspace.",
              file=sys.stderr)
        return 2
    workspace, derived_set, derived_name = _findings.resolve_target(target)
    profile_set = profile_set or derived_set
    profile = profile or derived_name

    if not required and not daemon_is_listening(default_socket_path()):
        return None
    answer = ask_daemon_validate(asked if isinstance(asked, str) else None,
                                 workspace, profile_set, profile)
    if not answer.reached:
        if not required:
            return None
        print(answer.unreachable, file=sys.stderr)
        return 2
    if not answer.ok:
        print(answer.error or "the daemon's validator did not run",
              file=sys.stderr)
        print(validation_attribution(answer), file=sys.stderr)
        return 2

    return _print_validation(answer, profile, profile_set, json_out)


def _print_validation(answer: RemoteValidation, profile: Optional[str],
                      profile_set: Optional[str], json_out: bool) -> int:
    """Print a daemon's findings as a local run prints its own; the exit code."""
    from . import findings as _findings

    if json_out:
        print(json.dumps(answer.findings, indent=2))
    else:
        if not answer.findings:
            print(_findings.clean_line(answer.scope or
                                       _findings.scope_label(profile),
                                       profile_set))
        for d in answer.findings:
            print(_findings.format_finding(d))
        print()
        print(validation_attribution(answer))
    return 1 if answer.errors else 0
