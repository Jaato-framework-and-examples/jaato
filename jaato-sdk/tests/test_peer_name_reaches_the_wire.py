"""``peer_name`` must be settable BY A CONSUMER, not merely by the server.

#592 landed the field complete server-side — validation, envelope, persistence
— and it was unreachable through the facade: zero occurrences in jaato-sdk,
no ``--peer-name`` in the ``session.new`` argv parser, and the router's
``create_session`` call did not pass it.  A session could be given an address
only by code that imports ``session_manager`` directly.

Found by the cascade-coordination example on the first commit it was pointed
at, precisely because an example can only use the public surface.  An
in-process harness would have called
``create_session(peer_name="reviewer")`` directly, got a validated address, and
gone GREEN — certifying a path no consumer can reach.

This is #585's defect class on a THIRD axis.  That parity test compares
IPCClient to IPCRecoveryClient; neither of them can see ``SessionManager``
growing a parameter no client can set.
"""
import asyncio
import inspect
from types import SimpleNamespace

import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.client.recovery import IPCRecoveryClient


def _sent_args(**kwargs):
    """Drive the REAL create_session and capture the argv it builds."""
    captured = {}

    async def _send_event(ev):
        captured["command"] = getattr(ev, "command", None)
        captured["args"] = list(getattr(ev, "args", []) or [])
        captured["payload"] = getattr(ev, "payload", None)
        raise _Stop()          # the wire is the assertion; stop before waiting

    class _Stop(Exception):
        pass

    client = IPCClient.__new__(IPCClient)
    client._send_event = _send_event
    try:
        asyncio.run(IPCClient.create_session(client, "n", **kwargs))
    except _Stop:
        pass
    return captured


def test_peer_name_becomes_a_wire_flag():
    args = _sent_args(peer_name="reviewer")["args"]
    assert "--peer-name" in args, (
        "peer_name never reaches the wire — the field exists on the server and "
        "no consumer can set it")
    assert args[args.index("--peer-name") + 1] == "reviewer"


def test_omitting_it_sends_no_flag():
    assert "--peer-name" not in _sent_args()["args"]


def test_both_clients_accept_it():
    for cls in (IPCClient, IPCRecoveryClient):
        assert "peer_name" in inspect.signature(cls.create_session).parameters, (
            f"{cls.__name__}.create_session cannot set a peer address")


def test_the_recovery_client_FORWARDS_it_rather_than_dropping_it():
    """Accepting a kwarg and discarding it passes a signature check while
    changing nothing — the same accept-and-drop shape guarded in #588."""
    seen = {}

    class _Inner:
        async def create_session(self, name=None, **kw):
            seen.update(kw)
            return "sid"

    rc = IPCRecoveryClient.__new__(IPCRecoveryClient)
    rc._client = _Inner()
    rc._check_can_send = lambda: None
    rc._session_id = None
    asyncio.run(rc.create_session("n", peer_name="reviewer"))
    assert seen.get("peer_name") == "reviewer", (
        "the recovery client accepted peer_name and dropped it")


# ------------------------------------------------- the axis that missed this

def test_every_create_session_kwarg_the_server_takes_is_reachable():
    """Server capability vs CLIENT SURFACE — the third parity axis.

    ``test_recovery_client_parity`` compares the two clients to each other, so
    a parameter added to ``SessionManager`` and to neither client looks
    perfectly consistent.  That is exactly how peer_name shipped unreachable.

    Server-side-only params are legitimate (in-process/daemon-extension
    concerns), so they are excused BY NAME with a reason.
    """
    from server.session_manager import SessionManager

    server_params = set(
        inspect.signature(SessionManager._create_session_impl).parameters)
    client_params = set(inspect.signature(IPCClient.create_session).parameters)

    NOT_CLIENT_SETTABLE = {
        "self", "client_id", "created_by", "provisioned", "env_overrides",
        "session_name", "workspace_path", "config_root", "apparmor",
        "initial_session_state", "budget_control", "budget_usage",
        "system_instruction_override", "suppress_base_instructions",
        "inline_profile_data", "profile_name", "agent_name", "agent_params",
    }
    unreachable = server_params - client_params - NOT_CLIENT_SETTABLE
    assert not unreachable, (
        f"SessionManager accepts {sorted(unreachable)} but no IPC consumer can "
        f"set them. Either wire them through session.new + IPCClient, or add "
        f"them to NOT_CLIENT_SETTABLE with a reason."
    )
