"""The facade can set the ``session.new`` confirmation budget (#1450, #899).

``create_session`` has always taken a ``timeout`` (default 60s) bounding the
wait for the daemon to confirm ``session.new``.  Every facade entry point --
``IPCClient.session``, ``IPCRecoveryClient.session``, ``WSClient.session``,
``WSRecoveryClient.session`` -- built its ``create_session`` kwargs from
profile / agent / agent_params / cascade_driver_id alone, so a facade caller
was fixed at 60s.  Under cascade load one confirmation measured 19.4s and
others passed 60s, and drivers dropped the facade to raise it.

``session_timeout`` is forwarded as ``create_session(timeout=)`` when set and
not at all when unset, so the default is ``create_session``'s own.  Each case
drives the real classmethod on a subclass whose transport methods record
instead of connecting, so what is asserted is what reaches
``create_session``.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, List

import pytest

from jaato_sdk.client.ipc import IPCClient
from jaato_sdk.client.recovery import IPCRecoveryClient
from jaato_sdk.client.ws import WSClient, WSRecoveryClient
from jaato_server.shared.tests.reversion import Reversion

_CONV = "jaato-sdk/jaato_sdk/client/convenience.py"
_WS = "jaato-sdk/jaato_sdk/client/ws.py"

REVERSIONS = [
    Reversion(
        target=_CONV,
        find=('    if session_timeout is not None:\n'
              '        kwargs["timeout"] = _check_session_timeout(session_timeout)\n'),
        replace='    _check_session_timeout(session_timeout or 1.0)\n',
        test="test_the_ipc_facades_forward_session_timeout",
        because="the builder validates the budget and forwards nothing",
    ),
    Reversion(
        target=_CONV,
        find=("        cascade_driver_id=cascade_driver_id, "
              "session_timeout=session_timeout)\n"),
        replace="        cascade_driver_id=cascade_driver_id)\n",
        test="test_the_ipc_facades_forward_session_timeout",
        because="open_session accepts session_timeout and drops it",
    ),
    Reversion(
        target=_WS,
        find=("        # The shared builder, so a knob forwarded over IPC is "
              "forwarded here\n"
              "        # too -- ``**_ignored`` above would otherwise swallow it "
              "silently.\n"
              "        create_kwargs = facade_create_kwargs(\n"
              "            profile=profile,\n"
              "            agent=agent,\n"
              "            agent_params=agent_params,\n"
              "            cascade_driver_id=cascade_driver_id,\n"
              "            session_timeout=session_timeout,\n"),
        replace=("        create_kwargs = facade_create_kwargs(\n"
                 "            profile=profile,\n"
                 "            agent=agent,\n"
                 "            agent_params=agent_params,\n"
                 "            cascade_driver_id=cascade_driver_id,\n"),
        test="test_the_ws_facades_forward_session_timeout",
        because="WSClient.session swallows session_timeout into **_ignored",
    ),
    Reversion(
        target=_CONV,
        find=('    if session_timeout is not None:\n'
              '        kwargs["timeout"] = _check_session_timeout(session_timeout)\n'),
        replace=('    kwargs["timeout"] = _check_session_timeout('
                 'session_timeout or 30.0)\n'),
        test="test_unset_session_timeout_leaves_create_sessions_60s_default",
        because="an unset budget no longer means create_session's own default",
    ),
    Reversion(
        target=_CONV,
        find="    if not math.isfinite(session_timeout) or session_timeout <= 0:\n",
        replace="    if not math.isfinite(session_timeout):\n",
        test="test_a_non_positive_session_timeout_is_refused",
        because="a zero or negative budget reaches asyncio.wait_for",
    ),
]


def _recording(base: type) -> type:
    """A subclass of *base* whose transport records instead of connecting."""

    class Recording(base):  # type: ignore[misc, valid-type]
        calls: List[Dict[str, Any]] = []

        async def connect(self, timeout: float = 5.0) -> bool:
            return True

        async def create_session(self, **kwargs: Any) -> str:
            type(self).calls.append(kwargs)
            return "sid-1"

        async def disconnect(self) -> None:
            return None

    Recording.calls = []
    return Recording


def _enter(ctx: Any) -> None:
    async def run() -> None:
        async with ctx:
            pass
    asyncio.run(run())


def _open(cls: type, **kw: Any) -> Dict[str, Any]:
    rec = _recording(cls)
    if issubclass(cls, WSClient) or cls is WSRecoveryClient:
        ctx = rec.session("ws://127.0.0.1:1", profile="p", **kw)
    else:
        ctx = rec.session(socket_path="/nonexistent.sock", auto_start=False,
                          profile="p", **kw)
    _enter(ctx)
    assert len(rec.calls) == 1
    return rec.calls[0]


@pytest.mark.parametrize("cls", [IPCClient, IPCRecoveryClient])
def test_the_ipc_facades_forward_session_timeout(cls: type) -> None:
    """(a) IPC: the budget reaches create_session as ``timeout``."""
    assert _open(cls, session_timeout=240.0)["timeout"] == 240.0


@pytest.mark.parametrize("cls", [WSClient, WSRecoveryClient])
def test_the_ws_facades_forward_session_timeout(cls: type) -> None:
    """(a) WS: the overrides swallow unknown kwargs, so this must be wired."""
    assert _open(cls, session_timeout=240.0)["timeout"] == 240.0


@pytest.mark.parametrize(
    "cls", [IPCClient, IPCRecoveryClient, WSClient, WSRecoveryClient])
def test_unset_session_timeout_leaves_create_sessions_60s_default(
        cls: type) -> None:
    """(b) Unset forwards nothing, so create_session's own 60s applies."""
    sent = _open(cls)
    assert "timeout" not in sent
    for client in (IPCClient, IPCRecoveryClient):
        default = inspect.signature(
            client.create_session).parameters["timeout"].default
        assert default == 60.0


@pytest.mark.parametrize("bad", [0, -1.0, float("inf"), float("nan")])
def test_a_non_positive_session_timeout_is_refused(bad: float) -> None:
    """Refused when the facade is built, before any connection."""
    with pytest.raises(ValueError):
        IPCClient.session(profile="p", auto_start=False, session_timeout=bad)


def test_a_non_number_session_timeout_is_refused() -> None:
    with pytest.raises(TypeError):
        IPCClient.session(profile="p", auto_start=False, session_timeout="60")
    with pytest.raises(TypeError):
        IPCClient.session(profile="p", auto_start=False, session_timeout=True)
