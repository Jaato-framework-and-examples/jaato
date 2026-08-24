"""The ``(ok, payload)`` executor contract must survive the daemon forward.

Plugin executors signal domain failure by returning a ``(False, payload)``
2-tuple.  That contract is carried by a Python ``tuple`` — and JSON has no
tuple type.  Before this guard, ``daemon.plugin_execute`` returned the raw
tuple as ``envelope.result``; it crossed the wire as ``[False, {...}]``;
``split_executor_result`` (which gates on ``isinstance(x, tuple)``) fell
through to ``return True, executor_result``; the flag demoted from FLAG to
DATA; and the payload reached the model as ``{"result": [False, {...}]}``.

A failing forwarded tool was then invisible to BOTH consumer-side checks:

- ``is_error`` — False, because the ``ok`` flag never surfaced.
- ``is_error_result`` — False, because the body check reads the OUTER dict,
  which had exactly one key, ``result``.

Found live by the cascade-coordination probe, which reads result BODY TEXT
rather than the flags; every flag-based consumer saw success.
"""

import asyncio
import json

import pytest

from server.runner.envelope import (
    TOOL_ERROR_TYPE,
    ExecutorOutcome,
    RequestEnvelope,
    ResponseEnvelope,
)
from server.runner_rpc_server import RunnerRPCServer
from shared.tool_result_builder import (
    normalize_result_dict,
    split_executor_result,
)
from jaato_sdk.plugins.model_provider.types import tool_result_is_error


FAILURE = {"status": "error", "error": "the executor said no"}


def _dispatch(handler, method="daemon.plugin_execute", args=None):
    """Run *handler* through the REAL dispatcher and a REAL JSON round-trip.

    The JSON hop is the point of the test — it is the step that destroyed
    the contract — so this must never be stubbed out.
    """
    srv = RunnerRPCServer()
    srv.register(method, handler)
    env = asyncio.run(
        srv.dispatch(RequestEnvelope(id=1, method=method, args=args or {}))
    )
    return ResponseEnvelope.from_dict(json.loads(json.dumps(env.to_dict())))


def _client_translate(env):
    """The ``rpc_client.daemon_plugin_execute`` return contract."""
    if (
        not env.ok
        and env.error is not None
        and env.error.type == TOOL_ERROR_TYPE
    ):
        return (False, env.result)
    if not env.ok or env.error is not None:
        raise AssertionError(f"unexpected transport failure: {env.error}")
    return env.result


def test_domain_failure_reaches_the_model_as_a_failure():
    """The whole chain: executor tuple → wire → model-facing result.

    Asserts on BOTH error signals, because the defect defeated both at
    once and either one alone would have passed for the wrong reason.
    """
    async def handler(args):
        ok, payload = split_executor_result((False, FAILURE))
        return ExecutorOutcome(ok=ok, payload=payload)

    env = _dispatch(handler)
    assert env.ok is False
    assert env.error.type == TOOL_ERROR_TYPE

    ok, data = split_executor_result(_client_translate(env))
    final = normalize_result_dict(data, ok=ok)

    assert ok is False, "the executor contract flag must survive the wire"
    assert tool_result_is_error(final), "the body check must see the failure"
    assert final == FAILURE, "payload must arrive intact"


def test_payload_is_not_double_wrapped():
    """The specific observed corruption, pinned by shape.

    ``{"result": [False, {...}]}`` is the fingerprint: an ``(ok, payload)``
    pair serialized whole into the result slot instead of being unpacked.
    """
    async def handler(args):
        return ExecutorOutcome(ok=False, payload=FAILURE)

    env = _dispatch(handler)
    ok, data = split_executor_result(_client_translate(env))
    final = normalize_result_dict(data, ok=ok)

    assert "result" not in final or final.get("result") != [False, FAILURE]
    assert not isinstance(data, list), (
        "payload arrived as a LIST — the tuple was serialized whole"
    )


def test_success_still_passes_through_unchanged():
    """The ok=True path must be untouched by the failure plumbing."""
    payload = {"status": "ok", "siblings": [{"name": "sibling-a"}]}

    async def handler(args):
        return ExecutorOutcome(ok=True, payload=payload)

    env = _dispatch(handler)
    assert env.ok is True and env.error is None

    ok, data = split_executor_result(_client_translate(env))
    assert ok is True and data == payload
    assert tool_result_is_error(normalize_result_dict(data, ok=ok)) is False


def test_plain_return_value_still_ok():
    """Handlers that never opted into the contract keep their semantics."""
    async def handler(args):
        return {"answer": 42}

    env = _dispatch(handler)
    assert env.ok is True and env.result == {"answer": 42}


def test_bare_tuple_from_a_handler_is_rejected_loudly():
    """A tuple cannot cross the wire intact, so it must never be shipped.

    Zero-distance enforcement: the next handler that expresses the
    contract BY SHAPE fails here rather than silently producing a
    success-looking envelope. A docstring would not have stopped it —
    the original bug was written against a module whose docstring
    described the correct behaviour.
    """
    async def handler(args):
        return (False, FAILURE)

    env = _dispatch(handler)
    assert env.ok is False
    assert env.error.type == "HandlerContractError"
    assert "ExecutorOutcome" in env.error.message


def test_crash_is_distinguishable_from_domain_failure():
    """A raised exception must NOT masquerade as ``(False, payload)``.

    The discriminator is ``error.type``: real exception class name for a
    crash, ``ToolError`` for a declared domain failure.  Without this the
    client cannot tell "the executor returned False" from "the executor
    exploded", and would hand the model a payload that does not exist.
    """
    async def handler(args):
        raise RuntimeError("kaboom")

    env = _dispatch(handler)
    assert env.ok is False
    assert env.error.type == "RuntimeError"
    assert env.error.type != TOOL_ERROR_TYPE

    with pytest.raises(AssertionError):
        _client_translate(env)
