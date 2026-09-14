"""The stdout noise filter must find its seam on BOTH mcp SDK generations.

An MCP server's stdout is also its JSON-RPC channel, so a server that logs
there feeds its log lines to the decoder.  The plugin filters them out before
decoding — but *where* the decode happens moved between SDK releases:

    mcp 1.x   ``types.JSONRPCMessage`` is a Pydantic model
              -> seam: ``JSONRPCMessage.model_validate_json``
    mcp 2.x   ``JSONRPCMessage`` is a PEP 604 ``UnionType`` with no
              ``model_validate_json`` at all
              -> seam: ``types.jsonrpc_message_adapter.validate_json``

Reading only the 1.x seam raised ``AttributeError: 'types.UnionType' object
has no attribute 'model_validate_json'`` out of the MCP thread's
``run_until_complete`` on any environment that resolved ``mcp>=2`` — which an
unpinned ``mcp[cli]`` dependency does by default.  Every MCP server in the
workspace was then unreachable, and the only evidence was a traceback in the
MCP thread's log.

These tests drive ``_ensure_mcp_patch`` against a STUB ``mcp.types`` of each
shape, so both generations are covered whichever one is installed here.
"""

import json
import logging
import types as pytypes

import pytest

from shared.plugins.mcp.plugin import MCPToolPlugin, SkipMessage


NOISE = "hello from a chatty server"
GOOD = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})


def _plugin():
    """A plugin instance with just enough state for the patch path."""
    p = MCPToolPlugin.__new__(MCPToolPlugin)
    p._mcp_patch_applied = False
    p._log_event = lambda *a, **k: None
    return p


class _Model:
    """Stand-in for mcp 1.x's ``JSONRPCMessage`` Pydantic model."""

    @classmethod
    def model_validate_json(cls, json_data, *args, **kwargs):
        return ("decoded", json_data)


class _Adapter:
    """Stand-in for mcp 2.x's ``jsonrpc_message_adapter`` TypeAdapter."""

    def validate_json(self, data, *args, **kwargs):
        return ("decoded", data)


def _types_1x():
    mod = pytypes.SimpleNamespace()
    mod.JSONRPCMessage = _Model
    return mod


def _types_2x():
    mod = pytypes.SimpleNamespace()
    # 2.x's JSONRPCMessage really is a UnionType — the object that has no
    # ``model_validate_json`` and whose absence used to crash the thread.
    mod.JSONRPCMessage = int | str
    mod.jsonrpc_message_adapter = _Adapter()
    return mod


def test_1x_seam_is_the_model_classmethod():
    p = _plugin()
    mod = _types_1x()
    assert p._install_jsonrpc_filter(mod) == "model"

    assert mod.JSONRPCMessage.model_validate_json(GOOD) == ("decoded", GOOD)
    with pytest.raises(SkipMessage):
        mod.JSONRPCMessage.model_validate_json(NOISE)


def test_2x_seam_is_the_type_adapter():
    p = _plugin()
    mod = _types_2x()
    assert p._install_jsonrpc_filter(mod) == "adapter"

    assert mod.jsonrpc_message_adapter.validate_json(GOOD) == ("decoded", GOOD)
    with pytest.raises(SkipMessage):
        mod.jsonrpc_message_adapter.validate_json(NOISE)


def test_2x_union_type_is_not_mistaken_for_the_model_seam():
    """The exact regression: reading ``.model_validate_json`` off a UnionType."""
    p = _plugin()
    mod = _types_2x()
    # No AttributeError, and it does NOT settle on the 1.x answer.
    assert p._install_jsonrpc_filter(mod) == "adapter"


def test_unknown_sdk_shape_reports_no_seam_instead_of_raising():
    """A future rename must cost the noise filter, never the MCP servers."""
    p = _plugin()
    assert p._install_jsonrpc_filter(pytypes.SimpleNamespace()) is None


def test_skip_message_is_a_value_error():
    """mcp 2.x's ``_parse_line`` catches ``ValueError`` and only ``ValueError``.

    A sentinel outside that hierarchy escapes the stdout reader and takes the
    connection down — the opposite of what the filter exists to do.
    """
    assert issubclass(SkipMessage, ValueError)


@pytest.mark.parametrize("line", ["", "   ", NOISE, "{not json", '{"a": 1}',
                                  '[1,2,3]', '{"jsonrpc": "1.0"}'])
def test_non_jsonrpc_lines_are_skipped(line):
    p = _plugin()
    mod = _types_2x()
    p._install_jsonrpc_filter(mod)
    with pytest.raises(SkipMessage):
        mod.jsonrpc_message_adapter.validate_json(line)


def test_bytes_are_decoded_before_inspection():
    p = _plugin()
    mod = _types_2x()
    p._install_jsonrpc_filter(mod)
    assert mod.jsonrpc_message_adapter.validate_json(GOOD.encode()) == (
        "decoded", GOOD.encode())
    with pytest.raises(SkipMessage):
        mod.jsonrpc_message_adapter.validate_json(NOISE.encode())


def test_2x_logger_filter_drops_only_our_sentinel():
    """The 2.x silencer must not swallow real parse failures.

    mcp 2.x reports a decode failure with ``logger.exception`` on its own
    module logger, so the filter is installed there — and a record carrying
    anything but :class:`SkipMessage` still reaches the operator.
    """
    stdio = pytest.importorskip("mcp.client.stdio")
    logger = logging.getLogger(stdio.__name__)
    before = list(logger.filters)
    try:
        _plugin()._silence_2x_logger()
        installed = [f for f in logger.filters if f not in before]
        assert installed, "no filter installed on mcp's stdio logger"
        drop = installed[-1]

        def rec(exc):
            r = logging.LogRecord(stdio.__name__, logging.ERROR, __file__, 1,
                                  "boom", (), (exc, exc("x"), None) if exc else None)
            return r

        assert drop(rec(SkipMessage)) is False        # ours — dropped
        assert drop(rec(ValueError)) is True          # a real parse error
        assert drop(rec(None)) is True                # not an exception record
    finally:
        for f in list(logger.filters):
            if f not in before:
                logger.removeFilter(f)


def test_patch_is_marked_applied_even_when_no_seam_exists():
    """A build we cannot patch must not be re-probed on every connect."""
    p = _plugin()
    p._ensure_mcp_patch()
    assert p._mcp_patch_applied is True


def test_an_unpatchable_sdk_never_kills_the_caller(monkeypatch):
    """This method used to be able to take the MCP thread down, and did.

    The filter is a convenience; MCP works without it.  So no shape a future
    SDK grows — an adapter whose ``validate_json`` cannot be assigned, a
    module that raises on attribute access — may cost the operator their
    servers again.  Driven through a REAL unpatchable adapter rather than a
    stubbed method, so the escape hatch is exercised, not asserted.
    """
    class Frozen:
        """A TypeAdapter stand-in whose attributes cannot be replaced."""
        __slots__ = ()

        def validate_json(self, data, *a, **k):
            return data

    mod = pytypes.SimpleNamespace()
    mod.JSONRPCMessage = int | str
    mod.jsonrpc_message_adapter = Frozen()

    p = _plugin()
    # It really is unpatchable ...
    with pytest.raises(AttributeError):
        p._install_jsonrpc_filter(mod)

    # ... and _ensure_mcp_patch survives it, logging instead of raising.
    logged = []
    p2 = _plugin()
    p2._log_event = lambda level, msg, **k: logged.append((level, msg))
    monkeypatch.setattr(p2, "_install_jsonrpc_filter",
                        lambda _m: (_ for _ in ()).throw(AttributeError("frozen")))
    p2._ensure_mcp_patch()                    # must not raise
    assert p2._mcp_patch_applied is True
    assert any("noise filter" in m for _, m in logged)
