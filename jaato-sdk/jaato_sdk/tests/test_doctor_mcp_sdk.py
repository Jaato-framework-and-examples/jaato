"""The doctor's mcp-sdk check — the guard against the NEXT decode-seam move.

``mcp[cli]`` is unpinned, so this environment's SDK generation is whatever pip
resolved.  The MCP plugin filters a server's stdout log lines out before
decoding them, and where that decode lives moved between mcp 1.x and 2.x.
Reading only the 1.x seam crashed the whole MCP thread on any 2.x install, and
nothing in the diagnostic surface said so.

The check reports on the seam the plugin found.  It is a WARN, never a FAIL:
with no seam MCP still works, only the noise filter is lost.
"""

import pytest

from jaato_sdk.doctor import check_mcp_sdk, PASS, WARN


def _one():
    checks = check_mcp_sdk()
    assert len(checks) == 1
    return checks[0]


def test_check_never_fails_and_names_itself():
    c = _one()
    assert c.name == "mcp sdk"
    assert c.status in (PASS, WARN)      # never FAIL — MCP works without it


def test_names_the_seam_when_one_is_found():
    pytest.importorskip("mcp")
    pytest.importorskip("shared.plugins.mcp.plugin")
    c = _one()
    if c.status is not PASS:
        pytest.skip(f"no seam in this environment: {c.detail}")
    assert "mcp " in c.detail                       # the resolved version
    assert ("'model'" in c.detail or "'adapter'" in c.detail)


def test_agrees_with_the_plugin_about_the_installed_sdk():
    """The check must report what the plugin will actually do, not a guess."""
    pytest.importorskip("mcp")
    plugin = pytest.importorskip("shared.plugins.mcp.plugin")
    from mcp import types as mcp_types

    seam = plugin.detect_jsonrpc_seam(mcp_types)
    c = _one()
    if seam is None:
        assert c.status is WARN
    else:
        assert c.status is PASS and f"'{seam}'" in c.detail
