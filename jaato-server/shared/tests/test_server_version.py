"""Tests for server version reporting and client version checking."""

import pytest
from unittest.mock import patch

from jaato_sdk.events import ConnectedEvent, ClientType


class TestServerVersionInConnectedEvent:
    """Verify that server_version round-trips through ConnectedEvent."""

    def test_server_version_in_server_info(self):
        """ConnectedEvent should carry server_version in server_info."""
        event = ConnectedEvent(
            protocol_version="1.0",
            server_info={
                "client_id": "test_1",
                "server_version": "0.2.27",
            },
        )
        assert event.server_info["server_version"] == "0.2.27"

    def test_server_version_absent_in_legacy_server(self):
        """Old servers won't include server_version — should be absent."""
        event = ConnectedEvent(
            protocol_version="1.0",
            server_info={"client_id": "test_1"},
        )
        assert event.server_info.get("server_version") is None


class TestGetServerVersion:
    """Verify _get_server_version reads from package metadata."""

    def test_ipc_get_server_version(self):
        from server.ipc import _get_server_version
        version = _get_server_version()
        # Must be a non-empty dotted version string
        assert isinstance(version, str)
        parts = version.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts)

    def test_websocket_get_server_version(self):
        from server.websocket import _get_server_version
        version = _get_server_version()
        assert isinstance(version, str)
        parts = version.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts)


class TestIncompatibleServerError:
    """Verify IncompatibleServerError attributes and message.

    The contract pivoted from package-version to wire-protocol-version
    in the gap-5 work — see ``docs/sdk-protocol-versioning.md``.
    """

    def test_attributes(self):
        from jaato_sdk.client.ipc import IncompatibleServerError
        # Constructor: (server_protocol, min_protocol, server_version=None)
        err = IncompatibleServerError("2.0", "1.0", server_version="0.2.10")
        assert err.server_protocol == "2.0"
        assert err.min_protocol == "1.0"
        assert err.server_version == "0.2.10"
        # Pre-1.0 alias kept for callers that read .min_version.
        assert err.min_version == "1.0"
        assert "2.0" in str(err)
        assert "1.0" in str(err)
        # Daemon package version surfaces in the diagnostic message.
        assert "0.2.10" in str(err)

    def test_is_exception(self):
        from jaato_sdk.client.ipc import IncompatibleServerError
        assert issubclass(IncompatibleServerError, Exception)


class TestIPCClientServerVersion:
    """Verify IPCClient stores and exposes server_version."""

    def test_server_version_none_before_connect(self):
        from jaato_sdk.client.ipc import IPCClient
        client = IPCClient(socket_path="/tmp/test.sock", auto_start=False, client_type=ClientType.TERMINAL)
        assert client.server_version is None


class TestRecoveryClientClassifiesIncompatibleAsPermanent:
    """Verify recovery client won't retry IncompatibleServerError."""

    def test_classify_incompatible_as_permanent(self):
        from jaato_sdk.client.recovery import IPCRecoveryClient
        from jaato_sdk.client.ipc import IncompatibleServerError

        client = IPCRecoveryClient(
            socket_path="/tmp/test.sock",
            auto_start=False,
            client_type=ClientType.TERMINAL,
        )
        # Major-version mismatch: incompatible wire shape.
        err = IncompatibleServerError("2.0", "1.0", server_version="0.5.0")
        assert client._classify_error(err) == "permanent"

    def test_classify_connection_refused_as_transient(self):
        """Sanity check: normal errors are still transient."""
        from jaato_sdk.client.recovery import IPCRecoveryClient

        client = IPCRecoveryClient(
            socket_path="/tmp/test.sock",
            auto_start=False,
            client_type=ClientType.TERMINAL,
        )
        err = ConnectionRefusedError("Connection refused")
        assert client._classify_error(err) == "transient"


class TestRecoveryClientServerVersionProperty:
    """Verify recovery client delegates server_version to inner client."""

    def test_server_version_none_without_client(self):
        from jaato_sdk.client.recovery import IPCRecoveryClient
        client = IPCRecoveryClient(
            socket_path="/tmp/test.sock",
            auto_start=False,
            client_type=ClientType.TERMINAL,
        )
        assert client.server_version is None

    def test_server_version_delegates_to_inner(self):
        from jaato_sdk.client.recovery import IPCRecoveryClient
        from jaato_sdk.client.ipc import IPCClient

        client = IPCRecoveryClient(
            socket_path="/tmp/test.sock",
            auto_start=False,
            client_type=ClientType.TERMINAL,
        )
        # Simulate a connected inner client
        inner = IPCClient(socket_path="/tmp/test.sock", auto_start=False, client_type=ClientType.TERMINAL)
        inner._server_version = "0.2.28"
        client._client = inner
        assert client.server_version == "0.2.28"
