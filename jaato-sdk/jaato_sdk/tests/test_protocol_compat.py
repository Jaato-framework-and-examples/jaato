"""Tests for the wire-protocol compat algorithm.

Pins the semver-flavoured semantics that ``IPCClient`` uses on
connect to refuse incompatible daemons. See
``docs/sdk-protocol-versioning.md`` for the bump policy.
"""

import pytest

from jaato_sdk.client.ipc import (
    IncompatibleServerError,
    _parse_protocol_version,
    _protocol_compatible,
)
from jaato_sdk.events import PROTOCOL_VERSION


class TestParseProtocolVersion:

    def test_basic(self):
        assert _parse_protocol_version("1.0") == (1, 0)
        assert _parse_protocol_version("2.5") == (2, 5)

    def test_extra_components_dropped(self):
        # "1.0.5" tolerated — third component is dropped, not parsed.
        assert _parse_protocol_version("1.0.5") == (1, 0)

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            _parse_protocol_version("1")
        with pytest.raises(ValueError):
            _parse_protocol_version("not-a-version")
        with pytest.raises(ValueError):
            _parse_protocol_version("a.b")


class TestProtocolCompatible:

    def test_exact_match(self):
        assert _protocol_compatible("1.0", "1.0") is True

    def test_server_newer_minor_ok(self):
        """Server has more fields, client ignores them."""
        assert _protocol_compatible("1.5", "1.2") is True
        assert _protocol_compatible("1.99", "1.0") is True

    def test_server_older_minor_refused(self):
        """Server lacks fields the client needs."""
        assert _protocol_compatible("1.0", "1.1") is False
        assert _protocol_compatible("1.2", "1.5") is False

    def test_different_major_refused(self):
        """Major mismatch is incompatible in either direction."""
        assert _protocol_compatible("2.0", "1.0") is False
        assert _protocol_compatible("1.0", "2.0") is False
        assert _protocol_compatible("0.9", "1.0") is False

    def test_malformed_refused(self):
        """Refuse rather than guess on garbage input."""
        assert _protocol_compatible("garbage", "1.0") is False
        assert _protocol_compatible("1.0", "garbage") is False
        assert _protocol_compatible("", "1.0") is False
        assert _protocol_compatible(None, "1.0") is False  # type: ignore[arg-type]


class TestIncompatibleServerErrorMessage:
    """The error message must be actionable — operator sees *why* it failed."""

    def test_major_mismatch_hints_at_wire_break(self):
        err = IncompatibleServerError("2.0", "1.0", server_version="0.7.0")
        msg = str(err)
        assert "major" in msg.lower()
        assert "2" in msg
        assert "1" in msg

    def test_minor_too_low_hints_at_missing_fields(self):
        err = IncompatibleServerError("1.0", "1.5", server_version="0.5.0")
        msg = str(err)
        assert "minor" in msg.lower()
        assert "missing" in msg.lower() or "below" in msg.lower()

    def test_diagnostic_server_version_in_message(self):
        err = IncompatibleServerError("2.0", "1.0", server_version="0.7.0")
        assert "0.7.0" in str(err)

    def test_unknown_server_version(self):
        """When daemon didn't report a package version, error stays useful."""
        err = IncompatibleServerError("2.0", "1.0", server_version=None)
        assert "unknown" in str(err)

    def test_attributes_match_old_api(self):
        """``min_version`` alias kept for pre-1.0 callers."""
        err = IncompatibleServerError("2.0", "1.0")
        assert err.server_protocol == "2.0"
        assert err.min_protocol == "1.0"
        assert err.min_version == "1.0"  # legacy alias


class TestProtocolVersionConstant:
    """The shipped PROTOCOL_VERSION must satisfy the SDK's own minimum."""

    def test_protocol_version_format(self):
        major, minor = _parse_protocol_version(PROTOCOL_VERSION)
        assert major >= 1
        assert minor >= 0

    def test_self_compat(self):
        """A daemon at PROTOCOL_VERSION must satisfy a client requiring the same."""
        from jaato_sdk.client.ipc import IPCClient
        assert _protocol_compatible(PROTOCOL_VERSION, IPCClient.MIN_PROTOCOL_VERSION) is True
