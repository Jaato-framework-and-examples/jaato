"""Pin test for the cascade_driver_id kwarg on create_headless_session.

Phase 2 PR-B (#167) added ``cascade_driver_id`` to the IPC path
(``_create_session_impl``) but missed ``create_headless_session``,
which premium's ``ActionContext.create_session`` calls directly.
Reactor-spawned sessions failed with ``TypeError: ... got an
unexpected keyword argument "cascade_driver_id"`` until PR #172
closed the wiring gap.

This pin guards against:

  1. Future signature changes that drop the kwarg
  2. Future refactors that forget to forward to ``_create_session_impl``

Both surfaces are tested via a lightweight stub of
``_create_session_impl`` so the test doesn't need to construct a
real JaatoServer / spawn a runner.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from server.session_manager import SessionManager


def test_create_headless_session_signature_accepts_cascade_driver_id() -> None:
    """The kwarg is in the signature.  Catches a future signature
    revert at lint speed without needing to instantiate SessionManager."""
    sig = inspect.signature(SessionManager.create_headless_session)
    assert "cascade_driver_id" in sig.parameters, (
        "create_headless_session signature missing 'cascade_driver_id' "
        f"kwarg.  Signature: {sig}"
    )
    param = sig.parameters["cascade_driver_id"]
    assert param.default is None, (
        f"cascade_driver_id default should be None for backward "
        f"compat (got {param.default!r})"
    )


def test_create_headless_session_forwards_cascade_driver_id_to_impl() -> None:
    """``create_headless_session`` must forward cascade_driver_id to
    its inner ``create_session`` call.  Without this, the kwarg is
    accepted but silently dropped before reaching PoolManager."""
    # Lightweight: skip __init__ entirely; patch create_session on
    # the instance so we only test the forwarding logic.
    sm = SessionManager.__new__(SessionManager)
    sm._HEADLESS_CLIENT_ID = "_headless"  # noqa: SLF001 — class attr access

    with patch.object(sm, "create_session", return_value="new-sid-1") as mock_create:
        result = sm.create_headless_session(
            profile_name="reactor-profile",
            agent_name="cascade_after_discovery",
            workspace_path="/tmp/ws",
            cascade_driver_id="cascade-abc123",
        )

    assert result == "new-sid-1"
    mock_create.assert_called_once()
    kwargs = mock_create.call_args.kwargs
    assert kwargs["cascade_driver_id"] == "cascade-abc123", (
        f"create_headless_session did NOT forward cascade_driver_id "
        f"to create_session; got kwargs={kwargs}"
    )


def test_create_headless_session_default_cascade_driver_id_is_none() -> None:
    """The standalone-session case (no cascade): omitting the kwarg
    forwards None to create_session — preserves Phase 1 semantics."""
    sm = SessionManager.__new__(SessionManager)
    sm._HEADLESS_CLIENT_ID = "_headless"

    with patch.object(sm, "create_session", return_value="new-sid-2") as mock_create:
        sm.create_headless_session(
            profile_name="reactor-profile",
            workspace_path="/tmp/ws",
        )

    kwargs = mock_create.call_args.kwargs
    assert kwargs.get("cascade_driver_id") is None
