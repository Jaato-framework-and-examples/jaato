"""Regression pins for ``envelope.session_env`` non-leakage.

``SessionInitEnvelope.session_env`` carries plaintext secrets (real
API keys, database passwords) resolved daemon-side from ``pass://``
/ ``vault://`` URIs.  The wire path (daemon → runner socketpair) is
private (FD-pass, not filesystem-visible), but any code that logs,
persists, or forwards envelope content to clients would silently
leak those secrets.

The audit behind PR #92 (the Y fix that introduced
``envelope.session_env``) verified that the current codebase has no
such leakage paths.  These tests pin the audit findings so a future
PR can't accidentally widen the exposure.

If a future change makes one of these assertions fail, that change
is leaking session_env somewhere it shouldn't.  The fix is either:
  - Remove the new exposure path (preferred); OR
  - Redact ``session_env`` before the new code path emits / persists.

These are NOT smoke tests for whether session_env works end-to-end —
that's in ``test_runner_session_workspace_env.py``.  These tests
deliberately verify the ABSENCE of session_env from sensitive
surfaces.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest

from shared.session_envelope import SessionInitEnvelope


_REDACTABLE_VALUE = "PLAINTEXT_SECRET_SHOULD_NEVER_LEAK"


def _envelope_with_secret() -> SessionInitEnvelope:
    """Build a SessionInitEnvelope carrying a recognizable secret in
    ``session_env``.  Tests scan persisted/logged/emitted output for
    this exact string."""
    return SessionInitEnvelope(
        session_id="sess-leak-pin",
        workspace_path="/tmp/ws",
        profile_name="test-profile",
        provider_name="anthropic",
        model_name="claude-sonnet-4-6",
        session_env={"ZHIPUAI_API_KEY": _REDACTABLE_VALUE},
    )


# ----------------------------------------------------------------------
# Wire round-trip: to_dict/from_dict carries session_env (this is the
# load-bearing path — daemon → runner over socketpair).  Pin that the
# field IS present on the wire so we don't regress the fix.
# ----------------------------------------------------------------------


def test_wire_round_trip_carries_session_env() -> None:
    env = _envelope_with_secret()
    serialized = env.to_dict()
    assert serialized["session_env"] == {"ZHIPUAI_API_KEY": _REDACTABLE_VALUE}
    decoded = SessionInitEnvelope.from_dict(serialized)
    assert decoded.session_env == {"ZHIPUAI_API_KEY": _REDACTABLE_VALUE}


# ----------------------------------------------------------------------
# Persistence: SessionRecord / WaypointRecord / subagent serialization
# must NOT include session_env.  The audit confirmed envelope is in-
# memory only; these tests pin the contract.
# ----------------------------------------------------------------------


def test_session_record_does_not_serialize_session_env() -> None:
    """The persisted ``SessionRecord`` dataclass in
    ``server.session_manager`` (the disk format under
    ``<workspace>/.jaato/sessions/<id>.json``) must not carry
    envelope content."""
    import server.session_manager as sm

    # SessionRecord is the persisted shape — fail loudly if it grows
    # an envelope-content field by accident.
    if hasattr(sm, "SessionRecord"):
        fields = {f for f in dir(sm.SessionRecord) if not f.startswith("_")}
        forbidden = {"session_env", "envelope"}
        leaks = forbidden & fields
        assert not leaks, (
            f"SessionRecord grew envelope-leaking fields: {leaks}.  "
            f"If this is intentional, add explicit redaction before "
            f"persistence."
        )


def test_file_session_state_serialization_drops_session_env(tmp_path) -> None:
    """``shared.plugins.session.file_session`` round-trips
    SessionState to disk.  Verify the serialized JSON does NOT
    contain a marker secret value we plant via the envelope."""
    try:
        from shared.plugins.session.file_session import FileSessionPlugin
    except ImportError:
        pytest.skip("FileSessionPlugin not importable in test env")

    # File-session writes SessionState to disk; SessionState should
    # not contain envelope/session_env.  We can't fully drive the
    # plugin without a runtime; instead, scan the module for any
    # serialize-or-dump function that references envelope.session_env.
    import inspect
    src = inspect.getsource(FileSessionPlugin)
    assert "session_env" not in src, (
        "FileSessionPlugin source references session_env — verify "
        "it's only used to populate state, not persist it."
    )


def test_waypoint_serialization_drops_session_env() -> None:
    """Waypoint snapshots (``shared.plugins.waypoint.manager``)
    must not persist envelope.session_env."""
    try:
        from shared.plugins.waypoint import manager as wp_manager
    except ImportError:
        pytest.skip("waypoint manager not importable")

    import inspect
    src = inspect.getsource(wp_manager)
    # The waypoint serializer writes per-session waypoint records.
    # If it ever references session_env, that's a leakage path.
    assert "session_env" not in src, (
        "waypoint manager source references session_env — verify "
        "it's not persisting envelope.session_env to disk."
    )


# ----------------------------------------------------------------------
# Event surface: client-facing events must not carry session_env.
# Audit verified ProfileSummary.env_var_names exposes only KEYS, not
# values; these tests pin that no Event class has a session_env field.
# ----------------------------------------------------------------------


def test_no_event_class_declares_session_env() -> None:
    """Scan all event classes in jaato-sdk for any field named
    session_env / env_overrides / envelope.  These names indicate
    a path where envelope content would reach a client."""
    try:
        from jaato_sdk import events
    except ImportError:
        pytest.skip("jaato-sdk events module not importable")

    import inspect
    forbidden_field_names = {"session_env", "env_overrides", "envelope"}
    leaks: list = []
    for name, obj in inspect.getmembers(events, inspect.isclass):
        if not name.endswith("Event") and not name.endswith("Request"):
            continue
        # Pydantic models expose fields via ``model_fields``.
        model_fields = getattr(obj, "model_fields", None)
        if model_fields:
            for field_name in model_fields:
                if field_name in forbidden_field_names:
                    leaks.append(f"{name}.{field_name}")
    assert not leaks, (
        f"Event classes declare envelope-leaking fields: {leaks}.  "
        f"If intentional, add explicit redaction in the event "
        f"emission path."
    )


# ----------------------------------------------------------------------
# Logging: bootstrap_session log lines must not include session_env
# values.  Audit found logging is safe today; this pins it.
# ----------------------------------------------------------------------


def test_bootstrap_session_logs_count_not_values(caplog) -> None:
    """When ``_apply_envelope_session_env`` applies values to
    ``os.environ``, the log line includes a COUNT, not the values."""
    import logging
    from server.runner.session import _apply_envelope_session_env

    env = _envelope_with_secret()
    caplog.set_level(logging.INFO, logger="server.runner.session")

    try:
        _apply_envelope_session_env(env)
        # Re-run via bootstrap to exercise the log line.  We can't
        # easily run full bootstrap here; instead, scan the source
        # of the bootstrap step 1b for the log statement and verify
        # it does NOT include the values dict.
        import inspect
        from server.runner import session as session_mod
        src = inspect.getsource(session_mod.bootstrap_session)
        # The log statement should use a COUNT (len()) of the dict,
        # not the dict itself.  Verify by checking the source: the
        # log call shouldn't %-format the dict.
        assert "%d session env keys" in src, (
            "bootstrap_session step 1b log statement should log key "
            "COUNT, not the dict itself"
        )
        # Verify the literal secret never appears in caplog.text.
        assert _REDACTABLE_VALUE not in caplog.text
    finally:
        # Clean up the env var we set.
        os.environ.pop("ZHIPUAI_API_KEY", None)


def test_envelope_to_dict_called_only_for_wire_serialization() -> None:
    """The audit found ``envelope.to_dict()`` is called only by the
    RPC wire serialization (private socketpair).  If a future log
    line starts dumping ``envelope.to_dict()`` it would print
    session_env values.  Lightweight pin via grep-like source scan."""
    import glob

    # Grep server/ + jaato-sdk/ for ``envelope.to_dict`` calls; verify
    # they're under runner_rpc_client.py / runner/rpc.py (wire) only.
    repo = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    hits: list = []
    for pattern in (
        os.path.join(repo, "jaato-server/**/*.py"),
        os.path.join(repo, "jaato-sdk/**/*.py"),
    ):
        for path in glob.glob(pattern, recursive=True):
            # Skip tests + build artifacts.
            if "/tests/" in path or "/build/" in path:
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                continue
            if "envelope.to_dict()" in content or "env.to_dict()" in content:
                hits.append(path)

    # Allowlist: known-safe wire-serialization sites.
    allowed_basenames = {
        "runner_rpc_client.py",
        "rpc.py",  # runner/rpc.py
        "session_envelope.py",  # docstring or self-reference
        "runner_spawn.py",  # bootstrap_session_threadsafe constructs + serializes
    }
    leaks: list = []
    for path in hits:
        basename = os.path.basename(path)
        if basename not in allowed_basenames:
            leaks.append(path)
    assert not leaks, (
        f"envelope.to_dict() called outside known wire-serialization "
        f"sites: {leaks}.  These callers would serialize session_env; "
        f"add to the allowlist if confirmed safe, or redact."
    )
