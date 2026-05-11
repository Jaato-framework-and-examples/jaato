"""Phase 4 §B regression pins: workspace .env → runner env propagation.

Closes the post-§7c gap where workspace `.env` values containing secret
URIs (`pass://`, `vault://`, `awssm://`) stayed unresolved in the
daemon's os.environ at runner-fork time.  Runners were forking without
the resolved values, and the provider plugin's `get_session_env(...)`
lookup got the literal URI back.

Two pins here:

1. ``_resolve_session_env`` is idempotent — pre-spawn call populates
   ``self._session_env``; the same method invoked again by
   ``initialize()`` step 1 is a no-op (no double resolution of
   secrets via re-running pass/vault/etc subprocesses).

2. ``_with_session_env`` applies the resolved values to ``os.environ``
   during the context, restores on exit.  This is what gives the
   ``RunnerSpawner.spawn`` fork() inside the context manager access to
   the resolved env via the inherited ``os.environ.copy()`` in
   ``runner_spawner.py:_compose_env``.

The integration counterpart (real openrouter + pass:// → working
session bootstrap) lives in the harness workspace
``jaato-tui-driven-tests`` per docs/design/phase4_env_propagation_audit.md.
"""

from __future__ import annotations

import os

import pytest

from server.core import JaatoServer


@pytest.fixture
def env_file(tmp_path):
    """Workspace .env file with one literal + one cross-reference."""
    f = tmp_path / ".env"
    f.write_text(
        "PHASE4_FOO=bar-resolved\n"
        # cross-reference resolves against sibling entries
        "PHASE4_BAZ=qux-${PHASE4_FOO}\n"
    )
    return f


def test_resolve_session_env_populates_session_env(env_file, tmp_path):
    """First call reads .env, runs expand_variables, fills _session_env."""
    server = JaatoServer(env_file=str(env_file), workspace_path=str(tmp_path))
    assert server._session_env == {}
    assert server._session_env_resolved is False
    server._resolve_session_env()
    assert server._session_env_resolved is True
    assert server._session_env["PHASE4_FOO"] == "bar-resolved"
    # ${VAR} cross-reference within .env expanded against sibling entries
    assert server._session_env["PHASE4_BAZ"] == "qux-bar-resolved"


def test_resolve_session_env_is_idempotent(env_file, tmp_path):
    """Second call is a no-op — important for the daemon-pre-spawn +
    runner-side initialize() double-call path.  Without idempotency
    the runner would re-resolve secrets (re-running `pass show`,
    re-fetching from vault, etc.), defeating the daemon-side
    resolution and risking divergence if the resolver hits a
    different backend state."""
    server = JaatoServer(env_file=str(env_file), workspace_path=str(tmp_path))
    server._resolve_session_env()
    # Simulate a mutation that should NOT be overwritten by a re-resolve.
    server._session_env["PHASE4_FOO"] = "mutated-after-resolve"
    server._resolve_session_env()
    assert server._session_env["PHASE4_FOO"] == "mutated-after-resolve"


def test_resolve_session_env_no_env_file(tmp_path):
    """env_file=None path: _session_env stays {} but flag still flips
    so subsequent calls (initialize() step 1) are no-ops."""
    server = JaatoServer(env_file=None, workspace_path=str(tmp_path))
    server._resolve_session_env()
    assert server._session_env_resolved is True
    assert server._session_env == {}


def test_with_session_env_propagates_to_environ_after_resolve(
    env_file, tmp_path, monkeypatch,
):
    """Pin the spawn contract: after _resolve_session_env, entering
    _with_session_env() places the resolved values in os.environ so
    a fork() inside the context inherits them.  This is the only
    propagation channel the runner subprocess has post-§7c — the
    envelope.env_overrides field is built but the runner does NOT
    apply it (grep `runner/session.py` = 0 matches for env_overrides
    application).
    """
    # Ensure the keys are NOT in os.environ to start (monkeypatch
    # auto-restores on test teardown).
    monkeypatch.delenv("PHASE4_FOO", raising=False)
    monkeypatch.delenv("PHASE4_BAZ", raising=False)
    assert "PHASE4_FOO" not in os.environ

    server = JaatoServer(env_file=str(env_file), workspace_path=str(tmp_path))
    server._resolve_session_env()

    with server._with_session_env():
        # Inside the context — a fork() here would inherit these values
        assert os.environ["PHASE4_FOO"] == "bar-resolved"
        assert os.environ["PHASE4_BAZ"] == "qux-bar-resolved"

    # After exit — values removed (they weren't in os.environ before)
    assert "PHASE4_FOO" not in os.environ
    assert "PHASE4_BAZ" not in os.environ


def test_with_session_env_restores_previous_value(env_file, tmp_path, monkeypatch):
    """If os.environ had a different value for one of the resolved
    keys before the with-block, _with_session_env must restore the
    previous value on exit (not delete it).  Protects against
    cross-session contamination when two sessions interleave their
    spawn paths."""
    monkeypatch.setenv("PHASE4_FOO", "pre-existing-value")

    server = JaatoServer(env_file=str(env_file), workspace_path=str(tmp_path))
    server._resolve_session_env()

    with server._with_session_env():
        assert os.environ["PHASE4_FOO"] == "bar-resolved"
    assert os.environ["PHASE4_FOO"] == "pre-existing-value"


def test_initialize_step1_treats_pre_resolved_env_as_authoritative(
    env_file, tmp_path,
):
    """Path B end-to-end semantics: when SessionManager calls
    _resolve_session_env() BEFORE the runner-spawn fork, the runner's
    server.initialize() step 1 must NOT clobber the daemon-resolved
    values.  This is the idempotency flag's reason to exist.

    The daemon-side pre-spawn resolution + os.environ apply (via
    _with_session_env) is the only reliable channel for getting
    secret-URI-resolved values into the runner; if initialize()
    re-ran dotenv_values + expand_variables runner-side, a resolver
    misregistration in the runner subprocess would re-introduce the
    literal `pass://...` value and break authentication.
    """
    server = JaatoServer(env_file=str(env_file), workspace_path=str(tmp_path))
    server._resolve_session_env()
    pre_init_snapshot = dict(server._session_env)

    # Mimic the runner-side: initialize() will internally call
    # _resolve_session_env again; the idempotency flag makes it a no-op.
    server._resolve_session_env()
    assert server._session_env == pre_init_snapshot
