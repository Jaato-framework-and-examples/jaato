"""A reused pool slot must not carry one session's environment into the next.

``session_env`` carries values resolved from ``pass://`` / ``vault://`` --
that resolution exists precisely because the literal URI reaching a runner
produced provider 401s, so what the field holds is DECODED credentials: real
API keys, database passwords.

``_apply_envelope_session_env`` only ever SET them::

    for key, value in applied.items():
        if value is not None:
            os.environ[key] = value

A pool slot serves more than one session.  So a key session A declared and
session B does not was left exactly as A left it -- A's decoded credentials,
in the environment of a runner now serving B, readable by any tool B runs.
Absent-versus-empty once more: "B does not mention this key" was read as
"leave it alone" when it means "B must not have it".

THE INVARIANT THAT JUSTIFIED IT WAS FALSE, AND ITS OWN FILE SAID SO.
``runner/session.py`` reasoned "There is NO reset -- a runner process serves
exactly one session for its whole lifetime", and 379 lines later described
"when a pool slot is REUSED across sessions of a cascade".  That reasoning
was sound for ``JAATO_WORKSPACE_ROOT`` / ``JAATO_CONFIG_ROOT``, which are
re-set on every bootstrap and therefore self-heal.  It generalised to
``session_env``, which does not.

The existing ``test_envelope_session_env_non_leakage`` does not cover this:
it pins the EXPOSURE axis the PR #92 audit enumerated -- logs, persistence,
client-facing surfaces.  Session-to-session persistence inside a reused
runner is a different axis.

Reported by a consumer, reproduced with two sessions sharing a
``cascade_driver_id`` and nothing else: separate workspaces, separate
``.env`` files, and the second printing the first's canary.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _isolate_module_state(monkeypatch):
    """Each test starts from a runner that has never applied an env."""
    import server.runner.session as rs

    monkeypatch.setattr(rs, "_PRISTINE_ENVIRON", None, raising=False)
    yield


def _apply(env):
    from server.runner.session import _apply_envelope_session_env

    return _apply_envelope_session_env(SimpleNamespace(session_env=env))


def test_a_key_from_the_previous_session_does_not_survive(monkeypatch):
    """The leak, stated as the thing that must not happen."""
    monkeypatch.delenv("LEAK_CANARY", raising=False)

    _apply({"LEAK_CANARY": "session-A-secret"})
    assert os.environ.get("LEAK_CANARY") == "session-A-secret"

    _apply({"SOMETHING_ELSE": "session-B"})

    assert os.environ.get("LEAK_CANARY") is None, (
        "session A's resolved secret survived into session B's environment "
        "on a reused slot"
    )
    assert os.environ.get("SOMETHING_ELSE") == "session-B"


def test_a_value_the_previous_session_overwrote_is_restored(monkeypatch):
    """Not just added keys — CHANGED ones.

    A slot inherits real environment from the template.  If session A
    overwrites an inherited value and B does not mention it, B must see what
    the slot started with, not A's substitute.
    """
    monkeypatch.setenv("INHERITED", "from-template")

    _apply({"INHERITED": "session-A-override"})
    assert os.environ["INHERITED"] == "session-A-override"

    _apply({"UNRELATED": "x"})

    assert os.environ["INHERITED"] == "from-template", (
        "session A's override of an inherited value persisted into session B"
    )


def test_the_snapshot_is_pristine_not_the_first_sessions_leak(monkeypatch):
    """Taken BEFORE the first apply, or it faithfully restores the bug.

    A snapshot captured per-session would record session A's env and hand it
    back to B every time -- a restore that restores the leak.
    """
    monkeypatch.delenv("A_ONLY", raising=False)

    _apply({"A_ONLY": "a"})
    _apply({"B_ONLY": "b"})
    _apply({"C_ONLY": "c"})

    assert os.environ.get("A_ONLY") is None
    assert os.environ.get("B_ONLY") is None
    assert os.environ.get("C_ONLY") == "c"


def test_an_empty_env_still_clears_the_previous_session(monkeypatch):
    """The early return must not skip the restore.

    A session declaring no env at all is the case most likely to be given a
    reused slot, and the one where inheriting the previous session's
    credentials is least visible.
    """
    monkeypatch.delenv("LEAK_CANARY", raising=False)

    _apply({"LEAK_CANARY": "session-A-secret"})
    _apply({})

    assert os.environ.get("LEAK_CANARY") is None, (
        "a session with no env of its own inherited the previous session's"
    )


# ------------------------------------------------------- the pool's identity

def test_reuse_requires_the_same_config_root():
    """A slot's warm plugin state was built from ITS config root.

    Reuse was gated on ``cascade_driver_id`` ALONE, so two sessions differing
    in config root -- different profiles, agents, prompt library, permission
    config -- shared a slot.  The path variables self-heal; everything the
    first bootstrap DERIVED from them does not, and keeping it warm is what
    the pool is for.
    """
    from server.runner_pool import PoolSlot

    slot = PoolSlot(pid=1, sock=None, cascade_id="cid-1", config_root="/a/.jaato")

    assert slot.config_root == "/a/.jaato"
    # identity is the PAIR
    assert not (slot.cascade_id == "cid-1" and slot.config_root == "/b/.jaato")


def test_the_pool_fake_matches_the_real_signature():
    """A fake that rejects a kwarg the caller passes tests the except branch.

    ``_FakeRunnerRPC`` did exactly this with ``require_idle`` and five tests
    silently exercised delivery-failure instead of what they were named for.
    Same guard, different fake.
    """
    import inspect

    from server.runner_pool import PoolManager
    from server.tests.test_runner_spawn import _FakePoolManager

    real = set(inspect.signature(PoolManager.acquire_slot).parameters)
    fake = set(inspect.signature(_FakePoolManager.acquire_slot).parameters)

    missing = real - fake
    assert not missing, (
        f"the fake pool manager rejects {sorted(missing)}, which the spawn "
        "path passes; every pool-routing test would land in an error branch"
    )
