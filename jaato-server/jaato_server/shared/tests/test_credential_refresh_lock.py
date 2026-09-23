"""Guards for issue #683 — a rotating refresh token raced by two refreshers.

THE DEFECT.  An OAuth refresh token *rotates*: the response replaces the
token that bought it, and the old one is void immediately.  So

    load -> stale? -> refresh -> save

is a read-modify-write on a shared file.  Two of them at once produce two
refreshes, each rotating the other's token away, and last-write-wins
decides which superseded credential ends up on disk.  The loser does not
fail then; it fails at its *next* refresh, and the user is logged out
with nothing in the failure naming the cause.

WHY A LOCK ALONE IS NOT THE FIX, and why that shapes these tests.  A lock
turns the race into a queue: N sessions still perform N refreshes, just
politely one after another, and the last one still wins for reasons
nobody can see.  What collapses the queue into a single refresh is
**re-reading the credential after acquiring the lock**.  So the headline
assertion here is not "the writes did not interleave" — it is
``refresh_count == 1``.  A test that only checked interleaving would pass
against a lock-without-re-read, which is the half-fix.

DETERMINISM.  Two processes racing a file lock is exactly the kind of
test that gets "stabilised" with a ``sleep`` and then proves nothing.
Every ordering below is pinned by a **file gate** — one side does not
proceed until the other has provably reached a named point — so the
interleaving the defect needs is forced rather than hoped for.  The only
polling is a gate waiting for a file to appear, and a gate that never
opens fails the test loudly instead of letting it pass by luck.

The decisive gate is this one, and it is what makes the test fail on
``main``:

    winner:  [holds lock] -> touch WINNER_INSIDE -> wait WAITER_LOADED
    waiter:  wait WINNER_INSIDE -> load (sees STALE) -> touch WAITER_LOADED

The waiter is guaranteed to have loaded the *stale* credential while the
winner still holds the lock.  Without the lock it then refreshes -> two
refreshes.  With the lock it blocks, re-reads, finds the winner's token
-> one refresh.  Neither outcome depends on a clock.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from jaato_server.shared.tests.reversion import Reversion

#: The half-fix, put back.  Deleting the re-read leaves the lock fully
#: intact -- which is precisely the shape that reads as fixed and is not:
#: N sessions still perform N refreshes, each rotating the previous one's
#: token away, just politely one at a time.  A race guard that does not
#: notice this is testing serialisation rather than the defect.
REVERSIONS = [
    Reversion(
        target="jaato-server/jaato_server/shared/credential_lock.py",
        find="""        state = load()
        if state is None:
            return None
        if not needs_refresh(state):
            logger.debug(
                "%s credential was refreshed by another holder; reusing it",
                label,
            )
            return state
        try:""",
        replace="""        try:""",
        test="test_two_processes_racing_a_rotating_token_refresh_once",
        because="a lock without the re-read: a queue of refreshes, not one",
    ),
]

from jaato_server.shared.credential_lock import (
    CredentialLockTimeout,
    InvalidGrantError,
    TransientRefreshError,
    classify_refresh_failure,
    credential_lock,
    has_expired,
    lock_path_for,
    refresh_under_lock,
    with_margin,
)

# Fabricated throughout.  Nothing here is, or resembles, a real credential.
FAKE_INITIAL_REFRESH = "fake-refresh-0"
FAKE_INITIAL_ACCESS = "fake-access-0"

#: How long a gate waits before declaring the other side never arrived.
#: Not a timing bet: this is the failure bound, not the success path — a
#: correct run opens every gate as soon as the peer reaches its point.
GATE_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def wait_for_gate(path: Path, timeout: float = GATE_TIMEOUT_SECONDS) -> None:
    """Block until ``path`` exists, or fail the test.

    A happens-before edge, not a delay: the peer creates the file when it
    reaches a named point in its own sequence.
    """
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"gate never opened: {path}")
        time.sleep(0.005)


def open_gate(path: Path) -> None:
    path.write_text("open")


# ---------------------------------------------------------------------------
# a fake rotating credential, in the shape of the real ones
# ---------------------------------------------------------------------------

def write_credential(path: Path, *, access: str, refresh: str, expires_at: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "access_token": access,
        "refresh_token": refresh,
        "expires_at": expires_at,
    }))


def read_credential(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def rotating_refresh(state: dict, *, counter_path: Path, serial: int) -> dict:
    """Stand-in for the provider: mints a new pair and voids the old one.

    Appends to ``counter_path`` so the number of refreshes performed
    across *all* processes is a fact on disk rather than an in-memory
    count that a subprocess could not report.
    """
    with open(counter_path, "a") as fh:
        fh.write(f"{os.getpid()}:{state['refresh_token']}\n")
    return {
        "access_token": f"fake-access-{serial}",
        "refresh_token": f"fake-refresh-{serial}",
        "expires_at": time.time() + 3600,
    }


def refresh_count(counter_path: Path) -> int:
    if not counter_path.exists():
        return 0
    return len([ln for ln in counter_path.read_text().splitlines() if ln.strip()])


# ---------------------------------------------------------------------------
# the cross-process race
# ---------------------------------------------------------------------------

_WORKER = textwrap.dedent(
    '''
    """One contender in the #683 race, driven by file gates."""
    import json, os, sys, time
    sys.path.insert(0, {server_dir!r})

    from jaato_server.shared.credential_lock import refresh_under_lock

    ROLE = sys.argv[1]
    CRED = {cred!r}
    COUNTER = {counter!r}
    WINNER_INSIDE = {winner_inside!r}
    WAITER_LOADED = {waiter_loaded!r}
    RESULT = sys.argv[2]
    GATE_TIMEOUT = {gate_timeout!r}

    def wait_for(p):
        deadline = time.monotonic() + GATE_TIMEOUT
        while not os.path.exists(p):
            if time.monotonic() >= deadline:
                raise SystemExit("gate never opened: " + p)
            time.sleep(0.005)

    def open_gate(p):
        with open(p, "w") as fh:
            fh.write("open")

    loads = {{"n": 0}}

    def load():
        if not os.path.exists(CRED):
            return None
        with open(CRED) as fh:
            state = json.load(fh)
        loads["n"] += 1
        if ROLE == "waiter" and loads["n"] == 1:
            # The decisive gate: this load has returned the STALE
            # credential, and the winner is still holding the lock.
            open_gate(WAITER_LOADED)
        return state

    def save(state):
        tmp = CRED + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh)
        os.replace(tmp, CRED)

    def refresh(state):
        with open(COUNTER, "a") as fh:
            fh.write("%d:%s\\n" % (os.getpid(), state["refresh_token"]))
        if ROLE == "winner":
            open_gate(WINNER_INSIDE)
            wait_for(WAITER_LOADED)
        return {{
            "access_token": "fake-access-%s" % ROLE,
            "refresh_token": "fake-refresh-%s" % ROLE,
            "expires_at": time.time() + 3600,
        }}

    if ROLE == "waiter":
        # Do not start until the winner is provably inside its refresh,
        # holding the lock.
        wait_for(WINNER_INSIDE)

    out = refresh_under_lock(
        credential_path=CRED,
        load=load,
        needs_refresh=lambda s: time.time() > s["expires_at"],
        refresh=refresh,
        save=save,
        label=ROLE,
    )
    with open(RESULT, "w") as fh:
        json.dump(out, fh)
    '''
)


def _server_dir() -> str:
    import jaato_server
    return str(Path(jaato_server.__file__).resolve().parent.parent)


def test_two_processes_racing_a_rotating_token_refresh_once(tmp_path: Path) -> None:
    """Two OS processes, one stale credential, exactly one refresh.

    The heart of #683.  The gates force the waiter to have read the stale
    credential while the winner still holds the lock, so on unfixed code
    both processes refresh and the second rotates the first's token away.
    """
    cred = tmp_path / "cred.json"
    counter = tmp_path / "refreshes.log"
    winner_inside = tmp_path / "gate.winner_inside"
    waiter_loaded = tmp_path / "gate.waiter_loaded"
    worker = tmp_path / "worker.py"

    write_credential(
        cred,
        access=FAKE_INITIAL_ACCESS,
        refresh=FAKE_INITIAL_REFRESH,
        expires_at=time.time() - 10,   # stale
    )
    worker.write_text(_WORKER.format(
        server_dir=_server_dir(),
        cred=str(cred),
        counter=str(counter),
        winner_inside=str(winner_inside),
        waiter_loaded=str(waiter_loaded),
        gate_timeout=GATE_TIMEOUT_SECONDS,
    ))

    procs = []
    for role in ("winner", "waiter"):
        procs.append(subprocess.Popen(
            [sys.executable, str(worker), role, str(tmp_path / f"result.{role}.json")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ))
    outs = [p.communicate(timeout=GATE_TIMEOUT_SECONDS * 3) for p in procs]

    for (role, proc, (out, err)) in zip(("winner", "waiter"), procs, outs):
        assert proc.returncode == 0, f"{role} failed: {err or out}"

    # THE assertion.  A lock without the re-read gives 2.
    assert refresh_count(counter) == 1, (
        "expected exactly one refresh across both processes, got "
        f"{refresh_count(counter)} — the waiter refreshed a token the "
        "winner had already rotated away"
    )

    # And both processes came away with the same, stored credential.
    stored = read_credential(cred)
    results = {
        role: json.loads((tmp_path / f"result.{role}.json").read_text())
        for role in ("winner", "waiter")
    }
    assert results["winner"] == results["waiter"] == stored


def test_the_gate_proves_the_waiter_read_the_stale_credential(tmp_path: Path) -> None:
    """The race test is only meaningful if its gate really fires.

    A concurrency test whose gate quietly never opened would pass while
    proving nothing, so assert the gate files exist after the run.
    """
    cred = tmp_path / "cred.json"
    counter = tmp_path / "refreshes.log"
    winner_inside = tmp_path / "gate.winner_inside"
    waiter_loaded = tmp_path / "gate.waiter_loaded"
    worker = tmp_path / "worker.py"

    write_credential(
        cred, access=FAKE_INITIAL_ACCESS, refresh=FAKE_INITIAL_REFRESH,
        expires_at=time.time() - 10,
    )
    worker.write_text(_WORKER.format(
        server_dir=_server_dir(), cred=str(cred), counter=str(counter),
        winner_inside=str(winner_inside), waiter_loaded=str(waiter_loaded),
        gate_timeout=GATE_TIMEOUT_SECONDS,
    ))
    procs = [
        subprocess.Popen(
            [sys.executable, str(worker), role, str(tmp_path / f"result.{role}.json")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for role in ("winner", "waiter")
    ]
    for p in procs:
        p.communicate(timeout=GATE_TIMEOUT_SECONDS * 3)

    assert winner_inside.exists(), "the winner never entered its refresh"
    assert waiter_loaded.exists(), (
        "the waiter never loaded — the race was never actually staged"
    )


# ---------------------------------------------------------------------------
# the in-process race (the daemon's own shape)
# ---------------------------------------------------------------------------

def test_two_threads_of_one_process_also_refresh_once(tmp_path: Path) -> None:
    """A daemon runs many sessions in ONE process.

    ``fcntl.flock`` attaches to the open file description, and each
    acquisition opens its own, so two threads contend exactly as two
    processes do.  That is the property that lets one mechanism serve
    both, and it is worth pinning: if someone "optimises" the lock into a
    per-process singleton fd, threads stop contending and this fails.
    """
    cred = tmp_path / "cred.json"
    counter = tmp_path / "refreshes.log"
    winner_inside = tmp_path / "gate.winner_inside"
    waiter_loaded = tmp_path / "gate.waiter_loaded"

    write_credential(
        cred, access=FAKE_INITIAL_ACCESS, refresh=FAKE_INITIAL_REFRESH,
        expires_at=time.time() - 10,
    )

    results: dict = {}

    def run(role: str) -> None:
        loads = {"n": 0}

        def load():
            state = read_credential(cred)
            if state is None:
                return None
            loads["n"] += 1
            if role == "waiter" and loads["n"] == 1:
                open_gate(waiter_loaded)
            return state

        def save(state):
            cred.write_text(json.dumps(state))

        def refresh(state):
            with open(counter, "a") as fh:
                fh.write(f"{role}:{state['refresh_token']}\n")
            if role == "winner":
                open_gate(winner_inside)
                wait_for_gate(waiter_loaded)
            return {
                "access_token": f"fake-access-{role}",
                "refresh_token": f"fake-refresh-{role}",
                "expires_at": time.time() + 3600,
            }

        if role == "waiter":
            wait_for_gate(winner_inside)

        results[role] = refresh_under_lock(
            credential_path=cred,
            load=load,
            needs_refresh=lambda s: time.time() > s["expires_at"],
            refresh=refresh,
            save=save,
            label=role,
        )

    threads = [threading.Thread(target=run, args=(r,)) for r in ("winner", "waiter")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=GATE_TIMEOUT_SECONDS * 2)
    for t in threads:
        assert not t.is_alive(), "a thread never finished — the lock deadlocked"

    assert refresh_count(counter) == 1
    assert results["winner"] == results["waiter"] == read_credential(cred)


# ---------------------------------------------------------------------------
# the lock itself
# ---------------------------------------------------------------------------

def test_lock_excludes_a_second_holder(tmp_path: Path) -> None:
    """Holding the lock makes a second acquisition time out, not proceed."""
    cred = tmp_path / "cred.json"
    cred.write_text("{}")
    with credential_lock(cred):
        with pytest.raises(CredentialLockTimeout):
            with credential_lock(cred, timeout=0.05):
                pass


def test_lock_is_released_when_the_body_raises(tmp_path: Path) -> None:
    """A refresh that blew up must not leave every other session stuck."""
    cred = tmp_path / "cred.json"
    cred.write_text("{}")
    with pytest.raises(ValueError):
        with credential_lock(cred):
            raise ValueError("boom")
    # Re-acquirable immediately.
    with credential_lock(cred, timeout=0.5):
        pass


def test_lock_file_is_a_sibling_not_the_credential_itself(tmp_path: Path) -> None:
    """The credential is rewritten by ``os.replace``, which swaps the inode.

    Locking the credential file itself would silently detach the lock
    from the thing it guards the moment somebody saved.
    """
    cred = tmp_path / "cred.json"
    assert lock_path_for(cred) != cred
    assert lock_path_for(cred).name == "cred.json.lock"


def test_no_lock_descriptor_is_held_at_rest(tmp_path: Path) -> None:
    """Nothing may hold a lock across a pool slot's ``fork()``.

    Pool slots fork from a template that has already imported the auth
    plugins; a descriptor held at import time would be inherited by every
    slot.  The invariant is that descriptors live only inside a call.
    """
    from jaato_server.shared import credential_lock as cl

    cred = tmp_path / "cred.json"
    cred.write_text("{}")
    assert cl._HELD_FDS == set()
    with credential_lock(cred):
        assert len(cl._HELD_FDS) == 1
    assert cl._HELD_FDS == set()


def test_importing_this_module_installs_no_fork_hook() -> None:
    """Importing an auth plugin must not put a handler on every fork().

    ``os.register_at_fork`` is a process-global side effect and cannot be
    undone.  ``RunnerSpawner`` forks and calls ``os.setsid()`` in the
    child, and after-fork handlers run *between* those two — so a hook
    installed merely because some module was imported is work on a
    critical path that never asked for it, in a window where this module
    has nothing to protect: no lock has ever been held, so none can be
    inherited.

    Registration is therefore deferred to the first acquisition, which
    is sufficient — a descriptor is only inheritable by a fork that
    happens while one is held.
    """
    import subprocess

    probe = (
        "import sys; sys.path.insert(0, %r); "
        "from jaato_server.shared import credential_lock as cl; "
        "print(cl._FORK_HOOK_INSTALLED)" % _server_dir()
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", (
        "importing shared.credential_lock registered an at-fork handler; "
        "it must wait until a lock is actually taken"
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork only")
def test_a_forked_child_does_not_inherit_a_held_lock(tmp_path: Path) -> None:
    """``fork()`` duplicates descriptors — including one holding a lock.

    A child forked while another thread held the lock would inherit a
    descriptor that *is* holding it, for a lock the child never took.
    The ``register_at_fork`` hook closes those in the child.
    """
    from jaato_server.shared import credential_lock as cl

    cred = tmp_path / "cred.json"
    cred.write_text("{}")
    read_fd, write_fd = os.pipe()

    with credential_lock(cred):
        pid = os.fork()
        if pid == 0:                                  # child
            try:
                held = len(cl._HELD_FDS)
                os.write(write_fd, str(held).encode())
            finally:
                os._exit(0)
        os.close(write_fd)
        seen = os.read(read_fd, 16).decode()
        os.waitpid(pid, 0)
    os.close(read_fd)

    assert seen == "0", (
        f"child reported {seen} inherited lock descriptors; it must report 0"
    )


# ---------------------------------------------------------------------------
# ask 4 — a transient failure is not a logout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,body,expected", [
    (400, '{"error":"invalid_grant"}', InvalidGrantError),
    (401, '{"error":"invalid_token"}', InvalidGrantError),
    (400, '{"error":"unauthorized_client"}', InvalidGrantError),
    # A bare 401 with no OAuth error code is a proxy or gateway, not the
    # authorization server disowning the credential.
    (401, "<html>407 Proxy Authentication Required</html>", TransientRefreshError),
    (429, "slow down", TransientRefreshError),
    (500, "internal error", TransientRefreshError),
    (502, "bad gateway", TransientRefreshError),
    (None, "Connection reset by peer", TransientRefreshError),
])
def test_only_an_explicit_oauth_error_code_reads_as_a_dead_credential(
    status, body, expected,
) -> None:
    """The asymmetry is deliberate and runs one way.

    Mistaking a dead grant for a transient costs one wasted retry.
    Mistaking a transient for a dead grant costs the user their session —
    which is the same symptom the lock exists to prevent, arriving by a
    different route.
    """
    assert classify_refresh_failure(status, body) is expected


def test_a_transient_failure_leaves_the_stored_credential_untouched(tmp_path: Path) -> None:
    cred = tmp_path / "cred.json"
    write_credential(
        cred, access=FAKE_INITIAL_ACCESS, refresh=FAKE_INITIAL_REFRESH,
        expires_at=time.time() - 10,   # hard expired: nothing to fall back on
    )
    before = cred.read_text()

    def boom(_state):
        raise TransientRefreshError("connection reset")

    with pytest.raises(TransientRefreshError):
        refresh_under_lock(
            credential_path=cred,
            load=lambda: read_credential(cred),
            needs_refresh=lambda s: True,
            still_usable=lambda s: not has_expired(s["expires_at"]),
            refresh=boom,
            save=lambda s: cred.write_text(json.dumps(s)),
        )

    assert cred.read_text() == before, "a failed refresh wrote to the credential file"


def test_a_transient_failure_inside_the_margin_keeps_serving_the_token(
    tmp_path: Path,
) -> None:
    """The margin's real payoff, and the ask-4 case that matters.

    A token inside the early-refresh margin is stale but *still valid*.
    A network blip in that window must not become a re-login prompt —
    the stored token still works, so hand it back.
    """
    cred = tmp_path / "cred.json"
    write_credential(
        cred, access=FAKE_INITIAL_ACCESS, refresh=FAKE_INITIAL_REFRESH,
        expires_at=time.time() + 60,   # stale by margin, not yet expired
    )

    def boom(_state):
        raise TransientRefreshError("connection reset")

    out = refresh_under_lock(
        credential_path=cred,
        load=lambda: read_credential(cred),
        needs_refresh=lambda s: with_margin(s["expires_at"]),
        still_usable=lambda s: not has_expired(s["expires_at"]),
        refresh=boom,
        save=lambda s: cred.write_text(json.dumps(s)),
    )
    assert out is not None
    assert out["access_token"] == FAKE_INITIAL_ACCESS


def test_an_invalid_grant_is_raised_even_inside_the_margin(tmp_path: Path) -> None:
    """A dead credential must not be papered over by the fallback above."""
    cred = tmp_path / "cred.json"
    write_credential(
        cred, access=FAKE_INITIAL_ACCESS, refresh=FAKE_INITIAL_REFRESH,
        expires_at=time.time() + 60,
    )

    def dead(_state):
        raise InvalidGrantError("invalid_grant")

    with pytest.raises(InvalidGrantError):
        refresh_under_lock(
            credential_path=cred,
            load=lambda: read_credential(cred),
            needs_refresh=lambda s: with_margin(s["expires_at"]),
            still_usable=lambda s: not has_expired(s["expires_at"]),
            refresh=dead,
            save=lambda s: cred.write_text(json.dumps(s)),
        )


def test_refresh_errors_stay_runtime_errors() -> None:
    """Existing callers catch ``RuntimeError``; this narrows, not replaces."""
    assert issubclass(TransientRefreshError, RuntimeError)
    assert issubclass(InvalidGrantError, RuntimeError)
    assert issubclass(CredentialLockTimeout, TransientRefreshError)


# ---------------------------------------------------------------------------
# the fast path
# ---------------------------------------------------------------------------

def test_a_fresh_credential_never_touches_the_lock(tmp_path: Path) -> None:
    """The common path must not serialise every request behind a file lock."""
    cred = tmp_path / "cred.json"
    write_credential(
        cred, access=FAKE_INITIAL_ACCESS, refresh=FAKE_INITIAL_REFRESH,
        expires_at=time.time() + 3600,
    )
    with credential_lock(cred):       # someone else is mid-refresh
        out = refresh_under_lock(
            credential_path=cred,
            load=lambda: read_credential(cred),
            needs_refresh=lambda s: with_margin(s["expires_at"]),
            refresh=lambda s: pytest.fail("refreshed a fresh credential"),
            save=lambda s: pytest.fail("saved a fresh credential"),
            timeout=0.05,
        )
    assert out["access_token"] == FAKE_INITIAL_ACCESS


def test_no_stored_credential_is_not_an_error(tmp_path: Path) -> None:
    """"Not logged in" is not this function's problem to raise about."""
    out = refresh_under_lock(
        credential_path=tmp_path / "absent.json",
        load=lambda: None,
        needs_refresh=lambda s: True,
        refresh=lambda s: pytest.fail("refreshed nothing"),
        save=lambda s: pytest.fail("saved nothing"),
    )
    assert out is None


# ---------------------------------------------------------------------------
# the margin (ask 3)
# ---------------------------------------------------------------------------

def test_stale_and_expired_are_different_questions() -> None:
    soon = time.time() + 60
    assert with_margin(soon) is True, "inside the margin, so due for refresh"
    assert has_expired(soon) is False, "but still valid — usable on a failure"


def test_margin_is_configurable(monkeypatch) -> None:
    from jaato_server.shared.credential_lock import REFRESH_MARGIN_ENV, refresh_margin_seconds

    monkeypatch.setenv(REFRESH_MARGIN_ENV, "900")
    assert refresh_margin_seconds() == 900.0
    # Nonsense falls back to the default rather than disabling the bound.
    monkeypatch.setenv(REFRESH_MARGIN_ENV, "not-a-number")
    assert refresh_margin_seconds() == 300.0
    monkeypatch.setenv(REFRESH_MARGIN_ENV, "-5")
    assert refresh_margin_seconds() == 300.0


# ---------------------------------------------------------------------------
# the providers are actually wired to it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_path", [
    "jaato_server.shared.plugins.model_provider.anthropic.oauth",
    "jaato_server.shared.plugins.model_provider.antigravity.oauth",
    "jaato_server.shared.plugins.model_provider.github_models.oauth",
])
def test_every_rotating_provider_reaches_the_shared_mechanism(module_path: str) -> None:
    """One definition, not three copies of it.

    The standing preference in this tree (``shared/completion_nudge.py``,
    ``shared/apparmor_label.py``): a mechanism lives in one module so the
    copies cannot drift.  A provider that grows its own lock should fail
    here.
    """
    import ast
    import importlib

    mod = importlib.import_module(module_path)
    source = Path(mod.__file__).read_text()
    tree = ast.parse(source)

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "jaato_server.shared.credential_lock"
        for alias in node.names
    }
    assert imported, f"{module_path} does not use shared.credential_lock"
    assert imported & {"credential_lock", "refresh_under_lock"}, (
        f"{module_path} imports {sorted(imported)} but takes no lock"
    )
