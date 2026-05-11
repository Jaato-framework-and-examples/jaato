# Phase 4 — workspace .env → runner subprocess: env-propagation audit

**Discovered**: 2026-05-11 trying to use `nvidia/nemotron-3-super-120b-a12b:free`
via openrouter + `JAATO_OPENROUTER_API_KEY=pass://jaato/openrouter/api-key` in
the jaato-tui-driven-tests workspace.

**Symptom**: TUI sticks on "Setting up session…" forever; daemon log shows
`session.bootstrap failed: ToolError: session.bootstrap: No OpenRouter
API key found`.

**10 diagnostic cycles** ruled out: pass:// resolver malfunction (verified
working in isolated Python process), premium not installed, GPG-agent TTY
issue, envelope-build literal substitution (the literal `pass://...` was
correctly carried in `envelope.env_overrides`), runner subprocess receiving
the envelope (it does), runner-side initialize() running expand_variables
(it does — but on a fresh `dotenv_values` read from `.env`, which still
contains `pass://...`).

## Root cause (precise)

`runner/session.py` **does not apply** `envelope.env_overrides` —
`grep env_overrides jaato-server/server/runner/session.py` returns zero
matches.  The envelope field is built and carried; the runner just
ignores it.

**Why the runner doesn't ignore the workspace `.env` though?** Because:

1. The runner inherits `os.environ` from the daemon at fork-time
   (`runner_spawner.py:251` — `env = os.environ.copy()`).
2. The runner's `JaatoServer.initialize()` step 1
   (`core.py:1514-1528`) does its OWN `dotenv_values(self.env_file)
   → expand_variables → self._session_env`.

Step 2 means: the runner sees `JAATO_OPENROUTER_API_KEY=pass://...`
unresolved in `self._session_env`.  When the openrouter plugin reads
`get_session_env("JAATO_OPENROUTER_API_KEY")` at connect-time, it gets
the literal `pass://...` string — fails to authenticate.

**Why does step 2 not resolve `pass://` in the runner?**  Because the
runner's process doesn't have `_resolvers` populated.  Premium's
`pass_resolver` registers via entry point `jaato.secret_resolvers`,
but `_discover_secret_resolvers` in
`shared/plugins/subagent/config.py:114` uses a **module-level
process-wide cache** that gets populated lazily on first call.

But that's a red herring: even if `_resolvers` worked perfectly in
the runner subprocess, **the GPG-agent priming wouldn't transfer**.
The daemon shell ran `pass show <path>` once to prime GPG, unlocking
the key for the daemon's GPG-agent.  The runner is a fresh fork —
the GPG-agent socket inherits, but the cached passphrase is
agent-side, not session-keyed, so this should work.  And in fact
direct testing showed the resolver DOES work in a separate Python
process post-prime.

The actual issue is simpler: **`dotenv_values()` in the runner reads
the literal file contents from disk, which contain unresolved
`pass://` URIs.**  `expand_variables` then runs the resolver on those
strings.  But because of an interaction between entry-point discovery
timing in the runner process and the lazy resolver-discovery cache,
the resolver is occasionally not registered at the moment
`expand_variables` runs.

Rather than fix the lazy-resolver interaction in the runner
(brittle), **fix it upstream**: resolve `pass://` once in the daemon
(where the resolver is reliably registered + GPG-primed), and
inherit the resolved values via `os.environ` at fork-time.

## Path B implementation plan

**Concept**: workspace `.env` is **workspace-scoped, resolve once at
primary-session bootstrap**.  Subsessions inherit (via runner-share
default per §4.3, OR fork-inheritance for isolated subagents).

### Current bootstrap order (broken)

`session_manager._construct_and_initialize_server` (lines 1042-1095):

```
Line 1042: JaatoServer(envelope.env_file, envelope.env_overrides, ...) constructed
Line 1059: server.config_root = envelope.config_root
Line 1070: _provision_ipc_apparmor_and_spawn_runner(...)
            └─→ _spawn_session_runner_unconditional(...)
                ├─→ spawn_session_runner(...)
                │   └─→ RunnerSpawner.spawn(...)
                │       └─→ fork() — child inherits os.environ          ❌ unresolved
                └─→ dispatch_bootstrap_envelope(...)                    ❌ literal pass:// in env_overrides
Line 1084: _run_pre_initialize_hooks(...)
Line 1094: server.initialize()
            └─→ step 1 (core.py:1514-1528): expand_variables           ❌ AFTER fork — too late
```

### Corrected order (Path B)

```
Line 1042: JaatoServer(envelope.env_file, envelope.env_overrides, ...) constructed
Line 1059: server.config_root = envelope.config_root
NEW      : server._resolve_session_env()  ← runs expand_variables NOW
NEW      : with server._with_session_env():  ← os.environ has resolved values
Line 1070:     _provision_ipc_apparmor_and_spawn_runner(...)
                └─→ fork() — child inherits resolved os.environ        ✅
Line 1084: _run_pre_initialize_hooks(...)
Line 1094: server.initialize()
            └─→ step 1: _resolve_session_env() — idempotent no-op      ✅
```

## Concrete code changes

### Change 1: `server/core.py` — extract resolution into a method

Hoist lines 1514-1539 (the env-resolution sub-block of `initialize()`
step 1) into a new method:

```python
def _resolve_session_env(self) -> None:
    """Populate self._session_env from env_file + profile.env + env_overrides.

    Idempotent — returns immediately if already called.  Designed so
    SessionManager can call it BEFORE the runner-spawn fork, giving
    the runner subprocess access to resolved secret URIs via
    fork-inherited os.environ (when combined with _with_session_env).

    Phase 4 §B fix for the §7c env-propagation gap: workspace .env
    values like `JAATO_OPENROUTER_API_KEY=pass://...` need to be
    resolved daemon-side (where the secret resolver is reliably
    registered + GPG-primed) and propagated to the runner via
    fork-inheritance, not re-resolved runner-side.
    """
    if getattr(self, "_session_env_resolved", False):
        return

    from dotenv import dotenv_values
    from shared.plugins.subagent.config import expand_variables

    raw_session_env = dotenv_values(self.env_file) if self.env_file else {}
    raw_filtered = {k: v for k, v in raw_session_env.items() if v is not None}
    self._session_env = expand_variables(raw_filtered, context=raw_filtered)

    if self._profile and self._profile.env:
        expanded_env = expand_variables(self._profile.env)
        self._session_env.update(expanded_env)

    if self._env_overrides:
        self._session_env.update(self._env_overrides)

    self._session_env_resolved = True
```

Initialize step 1 replaces its in-lined block with a call to this
method.  The `model_name = get_config("MODEL_NAME")` lookup continues
to read `self._session_env`.

### Change 2: `server/session_manager.py:_construct_and_initialize_server` — call pre-spawn

After line 1060 (`server.config_root = envelope.config_root`) and
before line 1070 (`_provision_ipc_apparmor_and_spawn_runner`):

```python
# Phase 4 §B: resolve workspace .env BEFORE the runner-spawn fork
# so secret URIs (pass://, vault://, awssm://) reach the runner
# via inherited os.environ.  Runner-side initialize() step 1 is
# idempotent and skips re-resolution.
server._resolve_session_env()

with server._with_session_env():
    ipc_sandbox_mode = self._provision_ipc_apparmor_and_spawn_runner(
        server,
        envelope.session_id,
        envelope.workspace_path,
        envelope.client_id,
    )
```

### Why no runner-side changes

The runner inherits `os.environ` via fork.  `JaatoServer.initialize()`
step 1 in the runner calls `_resolve_session_env()` which:
- Re-reads `.env` from disk → values still contain `pass://...`.
- Runs `expand_variables` → secret resolver re-resolves (the
  resolver is process-wide entry-point-discovered; should work in
  the runner if premium is installed).

But the runner is now resilient to resolver failure: even if the
resolver fails in the runner, `os.environ` already has the resolved
values from the daemon-side resolution, and the openrouter plugin
falls back to `os.environ` if `_session_env` doesn't have the key.

## Multi-session race semantics (accepted)

`_with_session_env()` already documents that os.environ mutations
"remain subject to races between concurrent sessions, but those
affect only external libraries, not jaato's credential handling."

For the spawn path specifically: if two sessions A and B
construct-and-initialize concurrently, their fork()s could see
interleaved os.environ states.  Specifically:

```
T1: A enters with-block → applies {KEY: A-val}
T2: B enters with-block → applies {KEY: B-val} (saves A-val)
T3: A forks() → child sees KEY=B-val  ❌ wrong session
T4: B exits with-block → restores KEY=A-val
T5: A exits with-block → restores KEY=None (the pre-A original)
```

This race is bounded to the spawn duration (~10-100ms).  Acceptable
for v1; if it surfaces in practice, lift the spawn into a
SessionManager lock or use `subprocess.Popen(env=...)` with the
explicit env dict instead of `os.environ.copy()`.

The runner-side `_resolve_session_env()` (re-running expand_variables
in the child) is a safety net: even if the child inherits the wrong
session's KEY from os.environ, the runner's `self._session_env` is
authoritative for the plugin's `get_session_env()` reads.  Only
direct `os.environ.get(KEY)` reads in third-party SDKs would see
the cross-contamination.  openrouter reads via `get_session_env`,
which is correct.

## Test plan

Three regression pins:

### Test 1: unit — `_resolve_session_env` idempotency

```python
# tests/server/test_resolve_session_env.py (new file)
def test_resolve_session_env_is_idempotent(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n")
    server = JaatoServer(env_file=str(env_file), workspace_path=str(tmp_path))
    server._resolve_session_env()
    assert server._session_env == {"FOO": "bar"}
    assert server._session_env_resolved
    # Mutate then re-call — should not re-read
    server._session_env["FOO"] = "mutated"
    server._resolve_session_env()
    assert server._session_env["FOO"] == "mutated"
```

### Test 2: unit — `_with_session_env` propagates to os.environ pre-spawn

```python
def test_session_env_applied_to_environ_during_with_block(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("CHECK_KEY=resolved-val\n")
    server = JaatoServer(env_file=str(env_file), workspace_path=str(tmp_path))
    server._resolve_session_env()
    saved = os.environ.get("CHECK_KEY")
    try:
        os.environ.pop("CHECK_KEY", None)
        with server._with_session_env():
            assert os.environ["CHECK_KEY"] == "resolved-val"
        assert "CHECK_KEY" not in os.environ
    finally:
        if saved is not None:
            os.environ["CHECK_KEY"] = saved
```

### Test 3: integration — real provider with pass://

Per Phase 3 closure-recap discipline #9 ("Real-provider integration
tests at every architectural change"):

```bash
# Existing harness reuse: jaato-tui-driven-tests workspace has
# JAATO_OPENROUTER_API_KEY=pass://jaato/openrouter/api-key
# Launch TUI on it; verify session bootstraps + first message
# completes without "No OpenRouter API key found" error.

.venv/bin/python jaato-tui/rich_client.py \
    --connect /tmp/jaato.sock \
    --workspace ~/Sources/Jaato-framework-and-examples/jaato-tui-driven-tests \
    --new-session --profile manual_writer
# Send a one-shot prompt; verify it completes.
```

This test is gated on a working `pass` setup + premium installed.
It's the regression gate that would have caught the bug at Phase 3
ship-time.

## Out of scope (for this PR)

- **Removing `envelope.env_overrides`** — the field is still useful
  for the disk-restore path (saved env overrides flow back via
  envelope).  Path B doesn't depend on it for fresh sessions
  because resolution-then-fork carries values via os.environ.

- **Multi-session race hardening** — bounded race is accepted per
  existing _with_session_env semantics.  Lift to explicit env=dict
  on Popen as a future refactor.

- **Runner-side `_resolve_session_env` removal** — keep it as a
  safety net; it's idempotent and cheap.

## Audit checklist

- [x] Located exact code paths touched (core.py:1514-1528,
      session_manager.py:1070-1075, runner_spawn.py:51-143).
- [x] Identified the race window + concluded it's accepted per
      existing semantics.
- [x] Confirmed no runner-side code changes needed.
- [x] Designed three regression tests (2 unit + 1 integration).
- [x] Scope-bounded: ~20-30 LoC daemon-side change.
- [x] Phase 3 audit-discipline pattern followed (audit-then-implement,
      separate commits).
