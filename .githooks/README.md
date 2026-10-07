# Git hooks

Version-controlled git hooks for this repo. Activate them once per clone:

```bash
git config core.hooksPath .githooks
```

(Run it from the repo root. This is per-clone local config, not committed —
each contributor opts in once.)

## `pre-commit`

Regenerates two generated files when a commit touches their sources, and
stages them into the same commit.

### `events.ts`

Keeps the generated TypeScript SDK events mirror in sync. When a commit
touches `jaato-sdk/jaato_sdk/events.py`, it regenerates
`jaato-sdk-ts/src/events.ts` (via `scripts/codegen_ts_events.py`) and stages
it — so the committed mirror is never stale and the **"Codegen: jaato-sdk-ts
events.ts staleness"** CI check (`.github/workflows/codegen-ts-events.yml`)
stays green.

The CI `--check` gate remains the authority; this hook just spares you a
red build + a follow-up "regenerate events.ts" commit.

- Needs `jaato-sdk` importable by the interpreter. Override it if it's in a
  venv: `PYTHON=.venv/bin/python git commit ...`
- Bypass for a one-off: `git commit --no-verify`.

### `authoring_snapshot.json`

When a staged path is a non-test `.py` file under
`jaato-server/jaato_server/server/` or `jaato-server/jaato_server/shared/`, or
is `jaato-server/pyproject.toml`, the hook runs
`python -m jaato_server.shared.scaffold.authoring_contracts --write` and
stages `jaato-sdk/jaato_sdk/scaffold/authoring_snapshot.json`.
That file ships in the jaato-sdk wheel and is what `jaato-scaffold new` reads
about providers and env vars when jaato-server is not installed beside it
(#1267).  It records the jaato-server version it was projected from, which is
why a version bump in `jaato-server/pyproject.toml` regenerates it too.
`test_authoring_does_not_load_introspection_1267.py` fails in CI when it is
stale.

- Needs both packages importable (`pip install -e jaato-sdk/ -e jaato-server/`).
- The scan takes a few seconds, so commits that touch server source pay it.
- Puts this checkout's `jaato-server/` and `jaato-sdk/` first on `PYTHONPATH`,
  so a git worktree regenerates its own snapshot, not the main clone's.
- It regenerates from the working tree, not the staged index (as the
  `events.ts` step does), so unstaged edits to source files are reflected.
- The guard `test_authoring_does_not_load_introspection_1267.py` (in the
  `suite (shared/tests)` CI job) remains the authority.

### `explain_snapshot.json`

When a staged path is a non-test `.py` file under `jaato-server/jaato_server/`
or `jaato-sdk/jaato_sdk/`, or is `jaato-server/pyproject.toml`, the hook runs
`python -m jaato_server.shared.scaffold.explain_snapshot --write` and stages
`jaato-sdk/jaato_sdk/scaffold/explain_snapshot.json`.  That file ships in the
jaato-sdk wheel and is what `jaato-scaffold explain` answers from when
jaato-server is not installed: every built-in topic jaato-server renders
without a workspace or the network.
`test_explain_snapshot_mirrors_the_server.py` fails in CI when it is stale.

- Needs both packages importable, with jaato-server's `[interactive]` extra:
  generation refuses an environment where an in-tree plugin did not load,
  and one holding a plugin or `explain` topic from another distribution.
- Renders in a subprocess with an empty `$HOME` and working directory, so
  nothing of yours (skills under `~/.claude`, a `.env`) reaches the file.
- Takes about 10 seconds.
