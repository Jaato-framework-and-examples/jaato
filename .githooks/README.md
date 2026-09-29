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

When a commit touches any non-test `.py` under
`jaato-server/jaato_server/{server,shared}/`, it reruns
`python -m jaato_server.shared.scaffold.authoring_contracts --write` and
stages `jaato-server/jaato_server/shared/scaffold/authoring_snapshot.json`.
That file is what `jaato-scaffold new` reads about providers and env vars
when the jaato-server source tree is absent (#1267); any env-var read or
provider contract can change it.

- The scan takes a few seconds, so commits that touch server source pay it.
- `PYTHONPATH` is pointed at this checkout's `jaato-server/`, so a git
  worktree regenerates its own snapshot, not the main clone's.
- It regenerates from the working tree, not the staged index (as the
  `events.ts` step does), so unstaged edits to source files are reflected.
- The guard `test_authoring_does_not_load_introspection_1267.py` (in the
  `suite (shared/tests)` CI job) remains the authority.
