# Git hooks

Version-controlled git hooks for this repo. Activate them once per clone:

```bash
git config core.hooksPath .githooks
```

(Run it from the repo root. This is per-clone local config, not committed —
each contributor opts in once.)

## `pre-commit`

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
