# Shipping GitHub Working Guidance into the Workspaces It Binds

**Status:** proposed. This follows the per-user-credentials epic
(#1225–#1228, all merged) and answers issue #1240. Nothing here is built
yet; the implementation is a follow-up. It is **application work, not daemon
work** — everything lives in `jaato-web-coder-server` (the BFF) and
`jaato-web-coder-ui`, and reaches a workspace as files. No `jaato-server/`
change, no framework instruction piece, no framework `github` plugin.

## 1. The problem

#1225–#1228 give a web session its user's GitHub token: the BFF holds the
grant and binds an account to a workspace, the workspace `.env` carries
`GH_TOKEN=app://github`, and the daemon resolves that reference to a
short-lived token at every spawn. They do not tell the model how to **use**
that access safely in a workspace other sessions may share.

The guidance that would tell it exists — the `gh-worker` profile and persona
under `jaato-web-coder-server/examples/gh/` (#1228) — but it reaches a session
only if a human copies it into a workspace by hand. A bound workspace opened
from the web gets the token and none of the rules.

## 2. Scope discipline: application, not daemon

The issue's central rule: **nothing that only matters to one application goes
into daemon scope.** GitHub is the web coder's concern. The daemon resolves an
opaque `app://github` reference (#1226) and knows nothing about GitHub, git,
worktrees or PRs. So this design proposes:

- **no** framework instruction piece (the `disclosure` / `security` layers are
  the framework's; a `github` layer would not be);
- **no** framework `github` plugin;
- **no** GitHub-aware code under `jaato-server/`.

The guidance is **content**, and the application already puts content into a
workspace two ways: the BFF writes `<ws>/.env` and `<ws>/.home/.gitconfig` at
bind time (`GitHubService.bind`, verified in `src/github.ts`), and the UI
stages files with `StageFilesRequest`, which accepts any workspace-relative
path without `..`, `.jaato/` included (`app/staging.ts`). Either route reaches
the workspace with **no daemon change**.

## 3. The rule-set (`.jaato/instructions/40-github.md`)

The issue's 13 candidate rules, refined and grouped, each with a one-line
rationale and cross-checked against what the epic actually delivers. This is
the content of the shipped instruction file.

### Isolation — a workspace can be shared between sessions

| # | Rule | Rationale / epic backing |
|---|------|--------------------------|
| 1 | **Work in your own git worktree, never in a shared checkout.** Layout: one shared clone per repo at `repos/<owner>/<repo>`, one worktree per session at `worktrees/<session_id>/<repo>`, both inside the workspace. | Two sessions in one clone overwrite each other's branch, index and working tree. Both paths sit inside the workspace so path containment (#710/#722) allows them. The session id comes from `get_environment(aspect="session")` → `session_id` (**verified**, `environment/plugin.py`). |
| 2 | **Every `cli` command starts at the workspace root.** Use `git -C <worktree>` or `cd <worktree> && …`. | A `cd` does not persist between `cli` calls (each is a fresh `subprocess.run`). |
| 3 | **Name branches per session**, e.g. `jaato/<session_id>/<topic>`. | Two sessions never push the same branch. |
| 4 | **Expect contention on the shared clone.** A concurrent `git fetch` can fail on a `*.lock`; retry it, never delete the lock. | Several worktrees share one `.git`; deleting a lock corrupts the other session's operation. |
| 5 | **Remove your worktree when done** (`git worktree remove`), or say you are leaving it for a follow-up. | A stale worktree is disk, not danger; `workspace.delete` reclaims it. Cleanup is best-effort (see Q2). |

### Tooling

| # | Rule | Rationale / epic backing |
|---|------|--------------------------|
| 6 | **Always use `gh`** for GitHub operations (issues, PRs, reviews, releases), and `gh api` for anything without a command. No `curl` with the token, no MCP server for GitHub. | The daemon's `app://` grant keeps `GH_TOKEN` for `cli`/`interactive_shell` only; MCP is never granted, so an MCP server gets no token anyway (#1228). |
| 7 | **Use `git` over HTTPS**; `gh` supplies the credential. No SSH keys, no remote URLs with the token in them. | `<ws>/.home/.gitconfig` seeds `helper = !gh auth git-credential` at bind (**verified**, `renderGitConfig` in `src/github.ts`), so `git` over HTTPS picks the token up with no URL rewriting. |

### The credential

| # | Rule | Rationale / epic backing |
|---|------|--------------------------|
| 8 | **Never print, echo or log the token.** No `env`, no `echo $GH_TOKEN`, never in a URL or a command's output. | The `app://` grant deliberately keeps `GH_TOKEN` **in** the `cli` subprocess environment, so `echo $GH_TOKEN` genuinely reveals it — the `/proc/environ` denials (#712) do not cover the model asking a subprocess to print its own env. This prose rule is the last line, and is load-bearing. |
| 9 | **On a 401, report it and stop.** No `gh auth login`. | The token is delivered per spawn by the platform; nothing on disk can fix it. `GH_PROMPT_DISABLED=1` / `GIT_TERMINAL_PROMPT=0` (the shipped profile) make a missing credential an error the model reads rather than a prompt nobody answers. |

### Acting on the user's behalf — outward-facing, often irreversible

| # | Rule | Rationale / epic backing |
|---|------|--------------------------|
| 10 | **Draft PRs by default**; the user merges. | The reversible default for an action other people see. |
| 11 | **Never force-push**, and never rewrite a branch you did not create. | *Enforceable*, not merely stated — a permission blacklist on `git push --force` / `-f` / `--force-with-lease` (see Q4). |
| 12 | **Confirm before anything other people see or that cannot be undone:** merging, closing, deleting a branch or release, commenting on someone else's issue or PR. | Judgement the model exercises per situation; partly enforceable, partly prose (see Q4). |
| 13 | **Only touch repositories the user named.** The token may reach more repositories than the task needs. | A GitHub App token's reach is the intersection of the user and the App's installations (#1225 §4); that can be more than one repo. |

Every rule is satisfiable with what the epic delivers. None assumes a
capability that is not there — the two that could (the session id for rule 1,
the credential helper for rule 7) are verified present.

## 4. The file set

| File | Purpose | Core? |
|------|---------|-------|
| `.jaato/instructions/40-github.md` | the rules above | **core** |
| `.jaato/profiles/gh-worker.yaml` (or an `inherits` fragment) | `GH_PROMPT_DISABLED=1` / `GIT_TERMINAL_PROMPT=0`, and the force-push permission blacklist | optional |
| `.jaato/bin/gh-worktree` | a shell helper the model runs through `cli` to open/close its worktree with the agreed layout and branch name — rules 1–5 as one command instead of prose | optional |

The split is not cosmetic. The **instruction file reaches every non-suppressing
session in the workspace** regardless of which profile it runs under (it is the
disk instruction layer). The **profile pieces reach only sessions that run under
that profile** — the non-interactive env and the blacklist
are profile config, and the application cannot force them into a profile the
user picked. So the instruction file is the one piece that is always correct to
ship; the profile is an opt-in the deployment (or the `gh` example) selects.

`GH_TOKEN=app://github` is **not** part of this set — the BFF already writes it
to `.env` at bind (#1227). The guidance rides alongside the binding, not inside
it.

## 5. The write / refresh lifecycle

The issue leans toward: *write at bind time, refresh on session start,
application-managed files with a header + version marker, overwrite our own
file but never one the user replaced (a user deletes the marker to keep their
own)*. Evaluated against the three triggers the issue tables:

| Trigger | For | Against |
|---------|-----|---------|
| **bind time** (BFF) | only bound workspaces get it; shares the binding's lifecycle and the containment/atomic-write machinery `bind()` already uses | needs a filesystem write, which the BFF has only when `workspaceRoot` is configured (single-host) |
| **session start** (UI) | always current — refreshes the managed file when the shipped version differs | one write per session start; only reaches sessions the UI opens |
| **workspace template** | simple | goes stale, only reaches new workspaces, appears where there is no GitHub |

**Verdict: endorse the leaning, with one sharpening the epic's own §6 forces.**
The epic's whole argument for `app://github` is that the token resolves at
**every** spawn — cascade stage, `session.wake`, a revived session — not only
browser-opened ones. Guidance staged **only** by the UI at session start would
leave every non-browser session with a credential and no rules. So the
**bind-time filesystem write is the primary**, because a file on disk in
`.jaato/instructions/` is read by every session in that workspace exactly as
the token is resolved for every session. The UI's session-start staging is the
**refresh** and the **split-host fallback** (where the BFF has no
`workspaceRoot` and already defers the `.env` write to the browser's
`config.update`).

**One source of the bytes.** To keep two writers from drifting, the file
content (bytes + header + version) is defined **once, in the BFF**. The
bind-time path writes it to disk directly; the UI-refresh path fetches it from
a BFF endpoint (e.g. `GET /api/github/guidance`) and stages it. Neither invents
content.

**The managed-file marker.** The first line is a machine-readable marker, e.g.:

```
<!-- jaato-managed: github-guidance v1 — delete this line to keep your own edits -->
```

On write: file absent → write; present with the marker and a **different**
version → overwrite; present with the marker and the **same** version → no-op;
present **without** the marker → the user replaced it, so **skip** and record a
note (never a silent skip — `bind()` already returns a `note` for a skipped
`.env` write, and this joins it). On bind-to-none, the managed file is removed
the way `GH_TOKEN` is removed from `.env`.

## 6. Two load-order traps, stated as constraints

Both are properties of how the daemon loads instructions; the design respects
them rather than changing them.

- **Content must be written into the workspace, not the daemon host.** The
  workspace and user instruction tiers are first-match-wins: a workspace's own
  `.jaato/instructions/` hides `~/.jaato/instructions/`. Writing the guidance
  to the daemon's home would be shadowed by any workspace that has its own
  instructions dir, and would appear in workspaces that have no GitHub. All
  writes go to `<ws>/.jaato/instructions/40-github.md`.
- **The shipped profile must not set `suppress_base_instructions: {disk: true}`.**
  That drops the disk instruction layer, taking the guidance with it. The
  optional `gh-worker` profile this design ships leaves the disk layer on.

## 7. The five open questions

### Q1 — BFF or UI does the writing?

**The BFF, at bind time, as the primary writer; the UI as refresh + split-host
fallback.** The hook is `GitHubService.bind` in `src/github.ts`, beside the
existing `_writeEnv` and `_seedGitConfig` calls — the guidance file shares the
binding's lifecycle (written on bind-to-account, removed on bind-to-none), its
containment (`_resolveWorkspace`, symlinks resolved) and its atomic write
(`atomicWrite`). The decisive reason over UI-only: a file on disk is read by
cascade / wake / revive sessions, which the UI never opens — matching the
token's own every-spawn reach (epic §6). The UI keeps a narrower role: staging
the BFF-served bytes to refresh on session start, and to cover the split-host
deployment where the BFF has no `workspaceRoot` (the same case where the `.env`
write already defers to the browser). Trade-off: two delivery moments, but one
source of bytes (§5) removes the drift risk.

### Q2 — worktree cleanup ownership?

**The helper plus prose; not an application-side sweep.** `.jaato/bin/gh-worktree`
gets a `close` subcommand (`git worktree remove` with the session's layout), so
rule 5 is one command; the persona tells the model to run it or say it is
leaving the worktree. An application-side sweep at session end is rejected: the
BFF/UI has no hook that fires for every session end (cascade, wake, revive,
orphan), a stale worktree is reclaimable disk rather than a leaked credential,
and the model may legitimately leave a worktree for a follow-up session —
forcing removal would destroy in-progress work. Cleanup is therefore
best-effort, which is the right posture for disk.

### Q3 — commit author in `.env`?

**No.** The BFF already seeds `<ws>/.home/.gitconfig` with `[user] name` +
noreply email at bind (**verified**, `renderGitConfig`), and `#1225`'s
`.home` makes that the subprocess `HOME`, so `git` reads it. Setting
`GIT_AUTHOR_NAME` / `GIT_AUTHOR_EMAIL` in `.env` would be a **second,
competing** source of identity — and those env vars **override** gitconfig, so
the two could disagree with the env pair winning silently. Keep the single
source: the gitconfig seeded at bind. The rule-set must **not** tell the model
to set author flags (they would shadow the gitconfig). A deployment wanting a
different author edits the gitconfig, not a second mechanism.

### Q4 — confirmation before outward-facing actions?

**Both, with the enforce/advise split made explicit.**

- **Rule 11 (never force-push) is enforced** by a permission **blacklist** on
  `git push --force` / `git push -f` / `git push --force-with-lease` in the
  shipped profile's `plugin_configs.permission.policy.blacklist.patterns`. A
  blacklist is a hard stop and works under a headless `ClientType.API` driver
  too.
- **Rules 10 / 12 / 13 are prose** — they are judgement, not command patterns.
  "Commenting on someone else's issue" depends on **whose** issue, which is not
  in the command string; "only repos the user named" is context. An enumerated
  pattern cannot express them.
- **A small set of unambiguous irreversible verbs get `ask` entries** —
  `gh pr merge`, `gh release delete`, `gh repo delete`. Stated cost: an `ask`
  under a headless driver has no channel to reach (the #950/#951 finding), so
  `ask` helps interactive deployments only; the force-push **blacklist** is
  what protects the headless case.

### Q5 — generalize to GitLab?

**Keep the content GitHub-shaped now; factor only the one seam that is
genuinely shared — an "application-managed instruction files" mechanism — so
GitLab is a data addition rather than a re-architecture.** The shared,
expensive-to-retrofit part is thin: (a) a managed-file writer with the
header + version + user-override logic (§5), and (b) the bind-time hook. Build
that generically. Everything else — the 13 rules, the `gh` helper, the
force-push blacklist — is GitHub-specific
content and should not be abstracted (a forge differs in MR vs PR, `glab` vs
`gh`). When GitLab lands (`app://gitlab`, `GL_TOKEN`), it registers its own
instruction file and profile fragment through the same mechanism. Trade-off:
abstracting the *rules* now would be speculative with only one forge in
existence; scoping the generality to the file-management mechanism captures the
reuse without the guesswork.

## 8. Phased implementation plan

A follow-up issue/PR can execute this; no code is written here.

**Phase 1 — BFF content + bind-time write.**
- Add `jaato-web-coder-server/src/managed-files.ts`: the managed-file writer
  (header + version + override detection + atomic write), reusing the
  containment/`atomicWrite` helpers from `github.ts`.
- Hold the `40-github.md` bytes (and, later, the helper) as a BFF asset with a
  version constant.
- Call the writer from `GitHubService.bind` in `src/github.ts`, beside
  `_writeEnv` / `_seedGitConfig`; write on bind-to-account, remove on
  bind-to-none; extend `BindResult.note` for a skipped/overwrote-user-file case.
- Tests (`jaato-web-coder-server/test/`): bind writes the file; a user-replaced
  file (marker deleted) is left alone; bind-to-none removes it; a version bump
  overwrites; a path outside `workspaceRoot` is skipped with a note.

**Phase 2 — UI refresh + split-host fallback.**
- BFF: `GET /api/github/guidance` serving `{version, files: [{path, bytes}]}`
  for the signed-in user (wired in `src/routes.ts` beside `handleGitHub`).
- UI: on `session.info` for a GitHub-bound workspace, fetch the guidance and
  stage any file whose version differs, via the existing `stageFiles` path in
  `app/staging.ts` (`stageQueued` / the `getClient().stageFiles("", …)` call).
- Tests: an e2e case that a bound session receives `40-github.md`; a mock BFF
  serving the endpoint.

**Phase 3 — profile + helper.**
- Ship `.jaato/bin/gh-worktree` (open/close) and the force-push blacklist +
  `ask` entries in the optional `gh-worker` profile (evolving
  `examples/gh/.jaato/profiles/gh-worker.yaml`).
- No scrub exemption is needed: the daemon grants the names it resolved from
  `app://` to `cli` and `interactive_shell` (#1228), so `GH_TOKEN=app://github`
  reaches `gh` in a profile-less workspace.

## 9. Honesty — what was verified, and what a reviewer should check

**Verified by reading the tree on this branch:**

- `GitHubService.bind` writes `<ws>/.env` (`_writeEnv` → `upsertEnvLine`) and
  seeds `<ws>/.home/.gitconfig` (`_seedGitConfig` → `renderGitConfig`), with
  containment via `_resolveWorkspace` (symlinks resolved on both sides, path
  strictly under `workspaceRoot`) and `atomicWrite` (temp-file + rename, mode
  0600). The guidance write has a real, correct home here. — `src/github.ts`.
- `renderGitConfig` already seeds `[user] name/email` **and**
  `helper = !gh auth git-credential`, so commit identity has one source
  (Q3's basis) and rule 7's credential helper exists. — `src/github.ts`.
- `get_environment(aspect="session")` returns `session_id` (and
  `env_session_id` from `JAATO_SESSION_ID`), so rule 1's per-session worktree
  naming has a real source. — `jaato_server/shared/plugins/environment/plugin.py:507–546`.
- The UI stages via `getClient().stageFiles("", payloads)` with
  workspace-relative paths, no `..` (`checkSizes` / `stagedName`), one request
  per batch, serialised. `.jaato/`-relative paths are within that contract. —
  `app/staging.ts`, `protocol/attachments.ts`.
- The `gh-worker` profile + persona exist and reach a session only by hand-copy
  (its README says so). An `app://` GH_TOKEN needs no scrub exemption there,
  because the daemon grants the names it resolved. — `examples/gh/`.

**Assumptions a reviewer should confirm before implementing:**

- **HOME → gitconfig wiring (#1225).** I read the `.gitconfig` *seed* but did
  not trace, on this branch, that `#1225`'s `.home` actually sets
  `HOME=<ws>/.home` for the `cli`/`interactive_shell` subprocesses (so `git`
  reads that gitconfig). The epic §8 asserts it; confirm the plumbing landed.
- **`$JAATO_SESSION_ID` inside a `cli` subprocess.** `JAATO_SESSION_ID` is
  `INTERNAL`-scoped ("the framework telling the environment plugin which
  session it is in") and I did **not** confirm it is present in the model-driven
  subprocess environment. The verified route to the session id is
  `get_environment(aspect="session")`, so the `gh-worktree` helper should take
  the session id as an argument the model supplies rather than reading
  `$JAATO_SESSION_ID` from the shell.
- **The instruction-loader tier.** That `<ws>/.jaato/instructions/*.md` is the
  disk layer and first-match-wins over `~/.jaato/instructions/` is asserted by
  the issue and CLAUDE.md; I did not trace the loader itself.
- **Staging overwrite semantics.** Whether the daemon's `StageFilesRequest`
  handler overwrites an existing workspace file or refuses it — I read the
  client side (`app/staging.ts`) but not the daemon-side `_write_staged_payload`
  overwrite behaviour. This matters for the Phase-2 refresh, which re-stages
  `40-github.md`; if staging refuses to overwrite, the refresh needs the BFF
  bind-time path or a different verb.
- **Force-push blacklist matching.** Whether the permission plugin's blacklist
  patterns reliably match `git push --force` / `-f` / `--force-with-lease`
  given the command-containment analyzer — not verified. Confirm the pattern
  form before relying on rule 11 being enforced rather than advised.
