<!-- jaato-managed: github-guidance v1 — delete this line to keep your own edits -->
# Working with GitHub in this workspace

This workspace is bound to a GitHub account. A per-user token reaches your
environment as `GH_TOKEN` at every turn; `gh` and `git` pick it up
automatically, so you never configure or log in. The workspace may be shared
between sessions running at the same time, so the rules below keep your work
from colliding with theirs and keep actions taken on the user's behalf safe.

## Isolation — a workspace can be shared between sessions

- **Work in your own git worktree, never in a shared checkout.** Layout: one
  shared clone per repo at `repos/<owner>/<repo>`, one worktree per session
  at `worktrees/<session_id>/<repo>`, both inside the workspace. Two sessions
  in one clone overwrite each other's branch, index and working tree. Your
  `<session_id>` comes from `get_environment(aspect="session")` (its
  `session_id` field). The `.jaato/bin/gh-worktree` helper does this layout
  in one command — `.jaato/bin/gh-worktree open <session_id> <owner>/<repo>`
  clones-or-reuses the shared clone and adds your worktree.
- **Every `cli` command starts at the workspace root.** Use
  `git -C <worktree>` or `cd <worktree> && …`. A `cd` does not persist
  between commands — each runs in a fresh shell.
- **Name branches per session**, e.g. `jaato/<session_id>/<topic>`, so two
  sessions never push the same branch. (`gh-worktree open` creates one for
  you.)
- **Expect contention on the shared clone.** A concurrent `git fetch` can
  fail on a `*.lock` file; retry it, and never delete the lock — several
  worktrees share one `.git`, and deleting a lock corrupts another session's
  operation.
- **Remove your worktree when done** (`git worktree remove`, or
  `.jaato/bin/gh-worktree close <session_id> <repo>`), or say you are leaving
  it for a follow-up. A stale worktree is disk, not danger.

## Tooling

- **Always use `gh`** for GitHub operations (issues, PRs, reviews, releases),
  and `gh api` for anything without a command. No `curl` with the token, no
  MCP server for GitHub.
- **Use `git` over HTTPS**; `gh` supplies the credential. No SSH keys, no
  remote URLs with the token in them.

## The credential

- **Never print, echo or log the token.** No `env`, no `echo $GH_TOKEN`,
  never in a URL or a command's output. Refer to it only implicitly, by letting
  `gh` / `git` read it from the environment.
- **On a 401, report it and stop.** No `gh auth login`. The token is
  delivered by the platform per turn; nothing on disk can fix it. Say clearly
  that the GitHub credential was rejected or missing, and stop that line of
  work.

## Acting on the user's behalf — outward-facing, often irreversible

- **Draft PRs by default**; the user merges. It is the reversible default for
  an action other people see.
- **Never force-push**, and never rewrite a branch you did not create.
- **Confirm before anything other people see or that cannot be undone:**
  merging, closing, deleting a branch or release, commenting on someone else's
  issue or PR.
- **Only touch repositories the user named.** The token may reach more
  repositories than the task needs.
