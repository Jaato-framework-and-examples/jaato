# Example: an agent that uses `gh` / `git` with a per-user token

Reference workspace for issue #1228 —
[docs/design/per-user-github-credentials.md](../../../docs/design/per-user-github-credentials.md)
§7. Once `GH_TOKEN` reaches a session, a few things have to line up for `gh`
and `git` to work; this example shows them, and
`jaato-scaffold explain gh` describes them.

- **`.jaato/profiles/gh-worker.yaml`**
  - **no scrub exemption**. `GH_TOKEN` is in the default `scrub_secret_env` set (#863), but a name the daemon resolved from `app://` is granted to `cli` and `interactive_shell` and kept through the scrub. A *literal* token is not granted and needs `plugin_configs.{cli,interactive_shell}.scrub_secret_env: [default, "!GH_TOKEN"]`. MCP is never granted (an MCP server gets no GitHub token).
  - the **non-interactive defaults** — `GH_PROMPT_DISABLED=1`, `GIT_TERMINAL_PROMPT=0` — so a missing credential is an error the model reads rather than a prompt nobody answers.
  - `GH_TOKEN=app://github` — the reference the daemon resolves to the real per-user token at every spawn (#1226). In a real web deployment this line usually lives in the workspace `.env`; it is in the profile here so the example is self-contained and `jaato-scaffold validate`-able.
- **`.jaato/agents/gh-worker.md`** — the persona: use `gh`; never print or echo the token; on a 401, report it rather than running `gh auth login`.
- **`.jaato/bin/gh-worktree`** — a shell helper the model runs through `cli`
  to open/close its own git worktree with the shared-clone /
  per-session-worktree / per-session-branch layout (rules 1–5 of the guidance
  file, below). It takes the session id as an argument — the model gets it from
  `get_environment(aspect="session")` — because `$JAATO_SESSION_ID` is not
  reliably present in a `cli` subprocess.
- **the force-push blacklist** — `plugin_configs.permission.policy.blacklist.patterns`
  hard-denies `git push --force` / `-f` / `--force-with-lease`, and the three
  irreversible verbs `gh pr merge` / `gh release delete` / `gh repo delete`
  (rules 10–12). See the profile comments for the enforce/advise split.

## The working guidance is shipped automatically for a bound workspace

In the web coder (`jaato-web-coder-server`), when a signed-in user binds a
GitHub account to a workspace the BFF writes
**`.jaato/instructions/40-github.md`** into that workspace beside the `.env`
and `.gitconfig` it already seeds. That file is the 13-rule guidance for using
`gh` / `git` safely in a workspace other sessions may share (isolation via
worktrees, tooling, the credential, acting on the user's behalf). It is the
disk instruction layer, so **every** session in the workspace reads it —
cascade, `session.wake`, a revived session — matching the token's own
every-spawn reach. See
[docs/design/github-workspace-guidance.md](../../../docs/design/github-workspace-guidance.md).

The guidance file carries a `<!-- jaato-managed: github-guidance v… -->` marker
on its first line: a bind refreshes an unedited copy when the shipped version
changes, and leaves a copy you made your own (marker deleted) untouched. This
example workspace ships that same file by hand so it is self-contained.
