# Example: an agent that uses `gh` / `git` with a per-user token

Reference workspace for issue #1228 —
[docs/design/per-user-github-credentials.md](../../../docs/design/per-user-github-credentials.md)
§7. Once `GH_TOKEN` reaches a session, three things have to line up for `gh`
and `git` to work; this example shows all three, and
`jaato-scaffold explain gh` describes them.

- **`.jaato/profiles/gh-worker.yaml`**
  - the **scrub exemption** — `plugin_configs.{cli,interactive_shell}.scrub_secret_env: [default, "!GH_TOKEN"]`. `GH_TOKEN` is in the default `scrub_secret_env` set (#863), so without this it is stripped from every model-driven subprocess before `gh` runs. MCP keeps `default` (an MCP server gets no GitHub token).
  - the **non-interactive defaults** — `GH_PROMPT_DISABLED=1`, `GIT_TERMINAL_PROMPT=0` — so a missing credential is an error the model reads rather than a prompt nobody answers.
  - `GH_TOKEN=app://github` — the reference the daemon resolves to the real per-user token at every spawn (#1226). In a real web deployment this line usually lives in the workspace `.env`; it is in the profile here so the example is self-contained and `jaato-scaffold validate`-able.
- **`.jaato/agents/gh-worker.md`** — the persona: use `gh`; never print or echo the token; on a 401, report it rather than running `gh auth login`.

Drop the `!GH_TOKEN` exemption from either surface and
`jaato-scaffold validate` reports **`gh_token_scrubbed_inert`** (warning):
the config is valid but the token is stripped before `gh` sees it.
