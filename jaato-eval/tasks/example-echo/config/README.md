# Read-only task definition

This is the task's `config_root`. In a real task it holds
`profiles/<set>/`, `agents/`, `completion_schemas/` and
`permissions.json`. It is passed to the daemon separately from the
workspace, so the agent under test cannot edit the configuration that
governs it.

**It is deliberately not named `.jaato`.** The manifest's `config_root`
field names the path, so a task is free to choose — and the jaato repo's
root `.gitignore` excludes `.jaato`, which would silently drop a task's
committed configuration from the repository. A task's config root is
data, not a developer's local settings.

The example ships no profiles, so running it needs a `worker` profile
resolvable from the user tier (`~/.jaato/profiles/`) — or point the task
at a real config root.
