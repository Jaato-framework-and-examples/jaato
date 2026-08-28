# Read-only task definition

This is the task's `config_root` — the conventional `.jaato/` layout, so
every jaato tool that already knows how to read a workspace can read a
task too (`jaato-scaffold validate`, `jaato-doctor`, `jaato-scaffold
explain sets`). Holding it anywhere else buys nothing and costs all of
that.

It holds `profiles/<set>/`, and in a larger task also `agents/`,
`completion_schemas/` and `permissions.json`. It is passed to the daemon
separately from the workspace, so the agent under test cannot edit the
configuration that governs it.

**The repo's root `.gitignore` excludes `.jaato`** — a developer's local
settings. A task's config root is not that: it is the read-only half of
the task definition and has to travel with the repository. The exclusion
is therefore negated for `jaato-eval/tasks/*/.jaato/` (both the directory
and its contents — git will not descend into an excluded directory, so
un-ignoring only the contents silently does nothing).

## The profile set

`profiles/_base_worker.yaml` is provider-agnostic and holds the parts of
the stage that must not vary across arms; `profiles/<set>/worker.yaml`
binds a provider and model. `JAATO_PROFILE_SET` selects the set, and the
engine writes it into each arm's workspace `.env` — which is how the
sweep's model axis reaches profile discovery without mutating the task.

Adding a second model to the sweep is one new directory beside
`openrouter_gpt5mini/`, not a change to this task.
