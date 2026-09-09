# AppArmor confinement

The framework provisions a kernel-enforced profile per runner subprocess.
`explain profile` carries the profile-level `apparmor` key; this file carries
the invariants behind it.

**No line numbers here, on purpose.** An earlier version of this material
pinned file paths with line ranges "valid as of server 0.6.131"; by 0.7.0 four
of five had moved (one by 500 lines) while every architectural claim still held.
Locate code by symbol — `grep -n 'def resolve_plugin_apparmor_rules'` — and
treat any line number you read anywhere as already wrong.

## Invariants

1. **Framework and plugins are confined to `config_root`; the workspace is
   tenant territory.** The framework writes its meta-state under
   `<config_root>/`. Agent tools write in the workspace as agent-mediated tenant
   operations gated by the permission plugin — not as framework-side grants.
   A plugin's `get_apparmor_rules` tempted to grant workspace writes for
   framework purposes is solving it at the wrong layer.

2. **Each plugin owns its fragment**, via a
   `get_apparmor_rules(*, workspace_path, session_id, config_root,
   plugin_config) -> List[str]` classmethod. The framework template carries the
   baseline; per-plugin rules override it by specificity.

3. **Specificity is per path being matched, not global.** A rule on
   `<cr>/sessions/` beats `<cr>/** r,` for that subtree while the broad rule
   still governs everything else under `<cr>`. Note the empirical finding that
   drove a rewrite of this subsystem: AppArmor does **not** let a more-specific
   *allow* override a less-specific *deny*, so carve-outs must be expressed as
   narrow denies rather than broad deny + specific allow.

4. **The mkdir parent chain must be granted.** To create
   `<cr>/a/b/c/`, each level's directory entry needs a grant. Two rules cover a
   subtree: `<base>/ rw,` and `<base>/** rw,`. A leaf-only rule fails on the
   parent's creation.

5. **The composer needs a populated registry** — it walks `profile.plugins` and
   looks each up to call its classmethod. Preserve registry-creation ordering
   ahead of the composer when touching server init.

## Diagnosing

A confined session failing to write shows up as `PermissionError` /
`EACCES` from inside the runner, often in a plugin's own path. Check, in order:
the composed profile that was written for the session; whether the plugin
exports `get_apparmor_rules` at all; whether the rule covers the parent chain;
and whether a broader deny outranks it.

Tenant-invented paths under `.jaato/` are denied by design, and the deny surface
is actively maintained — writes to `.jaato/templates/` were added to it
recently. Do not park application state there.
