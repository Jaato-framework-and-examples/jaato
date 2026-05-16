# Plugin AppArmor Contribution

Mechanism for plugins to contribute their own AppArmor rules to the per-session
profile, instead of the framework's `PROFILE_TEMPLATE` hardcoding plugin-specific
paths.

## Motivation

Today `jaato-server/server/apparmor.py:PROFILE_TEMPLATE` contains plugin-specific
rules baked into the framework body:

```
@{HOME}/.cache/huggingface/    rw,
@{HOME}/.cache/huggingface/**  rwk,
@{HOME}/.cache/torch/          rw,
@{HOME}/.cache/torch/**        rwk,
```

Those paths exist for the `references` plugin (sentence-transformers / HF
transformers / torch). The framework has no business knowing about them — the
plugin does. The same pattern would repeat for any future plugin that needs
host-tier paths (notebook IPython caches, ML providers using new cache
locations, etc.), each one adding lines to the framework template.

This couples plugin needs to the framework: every plugin's path requirements
become a framework-template change, reviewed by framework maintainers who may
not own the plugin.

## Constraint

The AppArmor profile is rendered ONCE per session, by the daemon, BEFORE the
runner subprocess self-confines. Plugin instances do not yet exist at this
point — they get instantiated inside the runner, post-confinement.

So plugin contribution must happen at the **class level**, via metadata the
daemon can query without instantiating the plugin.

## Design

### Plugin contract

The plugin base class gains an opt-in classmethod:

```python
class ToolPlugin(BasePlugin):
    @classmethod
    def get_apparmor_rules(
        cls,
        *,
        workspace_path: str,
        session_id: str,
        config_root: Optional[str],
        plugin_config: Dict[str, Any],
    ) -> List[str]:
        """Return AppArmor rule lines this plugin needs in the session profile.

        Called daemon-side at profile-render time, BEFORE the runner self-
        confines. Default returns ``[]`` — plugins opt in by overriding.

        Returned strings are spliced into the profile body verbatim. Each
        string is one rule (without trailing newline; the daemon handles
        formatting/indentation). Comments via ``#`` are allowed.
        """
        return []
```

References plugin migration target:

```python
class ReferencesPlugin(ToolPlugin):
    @classmethod
    def get_apparmor_rules(cls, *, workspace_path, session_id, config_root, plugin_config):
        return [
            "@{HOME}/.cache/huggingface/   rw,",
            "@{HOME}/.cache/huggingface/** rwk,",
            "@{HOME}/.cache/torch/         rw,",
            "@{HOME}/.cache/torch/**       rwk,",
        ]
```

### Daemon-side discovery-and-union

`AppArmorManager._render_profile` gains a new step before `.format()`:

1. Resolve the plugin list for the session (from `profile.plugins`, framework
   defaults, premium contributions).
2. For each plugin **class** in that list (loaded from the registry, not
   instantiated), call `cls.get_apparmor_rules(...)` with session context.
3. Concatenate the lists, dedup (string-equal), preserve discovery order.
4. Splice into the template via a new `{plugin_contributed_rules}` placeholder.

```
profile jaato-ws-{session_id} flags=(...) {{
  ... framework baseline ...

  # ---- plugin-contributed rules ----
  {plugin_contributed_rules}

  ... transitions, sub-profiles ...
}}
```

### Plugin loading semantics

All plugins in `profile.plugins` contribute, **regardless of discoverability**.
A `discoverable` plugin can be activated mid-session via `list_tools` /
`get_tool_schemas`; by then the profile is already loaded and immutable, so
the rules must already be there.

This is a deliberate over-grant: a plugin that's listed but never actually
used still gets its rules. The alternative (lazy profile reload) requires
unconfine→reload→reconfine, which is risky and partially unsupported by the
kernel mid-process.

### What stays in the framework template

After full migration, the template body contains only framework-baseline
rules — paths any session needs regardless of which plugins it loads:

- Workspace allow + `.jaato/` deny carve-outs
- Venv + source_root (editable installs)
- /tmp/jaato-{session_id}/*
- /dev/shm/sem.* + /dev/shm/psm_* (Python stdlib multiprocessing)
- /dev memory devices, /proc/self/**, /etc/passwd, /etc/nsswitch.conf
- Network inet/inet6 streams + dgrams
- Capability denies
- Profile transitions

What migrates out: HF cache (references), torch cache (references), refs
include glob (references), IPython caches (notebook, if any), any future
plugin-specific host paths.

## Migration plan

**Phase 0 — framework hook** (~30 LOC):
- Add `get_apparmor_rules` classmethod with default `[]` to `ToolPlugin` base.
- Add daemon-side discovery + union step in `_render_profile`.
- Add `{plugin_contributed_rules}` placeholder to PROFILE_TEMPLATE.
- Bump `_TEMPLATE_VERSION`.
- Tests: framework hook returns empty contribution when no plugin overrides.

**Phase 1 — references plugin migration** (~20 LOC):
- Override `get_apparmor_rules` in `ReferencesPlugin`.
- Remove HF cache + torch cache + refs include glob from framework template.
- Bump `_TEMPLATE_VERSION` again.
- Tests: rendered profile body matches the previous output byte-for-byte
  when references is in the plugin list; lines absent when it isn't.

**Phase 2 — other plugin candidates** (~5-10 LOC each):
- Audit each plugin for currently-framework-template rules that semantically
  belong to it. Likely candidates: notebook (IPython caches), any plugin
  reading from `~/.cache/<vendor>`.
- Migrate one PR per plugin.

**Phase 3 — review what remains in the template**:
- After plugin-specific rules are out, the template should read as a pure
  framework-baseline policy. Anything that still looks plugin-specific is a
  candidate for a later migration or a documentation comment explaining why
  it's framework-tier.

## Test plan

- Unit (`apparmor.py`): mock a fake plugin returning a known rule list;
  verify `_render_profile` splices it correctly.
- Unit: default base-class method returns `[]`; verify no contribution when
  plugin doesn't override.
- Unit: two plugins returning the same rule → dedup'd in output.
- Unit: invalid AppArmor syntax in a plugin's rules surfaces at profile-load
  time (kernel error), not silently — verify error messaging is greppable.
- Integration: full ReferencesPlugin migration produces byte-identical
  rendered profile vs pre-migration baseline (with references in the plugin
  list).
- End-to-end: spawn a session with references plugin loaded; verify the
  rendered profile is accepted by `apparmor_parser` and the runner bootstraps
  successfully.

## Backward compatibility

- Default base-class method returns `[]`, so unmodified plugins contribute
  nothing.
- Framework template retains all current rules through Phase 0 — migration
  is plugin-by-plugin in subsequent phases.
- External plugins (jaato-premium, third-party packages) work unchanged
  until they choose to override the method.

## Security considerations

- A plugin can grant overly broad rules (e.g., `/ rwk,`). But plugins only
  contribute if they're listed in the workspace's profile config
  (`.jaato/profiles/<name>.json`), which is protected by the existing
  `audit deny {workspace_path}/.jaato/profiles/** wlk` rule — a confined
  runner can't inject malicious plugin entries.
- Security review burden moves from "review framework template changes"
  (one file, many reviewers) to "review each plugin's rules" (per-plugin
  files, smaller diffs per change). Per-plugin diffs are easier to reason
  about; the trade-off favors auditability.
- Plugin contributions never override framework denies — AppArmor's
  default-deny-takes-priority semantics still apply.

## Open questions

- **Structured vs raw**: classmethod returns `List[str]` of raw AppArmor
  syntax. Could instead return a typed dict (`{"path": "...", "mode": "rwk"}`)
  the framework translates. Raw is simpler; typed catches more errors at
  Python level but adds API surface. Recommend raw for now; revisit if
  syntax errors become a real problem.
- **Plugin ordering**: discovery order matters for readability but not for
  AppArmor semantics. Preserve discovery order for now.
- **Premium plugins**: jaato-premium plugins use the same entry-point
  mechanism, so they participate naturally. No additional plumbing needed.

## Out of scope

- Lazy profile reload mid-session for late-discovered plugins (rejected:
  kernel doesn't support clean reload in confined process).
- AppArmor fragment files distributed alongside plugin Python (rejected:
  introduces parallel discovery mechanism, can't interpolate workspace
  context cleanly).
- Removing `/dev/shm` from framework template (it's stdlib-baseline, used
  by Python multiprocessing whether or not any plugin needs it).
