# Phase 5 §5.9 — supervisor-declared sub-profile tightening flags

**Parent plan:** `per_session_confined_runner_phase5_plan.md` §5.9
(Theme C, sourced from §4.3.9 item 5 in
`phase4_implementation_audits.md`).
**Predecessor:** §4.3.4 sub-profile template, §5.8 typed
`profile_payload` allow-list (PR #71 — same allow-list pattern).
**Status:** Draft (this commit ships the implementation).

## 1. Problem

The §4.3.4 sub-profile body is **daemon-trusted** — the runner can
request sub-profile creation, but the daemon decides the rules.
The default sub-profile rule-set (apparmor.py:1135-1270 in
`_render_sub_profile`) is conservative enough for most isolated
subagents (workspace allows, integrity write-denies, tool_hat
read-denies, no fragment-admit, no nested profiles), but two
common use cases want **further** tightening:

1. **Scratch-directory subagent.**  Confine the subagent to a
   workspace subpath (e.g., `<workspace>/scratch`) so it cannot
   read or write files outside that subtree.
2. **Read-only auditor subagent.**  Allow workspace reads but
   deny all workspace writes — useful for "investigate this
   bug, don't touch any files" agents.

Both are **additive tightenings** — the supervisor declares
LESS capability than the default sub-profile, never more.
§4.3.9 item 5 frames this:

> **Supervisor-declared sub-profile tightening flags.**  Today
> the daemon decides the sub-profile rules.  Phase 5: let the
> supervisor declare tightenings via wire shape (e.g.,
> `agent_params.isolated_workspace_subpath: "scratch"`) with
> daemon-side allow-list of permitted keys.

## 2. Trust model

Same as §5.8: runner is confined (AppArmor + cgroup), daemon is
not.  Tightenings flow runner → daemon through the
`subagent.spawn_isolated_runner` RPC.  The daemon must:

1. **Allow-list permitted keys** — refuse unknown flags
   (forward-compat: a new producer key requires a daemon-side
   change too; reject-not-ignore posture from §5.8).
2. **Per-key value validation** — reject malformed values that
   would corrupt the rendered profile body (path traversal, glob
   injection, oversized strings).
3. **Tightening direction enforcement** — every supported flag
   must strictly NARROW the sub-profile's capabilities relative
   to the default.  A supervisor cannot grant the subagent
   broader access than the default sub-profile.  This is the
   load-bearing security invariant of §5.9; it's verified by
   pinning the rendered profile body against expected diffs
   (snapshot tests).

## 3. Producer → daemon wire shape

The producer-side `subagent` plugin already forwards `agent_params`
as a free-form dict through the RPC handler.  The handler today
strips one control key (`isolated`) before passing the dict
downstream:

```python
forwarded_agent_params = {
    k: v for k, v in raw_agent_params.items() if k != "isolated"
}
```

§5.9 reuses this control-flag pattern: tightening keys are
prefixed `isolated_` (mirrors the existing `isolated` control
flag), stripped from `agent_params` before downstream forwarding,
and validated as a separate `sub_profile_tightenings` dict that
flows to `provision_sub_profile`.

**v1 keys (this commit):**

| Key | Type | Effect on rendered sub-profile |
|-----|------|-------------------------------|
| `isolated_workspace_subpath` | str | Replace `{workspace}/` allow with `{workspace}/{subpath}/`.  Path traversal + glob injection rejected. |
| `isolated_read_only_workspace` | bool | Downgrade workspace `rwkl` → `r`.  No writes anywhere under the workspace tree. |

Both keys can be combined: `subpath: "scratch" + read_only: true`
yields read-only access to `<workspace>/scratch/**`.

**Future v2+ keys** (out of scope here, follow the same allow-
list grow pattern):

- `isolated_deny_tmp_write` — downgrade `/tmp/jaato-*` write.
- `isolated_deny_memories_write` — downgrade
  `@{HOME}/.jaato/memories/**` write.
- `isolated_deny_premium_read` — drop the `premium_rules` block.

Adding a new flag requires THREE files to change in lockstep
(same forward-compat contract as §5.8): producer (no-op — already
forwards the whole dict), allow-list module (new entry + value
rule), renderer (new conditional block).

## 4. Why `agent_params.isolated_*` and not a nested dict

§4.3.9 item 5 uses `agent_params.isolated_workspace_subpath`
directly (no nesting).  Reasons to keep flat:

1. **Mirrors the existing `isolated` control flag.**  Already-
   established control-flag prefix is `isolated`; tightening
   flags follow the same convention.
2. **One strip-and-validate pass.**  Handler does a single
   "extract isolated_* keys, validate, pass the rest as template
   data" sweep.  Two-level nesting would need a separate strip
   for the nested dict.
3. **Caller ergonomics.**  Subagent specs in profile JSON write
   `"agent_params": {"isolated_workspace_subpath": "scratch"}` —
   one less indirection.

Trade-off: namespace pollution in `agent_params`.  Acceptable
because the `isolated_*` prefix is already reserved for control
flags and the allow-list module documents the full set.

## 5. Implementation surface (~250 LoC + ~300 LoC tests)

### 5.1 New module `sub_profile_tightenings_schema.py`

`server/runner_rpc_handlers/sub_profile_tightenings_schema.py`
mirrors the §5.8 pattern:

```python
SUB_PROFILE_TIGHTENING_KEYS: FrozenSet[str] = frozenset({
    "isolated_workspace_subpath",
    "isolated_read_only_workspace",
})

def extract_and_validate_tightenings(
    agent_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a dict containing ONLY the validated tightening
    keys.  Raises ValueError on any shape violation.  Leaves
    non-tightening keys (template data) in the caller's dict
    untouched — caller is responsible for the strip step."""
    ...
```

Per-key value rules:

- `isolated_workspace_subpath`:
  - Must be `str`.
  - Non-empty.
  - Length ≤ 256.
  - No path traversal: rejects `..` segments, absolute paths
    (`/abc`), Windows drive (`C:`).
  - No glob/wildcards: rejects `*`, `?`, `[`, `{`.
  - No leading/trailing slash.
  - No AppArmor-quote-breaking characters: `"`, newline, NUL.
- `isolated_read_only_workspace`:
  - Must be `bool` (explicit reject of int/str-truthy).

### 5.2 `AppArmorManager.provision_sub_profile` signature change

Add an optional `tightenings: Optional[Dict[str, Any]] = None`
kwarg.  Default `None` preserves backward compatibility — every
existing test call site continues to work unchanged.  When
non-None, threads through to `_render_sub_profile`.

### 5.3 `_render_sub_profile` extension

Take the same `tightenings` dict.  Modify the workspace allow
block based on the flags:

```python
ws_path = workspace_path
subpath = tightenings.get("isolated_workspace_subpath")
if subpath:
    ws_root = f"{workspace_path}/{subpath}"
else:
    ws_root = workspace_path

read_only = bool(tightenings.get("isolated_read_only_workspace"))
ws_perms = "r," if read_only else "rwkl,"
ws_dir_perms = "r," if read_only else "rw,"

workspace_block = f"""
  {ws_root}/   {ws_dir_perms}
  {ws_root}/** {ws_perms}
"""
```

NOTE on the integrity-deny block: the existing rules already
write-deny `{workspace}/.jaato/**` subtrees.  When subpath is
`scratch`, the deny rules at the broader `{workspace}/.jaato/**`
path remain (no effect inside `{workspace}/scratch/`).  If the
supervisor sets subpath to a path UNDER `.jaato/` itself, the
deny rules still apply and the subagent gets a useless profile —
that's the supervisor's bug, not ours; the validator's job is
narrow (reject malformed values), not policy (reject useless
combos).

### 5.4 `SessionManager._spawn_isolated_runner` plumbing

Pass `tightenings` through to `provision_sub_profile`.  Already
takes `agent_params` for forwarding to the new sub-runner; add a
sibling `sub_profile_tightenings` kwarg that the handler extracts
+ validates upstream.

### 5.5 `SpawnIsolatedRunnerHandler.handle` extension

After `validate_profile_payload` (§5.8 gate), before the
forwarded-agent-params strip:

```python
try:
    tightenings = extract_and_validate_tightenings(
        args.get("agent_params"),
    )
except ValueError as exc:
    raise ValueError(
        f"subagent.spawn_isolated_runner: "
        f"sub-profile tightening validation failed: {exc}"
    ) from exc

# Strip BOTH the existing 'isolated' flag AND every tightening
# key from forwarded_agent_params — tightenings are
# daemon-side control flags, not template data.
forwarded_agent_params = {
    k: v for k, v in raw_agent_params.items()
    if k != "isolated"
    and k not in SUB_PROFILE_TIGHTENING_KEYS
}
```

Pass `tightenings` to `_spawn_isolated_runner(...)` as a new
kwarg.

## 6. Test plan

### 6.1 Schema-level tests (new file
`tests/test_sub_profile_tightenings_schema.py`)

1. **Happy path no tightenings** — `None` / empty dict / dict
   with only non-tightening keys → returns `{}`.
2. **Happy path workspace subpath** — `{"isolated_workspace_subpath":
   "scratch"}` → returned dict has that key.
3. **Happy path read-only** — `{"isolated_read_only_workspace":
   True}` → returned dict has that key.
4. **Both flags together** — both keys validate independently.
5. **Path traversal rejection** — `".."`, `"a/../b"`, `"/abs"`.
6. **Glob rejection** — `"*"`, `"a*"`, `"a/[bc]"`.
7. **Empty / oversized subpath** — `""`, 257-char string.
8. **Forbidden chars** — `"a\"b"`, `"a\nb"`, `"a\x00b"`.
9. **Wrong-type subpath** — `123`, `True`, `None`.
10. **Wrong-type read-only** — `"yes"`, `1`, `0`.
11. **Tightening keys are stripped from caller's dict?  NO** —
    schema module returns a NEW dict; caller is responsible for
    the strip.  Test pins this contract.

### 6.2 Render-output snapshot tests
(extend `tests/test_apparmor.py` or sibling test file)

12. **No tightenings → body unchanged** — baseline.
13. **Workspace subpath narrows the allow rule** — assert the
    rendered body contains `{workspace}/scratch/   rw,` and
    `{workspace}/scratch/** rwkl,` (and does NOT contain the
    broader `{workspace}/   rw,` allow).
14. **Read-only → `rwkl` replaced with `r`** — assert no `rwkl`
    in the workspace allow block.
15. **Combined: subpath + read-only** — `{workspace}/scratch/
    r,` and `{workspace}/scratch/** r,` (no rwkl, no rw, scoped
    to subpath).
16. **Profile name unchanged** — tightenings don't affect the
    `jaato-ws-{parent}//{subagent}` name.
17. **Integrity-deny block unchanged** — `.jaato/**` denies
    persist regardless of tightenings.
18. **DROP comments still present** — fragment-admit drop,
    tool_hat drop, change_profile drop comments remain (these
    are §4.3.4 / v15 / §5.10e invariants the tightenings
    must not erode).

### 6.3 Handler integration

19. **`agent_params.isolated_workspace_subpath` is extracted** —
    assert the helper receives the tightening dict; forwarded
    agent_params has the key STRIPPED.
20. **Validation failure surfaces as ValueError** — bad value
    propagates as `subagent.spawn_isolated_runner:
    sub-profile tightening validation failed: ...`.
21. **Validator runs BEFORE provision** — invalid tightening
    never reaches `AppArmorManager.provision_sub_profile`.

### 6.4 Provisioning integration

22. **Tightenings flow into provision_sub_profile** — patched
    AppArmorManager records the kwargs; assert the tightening
    dict matches.

## 7. Out of scope (deferred)

- v2+ tightenings (deny_tmp_write, deny_memories_write,
  deny_premium_read).  Land incrementally as concrete use cases
  arise.
- Loosening flags (e.g., re-enable add_reference_fragment).
  §4.3.9 item 7 is a separate audit item — has its own opt-in
  surface to design.
- Producer-side ergonomics (profile-file syntax for declaring
  tightenings at spec time vs at spawn time).  Today the
  supervisor passes them through `agent_params` at the
  per-spawn API boundary — that's enough for v1.

## 8. Forward-compat contract

When a new tightening flag lands:

1. Add the key to `SUB_PROFILE_TIGHTENING_KEYS`.
2. Add the per-key validator rule in
   `extract_and_validate_tightenings`.
3. Add the rendering branch in `_render_sub_profile`.
4. Add (3) snapshot tests pinning the rendered effect.

The handler does not change — it routes through the schema +
provision API generically.

## 9. Cumulative test count

§5.9 adds **~22 new tests** (11 schema + 7 render + 4
integration).  §4.3 sub-track now sits at ~190 net new tests
(baseline 150 + §5.1b/§5.3/§5.8/§5.9 increments).
