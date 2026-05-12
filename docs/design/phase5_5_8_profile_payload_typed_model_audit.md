# Phase 5 §5.8 — `profile_payload` typed model / allow-list audit

**Parent plan:** `per_session_confined_runner_phase5_plan.md` §5.8
(Theme C, sourced from §4.3.9 item 10 in
`phase4_implementation_audits.md`).
**Predecessor:** §4.3 isolated-subagent track (PRs #56-#62), §5.1
isolated-default runtime limits (closed item 1 of the same audit
list).
**Status:** Draft (this commit ships the implementation).

## 1. Problem

`SpawnIsolatedRunnerHandler.handle()` accepts the `profile_payload`
RPC arg as `Dict[str, Any]`.  The only validation is
`isinstance(args["profile_payload"], dict)` — anything is allowed
inside.  Reconstruction goes through
`shared/plugins/subagent/config.py:build_inline_profile`, which is
intentionally permissive (it uses `.get(key, default)` and never
inspects unknown keys).

§4.3.9 item 10 frames this as a hardening gap:

> **`profile_payload` allow-list / typed model.**  Daemon-side
> handler accepts free-form dict; Phase 5 should add an allow-list
> or pydantic model to reject unknown fields.

## 2. Trust model

The runner is in Phase 4/5's threat model **less trusted than the
daemon** — it's the surface model-controlled code runs against
(cli, interactive_shell, mcp), confined by AppArmor + per-session
cgroup.  The daemon is **outside** the confinement: it holds
provider tokens (OAuth, API keys), the cgroup-management
capability, and the AppArmor profile-management capability.

The runner→daemon RPC surface is the **trust boundary**.  Every
RPC handler validates incoming args at the boundary — §3.2
(`PromptOperatorHandler`, `ApparmorFragmentHandler`) and §4.3.2
(`SpawnIsolatedRunnerHandler` itself) all do top-level type-check
+ confused-deputy echo-check.  But `profile_payload` is the one
handler arg that's deliberately structural (it's a profile
serialization) and the field-level checks were deferred to
§5.8.

## 3. Producer-side wire shape (single source of truth)

`shared/plugins/subagent/plugin.py:2283-2325` builds `profile_payload`
from a resolved `SubagentProfile`:

```python
profile_payload: Dict[str, Any] = {
    "name": profile.name,
    "description": profile.description,
    "model": profile.model,                       # Optional[str]
    "provider": profile.provider,                 # Optional[str]
    "plugins": list(profile.plugins),             # List[str]
    "plugin_configs": dict(profile.plugin_configs),  # Dict[str, Dict[str, Any]]
    "system_instructions": profile.system_instructions,  # Optional[str]
    "suppress_base_instructions": profile.suppress_base_instructions,  # bool
    "max_turns": profile.max_turns,               # int
    "env": dict(profile.env),                     # Dict[str, str]
}
# Conditionally:
profile_payload["gc"] = {"type": ..., "config": ...}              # if profile.gc
profile_payload["runtime_limits"] = profile.runtime_limits.to_dict()  # if set
# Plugin annotation:
profile_payload["plugins"] = [f"{n}(preload)" if n in preloaded else n ...]
```

The **producer never emits**: `inherits` (atomic specs;
build_inline_profile ignores it), `completion_payload_schema`,
`spawn_payload_schema`, `completion_artifacts`, `model_tiers`,
`preloaded_plugins` (encoded into `plugins` via `(preload)`).

This list of 12 keys (10 always + 2 conditional) is the wire
contract.  §5.8's allow-list mirrors it.

## 4. Risks of free-form pass-through (concrete)

### 4.1 Silent unknown-key acceptance

A compromised/buggy runner sends `{"name": ..., "ssrf_target":
"http://internal-mgmt-api"}`.  `build_inline_profile` drops the
extra key on the floor — no log, no failure.  The wire shape
becomes implicit (whatever fields downstream code happens to
`.get()`) rather than explicit.

Hardening goal: **the daemon refuses to accept any key it doesn't
recognize.** Makes the wire shape auditable + future-proof.

### 4.2 Type confusion within recognized keys

`build_inline_profile` does `data.get('plugins', [])` and feeds
the result straight into `parse_plugin_list`.  A runner that sends
`{"plugins": [123, {"shell": "rm -rf /"}, "../../etc/passwd"]}`
relies on downstream code to reject the bad shape — but downstream
code at the plugin-load seam may silently misinterpret, log the
payload (PII leak), or crash.

The element-type checks belong **at the trust boundary**, not 4
function calls deep where the failure mode is "downstream plugin
loader sees a dict and tries to call `.startswith()` on it".

### 4.3 Size / DoS surface

No length caps anywhere:

- `system_instructions: "A" * 10**8` → 100MB string round-trips
  through the RPC, gets logged at INFO if anything goes wrong,
  potentially gets included in the system prompt and tokenized
  (tokenizer DoS).
- `env: {f"K{i}": "V" * 10000 for i in range(10000)}` → 100MB env
  dict, deep-copied through profile resolution, applied to
  os.environ (silent failure on most platforms but RSS bloat in
  the daemon).
- `plugins: ["x"] * 10**7` → 10M-entry list iterated by every
  plugin-load loop downstream.

The runner is confined by cgroup memory; the **daemon is not**.
A malformed payload that pivots into a memory pressure attack
on the daemon would defeat that confinement asymmetry.

### 4.4 Nested-shape problems

`gc` and `runtime_limits` are themselves dicts.  `runtime_limits`
gets a structural check inside `build_inline_profile`
(`RuntimeLimits.from_dict` validates), but the failure mode there
is `ValueError` deep in profile reconstruction with a message
like `"unknown field 'X' in RuntimeLimits"` — not "wire-shape
violation at handler boundary".  Diagnostic clarity matters for
incident response.

`gc` carries `{type, config}` only.  Extra keys at this level
should be rejected at the boundary too (defense in depth — the
gc plugin resolver downstream might `.get()` unknown keys).

### 4.5 Cross-field invariants

Producer atomically omits `inherits` (atomic specs).  If a future
buggy/malicious producer sends `inherits=["dangerous_profile"]`,
the daemon silently ignores it — the **silent ignore is the
bug**.  Explicit rejection makes the violation visible.

## 5. Approach (no pydantic dep)

Project policy: no non-stdlib deps in the daemon-side critical
path (audit-driven discipline since Phase 2).  Hand-rolled
validator function, ~100 LoC, with explicit per-key type tuples +
nested-shape checks.

### 5.1 Schema module

Add `server/runner_rpc_handlers/profile_payload_schema.py`
exporting:

```python
# Single source of truth for what keys the daemon accepts.
PROFILE_PAYLOAD_ALLOWED_KEYS: FrozenSet[str] = frozenset({
    "name", "description",
    "model", "provider",
    "plugins", "plugin_configs",
    "system_instructions", "suppress_base_instructions",
    "max_turns", "env",
    "gc", "runtime_limits",
})

def validate_profile_payload(payload: Dict[str, Any]) -> None:
    """Raise ValueError on any shape violation.

    Called by SpawnIsolatedRunnerHandler.handle() before any
    profile-reconstruction work.  The raise propagates to the RPC
    dispatcher which surfaces a typed envelope on the wire.
    """
    ...
```

### 5.2 Per-key rules (full table)

| Key | Required | Type | Constraints |
|-----|----------|------|-------------|
| `name` | yes | non-empty str | len ≤ 256 |
| `description` | no | str-or-None | len ≤ 1024 |
| `model` | no | str-or-None | len ≤ 128 |
| `provider` | no | str-or-None | len ≤ 64 |
| `plugins` | no | list | len ≤ 64; each element str, len ≤ 128 |
| `plugin_configs` | no | dict | keys str (len ≤ 64); values dict; ≤ 64 keys |
| `system_instructions` | no | str-or-None | len ≤ 65536 |
| `suppress_base_instructions` | no | bool | — |
| `max_turns` | no | int | 1 ≤ x ≤ 1000 |
| `env` | no | dict | ≤ 128 entries; keys str (len ≤ 256); values str (len ≤ 4096) |
| `gc` | no | dict | only keys `{type, config}`; `type` str if present; `config` dict if present |
| `runtime_limits` | no | dict | structural pre-check; full validation deferred to `RuntimeLimits.from_dict` |

The caps are deliberately **generous** (10× normal usage) — the
goal is DoS prevention, not field-level policy.  Profiles that
need more should be split or reviewed.

### 5.3 Wire-up in handler

In `SpawnIsolatedRunnerHandler.handle()`, immediately after the
existing top-level checks and BEFORE the routed dispatch:

```python
# §5.8 — typed validation of profile_payload contents.
try:
    validate_profile_payload(args["profile_payload"])
except ValueError as exc:
    raise ValueError(
        f"subagent.spawn_isolated_runner: profile_payload "
        f"validation failed: {exc}"
    ) from exc
```

The wrap re-raises with the RPC-method-name prefix so error
messages match the existing handler convention (every other
ValueError in this handler is prefixed
`"subagent.spawn_isolated_runner: ..."`).

The RPC dispatcher translates `ValueError` to a typed error
envelope (existing contract from §3.2).  No new wire-shape
surface.

### 5.4 What this does NOT do

- **Does NOT** swap the wire shape for a typed dataclass.  The
  payload is and stays a JSON-serializable dict — RPC framing
  doesn't change.  The validator is purely a runtime gate.
- **Does NOT** validate semantic content (e.g., whether `model`
  names a real model, whether `provider` is registered).  Those
  belong further downstream (profile-load seam, provider plugin)
  where the relevant lookup tables exist.
- **Does NOT** change `build_inline_profile`.  That function
  stays permissive — the `session.new` inline-spec path uses it
  with operator-supplied dicts and has a different trust posture
  (operator is more-trusted than runner).  §5.8 only adds a
  validator at the **runner-RPC handler boundary**.
- **Does NOT** add a `pydantic` dependency.  Stdlib only.

## 6. Test plan

Add `server/runner_rpc_handlers/tests/test_profile_payload_schema.py`
with:

1. **happy_path:** payload mirroring the full producer-side shape
   (all 12 keys, gc + runtime_limits dicts) → no exception.
2. **producer_round_trip:** build a real `SubagentProfile`, run
   the producer-side construction (lift from `plugin.py:2283-2325`
   into a test helper), pass to validator → no exception.  This
   pins the producer ↔ validator contract: drift on either side
   breaks the test.
3. **unknown_top_level_key:** payload with `{"name": "x",
   "evil": "value"}` → ValueError mentioning `evil` and the
   word "unknown".
4. **unknown_gc_subkey:** `gc: {"type": "summarize", "evil":
   "x"}` → ValueError mentioning `gc.evil`.
5. **missing_required_name:** name absent → ValueError.
6. **wrong_type_name_int:** `name: 123` → ValueError.
7. **empty_name:** `name: ""` → ValueError.
8. **wrong_type_plugins_str:** `plugins: "cli,web"` → ValueError.
9. **plugin_element_non_str:** `plugins: [123]` → ValueError.
10. **plugin_configs_value_non_dict:** `plugin_configs: {"cli":
    "not a dict"}` → ValueError.
11. **env_value_non_str:** `env: {"K": 42}` → ValueError.
12. **gc_type_non_str:** `gc: {"type": 5}` → ValueError.
13. **runtime_limits_non_dict:** `runtime_limits: 42` → ValueError.
14. **max_turns_out_of_range:** `max_turns: 0` and
    `max_turns: 1_000_000` → ValueError.
15. **system_instructions_too_long:** `system_instructions:
    "A" * 70000` → ValueError mentioning the cap.
16. **plugins_too_many:** 65 entries → ValueError.
17. **env_too_many:** 129 entries → ValueError.
18. **env_value_too_long:** value of 5000 chars → ValueError.
19. **suppress_base_instructions_non_bool:**
    `suppress_base_instructions: "yes"` → ValueError.

And extend
`server/runner_rpc_handlers/tests/test_spawn_isolated_runner.py`
with:

20. **handler_calls_validator:** patch `validate_profile_payload`
    on the module surface; assert the handler invokes it once
    per request with the actual payload arg.
21. **invalid_payload_surfaces_as_value_error:** send a payload
    with an unknown key; assert handler raises ValueError; the
    message starts with `"subagent.spawn_isolated_runner:
    profile_payload validation failed:"`.

## 7. Forward-compatibility

When a future `SubagentProfile` field needs to ride this wire,
**three** files change in lockstep:

1. `shared/plugins/subagent/plugin.py:2283-2325` — producer adds
   the key to the `profile_payload` dict.
2. `shared/plugins/subagent/config.py:build_inline_profile` —
   consumer reads it during reconstruction.
3. `server/runner_rpc_handlers/profile_payload_schema.py` —
   validator allow-list grows by one entry; type rule added.

The validator's role is to make step 3 **mandatory** — a
producer can't quietly start riding new keys past an unupgraded
daemon.  This is the desired posture per the §4.3.9 framing:
"reject unknown fields."

Deployment ordering during a rollout: daemons should be
upgraded BEFORE runners that emit new fields.  This is the
standard daemon-first ordering for confinement systems
(orchestrator before workers).

## 8. Cumulative test count

§5.8 adds **21 new tests** (19 schema-level + 2 handler-integration)
to the §4.3 sub-track's existing ~150-test baseline.

## 9. Out of scope

- Re-using the validator for the `session.new` inline-spec path
  (different trust posture; operator-supplied not runner-supplied).
  Could fold in later if the inline-spec path grows a similar
  hardening requirement.
- Schema generation / introspection for the SDK side
  (`IPCClient.list_profile_schema()` or similar).  Defer until
  there's a concrete consumer.
- Tightening producer-side construction (defensive copies, type
  coercion).  Producer-side bugs surface as validator failures
  with the new validator, which is the desired audit trail.
