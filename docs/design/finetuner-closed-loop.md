# Fine-Tuner Closed Loop: Reliability Rule Patching

## Status

Design — implementation pending.

## Background

The **fine-tuner** agent today operates as an *assessor*: it consumes the OpenTelemetry stream of another running session, identifies gaps and misbehaviors, forks that session's conversation (read-only) to interrogate the model about its decisions, and produces a written assessment. It already extracts the input assets used by the analysed session — primarily its **profile** and **prompt** — and links findings back to them.

The role stops at "here is what went wrong and why." Translating findings into preventative configuration is left to the human.

## Goal

Evolve the fine-tuner from *assessor* to *true fine-tuner* by giving it the capability to **propose and apply reliability rules** to the analysed session's profile, so the same failure modes are prevented on the next run of that profile.

Reliability rules are the right intervention surface because:
- They are declarative, scoped, and side-effect-free until a tool call hits them.
- They map cleanly to behavioural failure patterns (retry loops, missing prerequisites, read-only stalls, repetitive calls) — exactly the categories the fine-tuner already detects from telemetry.
- They live in the profile, so they travel with the agent definition rather than being lost as ad-hoc instructions.

## Required Pieces

The closed loop has three components. The first is a precondition; the second is the new capability; the third is the validation step (already on the backlog as *fork replay*).

### 1. Profile-driven reliability config (precondition)

Today the reliability plugin reads its policies from a **single JSON file**:

1. `<workspace>/.jaato/reliability-policies.json`
2. `~/.jaato/reliability-policies.json`

It does **not** consult `plugin_configs.reliability` from the active profile. This means a patched profile would have no effect — patches would have to land in the workspace JSON file, which is global to the workspace and cannot vary per profile. That defeats the purpose of profiles.

**Required change:** the reliability plugin must accept policies passed via its `initialize(config)` call (the standard mechanism by which `plugin_configs.<plugin>` reaches a plugin), and merge them with the file-based config. Precedence:

```
profile plugin_configs.reliability  >  workspace JSON  >  user JSON  >  built-in defaults
```

Profile-supplied entries that share a `policy_id` with file-supplied ones replace them; otherwise they are additive. `pattern_detection` fields from the profile override matching fields from files.

Implementation surface:
- `shared/plugins/reliability/plugin.py::ReliabilityPlugin.initialize()` — read `config.get("pattern_detection")` and `config.get("prerequisite_policies")`, normalise via the existing `policy_config` parsers, and apply them after `load_file_policies()`.
- `shared/plugins/reliability/policy_config.py` — expose a pure parsing entry point that takes a dict (not a file path) so the same code path validates both sources.
- `get_config_schema()` on the plugin — declare the two top-level keys so introspection tools (and the fine-tuner) can discover the schema.

### 2. Profile patcher (the new capability)

A small, focused module that:

1. Loads a profile by name (using the existing `discover_profiles()` machinery, but operating on the **unresolved** on-disk file rather than the post-inheritance flattened object — patches must persist to the file the user actually edits).
2. Accepts a *patch object*: a partial `plugin_configs.reliability` dict containing one or more new prerequisite policies and/or pattern-detection overrides.
3. Validates the patch against the reliability schema **before** touching the file. Invalid → return errors, no write.
4. Merges the patch into the on-disk JSON, preserving formatting where reasonable (2-space indent, trailing newline, key order at top level).
5. Ensures `"reliability"` is present in the profile's `plugins` list (adds it if absent).
6. Returns a unified diff of the change so the caller (or a human approver) can see exactly what will be written.
7. Writes atomically (temp file + rename) only when the caller confirms.

Location: `shared/plugins/subagent/profile_patcher.py` — co-located with `config.py` since it operates on the same on-disk format. Deliberately *not* a method on `SubagentProfile`, because that dataclass represents the resolved (post-inheritance) form and is the wrong identity for a write-back operation.

**API sketch:**

```python
def propose_reliability_patch(
    profile_name: str,
    patch: dict,
    *,
    workspace_path: Path,
) -> ReliabilityPatchProposal:
    """Build a proposal without writing. Returns diff + validation result."""

def apply_reliability_patch(
    proposal: ReliabilityPatchProposal,
) -> None:
    """Persist a previously-built proposal atomically."""
```

`ReliabilityPatchProposal` carries: source path, original JSON, patched JSON, unified diff string, validation errors (empty on success), and a flag indicating whether `"reliability"` was added to the plugins list.

### 3. Fork replay (validation — separate backlog item)

Already tracked as `project_backlog_fork_replay.md`. Once available, the closed loop becomes:

```
1. detect failure pattern from telemetry
2. propose reliability patch (step 2 above)
3. fork-replay the failing turn against the patched profile
4. if the failure no longer reproduces → apply patch
   if it still reproduces  → discard patch, escalate to human
```

Until fork-replay lands, step 3 is replaced by "human reviews diff and confirms" — the patcher already returns a diff, so this is a usable interim flow.

## Tool surface for the fine-tuner

The fine-tuner needs two tools (both `discoverability="discoverable"`, neither auto-approved):

| Tool | Purpose |
|------|---------|
| `propose_reliability_patch` | Build a proposal for a target profile. Returns proposal id + diff. No filesystem write. |
| `apply_reliability_patch` | Apply a previously-built proposal by id. Permission-gated. |

Splitting propose/apply lets the fine-tuner present its reasoning + diff to the user (or to a fork-replay validator) before any disk mutation.

These tools belong in a new `shared/plugins/profile_patcher/` plugin rather than being grafted onto an existing plugin — it gives them a clean home, isolates the permission surface, and keeps `subagent/` focused on profile *resolution* rather than *mutation*.

## What this design deliberately does *not* do

- **No automatic patch application without confirmation.** Even with fork-replay validation, the default flow surfaces a diff. Silent mutation of a profile the human authored is too high-blast-radius.
- **No mutation of resolved profiles in memory.** Patches always go to the on-disk source file. In-memory mutation would be invisible after restart and would fight the inheritance system.
- **No "smart" merging of prerequisite policies with the same `policy_id`.** Replacement, not field-level merge — predictability beats cleverness here.
- **No new schema for the patch payload.** It is a literal subset of the existing `plugin_configs.reliability` schema, so the fine-tuner can author it from the same documentation a human would use (`docs/reliability-policies-config.md`).
- **No fallback heuristics.** If validation fails, the patcher returns errors and writes nothing. Per project policy, no defensive fallbacks.

## Open questions

1. Should the patcher also touch `agents/` markdown files (the new home for system instructions) when a failure suggests an instruction-level fix rather than a rule-level one? *Probably out of scope for v1 — instruction fixes are higher-judgement and harder to validate via replay.*
2. Should patches be journaled (append-only history of what the fine-tuner has done to a profile)? *Useful but not blocking; can be layered on later via the existing waypoint/ledger infrastructure.*
3. How should conflicts between fine-tuner-applied rules and human-authored rules in the same profile be surfaced? *v1: last-write-wins with the diff making the conflict visible. v2: per-rule provenance comments.*

## Related

- [Reliability Policies Configuration Guide](../reliability-policies-config.md)
- [Reliability Plugin Design](../reliability-plugin-design.md)
- Backlog: fork replay (memory: `project_backlog_fork_replay.md`)
- Backlog: conversation fork (memory: `project_backlog_conversation_fork.md`)
