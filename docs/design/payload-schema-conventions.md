# Payload-Schema Conventions — Authoring Guide

**Audience:** Developers authoring jaato profiles that declare a
`spawn_payload_schema` (input-side) or `completion_payload_schema`
(output-side) — or both.

**Scope:** Conventions every typed payload schema should follow,
the symmetric input/output framing, the failure modes each
convention closes, and how the schemas interact with other
`agent_params`-mediated patterns (notably agent continuity).

**Status:** Output-side conventions ratified server 0.6.27+ after
a non-determinism failure mode in kb-enablement-2.0 codegen
(2026-05-03). Input-side mechanism shipped earlier; conventions
documented here for the first time alongside the
output-side rules to surface the architectural symmetry.
Backward-compatible: schemas predating any of these conventions
continue to work.

---

## 1. The Symmetric Sandwich

Each agent has two boundaries the framework can validate via JSON
Schema:

```
                  ┌─────────────────────┐
   spawn          │                     │       completion
   ─────►         │      Agent          │       ────────►
   agent_params   │                     │       payload
                  └─────────────────────┘
                  ▲                     ▲
                  │                     │
        spawn_payload_schema   completion_payload_schema
        (input boundary)       (output boundary)
```

Both schemas are JSON Schema draft-07, both are profile-declared,
both are validated by the framework via `jsonschema.validate`,
both treat their respective payloads as typed structured data
rather than free-form input/output.

The symmetry is intentional: an agent declares the shape of what
it CONSUMES (spawn_payload_schema validates the dict the caller
passed) and the shape of what it PRODUCES (completion_payload_schema
validates the dict the agent emits via `signal_completion`). Both
boundaries get the same protection: malformed payloads are caught
and surfaced loudly at the boundary, not silently as
mid-execution prefetch failures or downstream cascade
inconsistencies.

### Where each schema lives

| Schema | Path | Resolver |
|---|---|---|
| `spawn_payload_schema`      | `<config_root>/spawn_schemas/<name>.json` (canonical) or `<workspace>/.jaato/spawn_schemas/<name>.json` (workspace-tier) | `shared/spawn_schema_loader.py` |
| `completion_payload_schema` | `<config_root>/completion_schemas/<name>.json` (canonical) or `<workspace>/.jaato/completion_schemas/<name>.json` (workspace-tier) | `shared/completion_schema_loader.py` |

Both resolvers accept either an inline dict OR a string path
(absolute / config_root-relative / workspace-relative / `~/.jaato/`-relative).
Both fall back to a backward-compat auto-prefix path
(`<root>/spawn_schemas/<path>` if the explicit form fails) and log
an INFO-level deprecation hint when the legacy path resolves.

### Where each schema is enforced

| Schema | Enforcement site | Failure mode |
|---|---|---|
| `spawn_payload_schema`      | `server/session_manager.py:~1109` — runs BEFORE session creation | Returns structured error to the caller; session is NOT created. The cascade hasn't happened yet, so retry is cheap. |
| `completion_payload_schema` | Provider-side (Anthropic/OpenAI/Google/Ollama/LM Studio constrain tool-call shape at sampling time) AND server-side via `jsonschema.validate` as defense-in-depth | Returns structured error to the model so it can self-correct on its next turn. Self-correction is non-deterministic — the rules in §4 below exist to limit the resulting drift. |

Boundary asymmetry: spawn validation can fail-loud-and-fast at
the caller; completion validation has to fail-and-let-model-retry,
which is where the determinism conventions matter most.

---

## 2. Background — Typed Payloads in 30 Seconds

When a profile sets `completion_payload_schema`, the framework
rebuilds the `signal_completion` tool's `parameters`:

- Legacy: `signal_completion(summary: str)` — free-form text.
- Typed:  `signal_completion(payload: <YourSchema>)` — structured
  JSON validated server-side.

When a profile sets `spawn_payload_schema`, the framework adds a
pre-creation gate:

- Legacy: `spawn_subagent(profile=<name>, agent_params=<dict>)` —
  any dict accepted; missing required keys surface at runtime.
- Typed:  `spawn_subagent(profile=<name>, agent_params=<dict>)`
  validated against the schema; missing required keys surface at
  the spawn boundary.

The two schemas declare what the agent EXPECTS and what the agent
EMITS. The framework validates both ends.

---

## 3. Conventions — Both Sides

### 3.1 Input side (`spawn_payload_schema`)

#### 3.1.1 Mirror the body-wired prefetch's required keys

If the profile body-wires a prefetch (`{{!py:scripts/prefetch_<x>.py}}`
in the agent's `.md`), the prefetch script reads keys from
`context.agent_params`. **The spawn schema's `required` array
should exactly mirror the keys the prefetch needs.**

This is what closes the field-drop variance class observed in
the 2026-05-01 load test (handoff_test pricing prefetch surfaced
5/10 rejections because upstream `case_intake → auto_underwriter`
was silently dropping fields). The schema makes the contract
explicit: caller can't spawn without the keys; missing-field bugs
fail at the spawn boundary instead of surfacing as deferred
prefetch errors halfway through the cascade.

Concrete example from `handoff_test/.jaato/spawn_schemas/pricing.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Pricing spawn payload",
  "description": "Required agent_params for spawn_subagent(profile='pricing'). Mirrors the required-fields list in .jaato/scripts/prefetch_pricing.py. Validated at the spawn boundary so the upstream model-driven case_data forwarding (case_intake → auto_underwriter) can't silently drop fields and surface as a deferred prefetch error.",
  "type": "object",
  "properties": { /* 16 case-data fields */ },
  "required": [
    "case_id", "tomador_dni", "tomador_codigo_postal", "tomador_profesion",
    "matricula", "marca", "modelo", /* ... 9 more ... */
  ]
}
```

The schema's `description` field names the prefetch script so
maintainers updating the prefetch know to update the schema in
the same change.

#### 3.1.2 Strictness depends on framework integration

Whether to set `additionalProperties: false` on a spawn schema is
the **load-bearing strictness decision**:

| `additionalProperties` | Behavior | When to use |
|---|---|---|
| Not set / `true` | Caller may pass extra keys; framework forwards them all to the agent | When the profile may consume `agent_params`-mediated framework features (continuity, future param-injection patterns); when forward-compat matters more than strict-shape |
| `false` | Caller may pass ONLY the listed properties; extra keys reject the spawn | When you want maximum strict-shape enforcement; when you're confident the listed properties cover all the parameters the agent + framework can ever consume |

Defaulting to `additionalProperties: true` (or omitting the field)
is the **safe default for spawn schemas**. The strict flag adds
brittleness against future framework features that may want to
pipe values through `agent_params` (see §6 on the continuity
interaction).

#### 3.1.3 Document the prefetch coupling

Prefetch scripts that read from `context.agent_params` are
load-bearing consumers of the spawn schema. The schema's
`description` should name the prefetch path; the prefetch's
docstring should reference the schema path. Both sides updated in
the same diff is the canonical contract.

### 3.2 Output side (`completion_payload_schema`)

#### 3.2.1 Always carry `warnings[]` and `errors[]`

**Every** `completion_payload_schema` should include two optional
string-array fields as standard escape hatches:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [/* your load-bearing fields */],
  "properties": {
    /* your data fields */,
    "warnings": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Advisory non-fatal notes the agent surfaced (skip decisions, defaulted values, ambiguities)."
    },
    "errors": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Hard failures the agent recovered from (degraded-mode signals)."
    }
  }
}
```

Both default-empty. Both excluded from canonical hash in
determinism tests (see §5).

The naming asymmetry vs spawn schemas is intentional:
- **Spawn schemas** default to `additionalProperties: true` (forward-compat)
- **Completion schemas** default to `additionalProperties: false` PLUS the `warnings[]`/`errors[]` escape hatches (strict-shape with sanctioned advisory channel)

Why the asymmetry: spawn schemas guard against caller errors at a
point where retry is cheap; completion schemas guard against
agent errors at a point where retry resampling drifts. Strict
output schema + sanctioned advisory fields is the combo that
keeps both load-bearing-fields strict AND advisory text from
being silently dropped.

#### 3.2.2 The failure mode the convention closes

Surfaced 2026-05-03 in kb-enablement-2.0 slice-3 codegen 5x
debugging:

- **Persona** instructs the agent to flag deviations: e.g.
  *"if a template doesn't apply because the entity has no value
  objects, add a warning entry to your completion payload."*
- **Schema** declared `additionalProperties: false` and DID NOT
  carry a `warnings[]` field.
- **Result:** when the agent followed its persona and emitted
  `warnings: ["ValueObject.java.tpl skipped: no domain.valueObjects"]`,
  schema validation rejected the payload.
- **Retry:** the model self-corrected by stripping the warnings
  field and re-emitting. That second emission's content was
  non-deterministic — different runs landed on slightly different
  load-bearing fields.

Two failure modes for the same root cause:

| Failure | Cause |
|---|---|
| **Retry-driven non-determinism** | Schema rejection forces a re-emit; the retry's content varies across runs even when the original load-bearing decision was deterministic. |
| **Silent debug-info loss** | If the agent silently strips advisory prose to fit the schema (instead of retrying), important context for human review disappears. |

Both are eliminated by giving the agent a place to put advisory
text by design.

---

## 4. Persona ↔ Schema Consistency

Whenever you write or edit an agent's `.jaato/agents/<name>.md`
persona, grep its phrasing against BOTH schemas:

| Persona phrase | Implication |
|---|---|
| "you receive `<field>` via agent_params" / "case_data fields are passed in" | Spawn schema must declare `<field>` (in `properties`, in `required` if mandatory). |
| "emit field X" | Completion schema must declare X (in `properties`, ideally in `required`). |
| "include warnings about Y" / "flag any deviation" | Completion schema must have `warnings[]`. |
| "report errors as ..." | Completion schema must have `errors[]` (or equivalent). |
| "include a summary of ..." | Completion schema needs a `summary` string field. |

A persona-schema mismatch is a non-determinism source even when
the rules the agent follows are deterministic. Catch this in
review, not in the 5x byte-identicality test.

---

## 5. Determinism-Test Canonicalization

`warnings` and `errors` are advisory by design. Strip them from the
canonical hash in byte-identicality tests:

```python
canonical = [
    {k: v for k, v in payload.items()
     if k not in ("timestamp", "warnings", "errors")}
    for payload in payloads
]
```

For nested arrays where individual entries carry advisory
sub-fields (e.g. `detected_capabilities[].feature` in
kb-enablement-2.0's discovery schema), strip per-entry too:

```python
for entry in canonical_payload["detected_capabilities"]:
    entry.pop("feature", None)  # advisory, derivable from capability_id
```

The remaining canonicalized payload should be byte-identical
across runs when the structural locks (suppress_base_instructions,
enable_thinking=false, body-wired prefetch, single-tool surface,
deterministic input) are in place.

Spawn schemas don't typically carry advisory fields (the caller's
agent_params is upstream-deterministic by construction), so
canonicalization conventions are an output-side concern.

---

## 6. Interaction with `agent_params`-Mediated Patterns

`agent_params` is shared infrastructure. Two patterns ride on it
today, more may be added later:

### 6.1 Continuity (`continuity_scope`)

Continuity-aware agents (see
[`agent-continuity.md`](./agent-continuity.md)) use
`agent_params["continuity_scope"]` to thread a scope id into the
rendered prompt. The memory plugin's `enrich_prompt` then surfaces
prior memories tagged with that scope.

If a profile declares `spawn_payload_schema` with
`additionalProperties: false` AND does not list `continuity_scope`,
**spawning that profile with `continuity_scope` in agent_params
WILL fail validation**. The framework rejects the spawn because
the schema treats the extra key as a violation.

Three resolution paths:

1. **Drop strictness** — set `additionalProperties: true` (or omit
   the field) on the spawn schema. Forward-compat for any
   `agent_params`-mediated framework feature; downside is the
   schema no longer rejects callers passing typo'd keys.
2. **Declare `continuity_scope` explicitly** as an optional
   property in the spawn schema. Keeps `additionalProperties: false`;
   adds one line per pattern the profile adopts. Strict but
   explicit.
3. **Author non-continuity-aware agents under strict schemas**;
   continuity-aware agents under permissive schemas. Per-profile
   choice; documented expectation.

For new profiles, **option 2 is the recommended default** when
strictness is desirable: explicitly list `continuity_scope` (and
any other framework-level params) as optional properties in the
spawn schema. The schema continues to reject typo'd keys; the
framework features that want to pass values through agent_params
keep working.

### 6.2 Future patterns

Any future framework feature that pipes values through
`agent_params` will face the same coupling. The bitter-lesson
form: a spawn schema with `additionalProperties: false` is a
forward-compat bet; the schema author is asserting that the
listed properties cover everything the agent + framework will
ever consume. That bet is fine when the profile is internal /
short-lived; less fine when the profile may live across multiple
framework versions.

The `spawn_payload_schema` itself is not the right place to
enumerate framework-level params (continuity_scope, etc.) — those
belong to the framework's contract, not the profile's. The
practical accommodation is either permissive `additionalProperties`
or explicit listings as the agent author absorbs new framework
patterns.

---

## 7. When the Standard Conventions Are NOT Enough

Convention covers the 95% case. Edge cases that need their own
schema fields:

### 7.1 Output-side edge cases

- **Structured per-item annotations.** When you need to emit
  warnings *attached to specific items* in an array (e.g.
  "this template was skipped for THIS reason"), add a
  `warnings_by_item: { "<item_id>": { "reason": "...", "code":
  "..." } }` field as a typed companion to the flat `warnings[]`
  array.
- **Enumerated codes vs prose.** When advisory information needs
  to drive automation downstream, replace string `warnings[]` with
  `warning_codes: [{"code": "...", "context": "..."}]` so the
  consumer can dispatch on the code without parsing prose.
- **Multi-severity buckets.** Distinguish `warnings[]` (advisory)
  from `errors[]` (recovered failures) from `notices[]` (purely
  informational). Add buckets as separate optional arrays; don't
  overload `warnings[]` semantics.

### 7.2 Input-side edge cases

- **Discriminated-union inputs.** Profiles that handle multiple
  case-types (e.g. one profile that processes BOTH "new request"
  AND "amendment") may use JSON Schema's `oneOf`/`anyOf` plus
  `discriminator` to validate per-type required-field sets.
  Document the discriminator field's enum values in the schema's
  description.
- **Optional alternates.** When several spelling variants of a
  field name are accepted (legacy + canonical), declare both as
  optional and validate via the prefetch script that exactly one
  is present.

Each of these is a deviation from the standard convention —
DOCUMENT the deviation in the schema's `description` field so
future maintainers know why the schema is shaped differently.

---

## 8. Migration / Audit

### 8.1 Output-side: completion-schema warnings/errors sweep

Existing schemas predating the warnings/errors convention work
fine — but each unpatched schema is a latent non-determinism source
for any agent whose persona surfaces advisory information.

To audit:

```bash
find . -path '*/.jaato/completion_schemas/*.json' \
    -not -path '*/node_modules/*' \
    -not -path '*/kb/*' \
    | while read schema; do
    if grep -q '"additionalProperties": false' "$schema" \
       && ! grep -q '"warnings"' "$schema"; then
        echo "MISSING warnings[]: $schema"
    fi
done
```

Patch shape (additive, low-risk):

```json
{
  "properties": {
    /* existing fields unchanged */,
    "warnings": { "type": "array", "items": { "type": "string" } },
    "errors":   { "type": "array", "items": { "type": "string" } }
  }
}
```

`required` array stays unchanged (the new fields are optional).

### 8.2 Input-side: spawn-schema strictness review

Existing spawn schemas with `additionalProperties: false` may
reject the framework-level params (e.g. `continuity_scope`) that
adopters might want to pass. Audit:

```bash
find . -path '*/.jaato/spawn_schemas/*.json' \
    -not -path '*/node_modules/*' \
    -not -path '*/kb/*' \
    | while read schema; do
    if grep -q '"additionalProperties": false' "$schema"; then
        echo "STRICT — review agent_params-mediated patterns: $schema"
    fi
done
```

For each strict schema, decide: keep strict and explicitly add
optional properties for framework params the profile may adopt,
OR relax to `additionalProperties: true` for forward-compat.

---

## 9. Reference Examples

### 9.1 Output-side (completion_payload_schema)

Schemas in the repo that follow the warnings/errors convention:

- `jaato-based-kb-enablement-2.0/.jaato/completion_schemas/discovery_result.schema.json`
  — declared `warnings[]` from slice 1; mirrors kb's own
  `discovery-agent.md` spec.
- `jaato-based-kb-enablement-2.0/.jaato/completion_schemas/prompt_extraction.schema.json`
  — slice 2.
- `jaato-based-kb-enablement-2.0/.jaato/completion_schemas/step_result.schema.json`
  — slice 3 (added 2026-05-03 after the lesson).
- `handoff_test/.jaato/completion_schemas/*.json` — partial coverage;
  see `project_backlog_completion_schema_warnings_audit.md` for the
  full sweep.

### 9.2 Input-side (spawn_payload_schema)

- `handoff_test/.jaato/spawn_schemas/pricing.json` — case_data
  forwarding contract; required-array mirrors
  `prefetch_pricing.py`. The header docstring names the
  prefetch script and the upstream caller (`case_intake → auto_underwriter`).
- `handoff_test/.jaato/spawn_schemas/kyc_aml.json` — same shape.
- `handoff_test/.jaato/spawn_schemas/antifraude.json` — same shape.

All three are `additionalProperties` open by omission, so
continuity-aware spawning would work without modification.

---

## 10. Related Documentation

- [`agent-continuity.md`](./agent-continuity.md) — the continuity
  pattern that interacts with spawn schemas via `agent_params`
  (see §6.1 above).
- Module docstrings: `shared/lifecycle_tools.py` (signal_completion),
  `shared/spawn_schema_loader.py`, `shared/completion_schema_loader.py`
  — quick reference alongside the actual implementations.
- `docs/design/agent-presentation-awareness.md` — sibling doc on
  how the model adapts output format based on client context.
- Project memory: `feedback_completion_schema_warnings_field`
  (the 2026-05-03 lesson, project-side).
- Project memory: `project_backlog_completion_schema_warnings_audit`
  (sweep-existing-schemas backlog).

---

**Convention version:** 2.0 (server 0.6.27+ for completion-side
warnings/errors; spawn-side mechanism shipped earlier; conventions
unified in this revision).
