# Completion-Payload Schema — Authoring Conventions

**Audience:** Developers authoring jaato profiles that declare a
`completion_payload_schema`.

**Scope:** What every typed completion schema should carry by default,
why those defaults exist, and what to strip from canonical hashes
when running byte-identicality tests.

**Status:** Conventions ratified server 0.6.27+ after a
non-determinism failure mode was diagnosed in the kb-enablement-2.0
codegen pipeline (2026-05-03).  Backward-compatible: schemas
predating this convention continue to work.

---

## 1. Background — Typed Completion in 30 Seconds

When a profile sets `completion_payload_schema` (inline dict or path
under `.jaato/completion_schemas/`), the framework rebuilds the
`signal_completion` tool's `parameters`:

- Legacy: `signal_completion(summary: str)` — free-form text.
- Typed:  `signal_completion(payload: <YourSchema>)` — structured
  JSON validated server-side via `jsonschema.validate`.

Providers that constrain tool-call shape at sampling time (Anthropic,
OpenAI, Google, Ollama, LM Studio) enforce the schema automatically.
The framework re-validates as defense-in-depth.

Validation failure returns a structured error to the model so it can
self-correct on its next turn — but the SELF-CORRECTION path is
non-deterministic (the model may produce different content the second
time), and that's where this document's recommendations come in.

---

## 2. The Convention — Always Carry `warnings[]` and `errors[]`

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

Both default-empty.  Both excluded from canonical hash in
determinism tests (see §5).

---

## 3. Why — The Failure Mode the Convention Closes

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
  field and re-emitting.  That second emission's content was
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
persona, grep its phrasing against the profile's
`completion_payload_schema`:

| Persona phrase | Schema implication |
|---|---|
| "emit field X" | Schema must declare X (in `properties`, ideally in `required`). |
| "include warnings about Y" / "flag any deviation" | Schema must have `warnings[]`. |
| "report errors as ..." | Schema must have `errors[]` (or equivalent). |
| "include a summary of ..." | Schema needs a `summary` string field. |

A persona-schema mismatch is a non-determinism source even when
the rules the agent follows are deterministic.  Catch this in
review, not in the 5x byte-identicality test.

---

## 5. Determinism-Test Canonicalization

`warnings` and `errors` are advisory by design.  Strip them from the
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

---

## 6. When `warnings`/`errors` Are NOT Enough

Convention covers the 95% case.  Edge cases that need their own
schema fields:

- **Structured per-item annotations.**  When you need to emit
  warnings *attached to specific items* in an array (e.g.
  "this template was skipped for THIS reason"), add a
  `warnings_by_item: { "<item_id>": { "reason": "...", "code":
  "..." } }` field as a typed companion to the flat `warnings[]`
  array.  Useful for downstream filtering / reporting.
- **Enumerated codes vs prose.**  When advisory information needs
  to drive automation downstream, replace string `warnings[]` with
  `warning_codes: [{"code": "...", "context": "..."}]` so the
  consumer can dispatch on the code without parsing prose.
- **Multi-severity buckets.**  Distinguish `warnings[]` (advisory)
  from `errors[]` (recovered failures) from `notices[]` (purely
  informational).  Add buckets as separate optional arrays; don't
  overload `warnings[]` semantics.

Each of these is a deviation from the standard convention — DOCUMENT
the deviation in the schema's `description` field so future
maintainers know why this schema is shaped differently.

---

## 7. Migration / Audit

Existing schemas predating this convention work fine — but each
unpatched schema is a latent non-determinism source for any agent
whose persona surfaces advisory information.

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

---

## 8. Reference Examples

Schemas in the repo that already follow the convention:

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

---

## 9. Related Documentation

- Module docstring: `shared/lifecycle_tools.py` — quick reference
  alongside the `signal_completion` tool implementation.
- `docs/design/agent-presentation-awareness.md` — sibling doc on
  how the model adapts output format based on client context.
- Project memory: `feedback_completion_schema_warnings_field`
  (this lesson, project-side).
- Project memory: `project_backlog_completion_schema_warnings_audit`
  (sweep-existing-schemas backlog).

---

**Convention version:** 1.0 (server 0.6.27+).
