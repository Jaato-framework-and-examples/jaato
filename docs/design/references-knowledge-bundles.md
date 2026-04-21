# References as Knowledge Bundles

Status: Proposed (2026-04)
Scope: `shared/plugins/references/`, `.jaato/references/` layout, `gen-references` indexer contract.

## Problem

Today the references plugin treats `.jaato/references/` as a flat catalog backed
by two shared artifacts:

- `embedding_config.json` — model / dim / sidecar filename
- `references.embeddings.npy` — a single matrix, one row per reference

Each reference JSON stores its row position in the matrix via
`"embedding": {"index": N, "source_hash": "sha256:…"}`.

Two real workflows break on this shape:

1. **Drop-in a new reference.** Copying `my-new-ref.json` into the directory is
   only half the job — until someone regenerates the sidecar and assigns an
   `index`, the new reference is invisible to semantic matching. The per-ref
   `index` also means inserting/removing a row requires rewriting every other
   reference's JSON to keep indices contiguous.
2. **Merge two independently-curated knowledge sets.** If two teammates each
   built their own `.jaato/references/` with their own model and sidecar,
   there's no unit of "bring your knowledge across" other than a manual rebuild
   of everything on one side.

## Goals

- Dropping a reference JSON into a directory is enough to make it usable
  (tag-matching immediately, semantic matching after a reconcile pass).
- Moving a *set* of references — reference JSONs + their sidecar + their
  embedding_config — from one workspace to another is a directory copy.
- Two people can work on independent knowledge sets and later merge them
  without losing provenance or re-embedding the world.

## Non-goals

- A network-attached reference store. Everything still lives on the local
  filesystem under `.jaato/`.
- A query-time embedding service. Embedding still runs locally via the
  `jaato.embedding` entry-point-provided provider.

## Design

### Three cooperating ideas

1. **Bundles as directories.** `.jaato/references/` is a *directory of
   bundles*. The root directory is one bundle. Each immediate subdirectory
   containing an `embedding_config.json` is another bundle. Each bundle is
   self-contained: its own references, its own sidecar matrix, its own
   model/dim stamp.
2. **Row ordering lives on the bundle.** `embedding_config.json` gains a
   `rows: ["ref-id-a", "ref-id-b", ...]` list — the single source of truth
   mapping matrix row → reference id. Per-reference `embedding.index` goes
   away. Only `embedding.source_hash` stays on the reference (as a staleness
   fingerprint).
3. **Reconcile on load + merge on demand.** At plugin startup the loader
   computes which references are missing, stale, or orphaned per bundle. If
   an embedding provider is available it fixes them in place. A new
   `references merge` user subcommand lets users fold one bundle into
   another.

### Bundle layout

```
.jaato/references/
├── embedding_config.json         ← root bundle manifest
├── references.embeddings.npy     ← root bundle sidecar
├── api-spec.json                 ← reference in root bundle
├── auth-guide.json
└── teammate-kb/                  ← another bundle (subdirectory)
    ├── embedding_config.json
    ├── references.embeddings.npy
    └── deploy-runbook.json
```

Bundle name defaults to the directory name (`teammate-kb`). The root bundle
has the name `""` (empty string) for internal bookkeeping and displays as
`(root)`.

A directory *is* a bundle if it contains an `embedding_config.json`. A
directory without one is just a container for references that participate
in the root bundle (this preserves the current flat layout for users who
never ran the indexer).

### `embedding_config.json` schema (v2)

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dimensions": 384,
  "embedding_sidecar": "references.embeddings.npy",
  "rows": ["api-spec", "auth-guide", "coding-standards"]
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `embedding_model` | yes | sentence-transformers identifier. A change invalidates *all* rows. |
| `embedding_dimensions` | yes | Must match the sidecar matrix's second axis. |
| `embedding_sidecar` | yes | Filename relative to this file. |
| `rows` | yes | Ordered list of reference ids. `len(rows) == matrix.shape[0]`. |

### Reference JSON schema (v2)

```json
{
  "id": "api-spec",
  "name": "API Specification",
  "description": "OpenAPI spec for the REST API",
  "type": "local",
  "path": "./docs/openapi.yaml",
  "mode": "auto",
  "tags": ["api", "endpoints"],
  "embedding": {
    "source_hash": "sha256:a1b2c3…"
  }
}
```

`embedding.index` is removed. Row position is derived from the bundle's
`rows` list. `embedding.source_hash` is optional; when present it is compared
against the reference's current metadata hash to detect staleness.

**Source hash contract.** `source_hash = sha256(name + "\n" + description +
"\n" + ",".join(sorted(tags)) + "\n" + (fetch_hint or ""))`. We hash
metadata, not content, because content can be large, remote, or derived; the
embedding is produced from the same metadata, so this is the right fingerprint
for "does the stored vector still match what this reference declares".

### Load-time reconcile

For each bundle the plugin computes:

- **missing** — references whose id is not in `rows`, or who have no
  `embedding` block at all.
- **stale** — references whose current metadata hash differs from their
  stored `embedding.source_hash`.
- **orphan** — ids in `rows` with no corresponding reference file.

If any set is non-empty and an embedding provider is available, the reconcile
pass:

1. Batch-embeds `missing ∪ stale` via `provider.embed_batch()`.
2. Builds a new matrix by: keeping rows whose id is in `rows ∖ (stale ∪
   orphan)`, appending new rows for `missing ∪ stale`, dropping `orphan`.
3. Writes `<bundle>/references.embeddings.npy.tmp` and
   `<bundle>/embedding_config.json.tmp`, fsyncs, then renames both atomically.
4. Updates each affected reference JSON to carry the new
   `embedding.source_hash`.

If no provider is available the reconcile logs a warning listing the
unindexed ids and returns; tag-based matching continues to work for the whole
catalog, semantic matching just skips those ids.

**Failure mode for one bad embed.** Skip the failed id + log; do not abort
the whole reconcile. Partial progress is preserved by the atomic swap only
running after the whole batch completes.

**Concurrency.** A `<bundle>/references.embeddings.npy.lock` file (advisory
`flock`) guards the reconcile pass so two jaato daemons starting against the
same workspace don't race. Lock is released after the atomic rename.

**Reconcile modes.** A `reconcile` key in `embedding_config.json` selects:

- `"eager"` (default) — run at `initialize()`, blocks until complete.
- `"lazy"` — defer until the first semantic query.
- `"off"` — never reconcile automatically; the operator runs
  `references reconcile` manually.

### Runtime semantic matching across bundles

At query time:

1. The query is embedded once with the active provider.
2. For each loaded bundle whose `embedding_model` matches the provider's
   model, run `matrix @ query_vec` and take the per-bundle top-K.
3. Merge per-bundle candidates into a global top-K by score, tagging each
   hit with its `bundle_name`.
4. Bundles whose model does not match the provider are skipped for semantic
   matching (logged at startup as `SKIPPED (model mismatch)`). They still
   contribute to tag-based matching.

### ID collisions

Ids are unique *within a bundle*. Across bundles, the plugin presents a
colliding id as `<bundle>/<id>` (root bundle ids stay bare). Internal
storage is a `(bundle, id)` tuple. `@<id>` prompt enrichment picks the root
bundle first, then any other bundle whose id matches, in deterministic
sorted-by-bundle-name order — users can always disambiguate with
`@bundle/id`.

### User commands

`references` gains three subcommands alongside the existing
`list | select | unselect | reload | help`:

- **`references bundles`** — one-line-per-bundle status:
  ```
  BUNDLES
    (root)          12 refs  model=all-MiniLM-L6-v2  dim=384  up-to-date
    teammate-kb      7 refs  model=all-MiniLM-L6-v2  dim=384  3 pending
    legacy-kb        3 refs  model=bge-large-en       dim=1024  SKIPPED (model mismatch)
  ```

- **`references reconcile [<bundle>]`** — force a reconcile pass. Without
  an argument, runs against every bundle. Useful after `reconcile: "off"`
  or when the provider became available after startup.

- **`references merge <source> [--into <bundle>] [--on-conflict reject|prefix|newer] [--re-embed] [--dry-run]`**
  — fold one bundle into another.
  - `<source>` is a bundle name currently loaded *or* a directory path.
  - Default `--into` is the root bundle.
  - Safe defaults: same `embedding_model` + `embedding_dimensions`
    required, `reject` on id collision.
  - `--re-embed` re-embeds the source bundle's references with the target
    bundle's model (requires the provider).
  - Output mirrors `reconcile`:
    ```
    Merged 'teammate-kb' into root.
      added:    7 refs (deploy-runbook, incident-playbook, …)
      renamed:  0
      skipped:  0
      sidecar:  references.embeddings.npy  (19 rows, 384d)
    ```

All three subcommands use the same atomic-swap writer path as the startup
reconcile — there is one and only one piece of code that rewrites a bundle's
sidecar.

### Why subcommands, not a separate CLI

The plugin already owns workspace path resolution, provider discovery, the
sidecar lock, and the model-driven catalog refresh (`share_with_model=True`).
A standalone CLI would need to reimplement all of that *and* earn its own
discovery channel. The subcommand approach:

- reuses `get_command_completions()` for tab completion on bundle names,
  conflict strategies, and `--into` targets;
- runs inside the session so the next model turn sees the updated catalog
  without a restart;
- keeps one atomic-write code path and one lock-file convention;
- leaves room for symmetric ops (`references split`, `references export`,
  `references rebuild-index`) in the same namespace.

## Migration plan

The work is split into three PRs. Each leaves the plugin in a shippable
state.

### PR 1 — Schema migration (this change)

- `embedding_config.json` grows a required `rows` list; loader reads it.
- `EmbeddingMetadata` drops `index`, keeps `source_hash`.
- `validate_reference_file` stops requiring `embedding.index`.
- `_init_semantic_matching` builds `index_to_source_id` from
  `config.embedding_rows` instead of per-reference `embedding.index`.
- `.jaato.example/references/embedding_config.json` is updated with the
  new shape and a concrete `rows` array.
- `gen-references` (lives in jaato-premium) must write the new shape from
  this point on.

No bundles yet — the root bundle is still the only bundle. Reconcile is a
no-op. This PR unblocks the index refactor and clears the way for bundle
scan logic.

### PR 2 — Multi-bundle loader + reconcile

- Bundle scanner: find all directories under `.jaato/references/` with an
  `embedding_config.json`.
- Per-bundle `(matrix, rows, model, dim)` in-memory state.
- Reconcile pass with atomic swap + advisory lock.
- `references bundles` and `references reconcile` subcommands.
- Cross-bundle top-K merge in the semantic matcher.

### PR 3 — Merge subcommand

- `references merge` with the flags listed above.
- `--re-embed` path exercising `provider.embed_batch()` against
  metadata-hashed source text.
- Completion support for bundle names and conflict strategies.

## Open questions

1. **Root bundle identity.** We use `""` internally; is there a cleaner
   sentinel? (Tentatively: keep empty string, display as `(root)`.)
2. **Per-bundle reconcile mode.** Each bundle carries its own `reconcile`
   setting. Should the plugin support a workspace-wide override via
   `plugin_configs.references.reconcile`? (Tentatively: yes, but each
   bundle's file wins.)
3. **Orphan handling on merge.** When merging, should orphans in the source
   bundle be silently dropped or surfaced? (Tentatively: drop + log; an
   orphan in a *source* bundle is already broken.)
